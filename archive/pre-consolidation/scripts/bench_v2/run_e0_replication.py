#!/usr/bin/env python
"""E0 — provenance replication of the committed 2-seed REAL-LLM
joint-search run (protocol §10). ZERO new API calls: every candidate in
the committed `outputs/free_amplitude_fixed_readout_v1/candidate_trace.csv`
(operations + verbatim theta) is re-evaluated offline on the regenerated
legacy dataset, and the recomputed numbers are compared against the
committed per-candidate metrics, the per-arm selections, and the
cross-seed summary. Purpose: verify provenance, not produce new evidence.

Writes outputs/bench_v2/E0/replication_report.json (+ COMPLETE.json on
success) and exits non-zero on any mismatch.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from llm_vqc.free_amplitude.candidate_schema import (  # noqa: E402
    CompleteCandidateProposal,
    complete_candidate_to_ir_and_theta,
)
from llm_vqc.free_amplitude.main_eval import (  # noqa: E402
    evaluate_on_test_fixed_theta,
)
from llm_vqc.free_amplitude.metrics import samplewise_rmse  # noqa: E402
from llm_vqc.free_amplitude.model import FixedReadoutQuantumModel  # noqa: E402
from llm_vqc.tasks.signal_suite import get_legacy_gauss_peak_task  # noqa: E402

SOURCE = REPO / "outputs" / "free_amplitude_fixed_readout_v1"
OUT_DIR = REPO / "outputs" / "bench_v2" / "E0"
VAL_TOLERANCE = 1e-9      # float64 recomputation of committed values
SUMMARY_TOLERANCE = 1e-9
N_QUBITS = 3
READOUT = 0


def _val_rmse_for(proposal: CompleteCandidateProposal, features, targets) -> float:
    ir, theta = complete_candidate_to_ir_and_theta(proposal, READOUT)
    model = FixedReadoutQuantumModel(ir, readout_qubit=READOUT)
    model.double()
    with torch.no_grad():
        model.q_layer.weights.copy_(torch.tensor(theta, dtype=torch.float64))
        pred = model(torch.tensor(features, dtype=torch.float64)).cpu().numpy().reshape(-1)
    return samplewise_rmse(pred, targets)


def main() -> int:
    rows = list(csv.DictReader((SOURCE / "candidate_trace.csv").open()))
    summary = json.loads((SOURCE / "cross_seed_summary.json").read_text())
    seeds = sorted({int(r["seed"]) for r in rows})
    task = get_legacy_gauss_peak_task()

    report: dict = {
        "source": str(SOURCE.relative_to(REPO)),
        "api_calls_made": 0,
        "n_trace_rows": len(rows),
        "seeds": seeds,
        "per_candidate_mismatches": [],
        "per_seed": {},
        "summary_checks": [],
    }
    checked = 0

    for seed in seeds:
        train_val, _ = task.build(seed)
        test_split, _ = task.build_test(seed)
        selections: dict[str, tuple[float, dict]] = {}

        for row in rows:
            if int(row["seed"]) != seed or row["outcome"] != "evaluated":
                continue
            operations = json.loads(row["operations"])
            proposal = CompleteCandidateProposal(n_qubits=N_QUBITS, operations=operations)
            recomputed = _val_rmse_for(
                proposal, train_val.val.features, train_val.val.targets
            )
            committed = float(row["val_rmse"])
            checked += 1
            if abs(recomputed - committed) > VAL_TOLERANCE:
                report["per_candidate_mismatches"].append({
                    "proposal_id": row["proposal_id"],
                    "committed_val_rmse": committed,
                    "recomputed_val_rmse": recomputed,
                })
            arm = row["arm"]
            best = selections.get(arm)
            if (
                not row["is_duplicate"] == "True"
                and (best is None or recomputed < best[0])
            ):
                selections[arm] = (recomputed, {"proposal": proposal, "row": row})

        per_seed = {}
        for arm, (val_rmse, info) in selections.items():
            test_metrics = evaluate_on_test_fixed_theta(
                info["proposal"], READOUT, test_split
            )
            per_seed[arm] = {
                "selected_candidate_hash": info["row"]["candidate_hash"],
                "selected_val_rmse": val_rmse,
                "recomputed_test_rmse": test_metrics["test_rmse"],
            }
        report["per_seed"][str(seed)] = per_seed

    # Cross-seed summary comparison: with n=2 and the committed
    # mean/std convention, {ci95_low, ci95_high} are exactly the two
    # per-seed values for both validation and test.
    for arm, key in (
        ("random", "random"), ("llm_open_loop", "llm_open_loop"),
        ("llm_closed_loop", "llm_closed_loop"),
    ):
        committed_val = {
            round(summary[key]["selected_validation_rmse"]["ci95_low"], 9),
            round(summary[key]["selected_validation_rmse"]["ci95_high"], 9),
        }
        committed_test = {
            round(summary[key]["selected_test_rmse"]["ci95_low"], 9),
            round(summary[key]["selected_test_rmse"]["ci95_high"], 9),
        }
        recomputed_val = {
            round(report["per_seed"][str(s)][arm]["selected_val_rmse"], 9) for s in seeds
        }
        recomputed_test = {
            round(report["per_seed"][str(s)][arm]["recomputed_test_rmse"], 9) for s in seeds
        }
        report["summary_checks"].append({
            "arm": arm,
            "validation_match": committed_val == recomputed_val,
            "test_match": committed_test == recomputed_test,
            "committed_val": sorted(committed_val),
            "recomputed_val": sorted(recomputed_val),
            "committed_test": sorted(committed_test),
            "recomputed_test": sorted(recomputed_test),
        })

    all_match = not report["per_candidate_mismatches"] and all(
        c["validation_match"] and c["test_match"] for c in report["summary_checks"]
    )
    report["n_candidates_checked"] = checked
    report["match"] = all_match

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "replication_report.json").write_text(json.dumps(report, indent=2) + "\n")
    if all_match:
        (OUT_DIR / "COMPLETE.json").write_text(json.dumps({
            "experiment": "E0", "status": "complete", "api_calls_made": 0,
            "n_candidates_checked": checked, "match": True,
        }, indent=2) + "\n")
        print(f"E0 replication: MATCH ({checked} candidates, "
              f"{len(report['summary_checks'])} arm summaries)")
        return 0
    print("E0 replication: MISMATCH — see outputs/bench_v2/E0/replication_report.json")
    print(json.dumps(report["summary_checks"], indent=2)[:2000])
    if report["per_candidate_mismatches"]:
        print(json.dumps(report["per_candidate_mismatches"][:5], indent=2))
    return 1


if __name__ == "__main__":
    sys.exit(main())
