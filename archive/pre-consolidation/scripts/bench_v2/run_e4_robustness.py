#!/usr/bin/env python
"""E4 — selected-circuit robustness (protocol §10): exact vs finite-shot
(1,024 / 4,096) vs the frozen depol_ro_v1 noisy profile, plus fixed-seed
transpiled costs, for the top selected circuits of completed E2/E3 cells.

Frozen selection rule (validation-only, no test tuning): for every
(experiment, task, n, arm), take the replicate whose SELECTED candidate
has the best validation metric, and evaluate that one frozen circuit on
the protected test split under each condition. Results are new,
predeclared robustness rows; they feed nothing back into search,
selection, or hyperparameters.

Writes outputs/bench_v2/E4/shot_sensitivity_<EID>.csv,
noise_profile_result_<EID>.csv, and COMPLETE.json when every available
completed source cell was processed.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from llm_vqc.bench_v2.cell_runner import _resolve_task  # noqa: E402
from llm_vqc.bench_v2.resources import (  # noqa: E402
    NOISE_PROFILE_V1,
    exact_eval,
    finite_shot_eval,
    noisy_eval,
)
from llm_vqc.bench_v2.space import READOUT_QUBIT, layered_ir_dict  # noqa: E402
from llm_vqc.bench_v2.track_a_evaluator import bench_v2_run_seed  # noqa: E402
from llm_vqc.bench_v2.track_b_evaluator import bench_v2_joint_run_seed  # noqa: E402
from llm_vqc.evaluation.seeds import derive_child_seed  # noqa: E402
from llm_vqc.evaluation.store import ResultStore  # noqa: E402
from llm_vqc.free_amplitude.candidate_schema import (  # noqa: E402
    CompleteCandidateProposal,
    complete_candidate_to_ir_and_theta,
)
from llm_vqc.free_amplitude.init_policy import free_amplitude_train_seed  # noqa: E402
from llm_vqc.free_amplitude.training import (  # noqa: E402
    FREE_AMPLITUDE_TRAINING_CONFIG_VERSION,
)
from llm_vqc.ir.schema import CircuitIR  # noqa: E402

RUNS_ROOT = REPO / "runs" / "bench_v2"
OUT_DIR = REPO / "outputs" / "bench_v2" / "E4"
SHOTS = (1024, 4096)
SOURCE_EXPERIMENTS = ("E2", "E3")
STRUCTURE_TRACK_ARMS_PREFIXES = ("random_structure", "evolutionary_structure",
                                 "greedy_growth", "llm_open_structure",
                                 "llm_archive_closed_structure", "ref_")


def _load_manifests(experiment: str) -> list[dict]:
    cells = RUNS_ROOT / experiment / "cells"
    return [
        json.loads(p.read_text())
        for p in sorted(cells.glob("*.json"))
        if json.loads(p.read_text()).get("status") == "complete"
    ] if cells.is_dir() else []


def _best_validation_cell(manifests: list[dict]) -> dict[tuple, dict]:
    best: dict[tuple, dict] = {}
    for m in manifests:
        key = (m["experiment"], m["task"], m["n_qubits"], m["arm"])
        value = m["selected"].get("val_metric_value")
        if value is None:
            continue
        if key not in best or value < best[key]["selected"]["val_metric_value"]:
            best[key] = m
    return best


def _selected_circuit(manifest: dict) -> tuple[CircuitIR, list[float]]:
    """Reload the frozen selected circuit + final angles from the cell's
    durable store."""
    store_path = REPO / manifest["store_path"] if not Path(
        manifest["store_path"]
    ).is_absolute() else Path(manifest["store_path"])
    store = ResultStore(store_path)
    arm = manifest["arm"]
    task, _is_cls, _vm = _resolve_task(manifest["task"], manifest["n_qubits"])
    task_name = task.spec.name
    selected_hash = manifest["selected"]["structural_hash"]

    if arm.startswith(STRUCTURE_TRACK_ARMS_PREFIXES):
        run_seed = bench_v2_run_seed(
            task_name, manifest["data_seed"], manifest["search_seed"]
        )
        train_seed = free_amplitude_train_seed(
            run_seed, selected_hash, FREE_AMPLITUDE_TRAINING_CONFIG_VERSION
        )
        blob = store.get_trained_weights(task_name, selected_hash, train_seed)
        if blob is None:
            raise RuntimeError(f"missing trained weights for {manifest['cell_id']}")
        ir = CircuitIR.model_validate(layered_ir_dict(blob["n_qubits"], blob["operations"]))
        theta = list(blob["learned_angles"])
    else:
        run_seed = bench_v2_joint_run_seed(
            task_name, manifest["data_seed"], manifest["search_seed"]
        )
        state = json.loads(store.load_run_state(manifest["cell_id"]))
        proposal = CompleteCandidateProposal.model_validate(state["best_candidate"])
        ir, theta = complete_candidate_to_ir_and_theta(proposal, READOUT_QUBIT)
        theta = list(theta)
    store.close()
    return ir, theta


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    processed = 0
    for experiment in SOURCE_EXPERIMENTS:
        manifests = _load_manifests(experiment)
        if not manifests:
            continue
        shot_rows, noise_rows = [], []
        for key, manifest in sorted(_best_validation_cell(manifests).items()):
            task, is_classification, _vm = _resolve_task(
                manifest["task"], manifest["n_qubits"]
            )
            if is_classification:
                # The robustness metric set below is regression-shaped;
                # T4's probability output still yields rmse-vs-labels,
                # which is what we report (documented).
                pass
            ir, theta = _selected_circuit(manifest)
            test_split, _diag = task.build_test(manifest["data_seed"])
            features, targets = test_split.features, test_split.targets
            exact = exact_eval(ir, theta, features, targets)
            base = {
                "experiment": key[0], "task": key[1], "n_qubits": key[2],
                "arm": key[3], "source_cell": manifest["cell_id"],
                "exact_rmse": exact["rmse"],
            }
            for shots in SHOTS:
                shot_seed = derive_child_seed(
                    manifest["data_seed"], "e4_shots", key[3], str(shots)
                )
                result = finite_shot_eval(
                    ir, theta, features, targets, shots=shots, shot_seed=shot_seed
                )
                shot_rows.append({**base, "shots": shots, "shot_seed": shot_seed,
                                  "shot_rmse": result["rmse"],
                                  "shot_minus_exact": result["rmse"] - exact["rmse"]})
            noisy = noisy_eval(ir, theta, features, targets)
            noise_rows.append({**base, "noise_profile": NOISE_PROFILE_V1["name"],
                               "noisy_rmse": noisy["rmse"],
                               "noisy_minus_exact": noisy["rmse"] - exact["rmse"]})
            processed += 1

        for name, rows in (
            (f"shot_sensitivity_{experiment}.csv", shot_rows),
            (f"noise_profile_result_{experiment}.csv", noise_rows),
        ):
            if rows:
                with (OUT_DIR / name).open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows)

    if processed:
        (OUT_DIR / "COMPLETE.json").write_text(json.dumps({
            "experiment": "E4", "status": "complete",
            "selected_circuits_evaluated": processed,
            "shots": list(SHOTS), "noise_profile": NOISE_PROFILE_V1,
        }, indent=2) + "\n")
        print(f"E4: evaluated {processed} selected circuits")
        return 0
    print("E4: no completed E2/E3 cells found")
    return 1


if __name__ == "__main__":
    sys.exit(main())
