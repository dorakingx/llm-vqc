#!/usr/bin/env python
"""Run the `mini_5gate_v1` comparison: five arms, four tasks, ten seeds.

Design, all of it stateable in one screen:

  * exactly 5 gates per circuit, never fewer or more;
  * gate set H, RX, RY, RZ (one wire) and CRX, CRY, CRZ (two wires);
  * angles in [-pi, pi] carried by the proposal and used verbatim -
    there is NO optimizer anywhere on this path;
  * amplitude encoding of a length-2**n signal, Pauli-Z read out on
    qubit 0, prediction mu_hat = (1 - <Z0>)/2;
  * every arm gets the same budget of B UNIQUE evaluated candidates;
    duplicates and grammar violations are recorded but cost no budget;
  * the winner is chosen on validation RMSE, and only that one circuit
    is ever scored on the held-out test split, once.

T3 and T4 have frozen profiles only at n=5, so they run there; T1 and T2
run at n=3. Track B has no training loop, so the qubit count barely moves
the wall clock - the honest saving comes from the fixed gate count and
the small budget, not from shrinking n.

Real LLM calls need OPENAI_API_KEY, OPENAI_MODEL and a positive
LLM_API_BUDGET_USD; without them the run refuses rather than silently
falling back to a mock.

    python scripts/mini5/run_mini5.py --budget 8 --seeds 10 --workers 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from llm_vqc.free_amplitude.candidate_schema import (  # noqa: E402
    CompleteCandidateProposal,
)
from llm_vqc.free_amplitude.main_eval import (  # noqa: E402
    evaluate_complete_candidate,
    evaluate_on_test_fixed_theta,
)
from llm_vqc.ir.budget import BudgetLedger  # noqa: E402
from llm_vqc.llm.global_ledger import GlobalSpendLedger, LedgerCapExceeded  # noqa: E402
from llm_vqc.mini5.arms import (  # noqa: E402
    ARM_NAMES,
    LLM_ARMS,
    MAX_PROPOSALS_PER_CELL,
    build_arm,
    grammar_issues,
)
from llm_vqc.mini5.space import EXACT_GATES, MINI_SPACE_VERSION  # noqa: E402
from llm_vqc.tasks.signal_suite import SignalProfile, SignalSuiteTask  # noqa: E402

OUT_DIR = REPO / "outputs" / "mini5"
LEDGER_PATH = REPO / "runs" / "mini5" / "spend_ledger.sqlite"
READOUT_QUBIT = 0

#: (task family, qubits, slide label). T3/T4 exist only at n=5.
TASKS = [
    ("gauss_peak", 3, "T1 Gaussian peak"),
    ("sin_freq", 3, "T2 Sinusoid frequency"),
    ("change_point", 5, "T3 Change point"),
    ("peak_count", 5, "T4 One vs two peaks"),
]

#: Cells run in separate PROCESSES, not threads: PennyLane/PyTorch model
#: construction is not thread-safe and deadlocks under a thread pool. The
#: spend ledger was built process-safe (SQLite, BEGIN IMMEDIATE) precisely
#: so a process pool can share one cumulative cap.
_WORKER: dict = {}


def _predict(proposal, split):
    """Forward the selected circuit at its own angles. Mirrors the test
    gate's own path so the AUROC check and the RMSE come from identical
    predictions."""
    from llm_vqc.free_amplitude.candidate_schema import (
        complete_candidate_to_ir_and_theta,
    )
    from llm_vqc.free_amplitude.main_eval import _forward, _load_theta
    from llm_vqc.free_amplitude.model import FixedReadoutQuantumModel

    ir, theta = complete_candidate_to_ir_and_theta(proposal, READOUT_QUBIT)
    model = FixedReadoutQuantumModel(ir, readout_qubit=READOUT_QUBIT)
    model.double()
    _load_theta(model, theta)
    return _forward(model, split.features)


def _auroc(predictions, targets) -> float:
    """Rank-based AUROC with tie handling; 0.5 when a class is absent."""
    scores = np.clip(np.asarray(predictions, dtype=float).ravel(), 0.0, 1.0)
    labels = np.asarray(targets, dtype=float).ravel()
    positives, negatives = labels == 1, labels == 0
    n_pos, n_neg = int(positives.sum()), int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    # average ranks within tied score groups
    sorted_scores = scores[order]
    start = 0
    for i in range(1, len(sorted_scores) + 1):
        if i == len(sorted_scores) or sorted_scores[i] != sorted_scores[start]:
            ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def log(message: str) -> None:
    print(message, flush=True)


def _worker(args) -> dict:
    """Top-level so it pickles. Each process builds its own provider once
    and reuses it, sharing the one on-disk cumulative ledger."""
    family, n_qubits, arm_name, seed, budget, mock, cap_usd = args

    def provider_factory():
        if arm_name not in LLM_ARMS:
            return None
        if "provider" not in _WORKER:
            if mock:
                from llm_vqc.mini5.mock import MockMini5Provider
                _WORKER["provider"] = MockMini5Provider()
            else:
                from llm_vqc.mini5.provider import build_mini5_provider
                ledger = GlobalSpendLedger(LEDGER_PATH, cap_usd=cap_usd,
                                           run_label="mini5")
                _WORKER["provider"] = build_mini5_provider(ledger)
        return _WORKER["provider"]

    return run_cell(family, n_qubits, arm_name, seed, budget, provider_factory)


def run_cell(family, n_qubits, arm_name, seed, budget, provider_factory) -> dict:
    """One (task, arm, seed) cell, start to finish."""
    started = time.perf_counter()
    data_seed, search_seed = 1000 + seed, 2000 + seed
    task = SignalSuiteTask(SignalProfile(family=family, n_qubits=n_qubits))
    train_val, _ = task.build(data_seed)   # (TrainValData, checks)
    rng = np.random.default_rng(search_seed)

    provider = provider_factory() if arm_name in LLM_ARMS else None
    arm = build_arm(arm_name, rng, n_qubits, budget, provider)
    ledger = BudgetLedger()

    evaluated: list[dict] = []
    seen: set[str] = set()
    proposals = 0
    stop_reason = "budget_exhausted"

    while len(evaluated) < budget:
        if proposals >= MAX_PROPOSALS_PER_CELL:
            stop_reason = "proposal_guard"
            break
        proposals += 1
        arm.telemetry.proposals = proposals
        proposal = arm.propose(len(evaluated) + 1)
        if proposal is None:
            arm.telemetry.invalid += 1
            continue

        issues = grammar_issues(proposal.operations, n_qubits)
        if issues:
            arm.telemetry.invalid += 1
            arm.telemetry.issues.extend(issues[:1])
            continue

        raw = {"n_qubits": n_qubits, "operations": proposal.operations}
        result = evaluate_complete_candidate(
            raw_proposal=raw,
            expected_n_qubits=n_qubits,
            readout_qubit=READOUT_QUBIT,
            train_val=train_val,
            proposal_id=f"{arm_name}-s{seed}-p{proposals}",
            ledger=ledger,
            seed=search_seed,
        )
        if not result.valid:
            arm.telemetry.invalid += 1
            arm.telemetry.issues.extend(i.code for i in result.validation_issues[:1])
            continue
        if result.candidate_hash in seen:
            arm.telemetry.duplicates += 1
            continue

        seen.add(result.candidate_hash)
        evaluated.append({
            "operations": result.operations,
            "val_rmse": float(result.val_rmse),
            "candidate_hash": result.candidate_hash,
            "gate_count": result.searched_body_gate_count,
            "depth": result.circuit_depth,
            "source": proposal.source,
            "order": len(evaluated) + 1,
        })
        arm.observe(result.operations, float(result.val_rmse))

    if not evaluated:
        return {
            "family": family, "n_qubits": n_qubits, "arm": arm_name, "seed": seed,
            "status": "failed", "stop_reason": stop_reason,
            "telemetry": asdict(arm.telemetry),
            "elapsed_seconds": time.perf_counter() - started,
        }

    # Selection happens on validation only; the test split is touched once,
    # after the choice is frozen.
    best = min(evaluated, key=lambda e: e["val_rmse"])
    proposal = CompleteCandidateProposal(n_qubits=n_qubits, operations=best["operations"])
    test_split, _ = task.build_test(data_seed)
    test = evaluate_on_test_fixed_theta(proposal, READOUT_QUBIT, test_split)

    # T4 is binary classification. RMSE against a 0/1 label is sqrt(Brier) -
    # a proper scoring rule, so it is defensible and keeps all four panels
    # on one axis. But it conflates calibration with discrimination: a
    # circuit that ranks the two classes perfectly while outputting 0.45
    # and 0.55 scores badly under RMSE and perfectly under AUROC. With
    # five gates and no optimizer, mu_hat tends to sit near 0.5, so that
    # failure mode is a live risk here rather than a theoretical one.
    # AUROC is therefore recorded alongside as a check; a disagreement
    # between the two is a finding to report, not noise to hide.
    test_auroc = None
    if task.is_classification:
        test_auroc = float(_auroc(_predict(proposal, test_split), test_split.targets))

    return {
        "family": family, "n_qubits": n_qubits, "arm": arm_name, "seed": seed,
        "status": "complete", "stop_reason": stop_reason,
        "budget": budget, "n_evaluated": len(evaluated),
        "selected_val_rmse": best["val_rmse"],
        "test_rmse": float(test["test_rmse"]),
        "test_auroc": test_auroc,
        "is_classification": task.is_classification,
        "selected_operations": best["operations"],
        "selected_gate_count": best["gate_count"],
        "selected_depth": best["depth"],
        "best_so_far": [
            min(e["val_rmse"] for e in evaluated[: k + 1]) for k in range(len(evaluated))
        ],
        "telemetry": asdict(arm.telemetry),
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget", type=int, default=8, help="unique candidates per arm")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--arms", nargs="*", default=list(ARM_NAMES))
    ap.add_argument("--mock", action="store_true",
                    help="offline plumbing check; never scientific evidence")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)

    spend_ledger = None
    if not args.mock and any(a in LLM_ARMS for a in args.arms):
        spend_ledger = GlobalSpendLedger.from_env(LEDGER_PATH, run_label="mini5")
        if spend_ledger is None:
            print("REFUSING: LLM_API_BUDGET_USD must be set to a positive value.")
            return 2
        log(f"[ledger] cap ${spend_ledger.cap_usd:.2f}, "
            f"already committed ${spend_ledger.totals().committed_usd:.6f}")

    cells = [
        (family, n, arm, seed)
        for family, n, _label in TASKS
        for arm in args.arms
        for seed in range(args.seeds)
    ]
    log(f"[mini5] {MINI_SPACE_VERSION}: {len(cells)} cells "
        f"({len(TASKS)} tasks x {len(args.arms)} arms x {args.seeds} seeds), "
        f"B={args.budget}, exactly {EXACT_GATES} gates, {args.workers} workers")

    started = time.perf_counter()
    rows, failures = [], 0
    cap = spend_ledger.cap_usd if spend_ledger is not None else 0.0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_worker, (f, n, a, s, args.budget, args.mock, cap)): (f, n, a, s)
            for (f, n, a, s) in cells
        }
        for done, future in enumerate(as_completed(futures), start=1):
            family, n, arm, seed = futures[future]
            try:
                row = future.result()
            except LedgerCapExceeded as exc:
                log(f"  STOP cap reached: {exc}")
                break
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                failures += 1
                log(f"  [{done}/{len(cells)}] {family} n={n} {arm} s{seed} "
                    f"FAILED {type(exc).__name__}: {exc}")
                continue
            rows.append(row)
            if row["status"] != "complete":
                failures += 1
            if done % 20 == 0 or done == len(cells):
                log(f"  [{done}/{len(cells)}] {time.perf_counter() - started:.0f}s elapsed")

    elapsed = time.perf_counter() - started
    payload = {
        "space_version": MINI_SPACE_VERSION,
        "exact_gates": EXACT_GATES,
        "budget_unique": args.budget,
        "seeds": args.seeds,
        "tasks": [{"family": f, "n_qubits": n, "label": lab} for f, n, lab in TASKS],
        "arms": args.arms,
        "mock": args.mock,
        "wall_clock_seconds": elapsed,
        "cells": rows,
    }
    if spend_ledger is not None:
        payload["spend"] = spend_ledger.summary()
    out = OUT_DIR / ("results_mock.json" if args.mock else "results.json")
    out.write_text(json.dumps(payload, indent=2) + "\n")

    log(f"[mini5] {len(rows)} cells in {elapsed:.1f}s ({elapsed / 60:.1f} min), "
        f"{failures} not complete -> {out.relative_to(REPO)}")
    if spend_ledger is not None:
        s = spend_ledger.summary()
        log(f"[ledger] calls={s['calls_settled']} in={s['input_tokens']} "
            f"out={s['output_tokens']} spent=${s['settled_usd']:.6f} "
            f"of ${s['cap_usd']:.2f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
