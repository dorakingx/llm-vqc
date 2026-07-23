#!/usr/bin/env python
"""Free-form parameterized-gate VQC search: amplitude encoding, fixed
quantum readout, zero classical parameters (`free_amplitude_fixed_readout_v1`).

**This entry point makes NO real or billed LLM API call.** `--arms` may
only name `random`, `scripted_open_loop`, or `scripted_closed_loop` (mock/
scripted, offline, zero-cost -- see `llm_vqc.free_amplitude.provider`). A
real-API arm is intentionally not wired up here; adding one later must
also honor the dirty-worktree guard already enforced below.

All search/train/report logic lives in `llm_vqc.free_amplitude` so it is
independently testable; this script only parses arguments, validates
configuration, wires up the task/provider/output directory, and prints the
console summary.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from llm_vqc.evaluation.store import IncompatibleResumeError, ResultStore  # noqa: E402
from llm_vqc.free_amplitude import reporting  # noqa: E402
from llm_vqc.free_amplitude.baselines import (  # noqa: E402
    evaluate_fixed_random_quantum_body_baseline,
    evaluate_quantum_no_body_baseline,
)
from llm_vqc.free_amplitude.provenance import collect_provenance  # noqa: E402
from llm_vqc.free_amplitude.provider import MockFreeGateProvider  # noqa: E402
from llm_vqc.free_amplitude.runner import (  # noqa: E402
    evaluate_protected_test,
    run_llm_closed_loop_arm,
    run_llm_open_loop_arm,
    run_random_arm,
)
from llm_vqc.free_amplitude.sampler import (  # noqa: E402
    sample_free_gate_proposal,
    search_space_size_report,
)
from llm_vqc.free_amplitude.tasks import (  # noqa: E402
    DATASET_PROFILES,
    AmplitudeDatasetProfile,
    AmplitudeGaussianPeakTask,
)
from llm_vqc.free_amplitude.training import FreeAmplitudeTrainingConfig  # noqa: E402

WALL_CLOCK_LIMIT_S = 300
MOCK_RUN_LABEL = "MOCK/NON-LLM AMPLITUDE-ENCODING SMOKE RUN"
MOCK_RUN_LABEL_EXTRA = "NOT A REAL LLM COMPARISON. NOT A SCIENTIFIC PERFORMANCE CLAIM."
_MOCK_SCRIPTED_ARMS = {"random", "scripted_open_loop", "scripted_closed_loop"}

_REPO_ROOT = Path(__file__).resolve().parent.parent


class ConfigError(Exception):
    """Raised for CLI configuration violations (Codex instruction section 20)."""


def _resolve_profile(args: argparse.Namespace) -> AmplitudeDatasetProfile:
    base = DATASET_PROFILES.get(args.dataset_profile)
    if base is None:
        raise ConfigError(
            f"unknown --dataset-profile {args.dataset_profile!r}; "
            f"known profiles: {sorted(DATASET_PROFILES)}"
        )
    sigma_min = args.sigma_min if args.sigma_min is not None else base.sigma_min
    sigma_max = args.sigma_max if args.sigma_max is not None else base.sigma_max
    return AmplitudeDatasetProfile(
        name=base.name, n_qubits=base.n_qubits, feature_count=base.feature_count,
        sigma_min=sigma_min, sigma_max=sigma_max,
        n_train=base.n_train, n_val=base.n_val, n_test=base.n_test,
    )


def validate_config(args: argparse.Namespace) -> None:
    if args.feature_count != 2**args.n_qubits:
        raise ConfigError(
            f"feature_count ({args.feature_count}) must equal 2**n_qubits "
            f"(2**{args.n_qubits} = {2**args.n_qubits})"
        )
    if not (0 <= args.readout_qubit < args.n_qubits):
        raise ConfigError(
            f"readout_qubit ({args.readout_qubit}) must satisfy "
            f"0 <= readout_qubit < n_qubits ({args.n_qubits})"
        )
    if not (1 <= args.max_gates <= 5):
        raise ConfigError(f"max_gates ({args.max_gates}) must satisfy 1 <= max_gates <= 5")
    for arm in args.arms:
        if arm not in _MOCK_SCRIPTED_ARMS:
            # A real-API arm name: block from a dirty worktree before doing
            # anything else (Codex instruction section 20). Currently no
            # real-API arm is implemented at all, so this only ever fires
            # for an arm name this script does not otherwise recognize.
            provenance = collect_provenance(_REPO_ROOT)
            if provenance["execution_git_dirty"]:
                raise ConfigError(
                    f"arm {arm!r} is not a recognized mock/scripted arm and this worktree is "
                    "dirty -- real API execution is blocked from a dirty worktree"
                )
            raise ConfigError(
                f"unknown arm {arm!r}; this implementation pass only supports: "
                f"{sorted(_MOCK_SCRIPTED_ARMS)}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-qubits", type=int, default=3)
    parser.add_argument("--feature-count", type=int, default=8)
    parser.add_argument("--readout-qubit", type=int, default=0)
    parser.add_argument("--max-gates", type=int, default=5)
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--arms", type=str, default="random,scripted_open_loop,scripted_closed_loop"
    )
    parser.add_argument("--dataset-profile", type=str, default="amplitude_n3_smoke_v1")
    parser.add_argument("--sigma-min", type=float, default=None)
    parser.add_argument("--sigma-max", type=float, default=None)
    parser.add_argument("--run-id", type=str, default="free_amplitude_fixed_readout_v1")
    parser.add_argument("--output", type=str, default="outputs/free_amplitude_fixed_readout_v1")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    try:
        validate_config(args)
        profile = _resolve_profile(args)
    except ConfigError as exc:
        print(f"BLOCKED: {exc}")
        sys.exit(1)

    t_start = time.monotonic()
    deadline = t_start + WALL_CLOCK_LIMIT_S

    output_dir = _REPO_ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = _REPO_ROOT / "runs" / "free_amplitude_fixed_readout_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    db_path = run_dir / f"{args.run_id}.sqlite"

    provenance = collect_provenance(_REPO_ROOT)
    task = AmplitudeGaussianPeakTask(profile)
    train_val, dataset_diag = task.build(seed=0)
    test_split, test_diag = task.build_test(seed=0)
    training_config = FreeAmplitudeTrainingConfig(epochs=args.epochs)

    try:
        store = ResultStore.open_or_create(
            db_path, run_id=args.run_id, config_json=str(vars(args)),
            config_reproducibility_fields={
                "dataset_profile": profile.name, "n_qubits": args.n_qubits,
                "epochs": args.epochs, "budget": args.budget, "max_gates": args.max_gates,
            },
            git_sha=provenance.get("execution_code_sha"),
            created_at=datetime.now(timezone.utc).isoformat(), allow_incompatible=args.resume,
        )
    except IncompatibleResumeError as exc:
        print(f"BLOCKED: {exc}")
        sys.exit(1)

    print(
        f"{MOCK_RUN_LABEL}\n{MOCK_RUN_LABEL_EXTRA}\n"
        f"profile={profile.name} n_qubits={args.n_qubits} feature_count={args.feature_count} "
        f"readout_qubit={args.readout_qubit} max_gates={args.max_gates} budget={args.budget} "
        f"epochs={args.epochs} arms={args.arms}"
    )

    task_name = args.run_id
    arm_outcomes = []
    for seed in range(args.seeds):
        if "random" in args.arms:
            outcome = run_random_arm(
                seed, task_name, args.n_qubits, args.readout_qubit, train_val,
                training_config, args.budget, store, deadline, args.max_gates,
            )
            arm_outcomes.append(outcome)
            print(f"[random seed={seed}] {outcome.ledger_summary} stop={outcome.stop_reason}")

        if "scripted_open_loop" in args.arms:
            provider = MockFreeGateProvider(
                seed=seed, n_qubits=args.n_qubits, max_gates=args.max_gates
            )
            outcome = run_llm_open_loop_arm(
                seed, task_name, args.n_qubits, args.readout_qubit, train_val,
                training_config, args.budget, store, provider, deadline, args.max_gates,
            )
            arm_outcomes.append(outcome)
            print(
                f"[scripted_open_loop seed={seed}] {outcome.ledger_summary} "
                f"stop={outcome.stop_reason} api_calls={outcome.api_successful_calls}"
            )

        if "scripted_closed_loop" in args.arms:
            provider = MockFreeGateProvider(
                seed=seed, n_qubits=args.n_qubits, max_gates=args.max_gates
            )
            outcome = run_llm_closed_loop_arm(
                seed, task_name, args.n_qubits, args.readout_qubit, train_val,
                training_config, args.budget, store, provider, deadline, args.max_gates,
            )
            arm_outcomes.append(outcome)
            print(
                f"[scripted_closed_loop seed={seed}] {outcome.ledger_summary} "
                f"stop={outcome.stop_reason} api_calls={outcome.api_successful_calls}"
            )

    for outcome in arm_outcomes:
        evaluate_protected_test(
            outcome, store, task_name, args.readout_qubit, test_split, run_seed=0
        )
        print(
            f"[{outcome.arm}] selected_hash={outcome.selected_structural_hash} "
            f"val_rmse={outcome.selected_val_rmse} test_rmse={outcome.protected_test_rmse}"
        )

    baselines = [
        evaluate_quantum_no_body_baseline(args.n_qubits, args.readout_qubit, train_val, test_split)
    ]
    rng = np.random.default_rng(0)
    frozen_proposal = sample_free_gate_proposal(rng, args.n_qubits, args.max_gates)
    baselines.append(
        evaluate_fixed_random_quantum_body_baseline(
            frozen_proposal, args.readout_qubit, seed=0, train_val=train_val, test_split=test_split
        )
    )
    for b in baselines:
        print(f"[baseline:{b.name}] val_rmse={b.val_rmse:.4f} test_rmse={b.test_rmse}")

    elapsed = time.monotonic() - t_start
    dataset_diagnostics = {"train_val": dataset_diag, "test": test_diag}
    search_space_report = search_space_size_report(args.n_qubits, args.max_gates)

    reporting.write_all_outputs(
        output_dir=output_dir, store=store, args=args, provenance=provenance,
        dataset_diagnostics=dataset_diagnostics, search_space_report=search_space_report,
        arm_outcomes=arm_outcomes, baseline_results=baselines, task_name=task_name,
        n_qubits=args.n_qubits, feature_count=args.feature_count, readout_qubit=args.readout_qubit,
        max_gates=args.max_gates, elapsed_seconds=elapsed, run_label=MOCK_RUN_LABEL,
    )
    store.close()

    n_train_features = train_val.train.features.shape[1]
    n_val_features = train_val.val.features.shape[1]
    n_test_features = test_split.features.shape[1]

    print(f"\n{MOCK_RUN_LABEL}")
    print(MOCK_RUN_LABEL_EXTRA)
    print(f"train/val/test feature counts: {n_train_features}/{n_val_features}/{n_test_features}")
    assert n_train_features == n_val_features == n_test_features == args.feature_count
    print("Zero real API calls were made.")
    print(f"Total elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
