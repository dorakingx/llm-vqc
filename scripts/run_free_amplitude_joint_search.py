#!/usr/bin/env python
"""Main-mode joint structure-and-theta search: amplitude encoding, fixed
quantum readout, NO optimizer, NO classical layers.

In main mode the LLM (here a mock/scripted provider -- **no real or billed
API call**) proposes a COMPLETE candidate: circuit structure AND the
numerical rotation angles, which are evaluated verbatim
(`mu_hat = (1 - <Z_0>) / 2`). The AdamW structure-search path is retained
only as a separately-labelled ablation (`scripts/
run_free_amplitude_fixed_readout.py`), never mixed with these results.

Config isolation (correction pass):
  - `n_qubits` and `feature_count` are DERIVED FROM the dataset profile,
    not taken from conflicting CLI flags;
  - a fresh run refuses to reuse an existing run store unless `--resume`
    is given (and the store's reproducibility config matches);
  - each seed builds its own data split and uses its own cache / ledger /
    protected-test evaluation.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_vqc.evaluation.store import IncompatibleResumeError, ResultStore  # noqa: E402
from llm_vqc.free_amplitude import main_reporting  # noqa: E402
from llm_vqc.free_amplitude.expressibility import IntrinsicDiagnosticsCache  # noqa: E402
from llm_vqc.free_amplitude.main_runner import (  # noqa: E402
    evaluate_protected_test_main,
    run_closed_loop_arm_main,
    run_open_loop_arm_main,
    run_random_arm_main,
)
from llm_vqc.free_amplitude.provenance import collect_provenance  # noqa: E402
from llm_vqc.free_amplitude.provider import MockCompleteCandidateProvider  # noqa: E402
from llm_vqc.free_amplitude.sampler import search_space_size_report  # noqa: E402
from llm_vqc.free_amplitude.tasks import (  # noqa: E402
    DATASET_PROFILES,
    AmplitudeDatasetProfile,
    AmplitudeGaussianPeakTask,
)

WALL_CLOCK_LIMIT_S = 600
MOCK_RUN_LABEL = "MOCK/NON-LLM JOINT-SEARCH RUN"
REAL_RUN_LABEL = "REAL-LLM JOINT-SEARCH RUN (OpenAI)"
MOCK_RUN_LABEL_EXTRA = "NOT A REAL LLM COMPARISON. NOT A SCIENTIFIC PERFORMANCE CLAIM."
REAL_RUN_LABEL_EXTRA = (
    "Real OpenAI proposals; n is small -- an integration demonstration, "
    "NOT a statistically powered performance claim."
)
_MOCK_SCRIPTED_ARMS = {"random", "scripted_open_loop", "scripted_closed_loop"}
_REAL_ARMS = {"real_open_loop", "real_closed_loop"}
#: Hard cap on total real, billed OUTBOUND request attempts across ALL
#: real arms and seeds in one execution -- retries count against this cap
#: too (the retry wrapper sits outside the call-count wrapper). Sized for
#: seeds x arms x budget nominal plus headroom for retries and invalid
#: proposals (which consume an LLM call but no search budget).
MAX_REAL_LLM_CALLS = 60

_REPO_ROOT = Path(__file__).resolve().parent.parent


class ConfigError(Exception):
    """Raised for CLI configuration violations."""


def _resolve_profile(args: argparse.Namespace) -> AmplitudeDatasetProfile:
    base = DATASET_PROFILES.get(args.dataset_profile)
    if base is None:
        raise ConfigError(
            f"unknown --dataset-profile {args.dataset_profile!r}; known: {sorted(DATASET_PROFILES)}"
        )
    sigma_min = args.sigma_min if args.sigma_min is not None else base.sigma_min
    sigma_max = args.sigma_max if args.sigma_max is not None else base.sigma_max
    return AmplitudeDatasetProfile(
        name=base.name, n_qubits=base.n_qubits, feature_count=base.feature_count,
        sigma_min=sigma_min, sigma_max=sigma_max,
        n_train=base.n_train, n_val=base.n_val, n_test=base.n_test,
    )


def validate_config(args: argparse.Namespace, profile: AmplitudeDatasetProfile) -> None:
    # n_qubits/feature_count come from the profile; if the user ALSO passed
    # them explicitly and they conflict, that is a hard error, not a silent
    # override.
    if args.readout_qubit is not None and not (0 <= args.readout_qubit < profile.n_qubits):
        raise ConfigError(
            f"readout_qubit ({args.readout_qubit}) must satisfy 0 <= readout_qubit < "
            f"n_qubits ({profile.n_qubits})"
        )
    if not (1 <= args.max_gates <= 5):
        raise ConfigError(f"max_gates ({args.max_gates}) must satisfy 1 <= max_gates <= 5")
    for arm in args.arms:
        if arm in _MOCK_SCRIPTED_ARMS:
            continue
        if arm in _REAL_ARMS:
            # Real, billed API arms may only run from a CLEAN worktree, so
            # the recorded execution_code_sha is the true implementation.
            provenance = collect_provenance(_REPO_ROOT)
            if provenance["execution_git_dirty"]:
                raise ConfigError(
                    f"real-API arm {arm!r} is blocked from a dirty worktree -- commit "
                    "the implementation first, then run from that clean SHA"
                )
            continue
        raise ConfigError(
            f"unknown arm {arm!r}; supported: "
            f"{sorted(_MOCK_SCRIPTED_ARMS | _REAL_ARMS)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-profile", type=str, default="amplitude_n3_smoke_v1")
    parser.add_argument("--readout-qubit", type=int, default=0)
    parser.add_argument("--max-gates", type=int, default=5)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument(
        "--arms", type=str, default="random,scripted_open_loop,scripted_closed_loop"
    )
    parser.add_argument("--sigma-min", type=float, default=None)
    parser.add_argument("--sigma-max", type=float, default=None)
    parser.add_argument("--run-id", type=str, default="free_amplitude_joint_search_v1")
    parser.add_argument(
        "--output", type=str, default="outputs/free_amplitude_fixed_readout_v1"
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    try:
        profile = _resolve_profile(args)
        validate_config(args, profile)
    except ConfigError as exc:
        print(f"BLOCKED: {exc}")
        sys.exit(1)

    # n_qubits/feature_count are AUTHORITATIVELY the profile's.
    n_qubits = profile.n_qubits
    feature_count = profile.feature_count
    readout_qubit = args.readout_qubit

    t_start = time.monotonic()
    deadline = t_start + WALL_CLOCK_LIMIT_S

    output_dir = _REPO_ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = _REPO_ROOT / "runs" / "free_amplitude_fixed_readout_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    db_path = run_dir / f"{args.run_id}.sqlite"

    # Fresh-run/resume isolation: a fresh run must NOT silently reuse an
    # existing store (which would resume a prior run's ledgers/caches). Only
    # --resume permits reusing an existing DB.
    if db_path.exists() and not args.resume:
        print(
            f"BLOCKED: run store {db_path} already exists. A fresh run must not reuse it "
            "(would resume a prior run's ledgers/caches). Pass --resume to continue that "
            "run, or use a different --run-id / remove the store first."
        )
        sys.exit(1)

    provenance = collect_provenance(_REPO_ROOT)

    # Real-API setup (only when a real arm was requested): load the local
    # gitignored .env, run preflight (key + explicit model + positive dollar
    # cap), and build ONE shared budget-enforced, call-capped provider chain
    # used by every real arm across every seed. Imported lazily so mock runs
    # never touch the OpenAI SDK.
    use_real = any(arm in _REAL_ARMS for arm in args.arms)
    shared_real_provider = None
    real_model = None
    real_budget = None
    if use_real:
        from dotenv import load_dotenv

        from llm_vqc.free_amplitude.openai_provider import (
            BudgetEnforcedProvider,
            CompleteCandidateOpenAIProvider,
            RealRunPreflightError,
            RetryingProvider,
            preflight_real_run,
        )
        from llm_vqc.llm.openai_provider import CallCountLimitedProvider

        load_dotenv(_REPO_ROOT / ".env")
        import os

        try:
            real_cfg = preflight_real_run(os.environ)
        except RealRunPreflightError as exc:
            print(f"BLOCKED: {exc}")
            sys.exit(1)
        real_model = real_cfg.model
        real_budget = real_cfg.budget
        inner = CompleteCandidateOpenAIProvider(api_key=real_cfg.api_key, model=real_cfg.model)
        # Chain order: retry OUTSIDE the caps, so every retry attempt is
        # individually call-counted and budget-checked.
        shared_real_provider = RetryingProvider(
            CallCountLimitedProvider(
                BudgetEnforcedProvider(inner, real_cfg.budget), max_calls=MAX_REAL_LLM_CALLS
            )
        )

    training_config_reproducibility = {
        "dataset_profile": profile.name, "n_qubits": n_qubits, "budget": args.budget,
        "max_gates": args.max_gates, "readout_qubit": readout_qubit, "mode": "main_joint_search",
    }
    try:
        store = ResultStore.open_or_create(
            db_path, run_id=args.run_id, config_json=str(vars(args)),
            config_reproducibility_fields=training_config_reproducibility,
            git_sha=provenance.get("execution_code_sha"),
            created_at=datetime.now(timezone.utc).isoformat(), allow_incompatible=False,
        )
    except IncompatibleResumeError as exc:
        print(f"BLOCKED: {exc}")
        sys.exit(1)

    run_label = REAL_RUN_LABEL if use_real else MOCK_RUN_LABEL
    run_label_extra = REAL_RUN_LABEL_EXTRA if use_real else MOCK_RUN_LABEL_EXTRA
    print(
        f"{run_label}\n{run_label_extra}\n"
        f"profile={profile.name} n_qubits={n_qubits} feature_count={feature_count} "
        f"readout_qubit={readout_qubit} max_gates={args.max_gates} budget={args.budget} "
        f"seeds={args.seeds} arms={args.arms}"
        + (
            f" model={real_model} dollar_cap=${real_budget.cap_usd:.2f} "
            f"max_real_calls={MAX_REAL_LLM_CALLS}"
            if use_real else ""
        )
    )

    task = AmplitudeGaussianPeakTask(profile)
    task_name = args.run_id
    diagnostics_cache = IntrinsicDiagnosticsCache()
    arm_outcomes = []
    dataset_diagnostics_by_seed = {}

    for seed in range(args.seeds):
        train_val, ds_diag = task.build(seed=seed)
        test_split, test_diag = task.build_test(seed=seed)
        dataset_diagnostics_by_seed[seed] = {"train_val": ds_diag, "test": test_diag}

        if "random" in args.arms:
            outcome = run_random_arm_main(
                seed, task_name, n_qubits, readout_qubit, train_val, args.budget, store,
                deadline, args.max_gates,
            )
            arm_outcomes.append(outcome)
        if "scripted_open_loop" in args.arms:
            provider = MockCompleteCandidateProvider(seed, n_qubits, args.max_gates)
            outcome = run_open_loop_arm_main(
                seed, task_name, n_qubits, readout_qubit, train_val, args.budget, store,
                provider, deadline, feature_count, args.max_gates,
            )
            arm_outcomes.append(outcome)
        if "scripted_closed_loop" in args.arms:
            provider = MockCompleteCandidateProvider(seed, n_qubits, args.max_gates)
            outcome = run_closed_loop_arm_main(
                seed, task_name, n_qubits, readout_qubit, train_val, args.budget, store,
                provider, deadline, feature_count, args.max_gates,
            )
            arm_outcomes.append(outcome)
        if "real_open_loop" in args.arms:
            outcome = run_open_loop_arm_main(
                seed, task_name, n_qubits, readout_qubit, train_val, args.budget, store,
                shared_real_provider, deadline, feature_count, args.max_gates,
            )
            arm_outcomes.append(outcome)
        if "real_closed_loop" in args.arms:
            outcome = run_closed_loop_arm_main(
                seed, task_name, n_qubits, readout_qubit, train_val, args.budget, store,
                shared_real_provider, deadline, feature_count, args.max_gates,
            )
            arm_outcomes.append(outcome)

        for outcome in arm_outcomes:
            if outcome.seed == seed:
                evaluate_protected_test_main(outcome, readout_qubit, test_split)
                stop = f" stop={outcome.stop_reason}" if outcome.stop_reason else ""
                print(
                    f"[seed={seed} {outcome.arm}] sel_val="
                    f"{outcome.selected_val_rmse} test={outcome.protected_test_rmse}{stop}"
                )

    elapsed = time.monotonic() - t_start

    main_reporting.write_all_outputs(
        output_dir=output_dir, args=args, provenance=provenance, profile=profile,
        arm_outcomes=arm_outcomes, dataset_diagnostics_by_seed=dataset_diagnostics_by_seed,
        search_space_report=search_space_size_report(n_qubits, args.max_gates),
        diagnostics_cache=diagnostics_cache, task=task, readout_qubit=readout_qubit,
        elapsed_seconds=elapsed, run_label=run_label,
    )
    store.close()

    print(f"\n{run_label}")
    print(run_label_extra)
    print("Mode: main joint structure-and-theta search (NO optimizer, NO classical layers).")
    if use_real:
        total_tokens = sum(
            (c.input_tokens or 0) + (c.output_tokens or 0)
            for o in arm_outcomes for c in o.candidates
        )
        total_latency = sum(
            c.api_latency_seconds or 0.0 for o in arm_outcomes for c in o.candidates
        )
        successful = sum(o.api_successful_calls for o in arm_outcomes)
        failed = sum(o.api_failed_calls for o in arm_outcomes)
        retried = getattr(shared_real_provider, "retried_logical_calls", 0)
        print(
            f"Real LLM calls: {successful} ok / {failed} failed / {retried} retried "
            f"(attempt cap {MAX_REAL_LLM_CALLS}); total tokens: {total_tokens}; "
            f"total API latency: {total_latency:.2f}s; model: {real_model}"
        )
    else:
        print("Zero real API calls were made.")
    print(f"Total elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
