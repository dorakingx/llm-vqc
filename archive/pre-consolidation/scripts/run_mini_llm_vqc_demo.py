#!/usr/bin/env python
"""Mini real-LLM-API VQC architecture-search demo (`mini_llm_api_vqc_demo_v1`).

An integration demonstration, not a statistically powered research
experiment: 1 seed, budget=2 per arm, 5 epochs per candidate, at most 4
real OpenAI calls. It exercises the full loop -- real LLM API call ->
compact architecture proposal -> deterministic CircuitIR conversion ->
VQC parameter training (AdamW) -> validation-RMSE-based selection ->
one-shot protected-test evaluation -- end to end in a few minutes.

All the actual search/train/select logic lives in
`llm_vqc.mini_demo.runner` (imported below) so it can be exercised by
`tests/` with a fake in-process provider; this file only wires up the
real OpenAI provider, CLI arguments, and console/file output. Mocks are
never used in this script itself -- only in unit tests.

Does NOT modify or read any T1/T2/HIGGS/qualification run directory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from llm_vqc.evaluation.store import ResultStore  # noqa: E402
from llm_vqc.evaluation.training import TrainingConfig  # noqa: E402
from llm_vqc.llm.openai_provider import CallCountLimitedProvider  # noqa: E402
from llm_vqc.mini_demo import reporting  # noqa: E402
from llm_vqc.mini_demo.prompts import PROMPT_VERSION  # noqa: E402
from llm_vqc.mini_demo.provenance import collect_provenance  # noqa: E402
from llm_vqc.mini_demo.provider import CompactSchemaOpenAIProvider  # noqa: E402
from llm_vqc.mini_demo.runner import (  # noqa: E402
    PreflightError,
    evaluate_protected_test,
    preflight_checks,
    run_llm_closed_loop_arm,
    run_llm_open_loop_arm,
    run_random_arm,
)
from llm_vqc.tasks.t1_gaussian import T1GaussianPeakTask  # noqa: E402

TASK_NAME = "T1_mini_demo"
TASK_DESCRIPTION = "T1 1D Gaussian-peak regression (predict the peak location, mu)"
MAX_REAL_LLM_CALLS = 4
WALL_CLOCK_LIMIT_S = 300

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["openai"], default="openai")
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="outputs/mini_llm_api_vqc_demo_v1")
    args = parser.parse_args()

    t_start = time.monotonic()
    deadline = t_start + WALL_CLOCK_LIMIT_S

    # Loads a local, gitignored .env if present (never overrides a real
    # already-exported env var); this is the only place any secret is
    # read, and it is never printed, logged, or written back out.
    load_dotenv(_REPO_ROOT / ".env")

    # Preflight: require OPENAI_API_KEY, an explicit OPENAI_MODEL (no silent
    # default), and a positive LLM_API_BUDGET_USD -- all BEFORE any provider
    # is constructed or any request issued. Raises PreflightError otherwise.
    try:
        preflight = preflight_checks(os.environ)
    except PreflightError as exc:
        print(f"BLOCKED: {exc}")
        sys.exit(1)
    api_key = preflight.api_key
    model = preflight.model
    budget = preflight.budget

    output_dir = _REPO_ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    run_dir = _REPO_ROOT / "runs" / "mini_llm_api_vqc_demo_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    db_path = run_dir / "results.sqlite"

    provenance = collect_provenance(_REPO_ROOT)
    if provenance.get("execution_git_dirty"):
        print(
            "WARNING: worktree is dirty -- artifacts will record "
            "execution_git_dirty=true. For a clean-SHA real run, commit the "
            "implementation first, then run from that committed SHA."
        )
    task = T1GaussianPeakTask()
    train_val = task.build(seed=args.seed)
    test_split = task.build_test(seed=args.seed)
    training_config = TrainingConfig(epochs=args.epochs, batch_size=16)

    store = ResultStore.open_or_create(
        db_path, run_id="mini_llm_api_vqc_demo_v1", config_json=json.dumps(vars(args)),
        config_reproducibility_fields={
            "task": TASK_NAME, "epochs": args.epochs, "seed": args.seed, "budget": args.budget,
        },
        git_sha=provenance.get("execution_code_sha"),
        created_at=_now_iso(), allow_incompatible=True,
    )

    real_provider = CompactSchemaOpenAIProvider(api_key=api_key, model=model)
    shared_provider = CallCountLimitedProvider(real_provider, max_calls=MAX_REAL_LLM_CALLS)

    print(
        f"Mini LLM-VQC API demo starting: model={model} budget_per_arm={args.budget} "
        f"epochs={args.epochs} seed={args.seed} max_real_calls={MAX_REAL_LLM_CALLS} "
        f"dollar_cap=${budget.cap_usd:.2f}"
    )

    random_outcome = run_random_arm(
        args.seed, TASK_NAME, train_val, training_config, args.budget, store, deadline
    )
    print(f"[random] {random_outcome.ledger_summary} stop={random_outcome.stop_reason}")

    open_loop_outcome = run_llm_open_loop_arm(
        args.seed, TASK_NAME, train_val, training_config, args.budget, store,
        shared_provider, deadline, TASK_DESCRIPTION, budget=budget,
    )
    print(
        f"[llm_open_loop] {open_loop_outcome.ledger_summary} stop={open_loop_outcome.stop_reason}"
    )

    closed_loop_outcome = run_llm_closed_loop_arm(
        args.seed, TASK_NAME, train_val, training_config, args.budget, store,
        shared_provider, deadline, TASK_DESCRIPTION, budget=budget,
    )
    print(
        f"[llm_closed_loop] {closed_loop_outcome.ledger_summary} "
        f"stop={closed_loop_outcome.stop_reason}"
    )

    for outcome in (random_outcome, open_loop_outcome, closed_loop_outcome):
        evaluate_protected_test(outcome, store, TASK_NAME, test_split, train_val.spec)
        print(
            f"[{outcome.arm}] selected_hash={outcome.selected_structural_hash} "
            f"val_rmse={outcome.selected_val_rmse} test_rmse={outcome.protected_test_rmse}"
        )

    elapsed = time.monotonic() - t_start
    real_calls_made = shared_provider.calls_made
    arm_outcomes = [random_outcome, open_loop_outcome, closed_loop_outcome]

    reporting.write_all_outputs(
        output_dir=output_dir, store=store,
        args=args, provenance=provenance, model=model, real_calls_made=real_calls_made,
        max_real_calls=MAX_REAL_LLM_CALLS, elapsed_seconds=elapsed, prompt_version=PROMPT_VERSION,
        arm_outcomes=arm_outcomes, task_name=TASK_NAME, run_status="corrected",
    )
    total_tokens = sum(
        (c.input_tokens or 0) + (c.output_tokens or 0)
        for o in arm_outcomes for c in o.candidates
    )
    total_api_latency = sum(
        c.api_latency_seconds or 0.0 for o in arm_outcomes for c in o.candidates
    )
    store.close()

    print("\nMini LLM-VQC API demo completed")
    print("Task: T1 Gaussian regression")
    print("Qubits: 2")
    print("Quantum parameters: 4")
    print("Total trainable parameters: 51")
    print(f"Epochs per candidate: {args.epochs}")
    print(f"Budget per arm: {args.budget}")
    print(f"{'Arm':<15}{'Best val RMSE':<16}{'Test RMSE':<12}{'Valid/API calls'}")
    for outcome, label in (
        (random_outcome, "Random"),
        (open_loop_outcome, "LLM Open"),
        (closed_loop_outcome, "LLM Closed"),
    ):
        has_val = outcome.selected_val_rmse is not None
        has_test = outcome.protected_test_rmse is not None
        val_s = f"{outcome.selected_val_rmse:.4f}" if has_val else "n/a"
        test_s = f"{outcome.protected_test_rmse:.4f}" if has_test else "n/a"
        n_valid = sum(1 for c in outcome.candidates if c.valid)
        print(f"{label:<15}{val_s:<16}{test_s:<12}{n_valid}/{len(outcome.candidates)}")
    print(f"LLM API calls: {real_calls_made}/{MAX_REAL_LLM_CALLS}")
    print(f"Total tokens: {total_tokens}")
    print(f"Total API latency: {total_api_latency:.2f}s")
    print(f"Total elapsed: {elapsed:.1f}s")
    print()
    print("This is an end-to-end integration demonstration with n=1 seed.")
    print("It does not establish that any search arm is superior.")


if __name__ == "__main__":
    main()
