#!/usr/bin/env python
"""Real-provider preflight: does the pinned model return parseable,
within-grammar candidate batches?

Deliberately narrow. This script:

* issues ONLY proposal calls and validates the JSON that comes back;
* never builds a task, never trains, never imports the protected-test
  gate (asserted below by module inspection);
* runs under its own small cumulative ledger cap (default $0.02) that is
  separate from the matrix cap;
* reports validity rates so a broken prompt or schema is caught before
  the matrix spends anything.

It must never be used to choose or switch models: it measures whether
output is *well formed*, not whether circuits are *good*. No validation
or test metric is computed here at all.

Usage:
    LLM_API_BUDGET_USD=0.02 python scripts/bench_v2/preflight_real_provider.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from llm_vqc.bench_v2.llm_arms import BATCH_SIZE, parse_candidate_batch  # noqa: E402
from llm_vqc.bench_v2.llm_prompts import (  # noqa: E402
    BENCH_V2_PROMPT_VERSION,
    joint_system_prompt,
    open_loop_user_prompt,
    structure_system_prompt,
)
from llm_vqc.bench_v2.llm_providers import build_real_provider  # noqa: E402
from llm_vqc.bench_v2.space import SpaceProfile, validate_layered_structure  # noqa: E402
from llm_vqc.free_amplitude.candidate_schema import (  # noqa: E402
    validate_complete_candidate,
)
from llm_vqc.llm.global_ledger import GlobalSpendLedger, LedgerCapExceeded  # noqa: E402

PREFLIGHT_LEDGER = REPO / "runs" / "bench_v2" / "preflight_ledger.sqlite"
DEFAULT_CAP = 0.02


def _assert_no_training_or_test_imports() -> None:
    """This module must not be able to reach training or the test gate."""
    banned = ("test_gate", "final_test", "track_a_evaluator", "cell_runner")
    source = Path(__file__).read_text()
    for name in banned:
        if f"import {name}" in source or f"from llm_vqc.bench_v2.{name}" in source:
            raise AssertionError(f"preflight must not import {name}")


def check_structure(provider, n_qubits: int) -> dict:
    profile = SpaceProfile("scalable_layered_v1", n_qubits)
    system = structure_system_prompt(profile, BATCH_SIZE)
    user = open_loop_user_prompt(1, 16)
    response = provider.complete(system, user, 1.0)
    candidates = parse_candidate_batch(response.raw_text)
    result = {
        "track": "structure", "n_qubits": n_qubits,
        "input_tokens": response.input_tokens, "output_tokens": response.output_tokens,
        "parseable": candidates is not None,
        "n_candidates": len(candidates) if candidates else 0,
        "valid": 0, "invalid": 0, "issues": [],
    }
    for cand in candidates or []:
        ir, issues = validate_layered_structure(cand.get("operations"), profile)
        if ir is not None:
            result["valid"] += 1
        else:
            result["invalid"] += 1
            result["issues"].extend(i.code for i in issues[:2])
    return result


def check_joint(provider, n_qubits: int) -> dict:
    system = joint_system_prompt(n_qubits, BATCH_SIZE)
    user = open_loop_user_prompt(1, 16)
    response = provider.complete(system, user, 1.0)
    candidates = parse_candidate_batch(response.raw_text)
    result = {
        "track": "joint", "n_qubits": n_qubits,
        "input_tokens": response.input_tokens, "output_tokens": response.output_tokens,
        "parseable": candidates is not None,
        "n_candidates": len(candidates) if candidates else 0,
        "valid": 0, "invalid": 0, "issues": [],
    }
    for cand in candidates or []:
        payload = dict(cand)
        payload.setdefault("n_qubits", n_qubits)
        outcome = validate_complete_candidate(payload, expected_n_qubits=n_qubits)
        if outcome.valid:
            result["valid"] += 1
        else:
            result["invalid"] += 1
            result["issues"].extend(i.code for i in outcome.issues[:2])
    return result


def main() -> int:
    _assert_no_training_or_test_imports()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cap-usd", type=float, default=DEFAULT_CAP)
    ap.add_argument("--out", type=Path,
                    default=REPO / "outputs" / "bench_v2" / "preflight_report.json")
    args = ap.parse_args()

    ledger = GlobalSpendLedger(PREFLIGHT_LEDGER, cap_usd=args.cap_usd,
                               run_label="bench_v2_preflight")
    print(f"preflight ledger {PREFLIGHT_LEDGER.name}: cap ${ledger.cap_usd:.4f}, "
          f"already committed ${ledger.totals().committed_usd:.6f}")

    checks: list[dict] = []
    stopped = None
    try:
        struct_provider, config = build_real_provider("structure", ledger=ledger,
                                                      cell_id="preflight_structure")
        print(f"model pinned: {config.model}  prompt: {BENCH_V2_PROMPT_VERSION}")
        for n in (3, 8):
            checks.append(check_structure(struct_provider, n))
            print(f"  structure n={n}: {checks[-1]['valid']}/{checks[-1]['n_candidates']} valid")
        joint_provider, _ = build_real_provider("joint", ledger=ledger,
                                                cell_id="preflight_joint")
        for n in (3, 5):
            checks.append(check_joint(joint_provider, n))
            print(f"  joint n={n}: {checks[-1]['valid']}/{checks[-1]['n_candidates']} valid")
    except LedgerCapExceeded as exc:
        stopped = f"preflight cap reached: {exc}"
        print(f"  STOP {stopped}")

    summary = ledger.summary()
    ok = (
        stopped is None
        and len(checks) == 4
        and all(c["parseable"] and c["n_candidates"] > 0 and c["valid"] > 0 for c in checks)
    )
    report = {
        "pinned_model": "gpt-5-nano-2025-08-07",
        "prompt_version": BENCH_V2_PROMPT_VERSION,
        "batch_size": BATCH_SIZE,
        "checks": checks,
        "ledger": summary,
        "stopped": stopped,
        "passed": ok,
        "note": "Schema/validity only. No task built, no training, no protected-test access.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"\ncalls={summary['calls_settled']} "
          f"in={summary['input_tokens']} out={summary['output_tokens']} "
          f"spent=${summary['settled_usd']:.6f} of ${summary['cap_usd']:.4f}")
    print("PREFLIGHT PASSED" if ok else "PREFLIGHT FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
