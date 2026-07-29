"""Track B: direct joint structure-and-theta evaluation (protocol §5).

A thin, import-audited facade over the preserved
`llm_vqc.free_amplitude.main_eval` — complete candidates carry gates,
wires, AND theta; theta is loaded verbatim; **no optimizer exists on this
path** (this module, like `main_eval`, never imports `torch.optim`; the
main-mode AST test extends to it). Candidate identity includes theta
(`candidate_hash`), so identical structure with different angles is a
fresh candidate while re-proposing identical angles is a duplicate.

This facade exists so every bench_v2 runner imports Track B through one
module (contract C04) with the bench_v2 seed-pairing helper attached,
rather than each runner re-plumbing main_eval.
"""

from __future__ import annotations

from llm_vqc.evaluation.seeds import derive_child_seed
from llm_vqc.free_amplitude.main_eval import (
    FixedThetaEvalResult,
    MainModeCache,
    evaluate_complete_candidate,
)

__all__ = [
    "FixedThetaEvalResult",
    "MainModeCache",
    "evaluate_complete_candidate",
    "bench_v2_joint_run_seed",
]


def bench_v2_joint_run_seed(task_name: str, data_seed: int, search_seed: int) -> int:
    """Track-B run seed: same pairing law as Track A but a distinct label,
    so joint-track candidate caches can never collide with structure-track
    caches for the same (task, seeds) block."""
    return derive_child_seed(data_seed, "bench_v2_joint_run", task_name, str(search_seed))
