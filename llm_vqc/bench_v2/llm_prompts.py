"""Versioned prompt construction + deterministic feedback summarizer for
the bench_v2 LLM arms (protocol §7).

**No test-derived information of any kind may enter any prompt** — the
builders here accept only grammar constants, budget state, and
validation-side archive/history records; there is no parameter through
which a test metric could arrive (the same by-construction quarantine the
rest of the pipeline uses). Locked by tests.

Prompt version: bump on ANY wording/format change; the version string is
logged with every call record.
"""

from __future__ import annotations

import json

from llm_vqc.bench_v2.space import (
    ENTANGLE_GATES,
    ENTANGLE_PATTERNS,
    ROT_GATES,
    SpaceProfile,
)

BENCH_V2_PROMPT_VERSION = "bench_v2_prompt_v1"

_STRUCTURE_SYSTEM = """You design variational quantum circuit ARCHITECTURES.

Model contract (fixed, not yours to change): amplitude encoding of a
2**n-point real signal on n qubits; your proposed body; measurement of
Pauli-Z on qubit 0; prediction mu_hat = (1 - <Z0>)/2 in [0,1]. Rotation
angles are NOT proposed by you - a fixed AdamW training protocol fits
them. Your goal: propose architectures whose TRAINED validation metric
(lower is better) is as small as possible.

Output format: a single JSON object, no prose, no markdown fences:
{"candidates": [{"operations": [OP, ...]}, ...]}
with exactly %(batch)d candidate architectures, each 1..%(max_ops)d operations.

Each OP is one of:
  {"type": "rot", "gates": [G], "wires": "all" | [q, ...]}
      with G in %(rot_gates)s
  {"type": "entangle", "pattern": P, "gate": E, "wires": "all",
   "center": q (star only), "pairs": [[a,b],...] (pairs only)}
      with P in %(patterns)s and E in %(ent_gates)s

Rules: n_qubits = %(n)d; wire indices 0..%(n_max)d; no "repeat" blocks;
diverse candidates are better than near-duplicates; duplicates of already
evaluated architectures waste budget."""

_JOINT_SYSTEM = """You design COMPLETE variational quantum circuits: gate
sequence AND numeric rotation angles. There is NO training step - your
angles are evaluated verbatim, so choose them deliberately.

Model contract (fixed): amplitude encoding of a 2**n-point real signal on
n qubits; your gate body; measurement of Pauli-Z on qubit 0; prediction
mu_hat = (1 - <Z0>)/2 in [0,1]. Goal: minimize the validation RMSE.

Output format: a single JSON object, no prose, no markdown fences:
{"candidates": [{"n_qubits": %(n)d, "operations": [OP, ...]}, ...]}
with exactly %(batch)d candidates, each 1..%(max_gates)d operations.

Each OP: {"gate": G, "wires": [...], "theta": angle-or-null}
  single-qubit G in ["H","RX","RY","RZ"] with wires [q];
  controlled G in ["CRX","CRY","CRZ"] with wires [control, target];
  theta in [-3.14159, 3.14159] for RX/RY/RZ/CRX/CRY/CRZ; null for H.

Rules: n_qubits = %(n)d; wire indices 0..%(n_max)d; identical structure
with identical angles counts as a duplicate and wastes budget."""


def structure_system_prompt(profile: SpaceProfile, batch_size: int) -> str:
    return _STRUCTURE_SYSTEM % {
        "batch": batch_size,
        "max_ops": profile.max_ops,
        "rot_gates": json.dumps(list(ROT_GATES)),
        "patterns": json.dumps(list(ENTANGLE_PATTERNS)),
        "ent_gates": json.dumps(list(ENTANGLE_GATES)),
        "n": profile.n_qubits,
        "n_max": profile.n_qubits - 1,
    }


def joint_system_prompt(n_qubits: int, batch_size: int, max_gates: int = 5) -> str:
    return _JOINT_SYSTEM % {
        "batch": batch_size, "n": n_qubits, "n_max": n_qubits - 1, "max_gates": max_gates,
    }


def summarize_archive(archive: list[dict], max_entries: int) -> str:
    """Deterministic compact numeric summary of the top-k archive: rank,
    validation metric, op/gate count, depth summary, and the operations
    themselves for the best few. Validation-side data only."""
    lines = []
    for rank, entry in enumerate(archive[:max_entries]):
        ops_repr = json.dumps(entry.get("operations"))
        lines.append(
            f"#{rank + 1}: val_metric={entry.get('val_metric'):.6f} "
            f"ops={entry.get('n_ops')} params={entry.get('n_params')} "
            f"operations={ops_repr}"
        )
    return "\n".join(lines) if lines else "(no evaluated candidates yet)"


def open_loop_user_prompt(proposal_round: int, remaining_unique: int) -> str:
    return (
        f"Round {proposal_round}. Remaining unique-evaluation budget: "
        f"{remaining_unique}. Propose the JSON now."
    )


def closed_loop_user_prompt(
    proposal_round: int,
    remaining_unique: int,
    archive: list[dict],
    failure_counts: dict[str, int],
    n_unique_evaluated: int,
    archive_top_k: int,
) -> str:
    failures = json.dumps(failure_counts, sort_keys=True)
    return (
        f"Round {proposal_round}. Remaining unique-evaluation budget: "
        f"{remaining_unique}. Unique architectures evaluated so far: "
        f"{n_unique_evaluated}.\n"
        f"Outcome counts: {failures}\n"
        f"Current top-{archive_top_k} archive (validation metric, lower is "
        f"better):\n{summarize_archive(archive, archive_top_k)}\n"
        "Learn from the archive: keep what works, vary what might improve "
        "it, avoid duplicates. Propose the JSON now."
    )
