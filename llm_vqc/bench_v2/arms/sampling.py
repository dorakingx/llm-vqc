"""Frozen sampling and mutation grammar for `scalable_layered_v1`
(protocol §4: "Freeze grammar version, mutation operators, size priors,
and validity rules before the main matrix").

Size prior: op-list length ~ Uniform{1..max_ops}. Per op: rot vs entangle
with probability 1/2 each; rot gate ~ Uniform(RX,RY,RZ,H) on "all" wires
w.p. 1/2 or a uniformly drawn non-empty wire subset; entangle gate ~
Uniform(CNOT,CZ,CRX,CRY,CRZ) with pattern ~ Uniform(line,ring,star,
pairs,all_to_all) on all wires (star gets a uniform center; pairs gets
1..floor(n/2) disjoint uniform pairs). Every draw is rejection-checked by
the SAME `validate_layered_structure` every other proposal source goes
through.

Mutation operators (evolutionary/greedy arms; uniform choice among the
applicable ones): ADD a sampled op (if below max_ops), REMOVE a uniform
op (if length > 1), REPLACE a uniform op with a fresh sample, RETYPE the
gate of a uniform op within its own family, REWIRE a uniform op (fresh
wires/pattern fields). Crossover: single-point over op lists, truncated
to max_ops.
"""

from __future__ import annotations

import numpy as np

from llm_vqc.bench_v2.space import (
    ENTANGLE_GATES,
    ENTANGLE_PATTERNS,
    ROT_GATES,
    SpaceProfile,
    validate_layered_structure,
)

MAX_SAMPLE_ATTEMPTS = 200


class LayeredSamplingError(Exception):
    pass


def _sample_wire_subset(rng: np.random.Generator, n_qubits: int) -> list[int]:
    size = int(rng.integers(1, n_qubits + 1))
    return sorted(rng.choice(n_qubits, size=size, replace=False).tolist())


def sample_rot_op(rng: np.random.Generator, n_qubits: int) -> dict:
    gate = str(rng.choice(ROT_GATES))
    wires = "all" if rng.random() < 0.5 else _sample_wire_subset(rng, n_qubits)
    return {"type": "rot", "gates": [gate], "wires": wires}


def sample_entangle_op(rng: np.random.Generator, n_qubits: int) -> dict:
    gate = str(rng.choice(ENTANGLE_GATES))
    pattern = str(rng.choice(ENTANGLE_PATTERNS))
    op: dict = {"type": "entangle", "pattern": pattern, "gate": gate, "wires": "all"}
    if pattern == "star":
        op["center"] = int(rng.integers(0, n_qubits))
    if pattern == "pairs":
        max_pairs = n_qubits // 2
        k = int(rng.integers(1, max_pairs + 1))
        wires = rng.permutation(n_qubits)[: 2 * k]
        op["pairs"] = [
            [int(wires[2 * i]), int(wires[2 * i + 1])] for i in range(k)
        ]
    return op


def sample_layered_op(rng: np.random.Generator, n_qubits: int) -> dict:
    if rng.random() < 0.5:
        return sample_rot_op(rng, n_qubits)
    return sample_entangle_op(rng, n_qubits)


def sample_layered_operations(rng: np.random.Generator, profile: SpaceProfile) -> list[dict]:
    """Draw one valid structure (op list), rejection-checked against the
    shared validator."""
    for _ in range(MAX_SAMPLE_ATTEMPTS):
        length = int(rng.integers(1, profile.max_ops + 1))
        ops = [sample_layered_op(rng, profile.n_qubits) for _ in range(length)]
        ir, _issues = validate_layered_structure(ops, profile)
        if ir is not None:
            return ops
    raise LayeredSamplingError(
        f"no valid sample for {profile.name} n={profile.n_qubits} "
        f"after {MAX_SAMPLE_ATTEMPTS} attempts"
    )


# --- mutation operators ----------------------------------------------------

MUTATION_KINDS = ("add", "remove", "replace", "retype", "rewire")


def _retype(rng: np.random.Generator, op: dict) -> dict:
    new = {k: (list(v) if isinstance(v, list) else v) for k, v in op.items()}
    if op["type"] == "rot":
        new["gates"] = [str(rng.choice(ROT_GATES))]
    else:
        new["gate"] = str(rng.choice(ENTANGLE_GATES))
    return new


def _rewire(rng: np.random.Generator, op: dict, n_qubits: int) -> dict:
    if op["type"] == "rot":
        fresh = sample_rot_op(rng, n_qubits)
        fresh["gates"] = list(op["gates"])
        return fresh
    fresh = sample_entangle_op(rng, n_qubits)
    fresh["gate"] = op["gate"]
    return fresh


def mutate_operations(
    rng: np.random.Generator, operations: list[dict], profile: SpaceProfile
) -> list[dict]:
    """One uniformly chosen applicable mutation; rejection-checked. Falls
    back to a fresh sample if no valid mutant emerges in the attempt cap
    (recorded implicitly: the fallback is still within-grammar)."""
    for _ in range(MAX_SAMPLE_ATTEMPTS):
        ops = [dict(op) for op in operations]
        applicable = [
            kind for kind in MUTATION_KINDS
            if not (kind == "add" and len(ops) >= profile.max_ops)
            and not (kind == "remove" and len(ops) <= 1)
        ]
        kind = str(rng.choice(applicable))
        if kind == "add":
            ops.insert(int(rng.integers(0, len(ops) + 1)),
                       sample_layered_op(rng, profile.n_qubits))
        elif kind == "remove":
            ops.pop(int(rng.integers(0, len(ops))))
        elif kind == "replace":
            ops[int(rng.integers(0, len(ops)))] = sample_layered_op(rng, profile.n_qubits)
        elif kind == "retype":
            i = int(rng.integers(0, len(ops)))
            ops[i] = _retype(rng, ops[i])
        else:  # rewire
            i = int(rng.integers(0, len(ops)))
            ops[i] = _rewire(rng, ops[i], profile.n_qubits)
        ir, _issues = validate_layered_structure(ops, profile)
        if ir is not None:
            return ops
    return sample_layered_operations(rng, profile)


def crossover_operations(
    rng: np.random.Generator, parent_a: list[dict], parent_b: list[dict],
    profile: SpaceProfile,
) -> list[dict]:
    """Single-point crossover, truncated to max_ops, rejection-checked;
    falls back to a mutation of parent_a if no valid child emerges."""
    for _ in range(MAX_SAMPLE_ATTEMPTS):
        cut_a = int(rng.integers(0, len(parent_a) + 1))
        cut_b = int(rng.integers(0, len(parent_b) + 1))
        child = [dict(op) for op in parent_a[:cut_a]] + [dict(op) for op in parent_b[cut_b:]]
        child = child[: profile.max_ops]
        if not child:
            continue
        ir, _issues = validate_layered_structure(child, profile)
        if ir is not None:
            return child
    return mutate_operations(rng, parent_a, profile)
