"""The `mini_5gate_v1` search space: exactly five gates, three or five
qubits, seven gate types, angles carried by the proposal itself.

This is a NEW namespace, deliberately separate from `bench_v2`. bench_v2
allows 1..max_ops operations and its own grammar; nothing here writes to
a bench_v2 store or changes a bench_v2 result. The point of the namespace
is that every arm — classical and LLM — faces the identical, trivially
stateable constraint set, so a slide can print the rules in four lines
and a reader can check any proposed circuit against them by eye.
"""

from __future__ import annotations

import numpy as np

#: Namespace tag written into every artifact this package produces.
MINI_SPACE_VERSION = "mini_5gate_v1"

#: Exactly this many gates. Not a maximum, not a range — every candidate
#: from every arm has this many, so circuit size is not a free variable
#: and cannot confound the comparison.
EXACT_GATES = 5

SINGLE_QUBIT_GATES = ("H", "RX", "RY", "RZ")
CONTROLLED_GATES = ("CRX", "CRY", "CRZ")
ALL_GATES = SINGLE_QUBIT_GATES + CONTROLLED_GATES

#: H is the only gate without an angle; every other gate carries one.
PARAMETRIC_GATES = tuple(g for g in ALL_GATES if g != "H")

THETA_LOW = -np.pi
THETA_HIGH = np.pi


def sample_operation(rng: np.random.Generator, n_qubits: int) -> dict:
    """One uniformly sampled gate. Controlled gates need two distinct
    wires, which requires n_qubits >= 2."""
    gate = str(rng.choice(ALL_GATES))
    if gate in CONTROLLED_GATES:
        control, target = rng.choice(n_qubits, size=2, replace=False)
        wires = [int(control), int(target)]
    else:
        wires = [int(rng.integers(n_qubits))]
    theta = None if gate == "H" else float(rng.uniform(THETA_LOW, THETA_HIGH))
    return {"gate": gate, "wires": wires, "theta": theta}


def sample_circuit(rng: np.random.Generator, n_qubits: int,
                   min_gates: int = EXACT_GATES,
                   max_gates: int = EXACT_GATES) -> list[dict]:
    """Uniform over the allowed lengths, then uniform over gates.

    With min == max this is the fixed-length space. With min < max the
    length itself becomes a search dimension - which is exactly the
    confound the fixed-length space was built to remove, so the variable
    setting exists to measure that confound, not to hide it.
    """
    k = int(rng.integers(min_gates, max_gates + 1))
    return [sample_operation(rng, n_qubits) for _ in range(k)]


def mutate_circuit(rng: np.random.Generator, ops: list[dict], n_qubits: int,
                   min_gates: int = EXACT_GATES,
                   max_gates: int = EXACT_GATES) -> list[dict]:
    """One local change.

    Three length-preserving kinds are always available: perturb one angle,
    replace one gate wholesale, or rewire one gate. `add` and `remove` are
    offered only when the length is allowed to vary, so at fixed length the
    mutation operator cannot smuggle in a size change.
    """
    out = [dict(op) for op in ops]
    kinds = ["theta_perturb", "gate_replace", "rewire"]
    if max_gates > min_gates:
        if len(out) < max_gates:
            kinds.append("add")
        if len(out) > min_gates:
            kinds.append("remove")
    kind = str(rng.choice(kinds))
    if kind == "add":
        out.insert(int(rng.integers(len(out) + 1)), sample_operation(rng, n_qubits))
        return out
    if kind == "remove":
        del out[int(rng.integers(len(out)))]
        return out
    i = int(rng.integers(len(out)))

    if kind == "theta_perturb" and out[i]["theta"] is not None:
        shifted = out[i]["theta"] + float(rng.normal(0.0, 0.3))
        # wrap into [-pi, pi] so a long random walk cannot drift out of range
        out[i]["theta"] = float((shifted + np.pi) % (2 * np.pi) - np.pi)
    elif kind == "rewire":
        gate = out[i]["gate"]
        if gate in CONTROLLED_GATES:
            control, target = rng.choice(n_qubits, size=2, replace=False)
            out[i]["wires"] = [int(control), int(target)]
        else:
            out[i]["wires"] = [int(rng.integers(n_qubits))]
    else:
        out[i] = sample_operation(rng, n_qubits)
    return out


def grammar_issues(ops: object, n_qubits: int,
                   min_gates: int = EXACT_GATES,
                   max_gates: int = EXACT_GATES) -> list[str]:
    """Return human-readable reasons `ops` is not a legal candidate.

    Applied identically to classical and LLM proposals, so an LLM cannot
    buy expressivity the samplers cannot reach, and a malformed LLM reply
    becomes an ordinary invalid proposal rather than a crash.
    """
    issues: list[str] = []
    if not isinstance(ops, list):
        return ["operations must be a list"]
    if not (min_gates <= len(ops) <= max_gates):
        want = (f"exactly {min_gates}" if min_gates == max_gates
                else f"{min_gates}..{max_gates}")
        issues.append(f"must contain {want} gates, got {len(ops)}")
    for k, op in enumerate(ops):
        if not isinstance(op, dict):
            issues.append(f"op {k}: not an object")
            continue
        gate = op.get("gate")
        wires = op.get("wires")
        theta = op.get("theta")
        if gate not in ALL_GATES:
            issues.append(f"op {k}: unknown gate {gate!r}")
            continue
        expected_wires = 2 if gate in CONTROLLED_GATES else 1
        if not isinstance(wires, list) or len(wires) != expected_wires:
            issues.append(f"op {k}: {gate} needs {expected_wires} wire(s)")
            continue
        if any(not isinstance(w, int) or not (0 <= w < n_qubits) for w in wires):
            issues.append(f"op {k}: wire out of range 0..{n_qubits - 1}")
            continue
        if expected_wires == 2 and wires[0] == wires[1]:
            issues.append(f"op {k}: control and target must differ")
        if gate == "H":
            if theta is not None:
                issues.append(f"op {k}: H takes no angle")
        else:
            if not isinstance(theta, (int, float)):
                issues.append(f"op {k}: {gate} needs a numeric angle")
            elif not (THETA_LOW - 1e-9 <= float(theta) <= THETA_HIGH + 1e-9):
                issues.append(f"op {k}: angle outside [-pi, pi]")
    return issues
