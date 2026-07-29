"""Search-space profiles for bench_v2 (protocol §4).

Two frozen grammars:

- `compact_free_v1` — the preserved free-gate space (1–5 explicit-wire
  gates from H/RX/RY/RZ/CRX/CRY/CRZ); validation and IR conversion are
  the preserved `llm_vqc.free_amplitude.schema` functions, reused as-is.
- `scalable_layered_v1` — n ∈ {3..8}; a structure proposal is an ordered
  list of at most `min(2n, 16)` layer operations drawn from the shared IR
  grammar (`rot` / `entangle` layers; NO repeat blocks), with the
  encoding (amplitude on all wires) and measurement (Z on qubit 0) fixed
  by the profile, never proposed. Every arm proposes in this exact
  grammar; validity = the shared IR validators plus the profile checks
  here — no arm has private actions.

The profile also owns the fixed **reference ansätze** (protocol §6):
RealAmplitudes-style (RY layer + CNOT line) and StronglyEntangling-style
(RZ-RY-RZ layers + CNOT ring) at depths 1 and 2, expressed in the same
grammar so their resource metrics and evaluation path are identical to
searched candidates'.
"""

from __future__ import annotations

from dataclasses import dataclass

from llm_vqc.ir.schema import CircuitIR
from llm_vqc.ir.validators import ValidationIssue, ValidationResult, validate_proposal

SCALABLE_LAYERED_V1 = "scalable_layered_v1"
COMPACT_FREE_V1 = "compact_free_v1"

SCALABLE_QUBITS = (3, 4, 5, 6, 8)
ROT_GATES = ("RX", "RY", "RZ", "H")
ENTANGLE_GATES = ("CNOT", "CZ", "CRX", "CRY", "CRZ")
ENTANGLE_PATTERNS = ("line", "ring", "star", "pairs", "all_to_all")
READOUT_QUBIT = 0


def max_ops_for(n_qubits: int) -> int:
    """Frozen rule (protocol §4): max_ops = min(2*n, 16)."""
    return min(2 * n_qubits, 16)


@dataclass(frozen=True)
class SpaceProfile:
    name: str
    n_qubits: int

    def __post_init__(self) -> None:
        if self.name not in (SCALABLE_LAYERED_V1, COMPACT_FREE_V1):
            raise ValueError(f"unknown space profile {self.name!r}")
        if self.name == SCALABLE_LAYERED_V1 and self.n_qubits not in SCALABLE_QUBITS:
            raise ValueError(
                f"{SCALABLE_LAYERED_V1} is frozen for n in {SCALABLE_QUBITS}, "
                f"got {self.n_qubits}"
            )

    @property
    def max_ops(self) -> int:
        return 5 if self.name == COMPACT_FREE_V1 else max_ops_for(self.n_qubits)

    @property
    def max_quantum_parameters(self) -> int:
        """Upper bound on trainable angles: every op parameterized, rot
        layers touching every wire (entangle layers contribute at most
        n pairs for ring/pairs; all_to_all can exceed n but is bounded by
        n*(n-1)/2)."""
        if self.name == COMPACT_FREE_V1:
            return 5
        per_op = max(self.n_qubits, self.n_qubits * (self.n_qubits - 1) // 2)
        return self.max_ops * per_op


def layered_ir_dict(n_qubits: int, operations: list[dict]) -> dict:
    """Assemble the full CircuitIR raw dict from a structure-only layered
    proposal: encoding and measurement are profile-fixed, NOT proposable."""
    return {
        "n_qubits": n_qubits,
        "encoding": {"type": "amplitude", "wires": "all"},
        "layers": operations,
        "measurements": {"wires": [READOUT_QUBIT], "observable": "Z"},
    }


def validate_layered_structure(
    raw_operations: object, profile: SpaceProfile
) -> tuple[CircuitIR | None, list[ValidationIssue]]:
    """Profile checks + shared IR validation; collects ALL issues."""
    issues: list[ValidationIssue] = []
    if not isinstance(raw_operations, list) or not raw_operations:
        issues.append(
            ValidationIssue(
                code="bench_v2.operations_not_list",
                path="operations",
                message="operations must be a non-empty list of layer objects",
            )
        )
        return None, issues
    if len(raw_operations) > profile.max_ops:
        issues.append(
            ValidationIssue(
                code="bench_v2.max_ops_exceeded",
                path="operations",
                message=(
                    f"{len(raw_operations)} operations exceed max_ops="
                    f"{profile.max_ops} for n_qubits={profile.n_qubits}"
                ),
            )
        )
    for i, op in enumerate(raw_operations):
        if isinstance(op, dict) and op.get("type") == "repeat":
            issues.append(
                ValidationIssue(
                    code="bench_v2.repeat_forbidden",
                    path=f"operations[{i}]",
                    message="repeat blocks are not part of scalable_layered_v1",
                )
            )
    if issues:
        return None, issues

    result: ValidationResult = validate_proposal(
        layered_ir_dict(profile.n_qubits, raw_operations)
    )
    if not result.valid:
        return None, list(result.issues)
    return result.ir, []


# --- Fixed reference ansätze (protocol §6) --------------------------------

REFERENCE_ARMS = (
    "ref_realamp_d1", "ref_realamp_d2", "ref_strongent_d1", "ref_strongent_d2",
)


def reference_operations(ref_name: str) -> list[dict]:
    """The reference circuit bodies, in the shared layered grammar."""
    realamp_block = [
        {"type": "rot", "gates": ["RY"], "wires": "all"},
        {"type": "entangle", "pattern": "line", "gate": "CNOT", "wires": "all"},
    ]
    strongent_block = [
        {"type": "rot", "gates": ["RZ", "RY", "RZ"], "wires": "all"},
        {"type": "entangle", "pattern": "ring", "gate": "CNOT", "wires": "all"},
    ]
    table = {
        "ref_realamp_d1": realamp_block,
        "ref_realamp_d2": realamp_block * 2,
        "ref_strongent_d1": strongent_block,
        "ref_strongent_d2": strongent_block * 2,
    }
    if ref_name not in table:
        raise ValueError(f"unknown reference ansatz {ref_name!r}")
    return [dict(op) for op in table[ref_name]]


def reference_ir(ref_name: str, profile: SpaceProfile) -> CircuitIR:
    ir, issues = validate_layered_structure(reference_operations(ref_name), profile)
    if ir is None:  # references are within-grammar by construction
        raise ValueError(f"reference {ref_name!r} failed validation: {issues}")
    return ir
