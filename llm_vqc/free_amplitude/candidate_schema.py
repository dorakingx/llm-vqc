"""The **main-mode** complete-candidate schema: the LLM (or the random
sampler) proposes BOTH the circuit structure AND the numerical rotation
angles `theta`, and the candidate is evaluated at exactly those angles with
NO optimizer.

```json
{
  "n_qubits": 3,
  "operations": [
    {"gate": "H", "wires": [1]},
    {"gate": "CRY", "wires": [1, 0], "theta": 1.247},
    {"gate": "RZ", "wires": [2], "theta": -0.381}
  ]
}
```

Rules (correction pass "Main experiment: LLM chooses gates and theta"):
  - `H` has no `theta` (a `theta` on an `H` op is rejected);
  - `RX/RY/RZ/CRX/CRY/CRZ` require exactly one finite `theta` in
    `[-pi, pi]` (missing, NaN, infinite, or out-of-range `theta` is
    rejected);
  - the proposed `theta` is evaluated verbatim -- no optimizer is ever
    constructed in main mode.

This is a DIFFERENT schema from `llm_vqc.free_amplitude.schema`
(structure-only): that one forbids `theta` entirely and is kept for the
separately-labelled structure-search + AdamW ablation. Two identity
hashes are defined here (correction pass "candidate_hash changes when
theta changes"):
  - `architecture_hash`: gates / order / wires only (theta-independent) --
    used to key the intrinsic expressibility/entanglement cache, which
    does not depend on the proposed angles;
  - `candidate_hash`: architecture plus canonical `theta` values -- so
    two proposals with the same structure but different angles are
    DIFFERENT candidates (they train-MSE / validate differently and must
    not collapse into one another).
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from llm_vqc.free_amplitude.schema import (
    MAX_OPERATIONS,
    MIN_OPERATIONS,
    PARAMETERIZED_GATES,
    SINGLE_WIRE_GATES,
    TWO_WIRE_GATES,
)
from llm_vqc.ir.canonicalize import canonical_json as ir_canonical_json
from llm_vqc.ir.canonicalize import structural_hash as ir_structural_hash
from llm_vqc.ir.schema import (
    CircuitIR,
    EncodingSpec,
    EntangleLayer,
    MeasurementSpec,
    RotationLayer,
)
from llm_vqc.ir.validators import ValidationIssue

COMPLETE_CANDIDATE_SCHEMA_VERSION = "free_amplitude_complete_v1"

#: Rounding for canonical theta values in `candidate_hash` -- enough
#: precision that meaningfully different angles hash differently, coarse
#: enough that float representation noise does not (9 decimals ~ 1e-9 rad).
CANONICAL_THETA_DECIMALS = 9

GateName = Literal["H", "RX", "RY", "RZ", "CRX", "CRY", "CRZ"]


class CompleteGateOperation(BaseModel):
    """One gate application carrying its own angle. `theta` is `None` for
    `H` and a finite float in `[-pi, pi]` for every parameterized gate --
    both enforced by `field_validator` below, so a bad `theta` is rejected
    at parse time, not silently repaired."""

    model_config = ConfigDict(extra="forbid")

    gate: GateName
    wires: list[int]
    theta: float | None = None

    @field_validator("theta")
    @classmethod
    def _theta_must_be_finite_if_present(cls, v: float | None) -> float | None:
        if v is not None and not math.isfinite(v):
            raise ValueError(f"theta must be finite, got {v!r}")
        return v


class CompleteCandidateProposal(BaseModel):
    """A complete candidate (structure + angles), as the main-mode LLM or
    random sampler emits it."""

    model_config = ConfigDict(extra="forbid")

    n_qubits: int = Field(ge=1)
    operations: list[CompleteGateOperation] = Field(
        min_length=MIN_OPERATIONS, max_length=MAX_OPERATIONS
    )


class CompleteCandidateValidationResult(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    valid: bool
    issues: list[ValidationIssue] = []
    proposal: CompleteCandidateProposal | None = None


def _pydantic_errors_to_issues(exc: ValidationError) -> list[ValidationIssue]:
    issues = []
    for error in exc.errors():
        path = ".".join(str(p) for p in error["loc"])
        issues.append(
            ValidationIssue(code=f"schema.{error['type']}", message=error["msg"], path=path)
        )
    return issues


def validate_complete_candidate(
    raw: dict | CompleteCandidateProposal, expected_n_qubits: int
) -> CompleteCandidateValidationResult:
    """Validate a raw complete candidate. Pydantic first (types, finite
    theta), then all semantic checks -- arity, wire bounds, control!=target,
    n_qubits match, adjacent-duplicate rejection, at-least-one-parameterized,
    and the theta presence/range rules -- collecting every violation."""
    if isinstance(raw, CompleteCandidateProposal):
        proposal = raw
    else:
        try:
            proposal = CompleteCandidateProposal.model_validate(raw)
        except ValidationError as exc:
            return CompleteCandidateValidationResult(
                valid=False, issues=_pydantic_errors_to_issues(exc)
            )

    issues = _validate_semantic(proposal, expected_n_qubits)
    if issues:
        return CompleteCandidateValidationResult(valid=False, issues=issues)
    return CompleteCandidateValidationResult(valid=True, issues=[], proposal=proposal)


def _validate_semantic(
    proposal: CompleteCandidateProposal, expected_n_qubits: int
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    if proposal.n_qubits != expected_n_qubits:
        issues.append(
            ValidationIssue(
                code="candidate.n_qubits_mismatch",
                message=(
                    f"proposal n_qubits={proposal.n_qubits} does not match this "
                    f"experiment's configured n_qubits={expected_n_qubits}"
                ),
                path="n_qubits",
            )
        )

    for i, op in enumerate(proposal.operations):
        path = f"operations[{i}]"
        expected_arity = 1 if op.gate in SINGLE_WIRE_GATES else 2
        if len(op.wires) != expected_arity:
            issues.append(
                ValidationIssue(
                    code="candidate.wrong_arity",
                    message=(
                        f"{op.gate} requires exactly {expected_arity} wire(s), "
                        f"got {len(op.wires)}"
                    ),
                    path=f"{path}.wires",
                )
            )
            continue

        for wire in op.wires:
            if not (0 <= wire < expected_n_qubits):
                issues.append(
                    ValidationIssue(
                        code="candidate.wire_out_of_bounds",
                        message=f"wire {wire} out of bounds for n_qubits={expected_n_qubits}",
                        path=f"{path}.wires",
                    )
                )

        if op.gate in TWO_WIRE_GATES and op.wires[0] == op.wires[1]:
            issues.append(
                ValidationIssue(
                    code="candidate.control_equals_target",
                    message=f"{op.gate} control and target must differ, got {op.wires}",
                    path=path,
                )
            )

        # Theta presence + range rules.
        if op.gate in PARAMETERIZED_GATES:
            if op.theta is None:
                issues.append(
                    ValidationIssue(
                        code="candidate.missing_theta",
                        message=f"{op.gate} requires a finite theta in [-pi, pi]",
                        path=f"{path}.theta",
                    )
                )
            elif not (-math.pi <= op.theta <= math.pi):
                issues.append(
                    ValidationIssue(
                        code="candidate.theta_out_of_range",
                        message=f"theta {op.theta} out of range [-pi, pi]",
                        path=f"{path}.theta",
                    )
                )
        else:  # H
            if op.theta is not None:
                issues.append(
                    ValidationIssue(
                        code="candidate.unexpected_theta",
                        message=f"{op.gate} must not carry a theta, got {op.theta}",
                        path=f"{path}.theta",
                    )
                )

    if not any(op.gate in PARAMETERIZED_GATES for op in proposal.operations):
        issues.append(
            ValidationIssue(
                code="candidate.no_parameterized_gate",
                message="at least one parameterized gate is required",
                path="operations",
            )
        )

    # Adjacent reducible duplicates: identical gate + wires AND identical
    # theta (two identical parameterized gates with different angles are
    # NOT trivially reducible, so only reject when the angle also matches).
    ops = proposal.operations
    for i in range(1, len(ops)):
        prev, cur = ops[i - 1], ops[i]
        if prev.gate == cur.gate and prev.wires == cur.wires and prev.theta == cur.theta:
            issues.append(
                ValidationIssue(
                    code="candidate.reducible_adjacent_duplicate",
                    message=(
                        f"operations[{i - 1}] and operations[{i}] are identical "
                        f"({cur.gate}{cur.wires} theta={cur.theta}) -- reducible"
                    ),
                    path=f"operations[{i}]",
                )
            )

    return issues


def complete_candidate_to_ir_and_theta(
    proposal: CompleteCandidateProposal, readout_qubit: int
) -> tuple[CircuitIR, list[float]]:
    """Expand a validated complete candidate into `(CircuitIR, theta_vector)`,
    preserving exact operation order. `theta_vector[k]` is the angle for the
    k-th PARAMETERIZED gate in operation order -- matching
    `llm_vqc.ir.expand.build_program`'s parameter-slot ordering exactly, so
    the model's `q_layer.weights[k]` receives the intended angle."""
    layers: list[RotationLayer | EntangleLayer] = []
    theta_vector: list[float] = []
    for op in proposal.operations:
        if op.gate in SINGLE_WIRE_GATES:
            layers.append(RotationLayer(gates=[op.gate], wires=list(op.wires)))
        else:
            control, target = op.wires
            layers.append(
                EntangleLayer(pattern="pairs", gate=op.gate, pairs=[(control, target)])
            )
        if op.gate in PARAMETERIZED_GATES:
            theta_vector.append(float(op.theta))

    ir = CircuitIR(
        n_qubits=proposal.n_qubits,
        encoding=EncodingSpec(type="amplitude", wires="all"),
        layers=layers,
        measurements=MeasurementSpec(observable="Z", wires=[readout_qubit]),
        metadata={
            "complete_candidate_schema_version": COMPLETE_CANDIDATE_SCHEMA_VERSION,
            "readout_qubit": str(readout_qubit),
        },
    )
    return ir, theta_vector


def architecture_ir(proposal: CompleteCandidateProposal, readout_qubit: int) -> CircuitIR:
    """The structure-only IR (theta-independent). Its `structural_hash` is
    the `architecture_hash`."""
    ir, _ = complete_candidate_to_ir_and_theta(proposal, readout_qubit)
    return ir


def architecture_hash(proposal: CompleteCandidateProposal, readout_qubit: int) -> str:
    """Hash of gates / order / wires only -- theta-independent. Two
    candidates that differ only in their angles share one architecture
    hash (and hence one cached expressibility/entanglement diagnostic)."""
    return ir_structural_hash(architecture_ir(proposal, readout_qubit))


def canonical_theta_values(proposal: CompleteCandidateProposal) -> list[float]:
    return [
        round(float(op.theta), CANONICAL_THETA_DECIMALS)
        for op in proposal.operations
        if op.gate in PARAMETERIZED_GATES
    ]


def candidate_hash(proposal: CompleteCandidateProposal, readout_qubit: int) -> str:
    """Hash of architecture PLUS canonical theta values -- so a change in
    any angle yields a different candidate identity."""
    arch_json = ir_canonical_json(architecture_ir(proposal, readout_qubit))
    payload = json.dumps(
        {"architecture": arch_json, "theta": canonical_theta_values(proposal)},
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_complete_candidate_json(
    raw_text: str,
) -> tuple[CompleteCandidateProposal | None, str | None]:
    """Parse+structural-validate raw LLM output against
    `CompleteCandidateProposal`. Never raises."""
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return None, f"response is not valid JSON: {exc}"
    if not isinstance(data, dict):
        return None, "parsed JSON is not an object"
    try:
        return CompleteCandidateProposal.model_validate(data), None
    except ValidationError as exc:
        return None, str(exc)
