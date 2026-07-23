"""The free-gate proposal schema (Codex instruction sections 7-8, 13):
an ordered sequence of 1-5 gates from a fixed 7-gate set, with no imposed
topology, layering, or entangler requirement.

```json
{
  "n_qubits": 3,
  "operations": [
    {"gate": "H", "wires": [1]},
    {"gate": "CRY", "wires": [1, 0]},
    {"gate": "RZ", "wires": [2]}
  ]
}
```

Deliberately NOT the full `CircuitIR` shape and NOT `llm_vqc.mini_demo`'s
compact schema (which fixes a 2-layer + 1-entangler template): this
experiment's whole point is a *free* ordered sequence, so the schema is
just "N operations, each one gate name + its wires" -- nothing about
layering, topology, or a required entangler.

`extra="forbid"` on `FreeGateOperation` is what makes "numerical parameter
values from the LLM are rejected" and "LLM-supplied parameter IDs are
rejected" (Codex instruction section 8) a structural guarantee rather than
a special-cased check: an operation object may only ever have `gate` and
`wires` keys -- any `theta`/`param_id`/anything else fails pydantic
validation before any semantic check runs.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from llm_vqc.ir.schema import (
    CircuitIR,
    EncodingSpec,
    EntangleLayer,
    MeasurementSpec,
    RotationLayer,
)
from llm_vqc.ir.validators import ValidationIssue

FREE_GATE_SCHEMA_VERSION = "free_amplitude_v1"

MIN_OPERATIONS = 1
MAX_OPERATIONS = 5

SINGLE_WIRE_GATES: tuple[str, ...] = ("H", "RX", "RY", "RZ")
TWO_WIRE_GATES: tuple[str, ...] = ("CRX", "CRY", "CRZ")
ALL_GATES: tuple[str, ...] = SINGLE_WIRE_GATES + TWO_WIRE_GATES
PARAMETERIZED_GATES: frozenset[str] = frozenset({"RX", "RY", "RZ", "CRX", "CRY", "CRZ"})

GateName = Literal["H", "RX", "RY", "RZ", "CRX", "CRY", "CRZ"]


class FreeGateOperation(BaseModel):
    """One gate application. `wires` is `[wire]` for single-wire gates or
    `[control, target]` (ordered) for the two-wire controlled-rotation
    gates -- arity is enforced by the semantic validator, not here (pydantic
    alone cannot make list length depend on the sibling `gate` value)."""

    model_config = ConfigDict(extra="forbid")

    gate: GateName
    wires: list[int]


class FreeGateProposal(BaseModel):
    """One free-gate circuit proposal, exactly as an LLM or the random
    sampler emits it -- before any experiment-level readout/encoding is
    attached."""

    model_config = ConfigDict(extra="forbid")

    n_qubits: int = Field(ge=1)
    operations: list[FreeGateOperation] = Field(
        min_length=MIN_OPERATIONS, max_length=MAX_OPERATIONS
    )


#: Strict JSON Schema shape for a future OpenAI Structured Outputs
#: `response_format` request. Not exercised by any real API call in this
#: implementation pass (Random search + mock/scripted providers only).
FREE_GATE_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "n_qubits": {"type": "integer"},
        "operations": {
            "type": "array",
            "minItems": MIN_OPERATIONS,
            "maxItems": MAX_OPERATIONS,
            "items": {
                "anyOf": [
                    {
                        "type": "object",
                        "properties": {
                            "gate": {"type": "string", "enum": list(SINGLE_WIRE_GATES)},
                            "wires": {
                                "type": "array", "minItems": 1, "maxItems": 1,
                                "items": {"type": "integer"},
                            },
                        },
                        "required": ["gate", "wires"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "gate": {"type": "string", "enum": list(TWO_WIRE_GATES)},
                            "wires": {
                                "type": "array", "minItems": 2, "maxItems": 2,
                                "items": {"type": "integer"},
                            },
                        },
                        "required": ["gate", "wires"],
                        "additionalProperties": False,
                    },
                ]
            },
        },
    },
    "required": ["n_qubits", "operations"],
    "additionalProperties": False,
}


class FreeGateValidationResult(BaseModel):
    """Mirrors `llm_vqc.ir.validators.ValidationResult`'s shape and
    "collect every issue, never fail-fast on the first one" contract."""

    model_config = {"arbitrary_types_allowed": True}

    valid: bool
    issues: list[ValidationIssue] = []
    proposal: FreeGateProposal | None = None


def _pydantic_errors_to_issues(exc: ValidationError) -> list[ValidationIssue]:
    issues = []
    for error in exc.errors():
        path = ".".join(str(p) for p in error["loc"])
        issues.append(
            ValidationIssue(code=f"schema.{error['type']}", message=error["msg"], path=path)
        )
    return issues


def validate_free_gate_proposal(
    raw: dict | FreeGateProposal,
    expected_n_qubits: int,
    require_at_least_one_parameterized: bool = True,
) -> FreeGateValidationResult:
    """Validate a raw free-gate proposal. Runs pydantic schema validation
    first (structural: field names/types/lengths); on success, runs every
    semantic check (arity, wire bounds, control!=target, n_qubits match,
    at-least-one-parameterized, adjacent-duplicate rejection) and collects
    *all* violations rather than stopping at the first one -- same
    discipline as `llm_vqc.ir.validators.validate_proposal`.
    """
    if isinstance(raw, FreeGateProposal):
        proposal = raw
    else:
        try:
            proposal = FreeGateProposal.model_validate(raw)
        except ValidationError as exc:
            return FreeGateValidationResult(valid=False, issues=_pydantic_errors_to_issues(exc))

    issues = _validate_semantic(proposal, expected_n_qubits, require_at_least_one_parameterized)
    if issues:
        return FreeGateValidationResult(valid=False, issues=issues)
    return FreeGateValidationResult(valid=True, issues=[], proposal=proposal)


def _validate_semantic(
    proposal: FreeGateProposal, expected_n_qubits: int, require_at_least_one_parameterized: bool
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    if proposal.n_qubits != expected_n_qubits:
        issues.append(
            ValidationIssue(
                code="free_gate.n_qubits_mismatch",
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
                    code="free_gate.wrong_arity",
                    message=(
                        f"{op.gate} requires exactly {expected_arity} wire(s), "
                        f"got {len(op.wires)}"
                    ),
                    path=f"{path}.wires",
                )
            )
            continue  # further per-op checks assume correct arity

        for wire in op.wires:
            if not (0 <= wire < expected_n_qubits):
                issues.append(
                    ValidationIssue(
                        code="free_gate.wire_out_of_bounds",
                        message=(
                            f"wire {wire} out of bounds for n_qubits={expected_n_qubits}"
                        ),
                        path=f"{path}.wires",
                    )
                )

        if op.gate in TWO_WIRE_GATES and len(op.wires) == 2 and op.wires[0] == op.wires[1]:
            issues.append(
                ValidationIssue(
                    code="free_gate.control_equals_target",
                    message=f"{op.gate} control and target must differ, got {op.wires}",
                    path=path,
                )
            )

    if require_at_least_one_parameterized:
        if not any(op.gate in PARAMETERIZED_GATES for op in proposal.operations):
            issues.append(
                ValidationIssue(
                    code="free_gate.no_parameterized_gate",
                    message="at least one parameterized gate is required",
                    path="operations",
                )
            )

    issues.extend(_check_adjacent_duplicates(proposal))
    return issues


def _check_adjacent_duplicates(proposal: FreeGateProposal) -> list[ValidationIssue]:
    """Reject immediately-adjacent identical (gate, wires) pairs -- e.g.
    `H(1); H(1)` or `CRX(2,0); CRX(2,0)` -- per Codex instruction section
    13's reducible-operation list. Rejected, not silently canonicalized:
    "reject or canonicalize consistently" is satisfied by always rejecting.
    """
    issues: list[ValidationIssue] = []
    ops = proposal.operations
    for i in range(1, len(ops)):
        prev, cur = ops[i - 1], ops[i]
        if prev.gate == cur.gate and prev.wires == cur.wires:
            issues.append(
                ValidationIssue(
                    code="free_gate.reducible_adjacent_duplicate",
                    message=(
                        f"operations[{i - 1}] and operations[{i}] are identical "
                        f"({cur.gate}{cur.wires}) -- adjacent duplicates are reducible "
                        "and must not be proposed"
                    ),
                    path=f"operations[{i}]",
                )
            )
    return issues


def free_gate_proposal_to_circuit_ir(proposal: FreeGateProposal, readout_qubit: int) -> CircuitIR:
    """Deterministically expand a (validated) free-gate proposal into the
    full `CircuitIR`, preserving exact operation order.

    Always produces: amplitude encoding on all `n_qubits` wires, one layer
    per operation (never grouped/reordered -- layer order IS operation
    order), and a measurement fixed to Z on `readout_qubit` only (the
    training/prediction path never sees any other wire's expectation
    value -- see `llm_vqc.free_amplitude.diagnostics` for the separate,
    diagnostics-only all-qubit measurement path).
    """
    layers: list[RotationLayer | EntangleLayer] = []
    for op in proposal.operations:
        if op.gate in SINGLE_WIRE_GATES:
            layers.append(RotationLayer(gates=[op.gate], wires=list(op.wires)))
        else:
            control, target = op.wires
            layers.append(
                EntangleLayer(pattern="pairs", gate=op.gate, pairs=[(control, target)])
            )

    return CircuitIR(
        n_qubits=proposal.n_qubits,
        encoding=EncodingSpec(type="amplitude", wires="all"),
        layers=layers,
        measurements=MeasurementSpec(observable="Z", wires=[readout_qubit]),
        metadata={
            "free_gate_schema_version": FREE_GATE_SCHEMA_VERSION,
            "readout_qubit": str(readout_qubit),
        },
    )


def parse_free_gate_json(raw_text: str) -> tuple[FreeGateProposal | None, str | None]:
    """Parse+validate raw LLM output text against `FreeGateProposal`'s
    structural schema only (not the semantic checks, which need
    `expected_n_qubits` -- call `validate_free_gate_proposal` for those).
    Returns `(proposal, None)` on success or `(None, error_message)` on any
    JSON/schema failure -- never raises.
    """
    import json

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return None, f"response is not valid JSON: {exc}"
    if not isinstance(data, dict):
        return None, "parsed JSON is not an object"
    try:
        return FreeGateProposal.model_validate(data), None
    except ValidationError as exc:
        return None, str(exc)
