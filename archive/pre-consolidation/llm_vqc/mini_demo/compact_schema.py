"""The compact 4-field architecture grammar for the mini LLM-API VQC demo.

Deliberately NOT the full `CircuitIR` JSON shape (see
`llm_vqc.ir.schema.CircuitIR` / `llm_vqc.llm.prompts._SCHEMA_DESCRIPTION`,
which is what every other LLM search arm in this repo asks a model to
emit). This experiment fixes the circuit's qubit count, encoding, and
measurement in advance and only asks the model to choose four discrete
fields:

    layer_1_gate: RX | RY | RZ           (2 trainable angles, one per wire)
    entangler:    CNOT | CZ | NONE       (fixed, no continuous parameter)
    entangler_direction: 0_to_1 | 1_to_0 (only meaningful for CNOT)
    layer_2_gate: RX | RY | RZ           (2 trainable angles, one per wire)

This keeps output tokens small and the invalid-proposal rate near zero
(a 3x3x2x3 = 54-point discrete grid has no way to violate wire bounds,
duplicate operands, or any of the other `llm_vqc.ir.validators` checks),
which is the entire point of using a compact schema instead of full IR
for a several-minutes, 4-call demo.

**Physical vs. syntactic duplicate detection.** `CZ` is symmetric under
qubit exchange (its unitary is diag(1,1,1,-1) regardless of which wire is
named "control") and `NONE` applies no gate at all, so for those two
choices `entangler_direction` has no physical effect. But
`llm_vqc.ir.canonicalize.structural_hash` hashes the *syntactic* IR (see
that module's docstring), so an `EntangleLayer(pairs=[(0,1)])` and one
with `pairs=[(1,0)]` would hash differently even when physically
identical (as they are for CZ). `canonicalize_compact` collapses
`entangler_direction` to a fixed value whenever the choice is
`CZ`/`NONE` *before* the CircuitIR is built, so the resulting IR (and
hence its structural hash) is the same for every proposal that is
physically the same circuit -- duplicate detection stays correct. `CNOT`
is not symmetric (control/target are physically distinguishable), so its
direction is never canonicalized away.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from llm_vqc.ir.schema import CircuitIR, EncodingSpec, EntangleLayer, MeasurementSpec, RotationLayer

RotationGate = Literal["RX", "RY", "RZ"]
Entangler = Literal["CNOT", "CZ", "NONE"]
Direction = Literal["0_to_1", "1_to_0"]

ROTATION_GATE_CHOICES: tuple[RotationGate, ...] = ("RX", "RY", "RZ")
ENTANGLER_CHOICES: tuple[Entangler, ...] = ("CNOT", "CZ", "NONE")
DIRECTION_CHOICES: tuple[Direction, ...] = ("0_to_1", "1_to_0")

#: Entangler choices whose two-qubit unitary (or absence of one) does not
#: depend on `entangler_direction` -- see module docstring.
_DIRECTION_INVARIANT_ENTANGLERS = frozenset({"CZ", "NONE"})
_CANONICAL_DIRECTION: Direction = "0_to_1"

N_QUBITS = 2
ENCODING_GATE = "RY"
MEASUREMENT_OBSERVABLE = "Z"
FIXED_WIRES = [0, 1]

#: Structural quantum-parameter count every valid compact proposal must
#: produce: 2 angles in layer 1 + 2 angles in layer 2, entangler never
#: parameterized (CNOT/CZ are both fixed gates; CRZ is not offered).
EXPECTED_QUANTUM_PARAMETER_COUNT = 4


class CompactArchitecture(BaseModel):
    """One compact architecture proposal, exactly as an LLM or the random
    arm emits it (before canonicalization or IR conversion)."""

    model_config = ConfigDict(extra="forbid")

    layer_1_gate: RotationGate
    entangler: Entangler
    entangler_direction: Direction
    layer_2_gate: RotationGate


#: Strict JSON Schema for OpenAI structured-output (`response_format`)
#: requests. All four fields are required (OpenAI's strict mode requires
#: every property to be listed in "required", even though none of these
#: fields are conceptually optional here anyway).
COMPACT_ARCHITECTURE_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "layer_1_gate": {"type": "string", "enum": list(ROTATION_GATE_CHOICES)},
        "entangler": {"type": "string", "enum": list(ENTANGLER_CHOICES)},
        "entangler_direction": {"type": "string", "enum": list(DIRECTION_CHOICES)},
        "layer_2_gate": {"type": "string", "enum": list(ROTATION_GATE_CHOICES)},
    },
    "required": ["layer_1_gate", "entangler", "entangler_direction", "layer_2_gate"],
    "additionalProperties": False,
}


def canonicalize_compact(proposal: CompactArchitecture) -> CompactArchitecture:
    """Return a copy with `entangler_direction` normalized to a fixed value
    whenever the entangler choice makes direction physically meaningless.

    Idempotent: canonicalizing an already-canonical proposal is a no-op.
    """
    if proposal.entangler in _DIRECTION_INVARIANT_ENTANGLERS:
        if proposal.entangler_direction == _CANONICAL_DIRECTION:
            return proposal
        return proposal.model_copy(update={"entangler_direction": _CANONICAL_DIRECTION})
    return proposal


def compact_to_circuit_ir(proposal: CompactArchitecture) -> CircuitIR:
    """Deterministically expand a (canonicalized) compact proposal into the
    full `CircuitIR` the rest of the repo's evaluation harness understands.

    Always produces: 2 qubits, angle-RY encoding on wires [0, 1], exactly
    4 quantum parameters (2 per rotation layer), Z measurement on wires
    [0, 1]. The entangler, if any, sits between the two rotation layers.
    """
    canonical = canonicalize_compact(proposal)

    layers: list[RotationLayer | EntangleLayer] = [
        RotationLayer(gates=[canonical.layer_1_gate], wires=list(FIXED_WIRES)),
    ]
    if canonical.entangler != "NONE":
        pair = (0, 1) if canonical.entangler_direction == "0_to_1" else (1, 0)
        layers.append(
            EntangleLayer(pattern="pairs", gate=canonical.entangler, pairs=[pair])
        )
    layers.append(RotationLayer(gates=[canonical.layer_2_gate], wires=list(FIXED_WIRES)))

    return CircuitIR(
        n_qubits=N_QUBITS,
        encoding=EncodingSpec(type="angle", gate=ENCODING_GATE, wires=list(FIXED_WIRES)),
        layers=layers,
        measurements=MeasurementSpec(observable=MEASUREMENT_OBSERVABLE, wires=list(FIXED_WIRES)),
        metadata={
            "compact_layer_1_gate": canonical.layer_1_gate,
            "compact_entangler": canonical.entangler,
            "compact_entangler_direction": canonical.entangler_direction,
            "compact_layer_2_gate": canonical.layer_2_gate,
        },
    )


def parse_compact_json(raw_text: str) -> tuple[CompactArchitecture | None, str | None]:
    """Parse+validate raw LLM output text against `CompactArchitecture`.

    Returns `(proposal, None)` on success or `(None, error_message)` on any
    JSON or schema failure -- never raises, so a caller can record a
    malformed LLM response as an ordinary invalid proposal rather than
    crashing the run.
    """
    import json

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return None, f"response is not valid JSON: {exc}"
    if not isinstance(data, dict):
        return None, "parsed JSON is not an object"
    try:
        return CompactArchitecture.model_validate(data), None
    except Exception as exc:  # pydantic.ValidationError, kept broad & message-only
        return None, str(exc)
