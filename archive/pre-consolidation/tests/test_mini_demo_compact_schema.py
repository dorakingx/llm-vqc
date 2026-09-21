"""Tests for the mini demo's compact 4-field architecture grammar
(`llm_vqc.mini_demo.compact_schema`): schema validation, deterministic
conversion to `CircuitIR`, exact quantum-parameter count, and
canonicalization for physically-inert `entangler_direction` choices.
"""

from __future__ import annotations

import itertools

import pytest
from pydantic import ValidationError

from llm_vqc.ir.canonicalize import structural_hash
from llm_vqc.ir.metrics import circuit_cost_summary
from llm_vqc.ir.validators import validate_proposal
from llm_vqc.mini_demo.compact_schema import (
    COMPACT_ARCHITECTURE_JSON_SCHEMA,
    DIRECTION_CHOICES,
    ENTANGLER_CHOICES,
    EXPECTED_QUANTUM_PARAMETER_COUNT,
    ROTATION_GATE_CHOICES,
    CompactArchitecture,
    canonicalize_compact,
    compact_to_circuit_ir,
    parse_compact_json,
)

ALL_COMBINATIONS = list(
    itertools.product(
        ROTATION_GATE_CHOICES, ENTANGLER_CHOICES, DIRECTION_CHOICES, ROTATION_GATE_CHOICES
    )
)


def _compact(l1="RY", ent="CNOT", direction="0_to_1", l2="RY") -> CompactArchitecture:
    return CompactArchitecture(
        layer_1_gate=l1, entangler=ent, entangler_direction=direction, layer_2_gate=l2
    )


# --- schema validation -----------------------------------------------------


def test_rejects_extra_fields():
    with pytest.raises(ValidationError):
        CompactArchitecture(
            layer_1_gate="RY", entangler="CNOT", entangler_direction="0_to_1",
            layer_2_gate="RY", extra_field="nope",
        )


@pytest.mark.parametrize(
    "field", ["layer_1_gate", "entangler", "entangler_direction", "layer_2_gate"]
)
def test_rejects_invalid_enum_value(field):
    kwargs = dict(
        layer_1_gate="RY", entangler="CNOT", entangler_direction="0_to_1", layer_2_gate="RY"
    )
    kwargs[field] = "NOT_A_VALID_CHOICE"
    with pytest.raises(ValidationError):
        CompactArchitecture(**kwargs)


def test_json_schema_enums_match_pydantic_choices():
    """The strict JSON Schema sent to OpenAI must offer exactly the same
    grammar the Random arm samples from -- Codex instruction section 4:
    "sample from the identical compact architecture grammar."""
    props = COMPACT_ARCHITECTURE_JSON_SCHEMA["properties"]
    assert set(props["layer_1_gate"]["enum"]) == set(ROTATION_GATE_CHOICES)
    assert set(props["layer_2_gate"]["enum"]) == set(ROTATION_GATE_CHOICES)
    assert set(props["entangler"]["enum"]) == set(ENTANGLER_CHOICES)
    assert set(props["entangler_direction"]["enum"]) == set(DIRECTION_CHOICES)
    assert COMPACT_ARCHITECTURE_JSON_SCHEMA["additionalProperties"] is False
    assert set(COMPACT_ARCHITECTURE_JSON_SCHEMA["required"]) == {
        "layer_1_gate", "entangler", "entangler_direction", "layer_2_gate",
    }


def test_parse_compact_json_rejects_non_json():
    proposal, err = parse_compact_json("not json at all")
    assert proposal is None
    assert err is not None


def test_parse_compact_json_rejects_non_object_json():
    proposal, err = parse_compact_json("[1, 2, 3]")
    assert proposal is None
    assert err is not None


def test_parse_compact_json_rejects_schema_violation():
    proposal, err = parse_compact_json('{"layer_1_gate": "RY", "entangler": "TOFFOLI"}')
    assert proposal is None
    assert err is not None


def test_parse_compact_json_accepts_well_formed_proposal():
    proposal, err = parse_compact_json(
        '{"layer_1_gate": "RX", "entangler": "CNOT", "entangler_direction": "1_to_0", '
        '"layer_2_gate": "RZ"}'
    )
    assert err is None
    assert proposal == CompactArchitecture(
        layer_1_gate="RX", entangler="CNOT", entangler_direction="1_to_0", layer_2_gate="RZ"
    )


# --- deterministic compact -> CircuitIR conversion -------------------------


@pytest.mark.parametrize("l1,ent,direction,l2", ALL_COMBINATIONS)
def test_every_grammar_point_yields_two_qubits_and_exactly_four_quantum_parameters(
    l1, ent, direction, l2
):
    ir = compact_to_circuit_ir(_compact(l1, ent, direction, l2))
    assert ir.n_qubits == 2
    validation = validate_proposal(ir)
    assert validation.valid, validation.issues
    cost = circuit_cost_summary(ir)
    assert cost.parameter_count == EXPECTED_QUANTUM_PARAMETER_COUNT == 4
    assert cost.n_qubits == 2
    assert cost.input_count == 2  # RY encoding on both wires


def test_conversion_is_deterministic():
    proposal = _compact("RZ", "CZ", "0_to_1", "RX")
    ir_a = compact_to_circuit_ir(proposal)
    ir_b = compact_to_circuit_ir(proposal)
    assert structural_hash(ir_a) == structural_hash(ir_b)


def test_cnot_direction_changes_the_trained_circuit():
    """CNOT is not symmetric -- 0_to_1 and 1_to_0 must remain distinct
    circuits (different structural hash)."""
    ir_forward = compact_to_circuit_ir(_compact("RY", "CNOT", "0_to_1", "RY"))
    ir_backward = compact_to_circuit_ir(_compact("RY", "CNOT", "1_to_0", "RY"))
    assert structural_hash(ir_forward) != structural_hash(ir_backward)


@pytest.mark.parametrize("entangler", ["CZ", "NONE"])
def test_direction_invariant_entanglers_produce_identical_circuit(entangler):
    """CZ is symmetric under qubit exchange and NONE applies no gate at
    all, so for these two choices `entangler_direction` must not affect
    the resulting circuit -- Codex instruction section 3."""
    ir_a = compact_to_circuit_ir(_compact("RY", entangler, "0_to_1", "RY"))
    ir_b = compact_to_circuit_ir(_compact("RY", entangler, "1_to_0", "RY"))
    assert structural_hash(ir_a) == structural_hash(ir_b)


def test_canonicalize_normalizes_direction_only_for_cz_and_none():
    cz = canonicalize_compact(_compact("RY", "CZ", "1_to_0", "RY"))
    none_ = canonicalize_compact(_compact("RY", "NONE", "1_to_0", "RY"))
    cnot = canonicalize_compact(_compact("RY", "CNOT", "1_to_0", "RY"))
    assert cz.entangler_direction == "0_to_1"
    assert none_.entangler_direction == "0_to_1"
    assert cnot.entangler_direction == "1_to_0"  # untouched: direction is physically meaningful


def test_canonicalize_is_idempotent():
    proposal = _compact("RY", "CZ", "1_to_0", "RY")
    once = canonicalize_compact(proposal)
    twice = canonicalize_compact(once)
    assert once == twice


def test_none_entangler_adds_no_two_qubit_gate():
    ir = compact_to_circuit_ir(_compact("RY", "NONE", "0_to_1", "RY"))
    cost = circuit_cost_summary(ir)
    assert cost.two_qubit_gate_count == 0


@pytest.mark.parametrize("entangler", ["CNOT", "CZ"])
def test_nonzero_entangler_adds_exactly_one_two_qubit_gate(entangler):
    ir = compact_to_circuit_ir(_compact("RY", entangler, "0_to_1", "RY"))
    cost = circuit_cost_summary(ir)
    assert cost.two_qubit_gate_count == 1
