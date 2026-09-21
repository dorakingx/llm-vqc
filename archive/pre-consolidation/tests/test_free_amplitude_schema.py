"""Tests for the free-gate proposal schema (Codex instruction sections
7-8, 13): exactly seven allowed gates, arity, wire bounds, ordered
control-target semantics, control != target, 1-5 operations, deterministic
parameter indexing, H consumes zero parameters, operation order preserved,
CRX/CRY/CRZ compile, duplicate detection, canonicalization.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llm_vqc.free_amplitude.schema import (
    ALL_GATES,
    MAX_OPERATIONS,
    MIN_OPERATIONS,
    SINGLE_WIRE_GATES,
    TWO_WIRE_GATES,
    FreeGateOperation,
    FreeGateProposal,
    free_gate_proposal_to_circuit_ir,
    parse_free_gate_json,
    validate_free_gate_proposal,
)
from llm_vqc.ir.canonicalize import structural_hash
from llm_vqc.ir.expand import build_program
from llm_vqc.ir.metrics import circuit_cost_summary
from llm_vqc.ir.validators import validate_proposal


def _proposal(ops: list[dict], n_qubits: int = 3) -> FreeGateProposal:
    return FreeGateProposal(n_qubits=n_qubits, operations=[FreeGateOperation(**op) for op in ops])


# --- exactly seven allowed gates --------------------------------------------


def test_exactly_seven_allowed_gates():
    assert set(ALL_GATES) == {"H", "RX", "RY", "RZ", "CRX", "CRY", "CRZ"}
    assert len(ALL_GATES) == 7


def test_single_and_two_wire_gate_sets_partition_all_gates():
    assert set(SINGLE_WIRE_GATES) | set(TWO_WIRE_GATES) == set(ALL_GATES)
    assert set(SINGLE_WIRE_GATES) & set(TWO_WIRE_GATES) == set()


def test_unknown_gate_name_rejected():
    with pytest.raises(ValidationError):
        FreeGateOperation(gate="SWAP", wires=[0, 1])


# --- arity -------------------------------------------------------------------


@pytest.mark.parametrize("gate", ["H", "RX", "RY", "RZ"])
def test_single_wire_gate_requires_exactly_one_wire(gate):
    ok = validate_free_gate_proposal(
        {"n_qubits": 3, "operations": [{"gate": gate, "wires": [0]}]}, 3
    )
    # H-only proposals fail the parameterized-gate check, not the arity check.
    assert ok.valid or gate == "H"
    bad = validate_free_gate_proposal(
        {"n_qubits": 3, "operations": [{"gate": gate, "wires": [0, 1]}]}, 3
    )
    assert not bad.valid
    assert any(i.code == "free_gate.wrong_arity" for i in bad.issues)


@pytest.mark.parametrize("gate", ["CRX", "CRY", "CRZ"])
def test_two_wire_gate_requires_exactly_two_wires(gate):
    bad = validate_free_gate_proposal(
        {"n_qubits": 3, "operations": [{"gate": gate, "wires": [0]}]}, 3
    )
    assert not bad.valid
    assert any(i.code == "free_gate.wrong_arity" for i in bad.issues)
    good = validate_free_gate_proposal(
        {"n_qubits": 3, "operations": [{"gate": gate, "wires": [0, 1]}]}, 3
    )
    assert good.valid


# --- wire bounds ---------------------------------------------------------


def test_wire_out_of_bounds_rejected():
    result = validate_free_gate_proposal(
        {"n_qubits": 3, "operations": [{"gate": "RY", "wires": [3]}]}, 3
    )
    assert not result.valid
    assert any(i.code == "free_gate.wire_out_of_bounds" for i in result.issues)


def test_negative_wire_rejected():
    result = validate_free_gate_proposal(
        {"n_qubits": 3, "operations": [{"gate": "RY", "wires": [-1]}]}, 3
    )
    assert not result.valid
    assert any(i.code == "free_gate.wire_out_of_bounds" for i in result.issues)


# --- ordered control-target semantics; control != target -------------------


def test_control_equals_target_rejected():
    result = validate_free_gate_proposal(
        {"n_qubits": 3, "operations": [{"gate": "CRX", "wires": [1, 1]}]}, 3
    )
    assert not result.valid
    assert any(i.code == "free_gate.control_equals_target" for i in result.issues)


def test_control_target_order_is_preserved_and_matters():
    forward = _proposal([{"gate": "CRX", "wires": [0, 1]}])
    backward = _proposal([{"gate": "CRX", "wires": [1, 0]}])
    ir_forward = free_gate_proposal_to_circuit_ir(forward, readout_qubit=0)
    ir_backward = free_gate_proposal_to_circuit_ir(backward, readout_qubit=0)
    assert structural_hash(ir_forward) != structural_hash(ir_backward)


# --- 1-5 operations ----------------------------------------------------------


def test_zero_operations_rejected():
    with pytest.raises(ValidationError):
        FreeGateProposal(n_qubits=3, operations=[])


def test_six_operations_rejected():
    ops = [FreeGateOperation(gate="RX", wires=[0])] + [
        FreeGateOperation(gate="RY", wires=[1]) for _ in range(5)
    ]
    with pytest.raises(ValidationError):
        FreeGateProposal(n_qubits=3, operations=ops)


def test_min_max_operations_constants():
    assert MIN_OPERATIONS == 1
    assert MAX_OPERATIONS == 5


def test_exactly_five_operations_accepted():
    ops = [{"gate": "RY", "wires": [i % 3]} for i in range(5)]
    result = validate_free_gate_proposal({"n_qubits": 3, "operations": ops}, 3)
    assert result.valid, result.issues


# --- deterministic parameter indexing; H consumes zero parameters ---------


def test_h_consumes_zero_parameters():
    proposal = _proposal([{"gate": "H", "wires": [1]}, {"gate": "RY", "wires": [0]}])
    ir = free_gate_proposal_to_circuit_ir(proposal, readout_qubit=0)
    cost = circuit_cost_summary(ir)
    assert cost.parameter_count == 1


def test_deterministic_parameter_indexing_matches_operation_order():
    # H(1) -> no param; CRY(1,0) -> theta_0; RZ(2) -> theta_1; CRX(2,0) -> theta_2
    proposal = _proposal(
        [
            {"gate": "H", "wires": [1]},
            {"gate": "CRY", "wires": [1, 0]},
            {"gate": "RZ", "wires": [2]},
            {"gate": "CRX", "wires": [2, 0]},
        ]
    )
    ir = free_gate_proposal_to_circuit_ir(proposal, readout_qubit=0)
    program = build_program(ir)
    # body order: H(no param), CRY(param 0), RZ(param 1), CRX(param 2)
    param_indices = [instr.param_index for instr in program.body]
    assert param_indices == [None, 0, 1, 2]


def test_each_parameterized_gate_consumes_exactly_one_parameter():
    proposal = _proposal(
        [
            {"gate": "RX", "wires": [0]},
            {"gate": "CRY", "wires": [0, 1]},
            {"gate": "RZ", "wires": [2]},
        ]
    )
    ir = free_gate_proposal_to_circuit_ir(proposal, readout_qubit=0)
    assert circuit_cost_summary(ir).parameter_count == 3


# --- operation order preserved ----------------------------------------------


def test_operation_order_preserved_in_ir_layers():
    proposal = _proposal(
        [{"gate": "RX", "wires": [0]}, {"gate": "RY", "wires": [1]}, {"gate": "RZ", "wires": [2]}]
    )
    ir = free_gate_proposal_to_circuit_ir(proposal, readout_qubit=0)
    gates_in_order = [layer.gates[0] for layer in ir.layers]
    assert gates_in_order == ["RX", "RY", "RZ"]


# --- at least one parameterized gate ----------------------------------------


def test_all_h_proposal_rejected_by_default():
    result = validate_free_gate_proposal(
        {"n_qubits": 3, "operations": [{"gate": "H", "wires": [0]}]}, 3
    )
    assert not result.valid
    assert any(i.code == "free_gate.no_parameterized_gate" for i in result.issues)


def test_all_h_proposal_accepted_when_not_required():
    result = validate_free_gate_proposal(
        {"n_qubits": 3, "operations": [{"gate": "H", "wires": [0]}]}, 3,
        require_at_least_one_parameterized=False,
    )
    assert result.valid


# --- duplicate detection / canonicalization (adjacent reducible ops) -------


@pytest.mark.parametrize(
    "gate,wires",
    [
        ("H", [0]), ("RX", [0]), ("RY", [1]), ("RZ", [2]),
        ("CRX", [0, 1]), ("CRY", [0, 1]), ("CRZ", [0, 1]),
    ],
)
def test_adjacent_identical_operation_rejected(gate, wires):
    ops = [{"gate": gate, "wires": wires}, {"gate": gate, "wires": wires}]
    result = validate_free_gate_proposal({"n_qubits": 3, "operations": ops}, 3)
    assert not result.valid
    assert any(i.code == "free_gate.reducible_adjacent_duplicate" for i in result.issues)


def test_non_adjacent_identical_operations_allowed():
    ops = [
        {"gate": "RY", "wires": [0]}, {"gate": "RZ", "wires": [1]}, {"gate": "RY", "wires": [0]},
    ]
    result = validate_free_gate_proposal({"n_qubits": 3, "operations": ops}, 3)
    assert result.valid, result.issues


def test_same_circuit_produces_same_structural_hash():
    p1 = _proposal([{"gate": "RY", "wires": [0]}, {"gate": "CRX", "wires": [0, 1]}])
    p2 = _proposal([{"gate": "RY", "wires": [0]}, {"gate": "CRX", "wires": [0, 1]}])
    ir1 = free_gate_proposal_to_circuit_ir(p1, readout_qubit=0)
    ir2 = free_gate_proposal_to_circuit_ir(p2, readout_qubit=0)
    assert structural_hash(ir1) == structural_hash(ir2)


# --- unknown fields rejected; no numeric/param-id fields from LLM ----------


def test_unknown_top_level_field_rejected():
    with pytest.raises(ValidationError):
        FreeGateProposal.model_validate(
            {"n_qubits": 3, "operations": [{"gate": "RY", "wires": [0]}], "extra_field": 1}
        )


def test_theta_field_on_operation_rejected():
    with pytest.raises(ValidationError):
        FreeGateOperation.model_validate({"gate": "RY", "wires": [0], "theta": 1.23})


def test_param_id_field_on_operation_rejected():
    with pytest.raises(ValidationError):
        FreeGateOperation.model_validate({"gate": "RY", "wires": [0], "param_id": "theta_0"})


# --- n_qubits mismatch -------------------------------------------------------


def test_n_qubits_mismatch_rejected():
    result = validate_free_gate_proposal(
        {"n_qubits": 5, "operations": [{"gate": "RY", "wires": [0]}]}, expected_n_qubits=3
    )
    assert not result.valid
    assert any(i.code == "free_gate.n_qubits_mismatch" for i in result.issues)


# --- parse_free_gate_json ----------------------------------------------------


def test_parse_free_gate_json_rejects_malformed_json():
    proposal, err = parse_free_gate_json("not json")
    assert proposal is None
    assert err is not None


def test_parse_free_gate_json_accepts_well_formed():
    raw = '{"n_qubits": 3, "operations": [{"gate": "RY", "wires": [0]}]}'
    proposal, err = parse_free_gate_json(raw)
    assert err is None
    assert proposal is not None
    assert proposal.n_qubits == 3


# --- CRX/CRY/CRZ compile correctly (redundant cross-check with IR test) ----


@pytest.mark.parametrize("gate", ["CRX", "CRY", "CRZ"])
def test_controlled_gate_compiles_through_free_gate_pipeline(gate):
    proposal = _proposal([{"gate": gate, "wires": [0, 1]}])
    ir = free_gate_proposal_to_circuit_ir(proposal, readout_qubit=0)
    validation = validate_proposal(ir)
    assert validation.valid, validation.issues
