"""Tests for the CRX/CRY IR extension (Codex instruction section 14):
CRX, CRY, CRZ must compile correctly through both the PennyLane and
Qiskit backends, and count consistently as parameterized two-qubit gates.
"""

from __future__ import annotations

import numpy as np
import pytest

from llm_vqc.ir.canonicalize import structural_hash
from llm_vqc.ir.compiler_pennylane import to_qnode
from llm_vqc.ir.compiler_qiskit import to_qiskit_bound
from llm_vqc.ir.metrics import circuit_cost_summary
from llm_vqc.ir.schema import (
    PARAMETERIZED_ENTANGLE_GATES,
    CircuitIR,
    EncodingSpec,
    EntangleLayer,
    MeasurementSpec,
    RotationLayer,
)
from llm_vqc.ir.validators import validate_proposal


def _amplitude_ir(gate: str) -> CircuitIR:
    return CircuitIR(
        n_qubits=3,
        encoding=EncodingSpec(type="amplitude", wires="all"),
        layers=[EntangleLayer(pattern="pairs", gate=gate, pairs=[(0, 1)])],
        measurements=MeasurementSpec(observable="Z", wires=[0]),
    )


@pytest.mark.parametrize("gate", ["CRX", "CRY", "CRZ"])
def test_controlled_rotation_gates_are_parameterized(gate):
    assert gate in PARAMETERIZED_ENTANGLE_GATES


@pytest.mark.parametrize("gate", ["CRX", "CRY", "CRZ"])
def test_controlled_rotation_gate_validates_and_compiles(gate):
    ir = _amplitude_ir(gate)
    validation = validate_proposal(ir)
    assert validation.valid, validation.issues

    cost = circuit_cost_summary(ir)
    assert cost.parameter_count == 1
    assert cost.two_qubit_gate_count == 1

    qnode = to_qnode(ir)
    inputs = np.zeros(8)
    inputs[0] = 1.0
    result = qnode(inputs, [0.3])
    assert len(result) == 1
    assert -1.0 <= float(result[0]) <= 1.0


@pytest.mark.parametrize("gate", ["CRX", "CRY", "CRZ"])
def test_controlled_rotation_gate_compiles_to_qiskit(gate):
    ir = _amplitude_ir(gate)
    inputs = [0.0] * 8
    inputs[0] = 1.0
    qc = to_qiskit_bound(ir, inputs=inputs, weights=[0.3])
    assert qc.depth() >= 1


def test_cnot_cz_remain_nonparameterized():
    from llm_vqc.ir.schema import NONPARAMETERIZED_ENTANGLE_GATES

    assert NONPARAMETERIZED_ENTANGLE_GATES == frozenset({"CNOT", "CZ"})


def test_all_five_entangle_gates_present():
    from llm_vqc.ir.schema import ENTANGLE_GATE_NAMES

    assert set(ENTANGLE_GATE_NAMES) == {"CNOT", "CZ", "CRX", "CRY", "CRZ"}


def test_structural_hash_distinguishes_crx_cry_crz():
    hashes = {gate: structural_hash(_amplitude_ir(gate)) for gate in ("CRX", "CRY", "CRZ")}
    assert len(set(hashes.values())) == 3


def test_two_qubit_free_body_with_multiple_controlled_gates():
    ir = CircuitIR(
        n_qubits=3,
        encoding=EncodingSpec(type="amplitude", wires="all"),
        layers=[
            RotationLayer(gates=["H"], wires=[1]),
            EntangleLayer(pattern="pairs", gate="CRY", pairs=[(1, 0)]),
            RotationLayer(gates=["RZ"], wires=[2]),
            EntangleLayer(pattern="pairs", gate="CRX", pairs=[(2, 0)]),
        ],
        measurements=MeasurementSpec(observable="Z", wires=[0]),
    )
    validation = validate_proposal(ir)
    assert validation.valid
    cost = circuit_cost_summary(ir)
    assert cost.parameter_count == 3  # CRY, RZ, CRX (H has no parameter)
    assert cost.two_qubit_gate_count == 2
