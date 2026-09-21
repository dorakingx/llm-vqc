"""Tests for the mini demo's fail-loud fixed-capacity check
(`llm_vqc.mini_demo.capacity.verify_fixed_capacity`): every candidate this
demo trains must have exactly 2 qubits, 4 quantum parameters, and 51 total
trainable parameters -- Codex instruction section 2.
"""

from __future__ import annotations

import itertools

import pytest

from llm_vqc.ir.schema import CircuitIR, EncodingSpec, MeasurementSpec, RotationLayer
from llm_vqc.mini_demo.capacity import (
    EXPECTED_N_QUBITS,
    EXPECTED_QUANTUM_PARAMETERS,
    EXPECTED_TOTAL_TRAINABLE_PARAMETERS,
    FixedCapacityViolation,
    verify_fixed_capacity,
)
from llm_vqc.mini_demo.compact_schema import (
    DIRECTION_CHOICES,
    ENTANGLER_CHOICES,
    ROTATION_GATE_CHOICES,
    CompactArchitecture,
    compact_to_circuit_ir,
)
from llm_vqc.tasks.t1_gaussian import T1GaussianPeakTask

TASK_SPEC = T1GaussianPeakTask().spec


@pytest.mark.parametrize(
    "l1,ent,direction,l2",
    list(
        itertools.product(
            ROTATION_GATE_CHOICES, ENTANGLER_CHOICES, DIRECTION_CHOICES, ROTATION_GATE_CHOICES
        )
    ),
)
def test_every_grammar_point_passes_the_capacity_check(l1, ent, direction, l2):
    proposal = CompactArchitecture(
        layer_1_gate=l1, entangler=ent, entangler_direction=direction, layer_2_gate=l2
    )
    ir = compact_to_circuit_ir(proposal)
    verify_fixed_capacity(ir, TASK_SPEC)  # must not raise


def test_expected_constants_match_codex_instruction():
    assert EXPECTED_N_QUBITS == 2
    assert EXPECTED_QUANTUM_PARAMETERS == 4
    assert EXPECTED_TOTAL_TRAINABLE_PARAMETERS == 51


def test_raises_on_wrong_qubit_count():
    bad_ir = CircuitIR(
        n_qubits=3,
        encoding=EncodingSpec(type="angle", gate="RY", wires=[0, 1, 2]),
        layers=[RotationLayer(gates=["RY"], wires=[0, 1, 2])],
        measurements=MeasurementSpec(observable="Z", wires=[0, 1, 2]),
    )
    with pytest.raises(FixedCapacityViolation, match="n_qubits"):
        verify_fixed_capacity(bad_ir, TASK_SPEC)


def test_raises_on_too_many_quantum_parameters():
    # Three independent RY layers on 2 wires each = 6 quantum parameters,
    # not 4 -- must be rejected loudly rather than silently trained.
    bad_ir = CircuitIR(
        n_qubits=2,
        encoding=EncodingSpec(type="angle", gate="RY", wires=[0, 1]),
        layers=[
            RotationLayer(gates=["RY"], wires=[0, 1]),
            RotationLayer(gates=["RY"], wires=[0, 1]),
            RotationLayer(gates=["RY"], wires=[0, 1]),
        ],
        measurements=MeasurementSpec(observable="Z", wires=[0, 1]),
    )
    with pytest.raises(FixedCapacityViolation, match="quantum parameters"):
        verify_fixed_capacity(bad_ir, TASK_SPEC)


def test_raises_on_too_few_quantum_parameters():
    bad_ir = CircuitIR(
        n_qubits=2,
        encoding=EncodingSpec(type="angle", gate="RY", wires=[0, 1]),
        layers=[RotationLayer(gates=["RY"], wires=[0, 1])],  # only 2 params
        measurements=MeasurementSpec(observable="Z", wires=[0, 1]),
    )
    with pytest.raises(FixedCapacityViolation, match="quantum parameters"):
        verify_fixed_capacity(bad_ir, TASK_SPEC)
