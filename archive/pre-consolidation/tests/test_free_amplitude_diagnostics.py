"""Tests for causal-cone and gradient diagnostics (Codex instruction
section 12): structural causal-cone membership vs. empirical near-zero
gradient are distinct, correctly computed concepts.
"""

from __future__ import annotations

from llm_vqc.free_amplitude.diagnostics import (
    causal_cone_summary,
    diagnostic_all_qubit_expectations,
    gradient_diagnostics,
    is_constant_prediction,
)
from llm_vqc.free_amplitude.schema import (
    free_gate_proposal_to_circuit_ir,
    validate_free_gate_proposal,
)


def _proposal(ops):
    return validate_free_gate_proposal({"n_qubits": 3, "operations": ops}, 3).proposal


def test_single_gate_directly_on_readout_is_in_causal_cone():
    p = _proposal([{"gate": "RY", "wires": [0]}])
    cone = causal_cone_summary(p, readout_qubit=0)
    assert cone.parameter_count_in_causal_cone == 1
    assert cone.fraction_in_causal_cone == 1.0


def test_gate_on_unconnected_wire_is_not_in_causal_cone():
    p = _proposal([{"gate": "RY", "wires": [0]}, {"gate": "RZ", "wires": [1]}])
    cone = causal_cone_summary(p, readout_qubit=0)
    assert cone.parameter_count_in_causal_cone == 1
    assert cone.total_quantum_parameter_count == 2
    assert cone.fraction_in_causal_cone == 0.5


def test_entangling_gate_pulls_a_wire_into_the_cone():
    p = _proposal([{"gate": "RZ", "wires": [1]}, {"gate": "CRX", "wires": [1, 0]}])
    cone = causal_cone_summary(p, readout_qubit=0)
    # Both RZ(1) and CRX(1,0) end up in the cone: CRX touches wire 0 (in the
    # cone from the start), which pulls wire 1 in, which then makes RZ(1) causal.
    assert cone.parameter_count_in_causal_cone == 2


def test_gate_after_the_only_connecting_gate_is_not_causal():
    """Order matters: a gate on wire 1 placed AFTER the only entangler that
    would connect wire 1 to wire 0 cannot influence q0's expectation --
    CRX(0,1) is causal (touches wire 0 directly); RZ(1) comes strictly
    after it and nothing later references wire 0 again, so RZ is outside
    the backward lightcone."""
    p = _proposal([{"gate": "CRX", "wires": [0, 1]}, {"gate": "RZ", "wires": [1]}])
    cone = causal_cone_summary(p, readout_qubit=0)
    assert cone.causal_operation_indices == {0}
    assert cone.parameter_count_in_causal_cone == 1
    assert cone.total_quantum_parameter_count == 2


def test_gradient_diagnostics_distinguishes_zero_from_near_zero():
    p = _proposal([{"gate": "RY", "wires": [0]}, {"gate": "RZ", "wires": [1]}])
    cone = causal_cone_summary(p, readout_qubit=0)
    grad_vector = [0.5, 1e-12]  # second param is (numerically) exactly zero
    diag = gradient_diagnostics(grad_vector, cone)
    assert diag.zero_gradient_parameter_count == 1
    assert diag.near_zero_gradient_parameter_count == 1
    assert diag.all_gradients_numerically_zero is False


def test_gradient_diagnostics_all_zero_case():
    p = _proposal([{"gate": "RY", "wires": [1]}])  # disconnected from readout
    cone = causal_cone_summary(p, readout_qubit=0)
    diag = gradient_diagnostics([0.0], cone)
    assert diag.all_gradients_numerically_zero is True
    assert diag.no_trainable_parameter_affects_readout is True


def test_gradient_diagnostics_returns_none_when_not_computed():
    p = _proposal([{"gate": "RY", "wires": [0]}])
    cone = causal_cone_summary(p, readout_qubit=0)
    diag = gradient_diagnostics(None, cone)
    assert diag.zero_gradient_parameter_count is None
    assert diag.near_zero_gradient_parameter_count is None
    assert diag.all_gradients_numerically_zero is None


def test_is_constant_prediction_detects_constant_output():
    import numpy as np

    assert is_constant_prediction(np.array([0.5, 0.5, 0.5, 0.5]))
    assert not is_constant_prediction(np.array([0.1, 0.9, 0.3, 0.7]))


def test_diagnostic_all_qubit_expectations_never_enters_prediction_path():
    """The diagnostic path is a SEPARATE QNode built from a copy of the IR
    -- it must return per-wire values without mutating the original IR."""
    p = _proposal([{"gate": "RY", "wires": [0]}, {"gate": "CRX", "wires": [0, 1]}])
    ir = free_gate_proposal_to_circuit_ir(p, readout_qubit=0)
    original_measurement_wires = list(ir.measurements.wires)

    inputs = [0.0] * 8
    inputs[0] = 1.0
    result = diagnostic_all_qubit_expectations(ir, inputs, [0.3, 0.4])

    assert ir.measurements.wires == original_measurement_wires  # unchanged
    assert isinstance(result, dict)
    assert set(result.keys()) == {0, 1, 2}
