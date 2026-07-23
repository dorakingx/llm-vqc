"""Tests for the fixed-readout model (Codex instruction sections 2-3):
only q0 enters the prediction, `mu_hat = (1 - z0) / 2`, prediction stays
in [0, 1], no `w_out`/`b_out`/classical output layer, classical trainable
parameter count is zero, the optimizer receives only quantum angles.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from llm_vqc.free_amplitude.init_policy import free_amplitude_init_policy
from llm_vqc.free_amplitude.model import FixedReadoutModelError, FixedReadoutQuantumModel
from llm_vqc.free_amplitude.schema import (
    free_gate_proposal_to_circuit_ir,
    validate_free_gate_proposal,
)
from llm_vqc.free_amplitude.training import (
    OptimizerContainsClassicalParameterError,
    assert_optimizer_has_only_quantum_params,
)
from llm_vqc.ir.schema import CircuitIR, EncodingSpec, MeasurementSpec, RotationLayer


def _model(readout_qubit: int = 0) -> FixedReadoutQuantumModel:
    p = {
        "n_qubits": 3,
        "operations": [
            {"gate": "H", "wires": [1]}, {"gate": "CRY", "wires": [1, 0]},
            {"gate": "RZ", "wires": [2]},
        ],
    }
    v = validate_free_gate_proposal(p, expected_n_qubits=3)
    ir = free_gate_proposal_to_circuit_ir(v.proposal, readout_qubit=readout_qubit)
    return FixedReadoutQuantumModel(ir, readout_qubit=readout_qubit)


def test_no_embed_head_or_output_weights():
    model = _model()
    assert not hasattr(model, "embed")
    assert not hasattr(model, "head")
    assert not hasattr(model, "w_out")
    assert not hasattr(model, "b_out")
    assert not hasattr(model, "output_head")


def test_classical_parameter_count_is_zero():
    model = _model()
    assert model.classical_parameter_count == 0


def test_only_quantum_parameters_exist():
    model = _model()
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    assert all(name.startswith("q_layer.") for name in names)


def test_prediction_formula_matches_1_minus_z0_over_2():
    model = _model()
    model.double()
    x = torch.rand(5, 8, dtype=torch.float64)
    x = x / x.norm(dim=1, keepdim=True)

    # Compute z0 directly via the underlying q_layer and compare.
    with torch.no_grad():
        z0 = model.q_layer(x)
        mu_hat = model(x)
    assert torch.allclose(mu_hat, (1.0 - z0) / 2.0)


def test_prediction_stays_in_unit_interval():
    model = _model()
    model.double()
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 8))
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    x_t = torch.tensor(x, dtype=torch.float64)
    with torch.no_grad():
        mu_hat = model(x_t)
    assert torch.all(mu_hat >= 0.0)
    assert torch.all(mu_hat <= 1.0)


def test_only_readout_qubit_wire_measured():
    ir = free_gate_proposal_to_circuit_ir(
        validate_free_gate_proposal(
            {"n_qubits": 3, "operations": [{"gate": "RY", "wires": [0]}]}, 3
        ).proposal,
        readout_qubit=2,
    )
    FixedReadoutQuantumModel(ir, readout_qubit=2)  # must construct without error
    assert ir.measurements.wires == [2]
    assert ir.measurements.observable == "Z"


def test_rejects_non_amplitude_encoding():
    ir = CircuitIR(
        n_qubits=3,
        encoding=EncodingSpec(type="angle", gate="RY", wires=[0, 1, 2]),
        layers=[RotationLayer(gates=["RY"], wires=[0])],
        measurements=MeasurementSpec(observable="Z", wires=[0]),
    )
    with pytest.raises(FixedReadoutModelError):
        FixedReadoutQuantumModel(ir, readout_qubit=0)


def test_rejects_multi_wire_measurement():
    ir = CircuitIR(
        n_qubits=3,
        encoding=EncodingSpec(type="amplitude", wires="all"),
        layers=[RotationLayer(gates=["RY"], wires=[0])],
        measurements=MeasurementSpec(observable="Z", wires=[0, 1]),
    )
    with pytest.raises(FixedReadoutModelError):
        FixedReadoutQuantumModel(ir, readout_qubit=0)


def test_optimizer_with_only_quantum_params_passes_assertion():
    model = _model()
    free_amplitude_init_policy(model, param_init_seed=1)
    optimizer = torch.optim.AdamW([model.q_layer.weights], lr=0.05)
    assert_optimizer_has_only_quantum_params(optimizer, model)  # must not raise


def test_optimizer_with_extra_classical_param_fails_assertion():
    model = _model()
    free_amplitude_init_policy(model, param_init_seed=1)
    fake_classical_param = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
    optimizer = torch.optim.AdamW([model.q_layer.weights, fake_classical_param], lr=0.05)
    with pytest.raises(OptimizerContainsClassicalParameterError):
        assert_optimizer_has_only_quantum_params(optimizer, model)
