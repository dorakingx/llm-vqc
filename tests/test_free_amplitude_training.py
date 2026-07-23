"""Tests for the fixed-readout training loop (Codex instruction section
21 "Gradient and training" checklist): loss backpropagates from the q0
measurement to a known nondegenerate angle, at least one known trainable
circuit changes its angle, zero-gradient circuits are identified honestly,
initial and learned angles are recorded, no classical parameter is
updated (there is none to update).
"""

from __future__ import annotations

import pytest
import torch

from llm_vqc.evaluation.seeds import train_seed_for_circuit
from llm_vqc.free_amplitude.schema import (
    free_gate_proposal_to_circuit_ir,
    validate_free_gate_proposal,
)
from llm_vqc.free_amplitude.tasks import AMPLITUDE_N3_SMOKE_V1, AmplitudeGaussianPeakTask
from llm_vqc.free_amplitude.training import (
    FreeAmplitudeTrainingConfig,
    train_fixed_readout_model,
)


@pytest.fixture(scope="module")
def train_val():
    task = AmplitudeGaussianPeakTask(AMPLITUDE_N3_SMOKE_V1)
    tv, _ = task.build(seed=0)
    return tv


def _ir(operations, readout_qubit=0):
    p = {"n_qubits": 3, "operations": operations}
    v = validate_free_gate_proposal(p, expected_n_qubits=3)
    return free_gate_proposal_to_circuit_ir(v.proposal, readout_qubit=readout_qubit)


def test_training_succeeds_and_reports_history(train_val):
    ir = _ir([{"gate": "RY", "wires": [0]}, {"gate": "CRX", "wires": [0, 1]}])
    cfg = FreeAmplitudeTrainingConfig(epochs=5)
    train_seed = train_seed_for_circuit(0, "test-label")
    output = train_fixed_readout_model(ir, 0, train_val, cfg, train_seed)
    assert output.success
    assert output.epochs_completed == 5
    assert len(output.train_loss_history) == 5
    assert len(output.val_metric_history) == 5


def test_at_least_one_known_trainable_circuit_changes_its_angle(train_val):
    """A circuit with a parameter directly connected to q0 (RY on q0
    itself) must change under training -- backprop from the q0
    measurement to a known nondegenerate angle."""
    ir = _ir([{"gate": "RY", "wires": [0]}])
    cfg = FreeAmplitudeTrainingConfig(epochs=5)
    train_seed = train_seed_for_circuit(0, "ry-on-q0")
    output = train_fixed_readout_model(ir, 0, train_val, cfg, train_seed)
    assert output.success
    initial = torch.tensor(output.initial_angles)
    learned = torch.tensor(output.learned_angles)
    assert not torch.allclose(initial, learned)


def test_initial_and_learned_angles_are_recorded(train_val):
    ir = _ir([{"gate": "RY", "wires": [0]}, {"gate": "RZ", "wires": [1]}])
    cfg = FreeAmplitudeTrainingConfig(epochs=3)
    train_seed = train_seed_for_circuit(0, "record-angles")
    output = train_fixed_readout_model(ir, 0, train_val, cfg, train_seed)
    assert output.initial_angles is not None
    assert output.learned_angles is not None
    assert len(output.initial_angles) == 2
    assert len(output.learned_angles) == 2
    assert output.angle_deltas == [
        learned - initial
        for learned, initial in zip(output.learned_angles, output.initial_angles, strict=True)
    ]


def test_gradient_flows_from_q0_measurement_to_connected_angle(train_val):
    """Direct check: the initial gradient of an RY(q0) angle is nonzero
    (loss backpropagates from the q0 measurement to this angle)."""
    ir = _ir([{"gate": "RY", "wires": [0]}])
    cfg = FreeAmplitudeTrainingConfig(epochs=1)
    train_seed = train_seed_for_circuit(0, "gradient-flow")
    output = train_fixed_readout_model(ir, 0, train_val, cfg, train_seed)
    assert output.success
    assert output.initial_gradient_norm is not None
    assert output.initial_gradient_norm > 0


def test_disconnected_parameter_has_zero_gradient(train_val):
    """A parameter on a wire that never causally connects to the readout
    qubit must have an honestly-reported (near-)zero gradient -- not
    silently hidden or fabricated as nonzero."""
    ir = _ir([{"gate": "RY", "wires": [1]}], readout_qubit=0)  # wire 1 never touches wire 0
    cfg = FreeAmplitudeTrainingConfig(epochs=2)
    train_seed = train_seed_for_circuit(0, "disconnected")
    output = train_fixed_readout_model(ir, 0, train_val, cfg, train_seed)
    assert output.success
    assert output.initial_gradient_vector is not None
    assert abs(output.initial_gradient_vector[0]) < 1e-9


def test_no_classical_parameter_exists_to_update(train_val):
    """There is no classical parameter in this model at all -- confirmed
    structurally via the model's own trainable-parameter names."""
    from llm_vqc.free_amplitude.model import FixedReadoutQuantumModel

    ir = _ir([{"gate": "RY", "wires": [0]}])
    model = FixedReadoutQuantumModel(ir, readout_qubit=0)
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    assert all(name.startswith("q_layer.") for name in names)
    assert model.classical_parameter_count == 0


def test_training_is_deterministic_given_the_same_train_seed(train_val):
    ir = _ir([{"gate": "RY", "wires": [0]}, {"gate": "CRZ", "wires": [0, 2]}])
    cfg = FreeAmplitudeTrainingConfig(epochs=3)
    train_seed = train_seed_for_circuit(0, "determinism-check")
    out1 = train_fixed_readout_model(ir, 0, train_val, cfg, train_seed)
    out2 = train_fixed_readout_model(ir, 0, train_val, cfg, train_seed)
    assert out1.initial_angles == out2.initial_angles
    assert out1.learned_angles == out2.learned_angles
    assert out1.train_loss_history == out2.train_loss_history


def test_training_config_records_explicit_betas_and_eps():
    cfg = FreeAmplitudeTrainingConfig()
    assert cfg.betas == (0.9, 0.999)
    assert cfg.eps == 1e-8
    assert cfg.early_stopping is False
    assert cfg.device == "cpu"
    assert cfg.dtype == "float64"
    assert cfg.epochs == 20
    assert cfg.batch_size == 16
    assert cfg.learning_rate == 0.05
    assert cfg.weight_decay == 1e-5
