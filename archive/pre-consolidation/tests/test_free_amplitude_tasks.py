"""Tests for the amplitude-encoded task profiles (Codex instruction
section 21 "Amplitude encoding" checklist): feature count must equal
`2 ** n_qubits`, no silent padding/truncation, nonzero-norm requirement,
deterministic normalization (single location), amplitude state has unit
norm (verified through the actual PennyLane embedding), 3 qubits require
8 features, 5 qubits require 32 features, amplitude encoding contains
zero trainable parameters.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
import pytest

from llm_vqc.free_amplitude.tasks import (
    AMPLITUDE_N3_SMOKE_V1,
    AMPLITUDE_N5_MAIN_V1,
    AmplitudeDatasetProfile,
    AmplitudeFeatureCountError,
    AmplitudeGaussianPeakTask,
    require_exact_feature_count,
    resolution_diagnostic,
)
from llm_vqc.ir.schema import CircuitIR, EncodingSpec, MeasurementSpec
from llm_vqc.tasks.base import assert_disjoint_splits

# --- feature count must equal 2 ** n_qubits ---------------------------------


def test_require_exact_feature_count_passes_when_matching():
    require_exact_feature_count(8, 3)  # must not raise


def test_require_exact_feature_count_raises_with_all_three_fields():
    with pytest.raises(AmplitudeFeatureCountError) as exc_info:
        require_exact_feature_count(7, 3)
    err = exc_info.value
    assert err.expected_feature_count == 8
    assert err.actual_feature_count == 7
    assert err.n_qubits == 3


def test_profile_construction_rejects_mismatched_feature_count():
    with pytest.raises(AmplitudeFeatureCountError):
        AmplitudeDatasetProfile(
            name="bad", n_qubits=3, feature_count=7, sigma_min=0.1, sigma_max=0.2
        )


def test_three_qubits_requires_eight_features():
    assert AMPLITUDE_N3_SMOKE_V1.n_qubits == 3
    assert AMPLITUDE_N3_SMOKE_V1.feature_count == 8


def test_five_qubits_requires_thirty_two_features():
    assert AMPLITUDE_N5_MAIN_V1.n_qubits == 5
    assert AMPLITUDE_N5_MAIN_V1.feature_count == 32


# --- no silent padding/truncation -------------------------------------------


def test_generated_splits_have_exact_feature_count_no_padding_or_truncation():
    task = AmplitudeGaussianPeakTask(AMPLITUDE_N3_SMOKE_V1)
    train_val, _ = task.build(seed=0)
    assert train_val.train.features.shape[1] == 8
    assert train_val.val.features.shape[1] == 8
    test_split, _ = task.build_test(seed=0)
    assert test_split.features.shape[1] == 8


# --- nonzero norm requirement ------------------------------------------------


def test_all_generated_samples_have_nonzero_l2_norm():
    task = AmplitudeGaussianPeakTask(AMPLITUDE_N3_SMOKE_V1)
    train_val, _ = task.build(seed=0)
    norms = np.linalg.norm(train_val.train.features, axis=1)
    assert np.all(norms > 1e-12)


# --- deterministic normalization; single location ---------------------------


def test_task_does_not_normalize_features_itself():
    """Features are NOT pre-normalized by the task -- normalization
    happens once, at the backend AmplitudeEmbedding step."""
    task = AmplitudeGaussianPeakTask(AMPLITUDE_N3_SMOKE_V1)
    train_val, diag = task.build(seed=0)
    norms = np.linalg.norm(train_val.train.features, axis=1)
    assert not np.allclose(norms, 1.0)  # raw features are not unit-norm
    assert diag["pre_normalization_l2_norm"]["train_mean"] > 0


def test_build_is_deterministic_given_the_same_seed():
    task = AmplitudeGaussianPeakTask(AMPLITUDE_N3_SMOKE_V1)
    tv1, _ = task.build(seed=0)
    tv2, _ = task.build(seed=0)
    assert np.allclose(tv1.train.features, tv2.train.features)
    assert np.allclose(tv1.train.targets, tv2.train.targets)


def test_train_val_test_splits_are_disjoint():
    task = AmplitudeGaussianPeakTask(AMPLITUDE_N3_SMOKE_V1)
    train_val, _ = task.build(seed=0)
    test_split, _ = task.build_test(seed=0)
    assert_disjoint_splits(train_val.train, train_val.val, test_split)  # must not raise


# --- amplitude state has unit norm (via the actual embedding) ---------------


def test_amplitude_embedding_produces_unit_norm_state():
    ir = CircuitIR(
        n_qubits=3, encoding=EncodingSpec(type="amplitude", wires="all"), layers=[],
        measurements=MeasurementSpec(observable="Z", wires=[0]),
    )
    from llm_vqc.ir.expand import build_program

    program = build_program(ir)

    dev = qml.device("default.qubit", wires=3)

    @qml.qnode(dev)
    def circuit(inputs):
        qml.AmplitudeEmbedding(inputs, wires=list(program.amplitude_wires), normalize=True)
        return qml.state()

    raw_features = np.array([3.0, -1.0, 0.5, 2.0, 0.0, 1.5, -2.0, 0.7])
    state = circuit(raw_features)
    norm = float(np.sum(np.abs(state) ** 2))
    assert abs(norm - 1.0) < 1e-9


# --- amplitude encoding contains zero trainable parameters ------------------


def test_amplitude_encoding_itself_contributes_zero_parameters():
    ir = CircuitIR(
        n_qubits=3, encoding=EncodingSpec(type="amplitude", wires="all"), layers=[],
        measurements=MeasurementSpec(observable="Z", wires=[0]),
    )
    from llm_vqc.free_amplitude.model import FixedReadoutQuantumModel

    model = FixedReadoutQuantumModel(ir, readout_qubit=0)
    assert model.quantum_parameter_count == 0


# --- resolution diagnostic ---------------------------------------------------


def test_resolution_diagnostic_computes_grid_spacing_and_ratio():
    result = resolution_diagnostic(feature_count=8, sigma_min=0.08)
    assert abs(result["grid_spacing"] - 1 / 7) < 1e-12
    assert result["sigma_min_over_grid_spacing"] == pytest.approx(0.08 / (1 / 7))


def test_resolution_diagnostic_warns_for_narrow_sigma():
    result = resolution_diagnostic(feature_count=8, sigma_min=0.01)
    assert result["warning"] is True
    assert result["message"] is not None


def test_resolution_diagnostic_never_changes_sigma_silently():
    # Calling the diagnostic must not mutate any profile constant.
    before = (AMPLITUDE_N3_SMOKE_V1.sigma_min, AMPLITUDE_N3_SMOKE_V1.sigma_max)
    resolution_diagnostic(feature_count=8, sigma_min=0.001)
    after = (AMPLITUDE_N3_SMOKE_V1.sigma_min, AMPLITUDE_N3_SMOKE_V1.sigma_max)
    assert before == after
