"""Tests for the mini demo's explicit, versioned initialization policy
(`llm_vqc.mini_demo.init_policy`, correction pass section 5): Xavier / zeros
/ Uniform[-pi, pi], deterministic, float64, CPU, with fail-loud checks.
"""

from __future__ import annotations

import math

import pytest
import torch

from llm_vqc.evaluation.model import HybridQNNModel
from llm_vqc.mini_demo.compact_schema import CompactArchitecture, compact_to_circuit_ir
from llm_vqc.mini_demo.init_policy import (
    EXPECTED_QUANTUM_PARAMETERS,
    EXPECTED_TOTAL_TRAINABLE_PARAMETERS,
    MINI_DEMO_INIT_VERSION,
    MiniDemoInitError,
    mini_demo_init_policy,
    verify_model_dtypes_and_device,
)

IR = compact_to_circuit_ir(
    CompactArchitecture(
        layer_1_gate="RY", entangler="CNOT", entangler_direction="0_to_1", layer_2_gate="RX"
    )
)


def _fresh_model() -> HybridQNNModel:
    return HybridQNNModel(IR, raw_feature_dim=21, head_out_dim=1)


def test_version_and_counts_constants():
    assert MINI_DEMO_INIT_VERSION == "mini_demo_init_v1"
    assert EXPECTED_QUANTUM_PARAMETERS == 4
    assert EXPECTED_TOTAL_TRAINABLE_PARAMETERS == 51


def test_deterministic_given_the_same_seed():
    m1 = _fresh_model()
    mini_demo_init_policy(m1, 12345)
    m2 = _fresh_model()
    mini_demo_init_policy(m2, 12345)
    assert torch.allclose(m1.q_layer.weights, m2.q_layer.weights)
    assert torch.allclose(m1.embed.weight, m2.embed.weight)
    assert torch.allclose(m1.head.weight, m2.head.weight)


def test_different_seed_gives_different_quantum_angles():
    m1 = _fresh_model()
    mini_demo_init_policy(m1, 1)
    m2 = _fresh_model()
    mini_demo_init_policy(m2, 2)
    assert not torch.allclose(m1.q_layer.weights, m2.q_layer.weights)


def test_biases_are_zero():
    m = _fresh_model()
    mini_demo_init_policy(m, 7)
    assert torch.all(m.embed.bias == 0)
    assert torch.all(m.head.bias == 0)


def test_quantum_angles_within_minus_pi_to_pi():
    m = _fresh_model()
    mini_demo_init_policy(m, 7)
    angles = m.q_layer.weights.detach()
    assert float(angles.min()) >= -math.pi
    assert float(angles.max()) <= math.pi


def test_all_trainable_tensors_are_float64_and_cpu():
    m = _fresh_model()
    # The quantum layer starts life as float32 (PennyLane default) -- the
    # policy must fix that.
    assert m.q_layer.weights.dtype == torch.float32
    mini_demo_init_policy(m, 7)
    for name, p in m.named_parameters():
        if p.requires_grad:
            assert p.dtype == torch.float64, name
            assert p.device.type == "cpu", name


def test_verify_helper_reports_ok():
    m = _fresh_model()
    mini_demo_init_policy(m, 7)
    result = verify_model_dtypes_and_device(m)
    assert result["all_float64"] is True
    assert result["all_cpu"] is True
    assert result["total_trainable_parameters"] == 51
    assert result["quantum_parameters"] == 4


def test_fail_loud_on_wrong_quantum_parameter_count():
    # A 3-layer circuit has 6 quantum parameters, not 4.
    from llm_vqc.ir.schema import CircuitIR, EncodingSpec, MeasurementSpec, RotationLayer

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
    model = HybridQNNModel(bad_ir, raw_feature_dim=21, head_out_dim=1)
    with pytest.raises(MiniDemoInitError):
        mini_demo_init_policy(model, 7)
