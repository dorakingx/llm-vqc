"""Phase 5 gate tests: cross-backend/transpile reproducibility, shot and
noise evaluation sanity (goal §17 Phase 5 gate; §19 transpilation
determinism item).
"""

from __future__ import annotations

import numpy as np
import pytest

from llm_vqc.bench_v2.resources import (
    COUPLING_PROFILES,
    NOISE_PROFILE_V1,
    exact_eval,
    finite_shot_eval,
    logical_resource_summary,
    noisy_eval,
    transpiled_resource_summary,
)
from llm_vqc.bench_v2.space import SpaceProfile, reference_ir

SPACE3 = SpaceProfile("scalable_layered_v1", 3)


@pytest.fixture(scope="module")
def selected():
    ir = reference_ir("ref_strongent_d1", SPACE3)
    rng = np.random.default_rng(5)
    from llm_vqc.ir.expand import build_program

    n_params = build_program(ir).num_parameters
    theta = rng.uniform(-np.pi, np.pi, size=n_params).tolist()
    features = rng.normal(0.4, 0.3, size=(8, 8)) + 0.5
    targets = rng.uniform(0.2, 0.8, size=8)
    return ir, theta, features, targets


def test_logical_summary_counts(selected):
    ir, *_ = selected
    summary = logical_resource_summary(ir)
    # RZ-RY-RZ on 3 wires = 9 one-qubit params/gates; ring CNOT = 3 CX.
    assert summary["one_qubit_count"] == 9
    assert summary["two_qubit_count"] == 3
    assert summary["parameter_count"] == 9
    assert summary["controlled_rotation_count"] == 0
    assert summary["logical_depth"] >= 1


@pytest.mark.parametrize("profile", COUPLING_PROFILES)
def test_transpile_deterministic_per_profile(selected, profile):
    ir, *_ = selected
    a = transpiled_resource_summary(ir, profile)
    b = transpiled_resource_summary(ir, profile)
    assert a == b  # fixed seed + fixed coupling -> identical result
    assert a["transpiled_depth"] >= 1
    assert a["transpiled_two_qubit_count"] >= 3


def test_line_routing_at_least_all_to_all_cost(selected):
    ir, *_ = selected
    line = transpiled_resource_summary(ir, "line")
    free = transpiled_resource_summary(ir, "all_to_all")
    assert line["transpiled_two_qubit_count"] >= free["transpiled_two_qubit_count"]


def test_finite_shot_converges_toward_exact(selected):
    ir, theta, features, targets = selected
    exact = exact_eval(ir, theta, features, targets)
    small = finite_shot_eval(ir, theta, features, targets, shots=256, shot_seed=1)
    large = finite_shot_eval(ir, theta, features, targets, shots=8192, shot_seed=1)
    # More shots -> closer to the exact expectation metric.
    assert abs(large["rmse"] - exact["rmse"]) <= abs(small["rmse"] - exact["rmse"]) + 0.02
    assert small["shots"] == 256 and large["shots"] == 8192


def test_finite_shot_seeded_reproducible(selected):
    ir, theta, features, targets = selected
    a = finite_shot_eval(ir, theta, features, targets, shots=1024, shot_seed=3)
    b = finite_shot_eval(ir, theta, features, targets, shots=1024, shot_seed=3)
    assert a == b


def test_noisy_eval_departs_from_exact_but_bounded(selected):
    ir, theta, features, targets = selected
    exact = exact_eval(ir, theta, features, targets)
    noisy = noisy_eval(ir, theta, features, targets)
    assert noisy["noise_profile"]["name"] == NOISE_PROFILE_V1["name"]
    assert np.isfinite(noisy["rmse"])
    # Depolarizing pulls predictions toward 0.5; the metric changes but
    # stays the same order of magnitude at these small error rates.
    assert abs(noisy["rmse"] - exact["rmse"]) < 0.25


def test_noisy_eval_deterministic(selected):
    ir, theta, features, targets = selected
    assert noisy_eval(ir, theta, features, targets) == noisy_eval(ir, theta, features, targets)
