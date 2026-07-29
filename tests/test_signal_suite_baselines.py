"""Classical sanity baselines: correctness smoke on small profiles.

The signal-processing estimators must clearly beat chance on their own
task (they see the true generative family), and the learned baselines
must produce finite metrics on every family — difficulty-context
machinery, exercised end to end.
"""

from __future__ import annotations

import numpy as np

from llm_vqc.tasks.signal_suite import SignalProfile, SignalSuiteTask
from llm_vqc.tasks.signal_suite.baselines import run_classical_baselines

SEED = 4321


def _run(family: str, n_qubits: int = 5):
    task = SignalSuiteTask(
        SignalProfile(family=family, n_qubits=n_qubits, n_train=48, n_val=48, n_test=96)
    )
    train_val, _ = task.build(SEED)
    test, _ = task.build_test(SEED)
    return task, run_classical_baselines(task, train_val, test)


def test_gauss_peak_signal_estimator_beats_chance():
    _, results = _run("gauss_peak")
    # Chance = predicting the mean of U(0.1, 0.9): RMSE ~ 0.23.
    assert results["signal_estimator"]["test"]["rmse"] < 0.1


def test_sin_freq_signal_estimator_beats_chance():
    _, results = _run("sin_freq")
    assert results["signal_estimator"]["test"]["rmse"] < 0.15


def test_change_point_signal_estimator_beats_chance():
    _, results = _run("change_point")
    # Chance = mean of U(0.2, 0.8): RMSE ~ 0.17.
    assert results["signal_estimator"]["test"]["rmse"] < 0.1


def test_peak_count_signal_estimator_beats_chance():
    _, results = _run("peak_count")
    assert results["signal_estimator"]["test"]["auc"] > 0.7


def test_learned_baselines_finite_on_all_families():
    for family in ("gauss_peak", "sin_freq", "change_point", "peak_count"):
        task, results = _run(family)
        linear_key = "logistic_regression" if task.is_classification else "ridge_regression"
        for name in (linear_key, "mlp_32", "signal_estimator"):
            for split in ("val", "test"):
                for value in results[name][split].values():
                    assert np.isfinite(value)


def test_meta_declares_no_advantage_purpose():
    _, results = _run("gauss_peak")
    assert "no quantum-advantage claims" in results["_meta"]["purpose"]
