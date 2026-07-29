"""Phase 1 gate tests for the bench_v2 signal suite (goal §19).

Locks in: deterministic regeneration, split disjointness (id- and
content-level), exact 2**n feature counts, target ranges, T2 aliasing
guard, T4 exact class balance, zero-norm guards, legacy-profile
delegation, and version-in-name (cache-key) behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest

from llm_vqc.tasks.base import TaskDataError, assert_disjoint_splits
from llm_vqc.tasks.signal_suite import (
    ALLOWED_PROFILES,
    SIGNAL_SUITE_VERSION,
    SignalProfile,
    SignalSuiteTask,
    get_legacy_gauss_peak_task,
    get_signal_task,
)
from llm_vqc.tasks.signal_suite.checks import (
    dataset_manifest,
    split_hash,
    verify_deterministic_regeneration,
)
from llm_vqc.tasks.signal_suite.generators import sin_freq_max
from llm_vqc.tasks.signal_suite.task import UnknownProfileError

SEED = 1234


def _small_task(family: str, n_qubits: int = 5) -> SignalSuiteTask:
    return SignalSuiteTask(
        SignalProfile(family=family, n_qubits=n_qubits, n_train=24, n_val=24, n_test=48)
    )


ALL_FAMILIES = ["gauss_peak", "sin_freq", "change_point", "peak_count"]


@pytest.mark.parametrize("family", ALL_FAMILIES)
def test_deterministic_regeneration(family):
    assert verify_deterministic_regeneration(_small_task(family), SEED)


@pytest.mark.parametrize("family", ALL_FAMILIES)
def test_different_seeds_differ(family):
    task = _small_task(family)
    a, _ = task.build(SEED)
    b, _ = task.build(SEED + 1)
    assert split_hash(a.train) != split_hash(b.train)


@pytest.mark.parametrize("family", ALL_FAMILIES)
def test_splits_disjoint_ids_and_content(family):
    task = _small_task(family)
    train_val, _ = task.build(SEED)
    test, _ = task.build_test(SEED)
    assert_disjoint_splits(train_val.train, train_val.val, test)
    # Content-level: no identical feature row across splits.
    rows = {a.tobytes() for a in np.round(train_val.train.features, 12)}
    for other in (train_val.val.features, test.features):
        for row in np.round(other, 12):
            assert row.tobytes() not in rows


@pytest.mark.parametrize("family,n_list", list(ALLOWED_PROFILES.items()))
def test_exact_pow2_feature_count(family, n_list):
    for n in n_list:
        task = SignalSuiteTask(
            SignalProfile(family=family, n_qubits=n, n_train=8, n_val=8, n_test=8)
        )
        train_val, _ = task.build(SEED)
        assert train_val.train.features.shape[1] == 2**n


@pytest.mark.parametrize("family", ALL_FAMILIES)
def test_targets_in_unit_interval(family):
    task = _small_task(family)
    train_val, _ = task.build(SEED)
    test, _ = task.build_test(SEED)
    for split in (train_val.train, train_val.val, test):
        assert split.targets.min() >= 0.0
        assert split.targets.max() <= 1.0


@pytest.mark.parametrize("family", ALL_FAMILIES)
def test_no_zero_norm_vectors(family):
    task = _small_task(family)
    train_val, _ = task.build(SEED)
    norms = np.linalg.norm(train_val.train.features, axis=1)
    assert norms.min() > 1e-9


def test_sin_freq_below_nyquist_all_profiles():
    for n in ALLOWED_PROFILES["sin_freq"]:
        feature_count = 2**n
        assert sin_freq_max(feature_count) < feature_count / 2.0
        task = SignalSuiteTask(
            SignalProfile(family="sin_freq", n_qubits=n, n_train=16, n_val=16, n_test=16)
        )
        _, diag = task.build(SEED)
        assert diag["train"]["aliasing"]["below_nyquist"] is True


def test_peak_count_exact_class_balance():
    task = _small_task("peak_count")
    train_val, _ = task.build(SEED)
    test, _ = task.build_test(SEED)
    for split in (train_val.train, train_val.val, test):
        labels = split.targets
        assert set(np.unique(labels)) == {0.0, 1.0}
        assert int(labels.sum()) == len(labels) // 2


def test_peak_count_rejects_odd_sample_count():
    task = SignalSuiteTask(
        SignalProfile(family="peak_count", n_qubits=5, n_train=7, n_val=8, n_test=8)
    )
    from llm_vqc.tasks.signal_suite.generators import SignalGenerationError

    with pytest.raises(SignalGenerationError):
        task.build(SEED)


def test_unknown_profile_rejected():
    with pytest.raises(UnknownProfileError):
        get_signal_task("peak_count", 8)  # T4 frozen only at n=5
    with pytest.raises(UnknownProfileError):
        get_signal_task("no_such_family", 5)


def test_task_name_contains_version():
    task = _small_task("gauss_peak")
    assert SIGNAL_SUITE_VERSION in task.spec.name  # version reaches cache keys via name


def test_manifest_fields_complete():
    manifest = dataset_manifest(_small_task("sin_freq"), SEED)
    for key in (
        "signal_suite_version", "split_hashes", "generator_config_sha256",
        "sizes", "diagnostics", "splits_disjoint",
    ):
        assert key in manifest
    assert manifest["splits_disjoint"] is True


def test_legacy_profile_is_the_committed_one():
    legacy = get_legacy_gauss_peak_task()
    assert legacy.profile.name == "amplitude_n3_smoke_v1"
    assert legacy.profile.sigma_min == pytest.approx(0.08)
    assert legacy.profile.sigma_max == pytest.approx(0.20)
    train_val, _ = legacy.build(SEED)
    assert train_val.train.features.shape[1] == 8


def test_val_metric_names_follow_protocol():
    assert _small_task("gauss_peak").val_metric_name == "rmse"
    assert _small_task("peak_count").val_metric_name == "brier"
    assert _small_task("peak_count").spec.metric_name == "auc"
    assert _small_task("peak_count").spec.lower_is_better is False
    assert _small_task("gauss_peak").spec.lower_is_better is True


def test_duplicate_sample_ids_rejected_by_base():
    task = _small_task("gauss_peak")
    train_val, _ = task.build(SEED)
    with pytest.raises(TaskDataError):
        assert_disjoint_splits(train_val.train, train_val.train)
