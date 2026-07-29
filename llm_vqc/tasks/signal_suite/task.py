"""Versioned task registry for the bench_v2 signal suite (protocol §3).

One `SignalSuiteTask` = one (family, n_qubits) profile with frozen
generator constants from `config.py`. Interface matches the preserved
`AmplitudeGaussianPeakTask`: `build(seed) -> (TrainValData, diagnostics)`
and `build_test(seed) -> (DataSplit, diagnostics)`, with the same
advance-the-stream construction that makes test disjointness structural
(test generation replays the train/val draws first, so the three splits
come from non-overlapping segments of one seeded stream).

The task *name* embeds `SIGNAL_SUITE_VERSION`, and every evaluator cache
key includes the task name — so a task-version bump automatically
invalidates every cache entry (protocol "task version in every cache key"
requirement, satisfied structurally).

The legacy E0/E1 profile (`gauss_peak_legacy_n3`) is a thin delegation to
the preserved `llm_vqc.free_amplitude.tasks.AmplitudeGaussianPeakTask`
with its committed `amplitude_n3_smoke_v1` profile — byte-identical
generation, not a reimplementation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from llm_vqc.evaluation.seeds import data_split_seed
from llm_vqc.free_amplitude.tasks import (
    AMPLITUDE_N3_SMOKE_V1,
    AmplitudeGaussianPeakTask,
    require_exact_feature_count,
    resolution_diagnostic,
)
from llm_vqc.tasks.base import (
    DataSplit,
    FittedPreprocessing,
    TaskSpec,
    TrainValData,
    assert_disjoint_splits,
)
from llm_vqc.tasks.signal_suite.config import (
    CHANGE_POINT_V1,
    GAUSS_PEAK_V1,
    N_TEST,
    N_TRAIN,
    N_VAL,
    PEAK_COUNT_V1,
    SIGNAL_SUITE_VERSION,
    SIN_FREQ_V1,
)
from llm_vqc.tasks.signal_suite.generators import (
    generate_change_point,
    generate_gauss_peak,
    generate_peak_count,
    generate_sin_freq,
    sin_freq_max,
)

FAMILIES = ("gauss_peak", "sin_freq", "change_point", "peak_count")

_GENERATORS = {
    "gauss_peak": (generate_gauss_peak, GAUSS_PEAK_V1),
    "sin_freq": (generate_sin_freq, SIN_FREQ_V1),
    "change_point": (generate_change_point, CHANGE_POINT_V1),
    "peak_count": (generate_peak_count, PEAK_COUNT_V1),
}

#: Which (family, n_qubits) profiles exist in bench_v2 (protocol §10).
#: peak_count/change_point exist only at n=5 (E2); gauss_peak/sin_freq
#: additionally cover the E3 scaling range.
ALLOWED_PROFILES: dict[str, tuple[int, ...]] = {
    "gauss_peak": (3, 4, 5, 6, 8),
    "sin_freq": (3, 4, 5, 6, 8),
    "change_point": (5,),
    "peak_count": (5,),
}


class UnknownProfileError(Exception):
    """Raised for a (family, n_qubits) combination outside the frozen matrix."""


def task_name(family: str, n_qubits: int) -> str:
    return f"sig_{family}_n{n_qubits}_{SIGNAL_SUITE_VERSION}"


@dataclass(frozen=True)
class SignalProfile:
    family: str
    n_qubits: int
    n_train: int = N_TRAIN
    n_val: int = N_VAL
    n_test: int = N_TEST

    @property
    def feature_count(self) -> int:
        return 2**self.n_qubits

    @property
    def name(self) -> str:
        return task_name(self.family, self.n_qubits)


def _identity_preprocessing() -> FittedPreprocessing:
    return FittedPreprocessing(kind="none", params={}, _transform_fn=lambda x, params: x)


class SignalSuiteTask:
    """One frozen signal-suite profile, generating length-2**n signals."""

    def __init__(self, profile: SignalProfile) -> None:
        if profile.family not in _GENERATORS:
            raise UnknownProfileError(f"unknown family {profile.family!r}")
        if profile.n_qubits not in ALLOWED_PROFILES[profile.family]:
            raise UnknownProfileError(
                f"{profile.family} is not frozen for n_qubits={profile.n_qubits} "
                f"(allowed: {ALLOWED_PROFILES[profile.family]})"
            )
        require_exact_feature_count(profile.feature_count, profile.n_qubits)
        self.profile = profile
        self.is_classification = profile.family == "peak_count"
        self.generator, self.generator_config = _GENERATORS[profile.family]
        self.spec = TaskSpec(
            name=profile.name,
            description=(
                f"signal_suite {profile.family}, n_qubits={profile.n_qubits}, "
                f"feature_count={profile.feature_count}, {SIGNAL_SUITE_VERSION}"
            ),
            raw_feature_dim=profile.feature_count,
            classical_head_out_dim=1,
            metric_name="auc" if self.is_classification else "rmse",
            lower_is_better=not self.is_classification,
            loss_name="mse",
        )
        #: Validation-selection metric (protocol §5): Brier for T4 (proper
        #: scoring rule on the probability output; same lower-is-better
        #: direction as RMSE), RMSE for the regression families.
        self.val_metric_name = "brier" if self.is_classification else "rmse"

    def _generate(self, rng: np.random.Generator, n_samples: int, id_prefix: str):
        features, targets, nuisance = self.generator(
            rng, n_samples, self.profile.feature_count, self.generator_config
        )
        sample_ids = tuple(f"{id_prefix}-{i}" for i in range(n_samples))
        return DataSplit(features=features, targets=targets, sample_ids=sample_ids), nuisance

    def _diagnostics(self, split: DataSplit, nuisance: dict) -> dict:
        norms = np.linalg.norm(split.features, axis=1)
        diag: dict = {
            "pre_normalization_l2_norm": {
                "mean": float(norms.mean()), "min": float(norms.min()), "max": float(norms.max()),
            },
            "target_stats": {
                "min": float(split.targets.min()), "max": float(split.targets.max()),
                "mean": float(split.targets.mean()), "std": float(split.targets.std()),
            },
            "generator_config": asdict(self.generator_config),
            "signal_suite_version": SIGNAL_SUITE_VERSION,
        }
        if self.profile.family == "gauss_peak":
            diag["resolution"] = resolution_diagnostic(
                self.profile.feature_count, self.generator_config.sigma_range[0]
            )
        if self.profile.family == "sin_freq":
            f_max = sin_freq_max(self.profile.feature_count)
            diag["aliasing"] = {
                "f_max_cycles": f_max,
                "nyquist_cycles": self.profile.feature_count / 2.0,
                "below_nyquist": f_max < self.profile.feature_count / 2.0,
            }
        if self.is_classification:
            diag["class_balance"] = nuisance.get("class_balance")
        return diag

    def build(self, seed: int) -> tuple[TrainValData, dict]:
        split_seed = data_split_seed(seed, self.spec.name)
        rng = np.random.default_rng(split_seed)
        train, train_nuis = self._generate(rng, self.profile.n_train, "train")
        val, val_nuis = self._generate(rng, self.profile.n_val, "val")
        assert_disjoint_splits(train, val)
        train_val = TrainValData(
            spec=self.spec, train=train, val=val, split_seed=split_seed,
            preprocessing=_identity_preprocessing(),
        )
        return train_val, {
            "train": self._diagnostics(train, train_nuis),
            "val": self._diagnostics(val, val_nuis),
        }

    def build_test(self, seed: int) -> tuple[DataSplit, dict]:
        split_seed = data_split_seed(seed, self.spec.name)
        rng = np.random.default_rng(split_seed)
        # Identical draw order to `build` so test occupies a disjoint
        # segment of the same stream.
        self._generate(rng, self.profile.n_train, "train")
        self._generate(rng, self.profile.n_val, "val")
        test, test_nuis = self._generate(rng, self.profile.n_test, "test")
        return test, {"test": self._diagnostics(test, test_nuis)}


def get_signal_task(family: str, n_qubits: int) -> SignalSuiteTask:
    return SignalSuiteTask(SignalProfile(family=family, n_qubits=n_qubits))


def get_legacy_gauss_peak_task() -> AmplitudeGaussianPeakTask:
    """The preserved E0/E1 legacy profile — exact committed behaviour."""
    return AmplitudeGaussianPeakTask(AMPLITUDE_N3_SMOKE_V1)
