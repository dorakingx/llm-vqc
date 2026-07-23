"""Amplitude-encoded Gaussian-peak regression task profiles (Codex
instruction sections 4-6).

**Why `feature_count == 2 ** n_qubits` exactly.** Amplitude encoding maps a
`2**n`-dimensional real vector onto the amplitudes of an `n`-qubit
statevector -- there is no meaningful way to encode more or fewer values
without padding (inventing data) or truncating (discarding data), both of
which this module explicitly refuses to do. A configuration or sample
whose feature count does not exactly match `2 ** n_qubits` is a
programming error, not something to silently correct, hence
`AmplitudeFeatureCountError` rather than a resize.

**Single normalization location.** This module deliberately does NOT
L2-normalize the generated feature vectors -- normalization happens once,
in the PennyLane `AmplitudeEmbedding(..., normalize=True)` call inside
`llm_vqc.ir.compiler_pennylane.to_qnode` (used by
`llm_vqc.free_amplitude.model.FixedReadoutQuantumModel`). This module only
*computes and records* the pre-normalization L2 norm, as a diagnostic --
see `llm_vqc.free_amplitude.model`'s docstring for the same point from the
model side.

**Consequence of amplitude normalization (Codex instruction section 6).**
Because encoding normalizes away overall scale, two samples differing only
by a positive multiplicative factor `A` become the *same* normalized
quantum state (before noise). This is acceptable here because the
regression target is the peak location `mu`, and `A` (amplitude/height) is
an explicit nuisance parameter in the generating formula below -- but it
means this model structurally cannot recover absolute signal magnitude,
and that limitation must be stated in any report using this task, not
just implied.

**Resolution matters.** A resolution profile with few grid points does not
resolve a very narrow Gaussian peak. `resolution_diagnostic` computes
`grid_spacing = 1 / (feature_count - 1)` and warns (does not silently
change anything) when the configured `sigma_min` is well below it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from llm_vqc.evaluation.seeds import data_split_seed
from llm_vqc.tasks.base import (
    DataSplit,
    FittedPreprocessing,
    TaskSpec,
    TrainValData,
    assert_disjoint_splits,
)

#: Below this ratio (sigma_min / grid_spacing), a Gaussian peak's width is
#: comparable to or smaller than the grid resolution -- an engineering
#: threshold (not derived from the task) chosen as "at least ~2 grid
#: points across one standard deviation" as a resolvability rule of thumb.
RESOLUTION_WARNING_RATIO = 2.0

_ZERO_NORM_ATOL = 1e-12
_MAX_REGENERATION_ATTEMPTS = 10


class AmplitudeFeatureCountError(Exception):
    """Raised when a feature vector's length does not exactly equal
    `2 ** n_qubits`. Carries the three fields required by Codex
    instruction section 4 so the caller can report them without
    re-deriving anything."""

    def __init__(self, expected_feature_count: int, actual_feature_count: int, n_qubits: int):
        self.expected_feature_count = expected_feature_count
        self.actual_feature_count = actual_feature_count
        self.n_qubits = n_qubits
        super().__init__(
            f"expected_feature_count={expected_feature_count}, "
            f"actual_feature_count={actual_feature_count}, n_qubits={n_qubits}"
        )


def require_exact_feature_count(feature_count: int, n_qubits: int) -> None:
    expected = 2**n_qubits
    if feature_count != expected:
        raise AmplitudeFeatureCountError(expected, feature_count, n_qubits)


@dataclass(frozen=True)
class AmplitudeDatasetProfile:
    """A named, versioned dataset configuration (Codex instruction section
    5: "Create named dataset profiles... Do not silently change sigma
    ranges at runtime.")."""

    name: str
    n_qubits: int
    feature_count: int
    sigma_min: float
    sigma_max: float
    n_train: int = 150
    n_val: int = 250
    n_test: int = 2000

    def __post_init__(self) -> None:
        require_exact_feature_count(self.feature_count, self.n_qubits)
        if self.sigma_min <= 0 or self.sigma_max <= 0 or self.sigma_min > self.sigma_max:
            raise ValueError(
                f"invalid sigma range: sigma_min={self.sigma_min}, sigma_max={self.sigma_max}"
            )


#: n_qubits=3, feature_count=8, sigma range chosen to be resolvable on the
#: 8-point grid (recommended default per Codex instruction section 5).
#: Its RMSE is NOT numerically comparable to the previous 21-point T1
#: experiment -- this is a different task profile with a different grid.
AMPLITUDE_N3_SMOKE_V1 = AmplitudeDatasetProfile(
    name="amplitude_n3_smoke_v1", n_qubits=3, feature_count=8, sigma_min=0.08, sigma_max=0.20,
)

#: n_qubits=5, feature_count=32. Sigma range is configurable at the CLI
#: (--sigma-min/--sigma-max); this default is only used when neither is
#: given.
AMPLITUDE_N5_MAIN_V1 = AmplitudeDatasetProfile(
    name="amplitude_n5_main_v1", n_qubits=5, feature_count=32, sigma_min=0.03, sigma_max=0.20,
)

DATASET_PROFILES: dict[str, AmplitudeDatasetProfile] = {
    AMPLITUDE_N3_SMOKE_V1.name: AMPLITUDE_N3_SMOKE_V1,
    AMPLITUDE_N5_MAIN_V1.name: AMPLITUDE_N5_MAIN_V1,
}


def resolution_diagnostic(feature_count: int, sigma_min: float) -> dict:
    """`grid_spacing = 1 / (feature_count - 1)`; warns (never silently
    changes anything) when `sigma_min` is well below it."""
    grid_spacing = 1.0 / (feature_count - 1)
    ratio = sigma_min / grid_spacing
    warning = ratio < RESOLUTION_WARNING_RATIO
    return {
        "feature_count": feature_count,
        "grid_spacing": grid_spacing,
        "sigma_min": sigma_min,
        "sigma_min_over_grid_spacing": ratio,
        "warning": warning,
        "message": (
            f"sigma_min={sigma_min} is only {ratio:.2f}x the grid spacing "
            f"({grid_spacing:.4f}); narrow peaks may not be resolvable on this grid."
            if warning else None
        ),
    }


def task_spec_for_profile(profile: AmplitudeDatasetProfile) -> TaskSpec:
    return TaskSpec(
        name=profile.name,
        description=(
            f"Amplitude-encoded 1D Gaussian-peak regression, n_qubits={profile.n_qubits}, "
            f"feature_count={profile.feature_count}"
        ),
        raw_feature_dim=profile.feature_count,
        classical_head_out_dim=1,  # unused by FixedReadoutQuantumModel; kept for TaskSpec parity
        metric_name="rmse",
        lower_is_better=True,
        loss_name="mse",
    )


def _generate_raw_split(
    rng: np.random.Generator, n_samples: int, profile: AmplitudeDatasetProfile, id_prefix: str
) -> tuple[DataSplit, list[float]]:
    """Returns `(split, pre_normalization_l2_norms)` -- the norms are
    computed here, once, purely for diagnostics; the returned
    `split.features` are NOT normalized (see module docstring)."""
    x_grid = np.linspace(0.0, 1.0, profile.feature_count)
    mu = rng.uniform(0.0, 1.0, size=n_samples)
    amplitude = rng.uniform(0.5, 1.5, size=n_samples)
    sigma = rng.uniform(profile.sigma_min, profile.sigma_max, size=n_samples)
    noise = rng.normal(0.0, 0.01, size=(n_samples, profile.feature_count))

    x = x_grid[None, :]
    mu_col, amp_col, sigma_col = mu[:, None], amplitude[:, None], sigma[:, None]
    y = amp_col / (sigma_col * np.sqrt(2 * np.pi)) * np.exp(
        -((x - mu_col) ** 2) / (2 * sigma_col**2)
    ) + noise

    pre_norms = np.linalg.norm(y, axis=1)
    zero_mask = pre_norms < _ZERO_NORM_ATOL
    attempts = 0
    while zero_mask.any() and attempts < _MAX_REGENERATION_ATTEMPTS:
        n_bad = int(zero_mask.sum())
        idx = np.where(zero_mask)[0]
        mu[idx] = rng.uniform(0.0, 1.0, size=n_bad)
        amplitude[idx] = rng.uniform(0.5, 1.5, size=n_bad)
        sigma[idx] = rng.uniform(profile.sigma_min, profile.sigma_max, size=n_bad)
        noise[idx] = rng.normal(0.0, 0.01, size=(n_bad, profile.feature_count))
        mu_col, amp_col, sigma_col = mu[:, None], amplitude[:, None], sigma[:, None]
        y = amp_col / (sigma_col * np.sqrt(2 * np.pi)) * np.exp(
            -((x - mu_col) ** 2) / (2 * sigma_col**2)
        ) + noise
        pre_norms = np.linalg.norm(y, axis=1)
        zero_mask = pre_norms < _ZERO_NORM_ATOL
        attempts += 1
    if zero_mask.any():
        raise ValueError(
            f"{int(zero_mask.sum())} sample(s) still have a numerically zero L2 norm "
            f"after {_MAX_REGENERATION_ATTEMPTS} regeneration attempts"
        )

    require_exact_feature_count(y.shape[1], profile.n_qubits)
    sample_ids = tuple(f"{id_prefix}-{i}" for i in range(n_samples))
    return DataSplit(features=y, targets=mu, sample_ids=sample_ids), pre_norms.tolist()


def _identity_preprocessing() -> FittedPreprocessing:
    # No preprocessing is applied here (normalization happens once, at the
    # backend AmplitudeEmbedding step) -- kept as a FittedPreprocessing for
    # interface parity with other tasks, per llm_vqc.tasks.base's own
    # convention for stateless transforms.
    return FittedPreprocessing(kind="none", params={}, _transform_fn=lambda x, params: x)


class AmplitudeGaussianPeakTask:
    """Builds `TrainValData`/test `DataSplit` for one named
    `AmplitudeDatasetProfile`. Unlike `llm_vqc.tasks.t1_gaussian`, features
    are NOT normalized here (see module docstring); pre-normalization L2
    norms are returned alongside for diagnostics."""

    def __init__(self, profile: AmplitudeDatasetProfile) -> None:
        self.profile = profile
        self.spec = task_spec_for_profile(profile)

    def build(self, seed: int) -> tuple[TrainValData, dict]:
        split_seed = data_split_seed(seed, self.spec.name)
        rng = np.random.default_rng(split_seed)

        train, train_norms = _generate_raw_split(rng, self.profile.n_train, self.profile, "train")
        val, val_norms = _generate_raw_split(rng, self.profile.n_val, self.profile, "val")
        assert_disjoint_splits(train, val)

        train_val = TrainValData(
            spec=self.spec, train=train, val=val, split_seed=split_seed,
            preprocessing=_identity_preprocessing(),
        )
        diagnostics = {
            "pre_normalization_l2_norm": {
                "train_mean": float(np.mean(train_norms)), "train_min": float(np.min(train_norms)),
                "train_max": float(np.max(train_norms)),
                "val_mean": float(np.mean(val_norms)), "val_min": float(np.min(val_norms)),
                "val_max": float(np.max(val_norms)),
            },
            "resolution": resolution_diagnostic(self.profile.feature_count, self.profile.sigma_min),
        }
        return train_val, diagnostics

    def build_test(self, seed: int) -> tuple[DataSplit, dict]:
        split_seed = data_split_seed(seed, self.spec.name)
        rng = np.random.default_rng(split_seed)
        # Advance the same rng stream past train+val (identical draw order
        # to `build`) so test is disjoint by construction.
        _generate_raw_split(rng, self.profile.n_train, self.profile, "train")
        _generate_raw_split(rng, self.profile.n_val, self.profile, "val")
        test, test_norms = _generate_raw_split(rng, self.profile.n_test, self.profile, "test")
        diagnostics = {
            "pre_normalization_l2_norm": {
                "test_mean": float(np.mean(test_norms)), "test_min": float(np.min(test_norms)),
                "test_max": float(np.max(test_norms)),
            }
        }
        return test, diagnostics
