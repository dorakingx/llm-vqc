"""Frozen generator constants for the bench_v2 signal suite.

`SIGNAL_SUITE_VERSION` participates in every cache key and dataset
manifest (protocol §3). Changing ANY constant in this module without
bumping the version is a protocol violation — the constants below were
frozen at Phase 1, before any v2 search experiment ran, per
`docs/research/BENCHMARK_V2_PROTOCOL.md`.

Design notes that the numbers encode:

- **T1 (`gauss_peak`)** uses one width range (0.08, 0.20) for every v2
  profile at every qubit count: peaks keep the same physical width and a
  larger grid (higher n) simply resolves them better — that resolution
  gain IS the scaling story for RQ4, rather than a per-n difficulty
  re-tuning that would make cross-n comparisons incoherent. At n=3 the
  lower end is under-resolved by the legacy resolution rule-of-thumb
  (ratio 0.56 < 2.0) exactly as the preserved legacy profile already
  accepted; the warning is recorded in every manifest, never silenced.
- **T2 (`sin_freq`)** frequency range is a profile *function* of the
  sample count N: `f ∈ [1, N/4]` cycles per unit interval — the maximum
  is half the Nyquist limit (N/2), so every admissible frequency is
  sampled at ≥ 4 points/cycle and the label is identifiable. The target
  is `(f-1)/(N/4-1)`, normalized to [0,1] within each profile.
- **T3 (`change_point`)** enforces a minimum pre/post level separation
  (0.4) so the change point is identifiable above noise, randomizes the
  step sign and both segment slopes so no fixed amplitude/sign convention
  leaks the label, and keeps the change point in [0.2, 0.8] so both
  segments always contain ≥ 20% of the samples.
- **T4 (`peak_count`)** requires ≥ 4 grid points between the two peak
  centers of a double-peak sample so the classes are physically
  distinguishable at the profile's resolution; classes are exactly
  balanced by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

SIGNAL_SUITE_VERSION = "signal_suite_v1"

#: Split sizes for every main/scaling v2 profile (protocol §3; provisional
#: 256/256/2048 confirmed or reduced only by the prewritten sizing rule).
N_TRAIN = 256
N_VAL = 256
N_TEST = 2048

#: Resolution rule-of-thumb ratio shared with the legacy task module.
RESOLUTION_WARNING_RATIO = 2.0

_ZERO_NORM_ATOL = 1e-12
_MAX_REGENERATION_ATTEMPTS = 10


@dataclass(frozen=True)
class GaussPeakConfig:
    """T1: y = baseline + A * exp(-(x-mu)^2 / (2 sigma^2)) + noise; target mu."""

    mu_range: tuple[float, float] = (0.10, 0.90)
    sigma_range: tuple[float, float] = (0.08, 0.20)
    amplitude_range: tuple[float, float] = (0.6, 1.4)
    baseline_range: tuple[float, float] = (0.0, 0.3)
    noise_sigma: float = 0.02


@dataclass(frozen=True)
class SinFreqConfig:
    """T2: y = A sin(2 pi f x + phi) + c + noise; target (f-1)/(N/4-1).

    `f_max` is derived per profile as N/4 (half Nyquist); it is not stored
    here because it is a pure function of the profile's feature count.
    """

    f_min: float = 1.0
    amplitude_range: tuple[float, float] = (0.5, 1.5)
    offset_range: tuple[float, float] = (-0.3, 0.3)
    noise_sigma: float = 0.05


@dataclass(frozen=True)
class ChangePointConfig:
    """T3: level a (+slope) before k*, level b (+slope) after; target k*."""

    changepoint_range: tuple[float, float] = (0.20, 0.80)
    pre_level_range: tuple[float, float] = (-1.0, 1.0)
    step_magnitude_range: tuple[float, float] = (0.4, 1.2)
    slope_range: tuple[float, float] = (-0.2, 0.2)
    noise_sigma: float = 0.03


@dataclass(frozen=True)
class PeakCountConfig:
    """T4: class 0 = one Gaussian peak, class 1 = two peaks; balanced."""

    location_range: tuple[float, float] = (0.12, 0.88)
    sigma_range: tuple[float, float] = (0.04, 0.12)
    amplitude_range: tuple[float, float] = (0.6, 1.4)
    baseline_range: tuple[float, float] = (0.0, 0.3)
    noise_sigma: float = 0.02
    min_separation_grid_points: int = 4


GAUSS_PEAK_V1 = GaussPeakConfig()
SIN_FREQ_V1 = SinFreqConfig()
CHANGE_POINT_V1 = ChangePointConfig()
PEAK_COUNT_V1 = PeakCountConfig()
