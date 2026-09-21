"""Pure generator functions for the four signal-suite task families.

Each generator draws `n_samples` signals of length `feature_count` from a
single `numpy.random.Generator` stream and returns `(features, targets,
nuisance_records)`. All randomness comes from the passed `rng`; a fixed
draw ORDER per family is part of the frozen contract (the split trick in
`task.py` — regenerate earlier splits to advance the stream — depends on
it, exactly as the preserved legacy task does).

Features are NEVER L2-normalized here: normalization happens once, inside
the backend `AmplitudeEmbedding(normalize=True)` (see
`llm_vqc.free_amplitude.model`). A shared zero-norm regeneration guard
redraws any sample whose pre-normalization norm is numerically zero and
raises after a bounded number of attempts rather than silently passing a
degenerate amplitude vector downstream.
"""

from __future__ import annotations

import numpy as np

from llm_vqc.tasks.signal_suite.config import (
    _MAX_REGENERATION_ATTEMPTS,
    _ZERO_NORM_ATOL,
    ChangePointConfig,
    GaussPeakConfig,
    PeakCountConfig,
    SinFreqConfig,
)


class SignalGenerationError(Exception):
    """Raised when a generator cannot produce non-degenerate samples."""


def _grid(feature_count: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, feature_count)


def _regenerate_zero_norm(features: np.ndarray, redraw) -> np.ndarray:
    """Redraw rows whose L2 norm is numerically zero. `redraw(indices)`
    must return replacement rows drawn from the same rng stream."""
    norms = np.linalg.norm(features, axis=1)
    zero = norms < _ZERO_NORM_ATOL
    attempts = 0
    while zero.any() and attempts < _MAX_REGENERATION_ATTEMPTS:
        idx = np.where(zero)[0]
        features[idx] = redraw(idx)
        norms = np.linalg.norm(features, axis=1)
        zero = norms < _ZERO_NORM_ATOL
        attempts += 1
    if zero.any():
        raise SignalGenerationError(
            f"{int(zero.sum())} sample(s) still have zero L2 norm after "
            f"{_MAX_REGENERATION_ATTEMPTS} regeneration attempts"
        )
    return features


def generate_gauss_peak(
    rng: np.random.Generator, n_samples: int, feature_count: int, cfg: GaussPeakConfig
) -> tuple[np.ndarray, np.ndarray, dict]:
    x = _grid(feature_count)[None, :]
    mu = rng.uniform(*cfg.mu_range, size=n_samples)
    sigma = rng.uniform(*cfg.sigma_range, size=n_samples)
    amp = rng.uniform(*cfg.amplitude_range, size=n_samples)
    base = rng.uniform(*cfg.baseline_range, size=n_samples)
    noise = rng.normal(0.0, cfg.noise_sigma, size=(n_samples, feature_count))
    y = base[:, None] + amp[:, None] * np.exp(
        -((x - mu[:, None]) ** 2) / (2 * sigma[:, None] ** 2)
    ) + noise

    def redraw(idx: np.ndarray) -> np.ndarray:
        k = len(idx)
        mu[idx] = rng.uniform(*cfg.mu_range, size=k)
        sigma[idx] = rng.uniform(*cfg.sigma_range, size=k)
        amp[idx] = rng.uniform(*cfg.amplitude_range, size=k)
        base[idx] = rng.uniform(*cfg.baseline_range, size=k)
        return base[idx, None] + amp[idx, None] * np.exp(
            -((x - mu[idx, None]) ** 2) / (2 * sigma[idx, None] ** 2)
        ) + rng.normal(0.0, cfg.noise_sigma, size=(k, feature_count))

    y = _regenerate_zero_norm(y, redraw)
    nuisance = {"sigma": sigma.tolist(), "amplitude": amp.tolist(), "baseline": base.tolist()}
    return y, mu, nuisance


def sin_freq_max(feature_count: int) -> float:
    """Frozen profile rule: f_max = N/4 cycles per unit interval — half the
    Nyquist limit N/2, i.e. >= 4 samples per cycle at the fastest
    admissible frequency."""
    return feature_count / 4.0


def generate_sin_freq(
    rng: np.random.Generator, n_samples: int, feature_count: int, cfg: SinFreqConfig
) -> tuple[np.ndarray, np.ndarray, dict]:
    x = _grid(feature_count)[None, :]
    f_max = sin_freq_max(feature_count)
    if f_max <= cfg.f_min:
        raise SignalGenerationError(
            f"feature_count={feature_count} leaves no admissible frequency range "
            f"(f_min={cfg.f_min}, f_max={f_max})"
        )
    freq = rng.uniform(cfg.f_min, f_max, size=n_samples)
    amp = rng.uniform(*cfg.amplitude_range, size=n_samples)
    phase = rng.uniform(0.0, 2 * np.pi, size=n_samples)
    offset = rng.uniform(*cfg.offset_range, size=n_samples)
    noise = rng.normal(0.0, cfg.noise_sigma, size=(n_samples, feature_count))
    y = amp[:, None] * np.sin(2 * np.pi * freq[:, None] * x + phase[:, None])
    y = y + offset[:, None] + noise

    def redraw(idx: np.ndarray) -> np.ndarray:
        k = len(idx)
        freq[idx] = rng.uniform(cfg.f_min, f_max, size=k)
        amp[idx] = rng.uniform(*cfg.amplitude_range, size=k)
        phase[idx] = rng.uniform(0.0, 2 * np.pi, size=k)
        offset[idx] = rng.uniform(*cfg.offset_range, size=k)
        return amp[idx, None] * np.sin(
            2 * np.pi * freq[idx, None] * x + phase[idx, None]
        ) + offset[idx, None] + rng.normal(0.0, cfg.noise_sigma, size=(k, feature_count))

    y = _regenerate_zero_norm(y, redraw)
    targets = (freq - cfg.f_min) / (f_max - cfg.f_min)
    nuisance = {
        "frequency_cycles": freq.tolist(), "amplitude": amp.tolist(),
        "phase": phase.tolist(), "offset": offset.tolist(), "f_max": f_max,
    }
    return y, targets, nuisance


def generate_change_point(
    rng: np.random.Generator, n_samples: int, feature_count: int, cfg: ChangePointConfig
) -> tuple[np.ndarray, np.ndarray, dict]:
    x = _grid(feature_count)[None, :]
    kstar = rng.uniform(*cfg.changepoint_range, size=n_samples)
    pre = rng.uniform(*cfg.pre_level_range, size=n_samples)
    step = rng.uniform(*cfg.step_magnitude_range, size=n_samples)
    sign = rng.choice([-1.0, 1.0], size=n_samples)
    post = pre + sign * step
    slope_pre = rng.uniform(*cfg.slope_range, size=n_samples)
    slope_post = rng.uniform(*cfg.slope_range, size=n_samples)
    noise = rng.normal(0.0, cfg.noise_sigma, size=(n_samples, feature_count))

    def compose(idx=None):
        if idx is None:
            k_, p_, po_, s1, s2 = kstar, pre, post, slope_pre, slope_post
            nz = noise
        else:
            k_, p_, po_, s1, s2 = kstar[idx], pre[idx], post[idx], slope_pre[idx], slope_post[idx]
            nz = rng.normal(0.0, cfg.noise_sigma, size=(len(idx), feature_count))
        before = p_[:, None] + s1[:, None] * x
        after = po_[:, None] + s2[:, None] * (x - k_[:, None])
        return np.where(x < k_[:, None], before, after) + nz

    y = compose()

    def redraw(idx: np.ndarray) -> np.ndarray:
        k = len(idx)
        kstar[idx] = rng.uniform(*cfg.changepoint_range, size=k)
        pre[idx] = rng.uniform(*cfg.pre_level_range, size=k)
        step_k = rng.uniform(*cfg.step_magnitude_range, size=k)
        sign_k = rng.choice([-1.0, 1.0], size=k)
        post[idx] = pre[idx] + sign_k * step_k
        slope_pre[idx] = rng.uniform(*cfg.slope_range, size=k)
        slope_post[idx] = rng.uniform(*cfg.slope_range, size=k)
        return compose(idx)

    y = _regenerate_zero_norm(y, redraw)
    nuisance = {
        "pre_level": pre.tolist(), "post_level": post.tolist(),
        "slope_pre": slope_pre.tolist(), "slope_post": slope_post.tolist(),
    }
    return y, kstar, nuisance


def generate_peak_count(
    rng: np.random.Generator, n_samples: int, feature_count: int, cfg: PeakCountConfig
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Balanced binary task: exactly floor(n/2) double-peak (label 1)
    samples, rest single-peak (label 0), in rng-shuffled order."""
    if n_samples % 2 != 0:
        raise SignalGenerationError(
            f"peak_count requires an even n_samples for exact class balance, got {n_samples}"
        )
    x = _grid(feature_count)[None, :]
    min_sep = cfg.min_separation_grid_points / (feature_count - 1)
    labels = np.zeros(n_samples)
    labels[: n_samples // 2] = 1.0
    rng.shuffle(labels)

    def one_peak_batch(k: int) -> np.ndarray:
        loc = rng.uniform(*cfg.location_range, size=k)
        sig = rng.uniform(*cfg.sigma_range, size=k)
        amp = rng.uniform(*cfg.amplitude_range, size=k)
        return amp[:, None] * np.exp(-((x - loc[:, None]) ** 2) / (2 * sig[:, None] ** 2))

    def two_peak_batch(k: int) -> np.ndarray:
        lo, hi = cfg.location_range
        first = rng.uniform(lo, hi - min_sep, size=k)
        # Second center drawn beyond the separation margin, still in range.
        second = np.array([rng.uniform(f + min_sep, hi) for f in first])
        out = np.zeros((k, x.shape[1]))
        for centers in (first, second):
            sig = rng.uniform(*cfg.sigma_range, size=k)
            amp = rng.uniform(*cfg.amplitude_range, size=k)
            out += amp[:, None] * np.exp(-((x - centers[:, None]) ** 2) / (2 * sig[:, None] ** 2))
        return out

    y = np.zeros((n_samples, feature_count))
    base = rng.uniform(*cfg.baseline_range, size=n_samples)
    single_idx = np.where(labels == 0.0)[0]
    double_idx = np.where(labels == 1.0)[0]
    y[single_idx] = one_peak_batch(len(single_idx))
    y[double_idx] = two_peak_batch(len(double_idx))
    y += base[:, None] + rng.normal(0.0, cfg.noise_sigma, size=(n_samples, feature_count))

    def redraw(idx: np.ndarray) -> np.ndarray:
        rows = np.zeros((len(idx), feature_count))
        for j, i in enumerate(idx):
            batch = one_peak_batch(1) if labels[i] == 0.0 else two_peak_batch(1)
            rows[j] = batch[0] + rng.uniform(*cfg.baseline_range) + rng.normal(
                0.0, cfg.noise_sigma, size=feature_count
            )
        return rows

    y = _regenerate_zero_norm(y, redraw)
    nuisance = {"class_balance": {"class0": int(len(single_idx)), "class1": int(len(double_idx))},
                "min_separation": min_sep}
    return y, labels, nuisance
