"""Classical sanity baselines for the signal suite (protocol §11).

These contextualize task difficulty on the SAME train/val/test splits the
quantum experiments use. They are difficulty checks, NOT evidence about
quantum models — **quantum-advantage claims are prohibited** (protocol
§11, contract C16), and nothing in this module feeds any search arm.

Per task: a linear model (ridge / logistic regression), a fixed-capacity
MLP (one hidden layer of 32 units, fixed seed, fixed iteration budget),
and an interpretable signal-processing estimator:

- T1 `gauss_peak`   — bounded Gaussian least-squares fit per sample;
- T2 `sin_freq`     — zero-padded periodogram argmax with parabolic
                      interpolation;
- T3 `change_point` — exact two-segment piecewise-mean least squares scan;
- T4 `peak_count`   — smoothed local-maximum count (>= 2 -> class 1).

All learned baselines fit on TRAIN only; the signal-processing estimators
are per-sample and use no training data at all.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor

from llm_vqc.evaluation.metrics import compute_metric
from llm_vqc.tasks.base import DataSplit, TrainValData
from llm_vqc.tasks.signal_suite.generators import sin_freq_max
from llm_vqc.tasks.signal_suite.task import SignalSuiteTask

_MLP_SEED = 20260729
_MLP_HIDDEN = (32,)
_MLP_MAX_ITER = 2000


def _amplitude_view(features: np.ndarray) -> np.ndarray:
    """L2-normalize rows — the exact information the quantum model sees
    after amplitude encoding. Baselines run on this view so the difficulty
    comparison is information-matched, not an artifact of the classical
    models seeing absolute scale that the encoding removes."""
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / norms


def _gauss_peak_estimator(features: np.ndarray) -> np.ndarray:
    x = np.linspace(0.0, 1.0, features.shape[1])

    def gauss(xv, mu, sigma, amp, base):
        return base + amp * np.exp(-((xv - mu) ** 2) / (2 * sigma**2))

    out = np.empty(len(features))
    for i, row in enumerate(features):
        mu0 = float(x[np.argmax(row)])
        try:
            popt, _ = curve_fit(
                gauss, x, row,
                p0=[mu0, 0.1, float(row.max() - row.min()), float(row.min())],
                bounds=([0.0, 1e-3, 0.0, -np.inf], [1.0, 1.0, np.inf, np.inf]),
                maxfev=2000,
            )
            out[i] = popt[0]
        except RuntimeError:  # non-convergence -> argmax fallback
            out[i] = mu0
    return out


def _sin_freq_estimator(features: np.ndarray) -> np.ndarray:
    n = features.shape[1]
    f_min, f_max = 1.0, sin_freq_max(n)
    pad = 8 * n  # zero-padding refines the frequency grid
    centered = features - features.mean(axis=1, keepdims=True)
    spectrum = np.abs(np.fft.rfft(centered, n=pad, axis=1))
    # Grid spacing on [0,1] is 1/(n-1) -> bin k corresponds to
    # k * (n-1) / pad cycles per unit interval.
    freqs = np.arange(spectrum.shape[1]) * (n - 1) / pad
    out = np.empty(len(features))
    for i, mag in enumerate(spectrum):
        k = int(np.argmax(mag[1:]) + 1)
        if 1 <= k < len(mag) - 1:  # parabolic interpolation around the peak
            alpha, beta, gamma = mag[k - 1], mag[k], mag[k + 1]
            denom = alpha - 2 * beta + gamma
            delta = 0.5 * (alpha - gamma) / denom if abs(denom) > 1e-12 else 0.0
            f_hat = (k + float(np.clip(delta, -0.5, 0.5))) * (n - 1) / pad
        else:
            f_hat = freqs[k]
        out[i] = (np.clip(f_hat, f_min, f_max) - f_min) / (f_max - f_min)
    return out


def _change_point_estimator(features: np.ndarray) -> np.ndarray:
    n = features.shape[1]
    x = np.linspace(0.0, 1.0, n)
    out = np.empty(len(features))
    for i, row in enumerate(features):
        best_cost, best_k = np.inf, 1
        for k in range(1, n):  # two-segment piecewise-constant least squares
            left, right = row[:k], row[k:]
            cost = float(((left - left.mean()) ** 2).sum() + ((right - right.mean()) ** 2).sum())
            if cost < best_cost:
                best_cost, best_k = cost, k
        out[i] = x[best_k - 1] + (x[1] - x[0]) / 2.0
    return np.clip(out, 0.0, 1.0)


def _peak_count_estimator(features: np.ndarray) -> np.ndarray:
    kernel = np.array([0.25, 0.5, 0.25])
    out = np.empty(len(features))
    for i, row in enumerate(features):
        smooth = np.convolve(row, kernel, mode="same")
        rng_span = float(smooth.max() - smooth.min())
        prominence = 0.15 * rng_span if rng_span > 0 else 0.0
        interior = smooth[1:-1]
        is_max = (interior > smooth[:-2]) & (interior > smooth[2:])
        is_tall = interior > (smooth.min() + prominence)
        out[i] = 1.0 if int((is_max & is_tall).sum()) >= 2 else 0.0
    return out


_SIGNAL_ESTIMATORS = {
    "gauss_peak": _gauss_peak_estimator,
    "sin_freq": _sin_freq_estimator,
    "change_point": _change_point_estimator,
    "peak_count": _peak_count_estimator,
}


def _score(task: SignalSuiteTask, predictions: np.ndarray, split: DataSplit) -> dict:
    if task.is_classification:
        clipped = np.clip(predictions, 0.0, 1.0)
        return {
            "auc": compute_metric("auc", clipped, split.targets),
            "brier": compute_metric("brier", clipped, split.targets),
            "balanced_accuracy_error": compute_metric(
                "balanced_accuracy_error", clipped, split.targets
            ),
        }
    return {
        "rmse": compute_metric("rmse", predictions, split.targets),
        "mae": float(np.mean(np.abs(predictions - split.targets))),
    }


def run_classical_baselines(
    task: SignalSuiteTask, train_val: TrainValData, test: DataSplit
) -> dict:
    """Fit/evaluate all three baseline families. Returns nested metrics:
    {baseline_name: {val: {...}, test: {...}}}."""
    x_train = _amplitude_view(train_val.train.features)
    x_val = _amplitude_view(train_val.val.features)
    x_test = _amplitude_view(test.features)
    y_train = train_val.train.targets

    results: dict[str, dict] = {}

    if task.is_classification:
        linear = LogisticRegression(C=1.0, max_iter=1000, random_state=_MLP_SEED)
        linear.fit(x_train, y_train.astype(int))
        lin_val = linear.predict_proba(x_val)[:, 1]
        lin_test = linear.predict_proba(x_test)[:, 1]
        mlp = MLPClassifier(
            hidden_layer_sizes=_MLP_HIDDEN, max_iter=_MLP_MAX_ITER, random_state=_MLP_SEED,
        )
        mlp.fit(x_train, y_train.astype(int))
        mlp_val = mlp.predict_proba(x_val)[:, 1]
        mlp_test = mlp.predict_proba(x_test)[:, 1]
        results["logistic_regression"] = {
            "val": _score(task, lin_val, train_val.val), "test": _score(task, lin_test, test),
        }
    else:
        linear = Ridge(alpha=1.0)
        linear.fit(x_train, y_train)
        mlp = MLPRegressor(
            hidden_layer_sizes=_MLP_HIDDEN, max_iter=_MLP_MAX_ITER, random_state=_MLP_SEED,
        )
        mlp.fit(x_train, y_train)
        lin_val, lin_test = linear.predict(x_val), linear.predict(x_test)
        mlp_val, mlp_test = mlp.predict(x_val), mlp.predict(x_test)
        results["ridge_regression"] = {
            "val": _score(task, lin_val, train_val.val), "test": _score(task, lin_test, test),
        }

    results["mlp_32"] = {
        "val": _score(task, mlp_val, train_val.val), "test": _score(task, mlp_test, test),
    }

    estimator = _SIGNAL_ESTIMATORS[task.profile.family]
    results["signal_estimator"] = {
        "val": _score(task, estimator(x_val), train_val.val),
        "test": _score(task, estimator(x_test), test),
    }
    results["_meta"] = {
        "mlp_hidden": list(_MLP_HIDDEN), "mlp_max_iter": _MLP_MAX_ITER, "seed": _MLP_SEED,
        "input_view": "l2_normalized (amplitude-encoding-equivalent information)",
        "purpose": "task-difficulty context only; no quantum-advantage claims",
    }
    return results
