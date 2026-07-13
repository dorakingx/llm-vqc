"""Task metric computation, kept separate and independently testable.

Pure functions of (predictions, targets) — no model, no training state,
no task object — so "metric correctness against independently computed
values" (a required Phase 2 test) can check these directly against a
hand-computed reference without standing up a whole training run.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


class MetricError(Exception):
    """Raised when a metric cannot be computed (e.g. a degenerate batch)."""


def rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean((predictions - targets) ** 2)))


def auc(predictions: np.ndarray, targets: np.ndarray) -> float:
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    if len(set(targets.tolist())) < 2:
        raise MetricError("AUC is undefined when targets contain a single class")
    return float(roc_auc_score(targets, predictions))


_METRIC_FUNCTIONS = {"rmse": rmse, "auc": auc}


def compute_metric(metric_name: str, predictions: np.ndarray, targets: np.ndarray) -> float:
    if metric_name not in _METRIC_FUNCTIONS:
        raise MetricError(f"unknown metric: {metric_name!r}")
    return _METRIC_FUNCTIONS[metric_name](predictions, targets)
