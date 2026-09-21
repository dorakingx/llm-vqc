"""Samplewise-safe regression metrics and prediction statistics.

**Never broadcast `(B,)` against `(B,1)`** (correction pass, blocking
fix). A model output of shape `(B, 1)` subtracted from a target of shape
`(B,)` silently broadcasts to `(B, B)` and averages over `B*B` elements --
a wrong loss that does not error. Every function here flattens both
arguments to 1-D and asserts they have identical shape before any
elementwise operation, so a shape mismatch fails loudly instead of
silently computing garbage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class SamplewiseShapeError(ValueError):
    """Raised when predictions and targets cannot be aligned samplewise."""


def _aligned_1d(predictions, targets) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(predictions, dtype=np.float64).reshape(-1)
    t = np.asarray(targets, dtype=np.float64).reshape(-1)
    if p.shape != t.shape:
        raise SamplewiseShapeError(
            f"predictions {p.shape} and targets {t.shape} are not samplewise-aligned "
            "after flattening; refusing to broadcast"
        )
    return p, t


def samplewise_mse(predictions, targets) -> float:
    p, t = _aligned_1d(predictions, targets)
    return float(np.mean((p - t) ** 2))


def samplewise_rmse(predictions, targets) -> float:
    return float(np.sqrt(samplewise_mse(predictions, targets)))


def samplewise_mae(predictions, targets) -> float:
    p, t = _aligned_1d(predictions, targets)
    return float(np.mean(np.abs(p - t)))


@dataclass
class PredictionStats:
    mean: float
    std: float
    min: float
    max: float
    constant_output: bool


#: A prediction is "constant" if its std across the batch is below this --
#: an engineering threshold (float64 rounding noise floor), not a
#: task-derived number.
CONSTANT_OUTPUT_STD_ATOL = 1e-9


def prediction_stats(predictions) -> PredictionStats:
    p = np.asarray(predictions, dtype=np.float64).reshape(-1)
    std = float(np.std(p))
    return PredictionStats(
        mean=float(np.mean(p)),
        std=std,
        min=float(np.min(p)),
        max=float(np.max(p)),
        constant_output=bool(std < CONSTANT_OUTPUT_STD_ATOL),
    )
