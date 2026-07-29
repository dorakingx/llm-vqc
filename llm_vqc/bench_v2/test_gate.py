"""Protected test gate (protocol §12) — the ONLY bench_v2 module allowed
to touch test data.

> "validation-selected protected-test RMSE": RMSE computed on a held-out
> test partition that is inaccessible to every search arm and inner
> tuning decision, evaluated only after the candidate/architecture has
> been selected using validation data, and never fed back into prompts,
> early stopping, hyperparameter choices, or protocol changes.

Called exactly once per completed replicate, after selection is frozen;
the returned record carries the evaluation timestamp and count so the
goal checker (C09) can verify the discipline from durable manifests.

No search/evaluator/arm module imports this one — enforced by AST tests.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import torch

from llm_vqc.evaluation.metrics import compute_metric
from llm_vqc.free_amplitude.model import FixedReadoutQuantumModel
from llm_vqc.ir.schema import CircuitIR
from llm_vqc.tasks.base import DataSplit


def _forward(model: FixedReadoutQuantumModel, features: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(features, dtype=torch.float64)).cpu().numpy().reshape(-1)


def _regression_metrics(pred: np.ndarray, targets: np.ndarray) -> dict:
    residuals = pred - targets
    rmse = float(np.sqrt(np.mean(residuals**2)))
    target_span = float(targets.max() - targets.min())
    ss_res = float((residuals**2).sum())
    ss_tot = float(((targets - targets.mean()) ** 2).sum())
    # Residual calibration over 4 equal-width target bins (protocol §9).
    bins = np.linspace(targets.min(), targets.max(), 5)
    bin_rmse = []
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        mask = (targets >= lo) & (targets <= hi)
        bin_rmse.append(
            float(np.sqrt(np.mean(residuals[mask] ** 2))) if mask.any() else None
        )
    return {
        "test_rmse": rmse,
        "test_mae": float(np.mean(np.abs(residuals))),
        "test_r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else None,
        "test_nrmse": rmse / target_span if target_span > 0 else None,
        "test_residual_rmse_by_target_bin": bin_rmse,
    }


def _classification_metrics(pred: np.ndarray, targets: np.ndarray) -> dict:
    clipped = np.clip(pred, 0.0, 1.0)
    return {
        "test_auc": compute_metric("auc", clipped, targets),
        "test_balanced_accuracy_error": compute_metric(
            "balanced_accuracy_error", clipped, targets
        ),
        "test_logloss": compute_metric("logloss", clipped, targets),
        "test_brier": compute_metric("brier", clipped, targets),
    }


def evaluate_selected_on_test(
    ir: CircuitIR,
    theta: list[float],
    test_split: DataSplit,
    is_classification: bool,
    readout_qubit: int = 0,
) -> dict:
    """Score one frozen selected candidate exactly once on the protected
    test partition. `theta` are the candidate's final angles: Track A's
    trained `learned_angles` from the shared cache, or Track B's verbatim
    proposed theta. Returns metrics + the gate timestamp."""
    model = FixedReadoutQuantumModel(ir, readout_qubit=readout_qubit)
    model.double()
    weights = model.q_layer.weights
    if weights.numel() != len(theta):
        raise ValueError(f"selected candidate needs {weights.numel()} angles, got {len(theta)}")
    with torch.no_grad():
        weights.copy_(torch.tensor(theta, dtype=torch.float64))
    pred = _forward(model, test_split.features)
    metrics = (
        _classification_metrics(pred, test_split.targets)
        if is_classification
        else _regression_metrics(pred, test_split.targets)
    )
    metrics["n_test_samples"] = len(test_split)
    metrics["test_gate_timestamp"] = datetime.now(UTC).isoformat()
    return metrics
