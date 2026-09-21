"""The fixed-readout model's own training loop (Codex instruction section
11).

Cannot reuse `llm_vqc.evaluation.training.train_model`: that function
hard-constructs a `HybridQNNModel` (classical embed + head) internally.
This experiment's `FixedReadoutQuantumModel` has no classical layer at
all, so its training loop only ever optimizes one tensor:
`model.q_layer.weights`. `assert_optimizer_has_only_quantum_params` makes
that a checkable invariant, not just an intention.

Loss (section 10): `loss = mean_k[(mu_hat_k - mu_k)^2]`, backpropagated via
`diff_method="backprop"` (set inside `FixedReadoutQuantumModel`) with no
classical output head assisting the gradient.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
from pydantic import BaseModel, Field

from llm_vqc.evaluation.metrics import rmse
from llm_vqc.evaluation.seeds import TrainingSeeds
from llm_vqc.free_amplitude.init_policy import free_amplitude_init_policy
from llm_vqc.free_amplitude.model import FixedReadoutQuantumModel
from llm_vqc.ir.schema import CircuitIR
from llm_vqc.tasks.base import TrainValData

FREE_AMPLITUDE_TRAINING_CONFIG_VERSION = "free_amplitude_training_v1"


class FreeAmplitudeTrainingConfig(BaseModel):
    """Every training hyperparameter explicit (Codex instruction section
    11) -- including AdamW's betas/eps, which PyTorch would otherwise
    silently default."""

    config_version: str = FREE_AMPLITUDE_TRAINING_CONFIG_VERSION
    epochs: int = Field(default=20, ge=1)
    batch_size: int = Field(default=16, ge=1)
    learning_rate: float = Field(default=0.05, gt=0)
    weight_decay: float = 1e-5
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    early_stopping: Literal[False] = False
    device: Literal["cpu"] = "cpu"
    dtype: Literal["float64"] = "float64"


class FreeAmplitudeTrainingDivergedError(Exception):
    """Raised internally when training produces a non-finite loss."""


class OptimizerContainsClassicalParameterError(Exception):
    """Raised if the optimizer would ever be given a non-quantum tensor --
    this must never happen; see `assert_optimizer_has_only_quantum_params`."""


@dataclass
class FreeAmplitudeTrainingOutput:
    success: bool
    epochs_completed: int
    train_loss_history: list[float] = field(default_factory=list)
    val_metric_history: list[float] = field(default_factory=list)
    final_val_metric: float | None = None
    initial_angles: list[float] | None = None
    learned_angles: list[float] | None = None
    angle_deltas: list[float] | None = None
    initial_gradient_norm: float | None = None
    final_gradient_norm: float | None = None
    # Full per-parameter gradient vectors (not just the norm), needed for
    # per-parameter zero/near-zero-gradient diagnostics (section 12) --
    # the norm alone cannot tell which specific angle(s) had ~0 gradient.
    initial_gradient_vector: list[float] | None = None
    final_gradient_vector: list[float] | None = None
    error_message: str | None = None
    wall_clock_seconds: float = 0.0


def assert_optimizer_has_only_quantum_params(
    optimizer: torch.optim.Optimizer, model: FixedReadoutQuantumModel
) -> None:
    """Fail loudly if the optimizer was ever given anything other than
    `model.q_layer.weights` -- the only tensor this experiment's model may
    train (Codex instruction section 11: "Add an assertion or test that
    the optimizer contains no classical parameters")."""
    optimized = [p for group in optimizer.param_groups for p in group["params"]]
    if len(optimized) != 1 or optimized[0] is not model.q_layer.weights:
        raise OptimizerContainsClassicalParameterError(
            "optimizer must contain exactly one tensor, model.q_layer.weights"
        )


def train_fixed_readout_model(
    ir: CircuitIR,
    readout_qubit: int,
    train_val: TrainValData,
    config: FreeAmplitudeTrainingConfig,
    train_seed: int,
    init_policy=None,
) -> FreeAmplitudeTrainingOutput:
    """Train a fresh `FixedReadoutQuantumModel` on `ir`, deterministic given
    `(ir, readout_qubit, train_val, config, train_seed)`. Never raises for
    ordinary training failures (divergence) -- reported in the returned
    output, matching `llm_vqc.evaluation.training.train_model`'s contract.
    """
    seeds = TrainingSeeds.from_train_seed(train_seed)
    start = time.perf_counter()

    model = FixedReadoutQuantumModel(ir, readout_qubit=readout_qubit)
    # Backward-compatible extension point (same pattern as
    # llm_vqc.evaluation.training.train_model's init_policy): bench_v2's
    # scalable space needs the identical Uniform[-pi, pi] init without the
    # compact profile's 1-5-parameter postcondition. Default behaviour is
    # byte-identical to before the kwarg existed.
    if init_policy is None:
        free_amplitude_init_policy(model, seeds.param_init)
    else:
        init_policy(model, seeds.param_init)
    initial_angles = model.q_layer.weights.detach().cpu().clone().tolist()

    optimizer = torch.optim.AdamW(
        [model.q_layer.weights],
        lr=config.learning_rate, betas=config.betas, eps=config.eps,
        weight_decay=config.weight_decay,
    )
    assert_optimizer_has_only_quantum_params(optimizer, model)
    loss_fn = torch.nn.MSELoss()

    x_train = torch.tensor(train_val.train.features, dtype=torch.float64)
    y_train = torch.tensor(train_val.train.targets, dtype=torch.float64).reshape(-1, 1)
    x_val = torch.tensor(train_val.val.features, dtype=torch.float64)
    y_val_np = np.asarray(train_val.val.targets, dtype=np.float64)

    minibatch_rng = np.random.default_rng(seeds.minibatch)
    n_train = len(x_train)

    train_loss_history: list[float] = []
    val_metric_history: list[float] = []
    initial_gradient_norm: float | None = None
    final_gradient_norm: float | None = None
    initial_gradient_vector: list[float] | None = None
    final_gradient_vector: list[float] | None = None

    try:
        for epoch in range(config.epochs):
            model.train()
            permutation = minibatch_rng.permutation(n_train)
            epoch_losses = []
            for batch_start in range(0, n_train, config.batch_size):
                batch_indices = permutation[batch_start : batch_start + config.batch_size]
                x_batch = x_train[batch_indices]
                y_batch = y_train[batch_indices]

                optimizer.zero_grad()
                prediction = model(x_batch)
                loss = loss_fn(prediction, y_batch)
                if not torch.isfinite(loss):
                    raise FreeAmplitudeTrainingDivergedError(
                        f"non-finite loss at epoch {epoch}: {float(loss.item())!r}"
                    )
                loss.backward()
                grad_vector = model.q_layer.weights.grad.detach().cpu().clone().tolist()
                grad_norm = float(model.q_layer.weights.grad.detach().norm().item())
                if initial_gradient_norm is None:
                    initial_gradient_norm = grad_norm
                    initial_gradient_vector = grad_vector
                final_gradient_norm = grad_norm
                final_gradient_vector = grad_vector
                optimizer.step()
                epoch_losses.append(float(loss.item()))
            train_loss_history.append(float(np.mean(epoch_losses)))

            model.eval()
            with torch.no_grad():
                val_prediction = model(x_val).cpu().numpy().reshape(-1)
            val_metric_history.append(rmse(val_prediction, y_val_np))

        elapsed = time.perf_counter() - start
        learned_angles = model.q_layer.weights.detach().cpu().clone().tolist()
        angle_deltas = [
            learned - initial
            for learned, initial in zip(learned_angles, initial_angles, strict=True)
        ]
        return FreeAmplitudeTrainingOutput(
            success=True,
            epochs_completed=config.epochs,
            train_loss_history=train_loss_history,
            val_metric_history=val_metric_history,
            final_val_metric=val_metric_history[-1],
            initial_angles=initial_angles,
            learned_angles=learned_angles,
            angle_deltas=angle_deltas,
            initial_gradient_norm=initial_gradient_norm,
            final_gradient_norm=final_gradient_norm,
            initial_gradient_vector=initial_gradient_vector,
            final_gradient_vector=final_gradient_vector,
            error_message=None,
            wall_clock_seconds=elapsed,
        )
    except (FreeAmplitudeTrainingDivergedError, RuntimeError, ValueError) as exc:
        elapsed = time.perf_counter() - start
        return FreeAmplitudeTrainingOutput(
            success=False,
            epochs_completed=len(train_loss_history),
            train_loss_history=train_loss_history,
            val_metric_history=val_metric_history,
            final_val_metric=None,
            initial_angles=initial_angles,
            learned_angles=None,
            angle_deltas=None,
            initial_gradient_norm=initial_gradient_norm,
            final_gradient_norm=final_gradient_norm,
            initial_gradient_vector=initial_gradient_vector,
            final_gradient_vector=final_gradient_vector,
            error_message=f"{type(exc).__name__}: {exc}",
            wall_clock_seconds=elapsed,
        )
