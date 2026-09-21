"""The quarantined final-test evaluation path for free-gate fixed-readout
candidates -- the `llm_vqc.free_amplitude` analog of `llm_vqc.evaluation.
final_test`.

**Deliberately never imported by `llm_vqc.free_amplitude.harness`** -- the
actual enforcement mechanism, not just a comment. Reloads the exact
learned quantum angles a circuit was already trained to; never retrains,
never touches validation data.
"""

from __future__ import annotations

import numpy as np
import torch

from llm_vqc.evaluation.metrics import rmse
from llm_vqc.free_amplitude.model import FixedReadoutQuantumModel
from llm_vqc.ir.canonicalize import structural_hash
from llm_vqc.ir.schema import CircuitIR
from llm_vqc.tasks.base import DataSplit


class FreeAmplitudeFinalTestError(Exception):
    """Raised when a trained model cannot be reconstructed or evaluated."""


class FreeAmplitudeFinalTestResult:
    """Plain, JSON-serializable result of one protected-test evaluation --
    not the shared `llm_vqc.evaluation.results.FinalTestResult` (that
    class carries no MAE field, and this experiment reports MAE too)."""

    def __init__(
        self, task_name: str, structural_hash_value: str, train_seed: int,
        test_rmse: float, test_mae: float, n_test_samples: int,
    ) -> None:
        self.task_name = task_name
        self.structural_hash = structural_hash_value
        self.train_seed = train_seed
        self.test_rmse = test_rmse
        self.test_mae = test_mae
        self.n_test_samples = n_test_samples

    def to_dict(self) -> dict:
        return {
            "task_name": self.task_name,
            "structural_hash": self.structural_hash,
            "train_seed": self.train_seed,
            "test_rmse": self.test_rmse,
            "test_mae": self.test_mae,
            "n_test_samples": self.n_test_samples,
        }


def evaluate_on_test(
    ir: CircuitIR,
    readout_qubit: int,
    learned_angles: list[float],
    test_split: DataSplit,
    task_name: str,
    train_seed: int,
) -> FreeAmplitudeFinalTestResult:
    """Reload `learned_angles` into a freshly-built `FixedReadoutQuantumModel`
    and score it once on `test_split`. Never retrains.
    """
    model = FixedReadoutQuantumModel(ir, readout_qubit=readout_qubit)
    model.double()
    if model.q_layer.weights.numel() != len(learned_angles):
        raise FreeAmplitudeFinalTestError(
            f"learned_angles has {len(learned_angles)} values but this IR needs "
            f"{model.q_layer.weights.numel()} -- the IR passed here must be identical "
            "to the one the weights were trained on"
        )
    with torch.no_grad():
        model.q_layer.weights.copy_(torch.tensor(learned_angles, dtype=torch.float64))
    model.eval()

    with torch.no_grad():
        x_test = torch.tensor(test_split.features, dtype=torch.float64)
        predictions = model(x_test).cpu().numpy().reshape(-1)

    targets = np.asarray(test_split.targets, dtype=np.float64)
    test_rmse = rmse(predictions, targets)
    test_mae = float(np.mean(np.abs(predictions - targets)))

    return FreeAmplitudeFinalTestResult(
        task_name=task_name,
        structural_hash_value=structural_hash(ir),
        train_seed=train_seed,
        test_rmse=test_rmse,
        test_mae=test_mae,
        n_test_samples=len(test_split),
    )
