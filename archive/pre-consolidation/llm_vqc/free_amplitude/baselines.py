"""Baselines and ablations, clearly separated from the main searched
model (Codex instruction section 19).

- **Main model** (not defined here -- it IS the search: `llm_vqc.
  free_amplitude.runner`'s three arms): amplitude encoding + free
  parameterized body + fixed q0 Z readout + zero classical parameters.
- **Quantum no-body baseline** (`evaluate_quantum_no_body_baseline`):
  amplitude encoding, no searched body at all, fixed q0 readout. Zero
  trainable parameters -- a reference point, not a normal searched
  candidate.
- **Fixed random quantum body** (`evaluate_fixed_random_quantum_body_baseline`):
  one fixed architecture, angles drawn from the same init policy but
  FROZEN (never trained) -- isolates whether training the angles adds
  value over the structure alone.
- **Classical-readout ablation** (`ClassicalReadoutAblationModel` /
  `train_classical_readout_ablation`): amplitude encoding + free quantum
  body + a trainable `Linear(1, 1)` + sigmoid output. Explicitly NOT part
  of the main experiment and never enabled by default -- every result this
  produces must carry the `"classical_readout_ablation"` label and must
  never be mixed with fixed-readout results without it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pennylane as qml
import torch

from llm_vqc.evaluation.metrics import rmse
from llm_vqc.evaluation.seeds import TrainingSeeds
from llm_vqc.free_amplitude.init_policy import free_amplitude_init_policy
from llm_vqc.free_amplitude.model import FixedReadoutQuantumModel
from llm_vqc.free_amplitude.schema import FreeGateProposal, free_gate_proposal_to_circuit_ir
from llm_vqc.free_amplitude.training import FreeAmplitudeTrainingConfig
from llm_vqc.ir.compiler_pennylane import to_qnode
from llm_vqc.ir.expand import build_program
from llm_vqc.ir.schema import CircuitIR, EncodingSpec, MeasurementSpec
from llm_vqc.tasks.base import DataSplit, TrainValData


@dataclass
class BaselineResult:
    name: str
    description: str
    trainable_parameter_count: int
    val_rmse: float
    val_mae: float
    test_rmse: float | None = None
    test_mae: float | None = None


def _mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(predictions) - np.asarray(targets))))


def _score_split(model: torch.nn.Module, split: DataSplit) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        x = torch.tensor(split.features, dtype=torch.float64)
        predictions = model(x).cpu().numpy().reshape(-1)
    targets = np.asarray(split.targets, dtype=np.float64)
    return rmse(predictions, targets), _mae(predictions, targets)


def evaluate_quantum_no_body_baseline(
    n_qubits: int, readout_qubit: int, train_val: TrainValData, test_split: DataSplit | None = None,
) -> BaselineResult:
    """Amplitude encoding straight into the fixed q0 readout, no searched
    gates at all. Zero trainable parameters -- nothing to train; this is a
    single deterministic forward pass, reported as a reference point."""
    ir = CircuitIR(
        n_qubits=n_qubits, encoding=EncodingSpec(type="amplitude", wires="all"), layers=[],
        measurements=MeasurementSpec(observable="Z", wires=[readout_qubit]),
    )
    model = FixedReadoutQuantumModel(ir, readout_qubit=readout_qubit)
    model.double()
    val_rmse, val_mae = _score_split(model, train_val.val)
    test_rmse = test_mae = None
    if test_split is not None:
        test_rmse, test_mae = _score_split(model, test_split)
    return BaselineResult(
        name="quantum_no_body",
        description="Amplitude encoding, no searched body, fixed q0 readout.",
        trainable_parameter_count=0, val_rmse=val_rmse, val_mae=val_mae,
        test_rmse=test_rmse, test_mae=test_mae,
    )


def evaluate_fixed_random_quantum_body_baseline(
    proposal: FreeGateProposal, readout_qubit: int, seed: int,
    train_val: TrainValData, test_split: DataSplit | None = None,
) -> BaselineResult:
    """One fixed architecture, angles initialized (same policy as the main
    experiment) but FROZEN -- never trained. Measures whether training the
    angles adds value over the structure alone."""
    ir = free_gate_proposal_to_circuit_ir(proposal, readout_qubit)
    model = FixedReadoutQuantumModel(ir, readout_qubit=readout_qubit)
    free_amplitude_init_policy(model, seed)
    model.q_layer.weights.requires_grad_(False)
    val_rmse, val_mae = _score_split(model, train_val.val)
    test_rmse = test_mae = None
    if test_split is not None:
        test_rmse, test_mae = _score_split(model, test_split)
    return BaselineResult(
        name="fixed_random_quantum_body",
        description=(
            "Fixed architecture, angles drawn from the same init policy as the main "
            "experiment but frozen (never trained)."
        ),
        trainable_parameter_count=0, val_rmse=val_rmse, val_mae=val_mae,
        test_rmse=test_rmse, test_mae=test_mae,
    )


class ClassicalReadoutAblationModel(torch.nn.Module):
    """**NOT part of the main experiment.** Amplitude encoding + free
    quantum body + a trainable `Linear(1, 1)` + sigmoid output -- adds
    exactly the classical output layer the main experiment's fixed-readout
    model deliberately excludes, to measure how much a learned readout
    combination (rather than the fixed `(1 - z0) / 2`) changes results.
    Every result produced by this class must carry the
    `"classical_readout_ablation"` label and must never be silently
    compared against fixed-readout results as if the two were the same
    condition."""

    def __init__(self, ir: CircuitIR, readout_qubit: int) -> None:
        super().__init__()
        program = build_program(ir)
        if ir.measurements.wires != [readout_qubit] or ir.measurements.observable != "Z":
            raise ValueError(
                "expects a single Z(readout_qubit) measurement, same as the main model"
            )
        qnode = to_qnode(ir, diff_method="backprop")
        self.q_layer = qml.qnn.TorchLayer(qnode, {"weights": (program.num_parameters,)})
        self.output_head = torch.nn.Linear(1, 1, dtype=torch.float64)

    def forward(self, raw_features: torch.Tensor) -> torch.Tensor:
        z0 = self.q_layer(raw_features)
        return torch.sigmoid(self.output_head(z0))


def train_classical_readout_ablation(
    ir: CircuitIR, readout_qubit: int, train_val: TrainValData,
    config: FreeAmplitudeTrainingConfig, train_seed: int,
) -> tuple[ClassicalReadoutAblationModel, float]:
    """Trains BOTH the quantum angles and the classical output head
    together. Returns `(trained_model, final_val_rmse)`. Ablation-only --
    never used by the main experiment's search arms."""
    seeds = TrainingSeeds.from_train_seed(train_seed)
    model = ClassicalReadoutAblationModel(ir, readout_qubit=readout_qubit)
    model.double()
    generator = torch.Generator(device="cpu").manual_seed(int(seeds.param_init) & 0x7FFFFFFF)
    with torch.no_grad():
        model.q_layer.weights.uniform_(-3.141592653589793, 3.141592653589793, generator=generator)
        torch.nn.init.xavier_uniform_(model.output_head.weight, generator=generator)
        torch.nn.init.zeros_(model.output_head.bias)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, betas=config.betas, eps=config.eps,
        weight_decay=config.weight_decay,
    )
    loss_fn = torch.nn.MSELoss()
    x_train = torch.tensor(train_val.train.features, dtype=torch.float64)
    y_train = torch.tensor(train_val.train.targets, dtype=torch.float64).reshape(-1, 1)
    minibatch_rng = np.random.default_rng(seeds.minibatch)
    n_train = len(x_train)

    for _ in range(config.epochs):
        model.train()
        permutation = minibatch_rng.permutation(n_train)
        for batch_start in range(0, n_train, config.batch_size):
            batch_indices = permutation[batch_start : batch_start + config.batch_size]
            optimizer.zero_grad()
            prediction = model(x_train[batch_indices])
            loss = loss_fn(prediction, y_train[batch_indices])
            loss.backward()
            optimizer.step()

    val_rmse, _ = _score_split(model, train_val.val)
    return model, val_rmse
