"""Explicit, versioned parameter-initialization policy (Codex instruction
section 11): `theta_j ~ Uniform[-pi, pi]` for every parameterized gate,
deterministic given `(experiment_seed, canonical_architecture_hash,
training_config_version)`.

Only one tensor exists to initialize -- `model.q_layer.weights` -- since
`FixedReadoutQuantumModel` has no classical layer at all. `model.double()`
is still required: PennyLane's `TorchLayer` initializes its weights as
`float32` by default regardless of the surrounding model's dtype.
"""

from __future__ import annotations

import math

import torch

from llm_vqc.evaluation.seeds import derive_child_seed
from llm_vqc.free_amplitude.model import FixedReadoutQuantumModel

FREE_AMPLITUDE_INIT_VERSION = "free_amplitude_init_v1"

MIN_QUANTUM_PARAMETERS = 1
MAX_QUANTUM_PARAMETERS = 5


class FreeAmplitudeInitError(RuntimeError):
    """Raised when the initialized model violates the fixed-readout
    contract (parameter count, dtype, or classical-parameter presence)."""


def free_amplitude_train_seed(run_seed: int, structural_hash: str, config_version: str) -> int:
    """Deterministic per-candidate training seed, a pure function of
    `(run_seed, structural_hash, config_version)` -- Codex instruction
    section 11's explicit "experiment seed; canonical architecture hash;
    training configuration version" requirement (one label more than the
    shared `llm_vqc.evaluation.seeds.train_seed_for_circuit`, which does
    not carry a config-version component)."""
    return derive_child_seed(run_seed, "free_amplitude_train_seed", structural_hash, config_version)


def free_amplitude_init_policy(model: FixedReadoutQuantumModel, param_init_seed: int) -> None:
    """Apply `free_amplitude_init_v1` to `model` in place. Deterministic
    given `param_init_seed` alone (a local `torch.Generator`; never touches
    or depends on global RNG state)."""
    model.double()
    model.cpu()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(param_init_seed) & 0x7FFFFFFF)

    with torch.no_grad():
        model.q_layer.weights.uniform_(-math.pi, math.pi, generator=generator)

    _verify_postconditions(model)


def _verify_postconditions(model: FixedReadoutQuantumModel) -> None:
    n_quantum = int(model.q_layer.weights.numel())
    if not (MIN_QUANTUM_PARAMETERS <= n_quantum <= MAX_QUANTUM_PARAMETERS):
        raise FreeAmplitudeInitError(
            f"expected {MIN_QUANTUM_PARAMETERS}-{MAX_QUANTUM_PARAMETERS} quantum "
            f"parameters, got {n_quantum}"
        )

    trainable = list(model.named_parameters())
    for name, param in trainable:
        if not param.requires_grad:
            continue
        if not name.startswith("q_layer."):
            raise FreeAmplitudeInitError(
                f"unexpected classical trainable tensor {name!r} -- this model must have "
                "zero classical trainable parameters"
            )
        if param.dtype != torch.float64:
            raise FreeAmplitudeInitError(
                f"tensor {name!r} is {param.dtype}, expected torch.float64"
            )
        if param.device.type != "cpu":
            raise FreeAmplitudeInitError(f"tensor {name!r} is on {param.device}, expected CPU")

    if model.classical_parameter_count != 0:
        raise FreeAmplitudeInitError(
            f"expected classical_parameter_count == 0, got {model.classical_parameter_count}"
        )


def verify_model_dtypes_and_device(model: FixedReadoutQuantumModel) -> dict:
    """Non-raising recordable verification summary for process metrics."""
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    return {
        "all_float64": all(p.dtype == torch.float64 for _, p in trainable),
        "all_cpu": all(p.device.type == "cpu" for _, p in trainable),
        "n_trainable_tensors": len(trainable),
        "quantum_parameter_count": model.quantum_parameter_count,
        "classical_parameter_count": model.classical_parameter_count,
    }
