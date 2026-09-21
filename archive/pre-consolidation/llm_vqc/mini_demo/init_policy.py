"""Explicit, versioned parameter-initialization policy for the mini demo.

The correction pass makes initialization an *explicitly declared, recorded*
part of the experiment rather than whatever PyTorch/PennyLane happen to do
by default (the earlier real run relied on dependency defaults, and its
quantum-layer weights were silently `float32` -- see below). Every
candidate in every arm is trained through this one shared policy, passed
to `evaluate_candidate(..., init_policy=...)`.

Policy `mini_demo_init_v1`:
  - embedding weights: Xavier (Glorot) uniform;
  - embedding biases:  zeros;
  - output-head weights: Xavier uniform;
  - output-head bias:  zero;
  - quantum angles:    Uniform[-pi, pi];
  - deterministic: all draws come from one `torch.Generator` seeded with
    the `param_init_seed` that `train_model` derives from the per-candidate
    `train_seed` (`TrainingSeeds.from_train_seed`), so the same circuit
    always initializes identically -- in every arm;
  - CPU only, and every trainable tensor forced to `torch.float64`
    (PennyLane's `TorchLayer` initializes its weights as `float32` by
    default, so `model.double()` is required to guarantee uniform
    float64 -- this is exactly the defect this policy closes).

The policy hard-asserts its own postconditions (fail loud): exactly 4
quantum parameters, exactly 51 total trainable parameters, every trainable
tensor `float64` and on CPU. A violation raises `MiniDemoInitError` and
aborts the run rather than training a quietly-wrong model.
"""

from __future__ import annotations

import math

import torch

from llm_vqc.evaluation.model import HybridQNNModel

MINI_DEMO_INIT_VERSION = "mini_demo_init_v1"

EXPECTED_QUANTUM_PARAMETERS = 4
EXPECTED_TOTAL_TRAINABLE_PARAMETERS = 51


class MiniDemoInitError(RuntimeError):
    """Raised when the initialized model violates the mini-demo contract
    (parameter counts, dtype, or device)."""


def mini_demo_init_policy(model: HybridQNNModel, param_init_seed: int) -> None:
    """Apply `mini_demo_init_v1` to `model` in place. Deterministic given
    `param_init_seed` alone (uses a local `torch.Generator`, so it never
    depends on or perturbs global RNG state)."""
    # Guarantee uniform float64 across ALL parameters first (converts the
    # TorchLayer's default-float32 quantum weights up to float64).
    model.double()
    model.cpu()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(param_init_seed) & 0x7FFFFFFF)

    with torch.no_grad():
        torch.nn.init.xavier_uniform_(model.embed.weight, generator=generator)
        torch.nn.init.zeros_(model.embed.bias)
        torch.nn.init.xavier_uniform_(model.head.weight, generator=generator)
        torch.nn.init.zeros_(model.head.bias)
        model.q_layer.weights.uniform_(-math.pi, math.pi, generator=generator)

    _verify_postconditions(model)


def _verify_postconditions(model: HybridQNNModel) -> None:
    quantum_params = int(model.q_layer.weights.numel())
    if quantum_params != EXPECTED_QUANTUM_PARAMETERS:
        raise MiniDemoInitError(
            f"expected {EXPECTED_QUANTUM_PARAMETERS} quantum parameters, got {quantum_params}"
        )

    trainable = [p for p in model.parameters() if p.requires_grad]
    total = sum(p.numel() for p in trainable)
    if total != EXPECTED_TOTAL_TRAINABLE_PARAMETERS:
        raise MiniDemoInitError(
            f"expected {EXPECTED_TOTAL_TRAINABLE_PARAMETERS} total trainable parameters, "
            f"got {total}"
        )

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dtype != torch.float64:
            raise MiniDemoInitError(
                f"trainable tensor {name!r} is {param.dtype}, expected torch.float64"
            )
        if param.device.type != "cpu":
            raise MiniDemoInitError(
                f"trainable tensor {name!r} is on {param.device}, expected CPU"
            )


def verify_model_dtypes_and_device(model: HybridQNNModel) -> dict:
    """Return a recordable verification summary (does not raise) for the
    process-metrics output -- complements the fail-loud checks in
    `mini_demo_init_policy` by producing an auditable result value."""
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    return {
        "all_float64": all(p.dtype == torch.float64 for _, p in trainable),
        "all_cpu": all(p.device.type == "cpu" for _, p in trainable),
        "n_trainable_tensors": len(trainable),
        "total_trainable_parameters": sum(p.numel() for _, p in trainable),
        "quantum_parameters": int(model.q_layer.weights.numel()),
    }
