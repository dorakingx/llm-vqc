"""bench_v2 parameter-initialization policy.

Identical distribution to the preserved `free_amplitude_init_v1`
(`theta_j ~ Uniform[-pi, pi]`, local generator, float64, CPU, zero
classical parameters) — the ONLY difference is the parameter-count
postcondition: the scalable layered space allows more than the compact
profile's 5 angles, bounded instead by the space profile's
`max_quantum_parameters`. Version-tagged so the training config provenance
distinguishes the two policies.
"""

from __future__ import annotations

import math

import torch

from llm_vqc.free_amplitude.init_policy import FreeAmplitudeInitError
from llm_vqc.free_amplitude.model import FixedReadoutQuantumModel

BENCH_V2_INIT_VERSION = "bench_v2_init_v1"


def make_bench_v2_init_policy(max_quantum_parameters: int):
    """Build an init policy bounded by the space profile's parameter cap."""

    def policy(model: FixedReadoutQuantumModel, param_init_seed: int) -> None:
        model.double()
        model.cpu()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(param_init_seed) & 0x7FFFFFFF)
        with torch.no_grad():
            model.q_layer.weights.uniform_(-math.pi, math.pi, generator=generator)

        n_quantum = int(model.q_layer.weights.numel())
        if not (0 <= n_quantum <= max_quantum_parameters):
            raise FreeAmplitudeInitError(
                f"expected 0-{max_quantum_parameters} quantum parameters, got {n_quantum}"
            )
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if not name.startswith("q_layer."):
                raise FreeAmplitudeInitError(
                    f"unexpected classical trainable tensor {name!r}"
                )
            if param.dtype != torch.float64:
                raise FreeAmplitudeInitError(f"{name!r} is {param.dtype}, expected float64")
        if model.classical_parameter_count != 0:
            raise FreeAmplitudeInitError(
                f"classical_parameter_count must be 0, got {model.classical_parameter_count}"
            )

    return policy
