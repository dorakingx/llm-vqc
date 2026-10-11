"""Tensor-contraction re-implementation of the frozen QAE trainer.

`conditions.evaluate_architecture` builds a dense 2^n x 2^n matrix for every
gate, which dominates run time at n = 8. This module applies each gate as a
local tensor contraction instead. The optimizer, initialisation, training
seed, epochs, learning rate, loss and best-validation checkpoint are the
same; only floating-point summation order differs, so results agree with the
frozen trainer to round-off (checked in `tests/test_qae_baselines.py`).
"""
from __future__ import annotations

import numpy as np
import torch

from llm_vqc.experiments.qae_robustness.conditions import (
    Space,
    verify_capacity,
)
from llm_vqc.experiments.qae_tfim.pilot import EPOCHS, LEARNING_RATE, EvalResult


def _apply(states: torch.Tensor, architecture: list[dict], params: torch.Tensor,
           n: int) -> torch.Tensor:
    batch = states.shape[0]
    psi = states.reshape((batch,) + (2,) * n)
    p = 0
    for op in architecture:
        if op["type"] == "R":
            theta = params[p]
            p += 1
            c = torch.cos(theta / 2).to(torch.complex128)
            s = torch.sin(theta / 2).to(torch.complex128)
            ax = 1 + op["q"]
            a0 = psi.select(ax, 0)
            a1 = psi.select(ax, 1)
            if op["axis"] == "X":
                b0 = c * a0 - 1j * s * a1
                b1 = -1j * s * a0 + c * a1
            elif op["axis"] == "Y":
                b0 = c * a0 - s * a1
                b1 = s * a0 + c * a1
            else:
                b0 = (c - 1j * s) * a0
                b1 = (c + 1j * s) * a1
            psi = torch.stack((b0, b1), dim=ax)
        else:
            cax, tax = 1 + op["c"], 1 + op["t"]
            zero = psi.select(cax, 0)
            one = psi.select(cax, 1)
            # target axis index inside the sliced tensor
            t_in = tax - 1 if tax > cax else tax
            one = one.flip(t_in)
            psi = torch.stack((zero, one), dim=cax)
    return psi.reshape(batch, 2**n)


def evaluate_fast(architecture: list[dict], train_states: np.ndarray,
                  val_states: np.ndarray, test_states: np.ndarray, *, seed: int,
                  n: int, epochs: int = EPOCHS,
                  learning_rate: float = LEARNING_RATE) -> EvalResult:
    space = Space(n)
    verify_capacity(architecture, space)
    torch.manual_seed(seed)
    parameters = torch.nn.Parameter(torch.empty(space.n_rotations, dtype=torch.float64))
    torch.nn.init.uniform_(parameters, -0.05, 0.05)
    optimizer = torch.optim.Adam([parameters], lr=learning_rate)
    train = torch.tensor(train_states, dtype=torch.complex128)
    val = torch.tensor(val_states, dtype=torch.complex128)
    test = torch.tensor(test_states, dtype=torch.complex128)
    trash = space.trash

    def fid(states: torch.Tensor) -> torch.Tensor:
        out = _apply(states, architecture, parameters, n).reshape((-1,) + (2,) * n)
        idx = [slice(None)] + [slice(None)] * n
        for q in trash:
            idx[1 + q] = 0
        kept = out[tuple(idx)]
        return (kept.abs() ** 2).reshape(out.shape[0], -1).sum(dim=1).mean()

    best_val = float("inf")
    best = None
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = 1 - fid(train)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            v = 1 - fid(val)
        if float(v) < best_val:
            best_val = float(v)
            best = parameters.detach().clone()
    with torch.no_grad():
        parameters.copy_(best)
        tr, va, te = fid(train), fid(val), fid(test)
    return EvalResult(train_loss=float(1 - tr), val_loss=float(1 - va),
                      test_loss=float(1 - te), train_fid=float(tr),
                      val_fid=float(va), test_fid=float(te))
