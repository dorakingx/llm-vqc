"""Prior-matched non-LLM proposal distributions (see
`docs/research/QAE_PRIOR_BASELINES_PROTOCOL.md`)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from llm_vqc.experiments.qae_robustness.conditions import AXES, Space, verify_capacity

ARMS = ("Uniform", "RY", "NN", "RY+NN", "Hand", "LLM-shuffled")
POOL_SIZE = {"Uniform": 64, "RY": 32, "NN": 32, "RY+NN": 32}
FAMILY_INDEX = {"TFIM": 0, "XXZ": 1}
ROOT = Path(__file__).resolve().parents[3]


def rng_for(arm: str, n: int, family: str, seed: int) -> np.random.Generator:
    return np.random.default_rng([20261011, ARMS.index(arm), n, FAMILY_INDEX[family], seed])


def sample(rng: np.random.Generator, space: Space, *, ry_only: bool, nn_only: bool) -> list[dict]:
    pairs = [p for p in space.directed_pairs if not nn_only or abs(p[0] - p[1]) == 1]
    ops: list[dict] = [
        {"type": "R", "axis": "Y" if ry_only else str(rng.choice(AXES)),
         "q": int(rng.choice(space.wires))}
        for _ in range(space.n_rotations)
    ]
    for _ in range(space.n_cnots):
        c, t = pairs[int(rng.integers(len(pairs)))]
        ops.append({"type": "CX", "c": c, "t": t})
    arch = [ops[i] for i in rng.permutation(space.n_gates)]
    verify_capacity(arch, space)
    return arch


def hand_templates(n: int) -> dict[str, list[dict]]:
    h = n // 2

    def ry_layer():
        return [{"type": "R", "axis": "Y", "q": q} for q in range(n)]

    def build(reverse: bool) -> list[dict]:
        def cx(c, t):
            return {"type": "CX", "c": t, "t": c} if reverse else {"type": "CX", "c": c, "t": t}
        arch = ry_layer()
        arch += [cx(i + 1, i) for i in range(n - 2, -1, -1)]
        arch += ry_layer()
        arch += [cx(h, h - 1)]
        arch += ry_layer()
        verify_capacity(arch, Space(n))
        return arch

    return {"ladder-in": build(False), "ladder-out": build(True)}


def llm_open_pool(condition_key: str) -> dict[str, list[dict]]:
    if condition_key == "reference":
        path = ROOT / "outputs/qae_tfim_neutral_v5/llm_open_pool.json"
    else:
        path = ROOT / f"outputs/qae_robustness/{condition_key}/llm_open_pool.json"
    return json.loads(path.read_text())["candidates"]


def pool_for(arm: str, condition_key: str, n: int, family: str, seed: int) -> list[tuple[str, list[dict]]]:
    space = Space(n)
    if arm == "Hand":
        return list(hand_templates(n).items())
    rng = rng_for(arm, n, family, seed)
    if arm == "LLM-shuffled":
        out = []
        for rep in range(4):
            for name, arch in llm_open_pool(condition_key).items():
                perm = rng.permutation(len(arch))
                out.append((f"{name}~{rep}", [dict(arch[i]) for i in perm]))
        return out
    flags = {"Uniform": (False, False), "RY": (True, False),
             "NN": (False, True), "RY+NN": (True, True)}[arm]
    return [(f"{arm}{k + 1}", sample(rng, space, ry_only=flags[0], nn_only=flags[1]))
            for k in range(POOL_SIZE[arm])]
