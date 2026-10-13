"""The tensor-contraction trainer must match the frozen dense trainer."""
import numpy as np
import pytest

from llm_vqc.experiments.qae_baselines.arms import ARMS, hand_templates, pool_for
from llm_vqc.experiments.qae_baselines.fast_eval import evaluate_fast
from llm_vqc.experiments.qae_robustness.conditions import (
    Space,
    evaluate_architecture,
    sample_neutral_arch,
    state_sets,
    train_seed,
    verify_capacity,
)


@pytest.mark.parametrize("n,family", [(4, "TFIM"), (4, "XXZ"), (6, "TFIM")])
def test_fast_trainer_matches_dense(n, family):
    rng = np.random.default_rng(7)
    train, val, test = state_sets(0, family, n)
    for _ in range(3):
        arch = sample_neutral_arch(rng, Space(n))
        seed = train_seed(0, arch)
        a = evaluate_architecture(arch, train, val, test, seed=seed, n=n, epochs=10)
        b = evaluate_fast(arch, train, val, test, seed=seed, n=n, epochs=10)
        assert abs(a.val_fid - b.val_fid) < 1e-10
        assert abs(a.test_fid - b.test_fid) < 1e-10


@pytest.mark.parametrize("n", [4, 6, 8])
def test_pools_respect_capacity(n):
    for arm in ARMS:
        if arm == "LLM-shuffled":
            continue
        for _, arch in pool_for(arm, "x", n, "TFIM", 0):
            verify_capacity(arch, Space(n))
    for arch in hand_templates(n).values():
        verify_capacity(arch, Space(n))


def test_restricted_arms():
    for _, arch in pool_for("RY+NN", "x", 6, "TFIM", 3):
        assert all(op["axis"] == "Y" for op in arch if op["type"] == "R")
        assert all(abs(op["c"] - op["t"]) == 1 for op in arch if op["type"] == "CX")
