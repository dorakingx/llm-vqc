"""Tests for the Random arm's canonical-architecture sampler
(`llm_vqc.mini_demo.sampler`).

Correction pass (section 10): the Random arm samples uniformly over the 36
*canonical* physical architectures, not the 54 raw field combinations.
"""

from __future__ import annotations

import numpy as np

from llm_vqc.mini_demo.compact_schema import (
    DIRECTION_CHOICES,
    ENTANGLER_CHOICES,
    ROTATION_GATE_CHOICES,
    canonicalize_compact,
)
from llm_vqc.mini_demo.sampler import (
    CANONICAL_ARCHITECTURES,
    enumerate_canonical_architectures,
    sample_compact_architecture,
)


def test_exactly_36_canonical_architectures():
    architectures = enumerate_canonical_architectures()
    assert len(architectures) == 36
    # 3 layer-1 gates x 4 canonical entanglers x 3 layer-2 gates.
    assert len(ROTATION_GATE_CHOICES) ** 2 * 4 == 36


def test_enumeration_has_no_physical_duplicates():
    architectures = enumerate_canonical_architectures()
    keys = {
        (a.layer_1_gate, a.entangler, a.entangler_direction, a.layer_2_gate)
        for a in architectures
    }
    assert len(keys) == 36


def test_every_enumerated_architecture_is_already_canonical():
    for a in enumerate_canonical_architectures():
        assert canonicalize_compact(a) == a


def test_cz_and_none_only_appear_with_canonical_direction():
    for a in enumerate_canonical_architectures():
        if a.entangler in ("CZ", "NONE"):
            assert a.entangler_direction == "0_to_1"


def test_both_cnot_directions_are_present():
    directions = {
        a.entangler_direction for a in enumerate_canonical_architectures() if a.entangler == "CNOT"
    }
    assert directions == {"0_to_1", "1_to_0"}


def test_deterministic_given_the_same_seed():
    a = sample_compact_architecture(np.random.default_rng(42))
    b = sample_compact_architecture(np.random.default_rng(42))
    assert a == b


def test_sampler_only_ever_returns_canonical_architectures():
    rng = np.random.default_rng(0)
    canonical_set = set(id(a) for a in CANONICAL_ARCHITECTURES)
    for _ in range(200):
        drawn = sample_compact_architecture(rng)
        assert canonicalize_compact(drawn) == drawn
        assert id(drawn) in canonical_set


def test_sampler_eventually_covers_all_36_canonical_architectures():
    rng = np.random.default_rng(7)
    seen = {
        (p.layer_1_gate, p.entangler, p.entangler_direction, p.layer_2_gate)
        for p in (sample_compact_architecture(rng) for _ in range(4000))
    }
    assert len(seen) == 36


def test_choice_constants_still_describe_the_underlying_grammar():
    # The sampler draws canonical architectures, but the *grammar* (the enum
    # of allowed field values sent to the LLM) is unchanged.
    assert set(ROTATION_GATE_CHOICES) == {"RX", "RY", "RZ"}
    assert set(ENTANGLER_CHOICES) == {"CNOT", "CZ", "NONE"}
    assert set(DIRECTION_CHOICES) == {"0_to_1", "1_to_0"}
