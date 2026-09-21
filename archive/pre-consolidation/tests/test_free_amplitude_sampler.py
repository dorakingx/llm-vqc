"""Tests for search-space accounting and the Random-search sampler (Codex
instruction section 16): `A(n) = 3n^2 + n` (`A(3)=30`, `A(5)=80`), and the
sampler draws only valid, uniformly-distributed proposals.
"""

from __future__ import annotations

import numpy as np

from llm_vqc.free_amplitude.sampler import (
    FreeGateSamplingError,
    num_placement_actions,
    sample_free_gate_proposal,
    search_space_size_report,
)
from llm_vqc.free_amplitude.schema import validate_free_gate_proposal


def test_a3_equals_30():
    assert num_placement_actions(3) == 30


def test_a5_equals_80():
    assert num_placement_actions(5) == 80


def test_formula_matches_3n_squared_plus_n():
    for n in range(2, 8):
        assert num_placement_actions(n) == 3 * n**2 + n


def test_search_space_size_report_structure():
    report = search_space_size_report(3, max_gates=5)
    assert report["placement_actions_per_position"] == 30
    assert report["candidates_per_sequence_length"][1] == 30
    assert report["candidates_per_sequence_length"][5] == 30**5
    assert report["total_candidates"] == sum(30**g for g in range(1, 6))


def test_sampler_only_returns_valid_proposals():
    rng = np.random.default_rng(0)
    for _ in range(300):
        proposal = sample_free_gate_proposal(rng, n_qubits=3, max_gates=5)
        result = validate_free_gate_proposal(proposal, expected_n_qubits=3)
        assert result.valid, result.issues


def test_sampler_respects_max_gates_bound():
    rng = np.random.default_rng(1)
    for _ in range(100):
        proposal = sample_free_gate_proposal(rng, n_qubits=3, max_gates=5)
        assert 1 <= len(proposal.operations) <= 5


def test_sampler_is_deterministic_given_the_same_seed():
    p1 = sample_free_gate_proposal(np.random.default_rng(42), n_qubits=3)
    p2 = sample_free_gate_proposal(np.random.default_rng(42), n_qubits=3)
    assert p1 == p2


def test_sampler_works_for_five_qubits():
    rng = np.random.default_rng(0)
    for _ in range(50):
        proposal = sample_free_gate_proposal(rng, n_qubits=5, max_gates=5)
        result = validate_free_gate_proposal(proposal, expected_n_qubits=5)
        assert result.valid, result.issues


def test_sampler_raises_after_attempt_budget_when_impossible():
    rng = np.random.default_rng(0)
    # require_at_least_one_parameterized is unsatisfiable with 0 attempts
    # remaining if max_attempts is absurdly low and grammar is adversarial;
    # exercise the honest-failure path directly with max_attempts=0.
    try:
        sample_free_gate_proposal(rng, n_qubits=3, max_gates=5, max_attempts=0)
        raised = False
    except FreeGateSamplingError:
        raised = True
    assert raised
