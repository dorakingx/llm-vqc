"""Tests for the mini demo's prompt templates and feedback channel.

Correction pass (section 4): closed-loop feedback is a single deterministic
JSON-like block that ALWAYS carries all six fields (architecture, validation
RMSE-or-null, validity, duplicate status, depth-or-null, two-qubit-gate
count-or-null) -- never omitted for duplicate/invalid proposals, and never
any protected-test field.
"""

from __future__ import annotations

import inspect
import json

from llm_vqc.mini_demo.prompts import (
    build_closed_loop_feedback_user_prompt,
    build_closed_loop_first_user_prompt,
    build_open_loop_user_prompt,
    build_system_prompt,
)
from llm_vqc.search.feedback import SearchFeedback


def _parse_feedback_json(prompt: str) -> dict:
    """Extract the single JSON object embedded in the feedback prompt."""
    start = prompt.index("{")
    end = prompt.rindex("}") + 1
    return json.loads(prompt[start:end])


def _base_kwargs(**overrides):
    kwargs = dict(
        prior_layer_1_gate="RY", prior_entangler="CNOT", prior_entangler_direction="0_to_1",
        prior_layer_2_gate="RX", valid=True, is_duplicate=False, val_rmse=0.1234,
        circuit_depth=4, two_qubit_gate_count=1,
    )
    kwargs.update(overrides)
    return kwargs


REQUIRED_FEEDBACK_KEYS = {
    "prior_architecture", "valid", "is_duplicate", "validation_rmse",
    "circuit_depth", "two_qubit_gate_count",
}


def test_closed_loop_feedback_prompt_has_no_test_metric_parameter():
    params = set(inspect.signature(build_closed_loop_feedback_user_prompt).parameters)
    assert not any("test" in p.lower() for p in params), params


def test_search_feedback_has_no_test_metric_field():
    fields = set(SearchFeedback.model_fields)
    assert not any("test" in f.lower() for f in fields), fields


def test_valid_nonduplicate_feedback_has_all_fields():
    prompt = build_closed_loop_feedback_user_prompt(**_base_kwargs())
    data = _parse_feedback_json(prompt)
    assert set(data.keys()) == REQUIRED_FEEDBACK_KEYS
    assert data["valid"] is True
    assert data["is_duplicate"] is False
    assert data["validation_rmse"] == 0.1234
    assert data["circuit_depth"] == 4
    assert data["two_qubit_gate_count"] == 1


def test_valid_duplicate_feedback_still_includes_rmse_and_metrics():
    # The correction: a duplicate must NOT drop RMSE/depth/2q -- the cached
    # result carries them, so they are reported.
    prompt = build_closed_loop_feedback_user_prompt(
        **_base_kwargs(is_duplicate=True, val_rmse=0.2222, circuit_depth=4, two_qubit_gate_count=1)
    )
    data = _parse_feedback_json(prompt)
    assert set(data.keys()) == REQUIRED_FEEDBACK_KEYS
    assert data["is_duplicate"] is True
    assert data["validation_rmse"] == 0.2222
    assert data["circuit_depth"] == 4
    assert data["two_qubit_gate_count"] == 1


def test_invalid_feedback_reports_null_metrics_but_all_keys_present():
    prompt = build_closed_loop_feedback_user_prompt(
        **_base_kwargs(valid=False, val_rmse=None, circuit_depth=None, two_qubit_gate_count=None)
    )
    data = _parse_feedback_json(prompt)
    assert set(data.keys()) == REQUIRED_FEEDBACK_KEYS
    assert data["valid"] is False
    assert data["validation_rmse"] is None
    assert data["circuit_depth"] is None
    assert data["two_qubit_gate_count"] is None


def test_failed_training_feedback_reports_null_rmse_with_all_keys():
    # A valid proposal whose training failed: no RMSE, but depth/2q known.
    prompt = build_closed_loop_feedback_user_prompt(
        **_base_kwargs(valid=True, val_rmse=None, circuit_depth=4, two_qubit_gate_count=1)
    )
    data = _parse_feedback_json(prompt)
    assert set(data.keys()) == REQUIRED_FEEDBACK_KEYS
    assert data["validation_rmse"] is None
    assert data["circuit_depth"] == 4


def test_feedback_never_contains_protected_test_fields():
    for kwargs in (
        _base_kwargs(),
        _base_kwargs(is_duplicate=True),
        _base_kwargs(valid=False, val_rmse=None, circuit_depth=None, two_qubit_gate_count=None),
    ):
        prompt = build_closed_loop_feedback_user_prompt(**kwargs)
        lowered = prompt.lower()
        assert "protected" not in lowered
        assert "test_rmse" not in lowered
        assert "test rmse" not in lowered


def test_feedback_is_deterministic():
    a = build_closed_loop_feedback_user_prompt(**_base_kwargs())
    b = build_closed_loop_feedback_user_prompt(**_base_kwargs())
    assert a == b


def test_feedback_asks_for_a_different_or_improved_circuit():
    prompt = build_closed_loop_feedback_user_prompt(**_base_kwargs())
    assert "different or improved" in prompt.lower()


def test_open_loop_prompt_never_varies():
    assert build_open_loop_user_prompt() == build_open_loop_user_prompt()


def test_closed_loop_first_prompt_carries_no_feedback():
    prompt = build_closed_loop_first_user_prompt()
    assert "rmse" not in prompt.lower()
    assert "{" not in prompt  # no feedback JSON block on the first turn


def test_system_prompt_states_task_grammar_and_strict_json_only():
    prompt = build_system_prompt("T1 1D Gaussian-peak regression")
    assert "gaussian" in prompt.lower() or "regression" in prompt.lower()
    assert "lower validation rmse is better" in prompt.lower()
    assert "layer_1_gate" in prompt
    assert "entangler" in prompt
    assert "no markdown" in prompt.lower() or "no code" in prompt.lower()
    assert "2 qubits" in prompt.lower() or "two qubits" in prompt.lower()
