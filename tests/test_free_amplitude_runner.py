"""Search-isolation tests for the free-amplitude 3-arm runner (Codex
instruction section 21 "Search isolation" checklist): open-loop prompts
have no history, closed-loop receives only its own history, protected-test
data never enters prompts, no network calls in tests, no raw
responses/secrets in outputs. Every provider here is a fake, in-process
stand-in.
"""

from __future__ import annotations

import ast
import json
import pathlib
import time

import pytest

from llm_vqc.evaluation.store import ResultStore
from llm_vqc.free_amplitude import reporting
from llm_vqc.free_amplitude.provenance import collect_provenance
from llm_vqc.free_amplitude.runner import (
    arm_task_name,
    evaluate_protected_test,
    run_llm_closed_loop_arm,
    run_llm_open_loop_arm,
    run_random_arm,
)
from llm_vqc.free_amplitude.sampler import search_space_size_report
from llm_vqc.free_amplitude.tasks import AMPLITUDE_N3_SMOKE_V1, AmplitudeGaussianPeakTask
from llm_vqc.free_amplitude.training import FreeAmplitudeTrainingConfig
from llm_vqc.llm.provider import LLMResponse

TASK_NAME = "amp_test"

_ARCH_A = '{"n_qubits": 3, "operations": [{"gate": "RY", "wires": [0]}]}'
_ARCH_B = (
    '{"n_qubits": 3, "operations": '
    '[{"gate": "RZ", "wires": [1]}, {"gate": "CRX", "wires": [1, 0]}]}'
)


class _ScriptedProvider:
    model_name = "scripted-test"

    def __init__(self, responses):
        self._responses = responses
        self.calls_made = 0
        self.prompts_seen = []

    def complete(self, system_prompt, user_prompt, temperature):
        self.prompts_seen.append(user_prompt)
        text = self._responses[self.calls_made]
        self.calls_made += 1
        return LLMResponse(
            raw_text=text, model=self.model_name, input_tokens=10, output_tokens=10,
            estimated_cost_usd=0.0, latency_seconds=0.0,
        )


@pytest.fixture(scope="module")
def train_val():
    task = AmplitudeGaussianPeakTask(AMPLITUDE_N3_SMOKE_V1)
    tv, _ = task.build(seed=0)
    return tv


@pytest.fixture(scope="module")
def test_split():
    task = AmplitudeGaussianPeakTask(AMPLITUDE_N3_SMOKE_V1)
    test, _ = task.build_test(seed=0)
    return test


def _store(tmp_path, name="r.sqlite"):
    return ResultStore.open_or_create(
        tmp_path / name, run_id="run", config_json="{}",
        config_reproducibility_fields={"t": 1}, git_sha=None, created_at="t0",
        allow_incompatible=True,
    )


def _cfg(epochs=2):
    return FreeAmplitudeTrainingConfig(epochs=epochs)


# --- open-loop has no history ------------------------------------------------


def test_open_loop_prompts_are_identical_across_calls(tmp_path, train_val):
    store = _store(tmp_path)
    provider = _ScriptedProvider([_ARCH_A, _ARCH_B])
    run_llm_open_loop_arm(
        0, TASK_NAME, 3, 0, train_val, _cfg(), 2, store, provider, time.monotonic() + 60,
    )
    store.close()
    assert len(provider.prompts_seen) == 2
    assert provider.prompts_seen[0] == provider.prompts_seen[1]


# --- closed-loop receives only its own history ------------------------------


def test_closed_loop_second_prompt_reflects_first_proposal_only(tmp_path, train_val):
    store = _store(tmp_path)
    provider = _ScriptedProvider([_ARCH_A, _ARCH_B])
    run_llm_closed_loop_arm(
        0, TASK_NAME, 3, 0, train_val, _cfg(), 2, store, provider, time.monotonic() + 60,
    )
    store.close()
    first_prompt, second_prompt = provider.prompts_seen
    assert "{" not in first_prompt  # first turn: no feedback block at all
    feedback = json.loads(second_prompt[second_prompt.index("{"): second_prompt.rindex("}") + 1])
    assert feedback["prior_proposal"]["operations"] == [{"gate": "RY", "wires": [0]}]
    assert feedback["validation_rmse"] is not None


def test_closed_loop_history_is_arm_local_not_cross_arm(tmp_path, train_val):
    """Running Random and Open-loop first must not change what Closed-loop's
    own first-proposal feedback looks like -- there is no shared state to
    leak across arms in the first place (each has its own namespaced cache
    and its own local `prior_result`)."""
    store = _store(tmp_path)
    run_random_arm(0, TASK_NAME, 3, 0, train_val, _cfg(), 2, store, time.monotonic() + 60)
    open_provider = _ScriptedProvider([_ARCH_A, _ARCH_A])
    run_llm_open_loop_arm(
        0, TASK_NAME, 3, 0, train_val, _cfg(), 2, store, open_provider, time.monotonic() + 60,
    )
    closed_provider = _ScriptedProvider([_ARCH_A, _ARCH_B])
    run_llm_closed_loop_arm(
        0, TASK_NAME, 3, 0, train_val, _cfg(), 2, store, closed_provider, time.monotonic() + 60,
    )
    store.close()
    # Closed-loop's prompts must be identical to a fresh run with no other
    # arm run first (verified separately below) -- here we just confirm no
    # exception / no reference to another arm's data appears.
    for prompt in closed_provider.prompts_seen:
        assert "random" not in prompt.lower()
        assert "open_loop" not in prompt.lower()


def test_closed_loop_prompts_identical_regardless_of_arm_execution_order(tmp_path, train_val):
    store1 = _store(tmp_path, "order1.sqlite")
    run_random_arm(0, TASK_NAME, 3, 0, train_val, _cfg(), 2, store1, time.monotonic() + 60)
    closed_1 = _ScriptedProvider([_ARCH_A, _ARCH_B])
    run_llm_closed_loop_arm(
        0, TASK_NAME, 3, 0, train_val, _cfg(), 2, store1, closed_1, time.monotonic() + 60,
    )
    store1.close()

    store2 = _store(tmp_path, "order2.sqlite")
    closed_2 = _ScriptedProvider([_ARCH_A, _ARCH_B])
    run_llm_closed_loop_arm(
        0, TASK_NAME, 3, 0, train_val, _cfg(), 2, store2, closed_2, time.monotonic() + 60,
    )
    store2.close()

    assert closed_1.prompts_seen == closed_2.prompts_seen


# --- arm-local duplicate detection -------------------------------------------


def test_cross_arm_same_architecture_is_not_a_duplicate(tmp_path, train_val):
    store = _store(tmp_path)
    open_provider = _ScriptedProvider([_ARCH_A, _ARCH_A])
    open_outcome = run_llm_open_loop_arm(
        0, TASK_NAME, 3, 0, train_val, _cfg(), 2, store, open_provider, time.monotonic() + 60,
    )
    closed_provider = _ScriptedProvider([_ARCH_A, _ARCH_B])
    closed_outcome = run_llm_closed_loop_arm(
        0, TASK_NAME, 3, 0, train_val, _cfg(), 2, store, closed_provider, time.monotonic() + 60,
    )
    store.close()

    assert open_outcome.candidates[0].is_duplicate is False
    assert open_outcome.candidates[1].is_duplicate is True  # within-arm duplicate

    # Closed-loop's first proposal (also architecture A) must NOT be
    # flagged duplicate -- it's a fresh candidate in its own namespace.
    assert closed_outcome.candidates[0].is_duplicate is False
    assert closed_outcome.candidates[0].valid is True


def test_arm_task_namespace_differs_per_arm():
    assert arm_task_name("t", "random") != arm_task_name("t", "llm_open_loop")
    assert arm_task_name("t", "llm_open_loop") != arm_task_name("t", "llm_closed_loop")


# --- protected-test data never enters prompts / search ----------------------


def test_protected_test_rmse_not_computed_during_search(tmp_path, train_val):
    store = _store(tmp_path)
    outcome = run_random_arm(0, TASK_NAME, 3, 0, train_val, _cfg(), 2, store, time.monotonic() + 60)
    store.close()
    assert outcome.protected_test_rmse is None


def test_protected_test_only_reachable_via_dedicated_function(tmp_path, train_val, test_split):
    store = _store(tmp_path)
    outcome = run_random_arm(0, TASK_NAME, 3, 0, train_val, _cfg(), 2, store, time.monotonic() + 60)
    evaluate_protected_test(outcome, store, TASK_NAME, 0, test_split, run_seed=0)
    store.close()
    assert outcome.protected_test_rmse is not None and outcome.protected_test_rmse > 0


def test_closed_loop_feedback_prompt_never_mentions_test():
    from llm_vqc.free_amplitude.prompts import build_closed_loop_feedback_user_prompt

    prompt = build_closed_loop_feedback_user_prompt(
        prior_operations=[{"gate": "RY", "wires": [0]}], valid=True, is_duplicate=False,
        val_rmse=0.1, searched_body_gate_count=1, quantum_parameter_count=1,
        controlled_gate_count=0, compiled_depth=2, initial_gradient_norm=0.1,
        final_gradient_norm=0.05, zero_gradient_parameter_count=0,
        parameter_count_in_causal_cone=1, best_so_far_val_rmse=0.1,
        best_so_far_operations=[{"gate": "RY", "wires": [0]}], remaining_budget=1,
    )
    assert "protected" not in prompt.lower()
    assert "test_rmse" not in prompt.lower()


# --- no network calls in tests -----------------------------------------------


def test_free_amplitude_package_contains_no_network_or_sdk_imports():
    forbidden = {"requests", "httpx", "urllib", "urllib2", "socket", "openai", "anthropic"}
    pkg = pathlib.Path("llm_vqc/free_amplitude")
    for path in pkg.glob("*.py"):
        tree = ast.parse(path.read_text())
        targets = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                targets.append((node.module or "").split(".")[0])
            elif isinstance(node, ast.Import):
                targets.extend(a.name.split(".")[0] for a in node.names)
        assert not (set(targets) & forbidden), (path.name, targets)


# --- no raw responses/secrets in written outputs -----------------------------


def test_written_outputs_contain_no_raw_responses_or_secrets(tmp_path, train_val, test_split):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    store = _store(tmp_path, "secrets.sqlite")

    fake_secret = "sk-FAKE1234567890ABCDEFSECRET"
    raw_marker = "RAW_RESPONSE_MARKER_SHOULD_NEVER_APPEAR"
    provider = _ScriptedProvider([f"not json {raw_marker} {fake_secret}"])
    outcome = run_llm_open_loop_arm(
        0, TASK_NAME, 3, 0, train_val, _cfg(), 1, store, provider, time.monotonic() + 60,
    )
    evaluate_protected_test(outcome, store, TASK_NAME, 0, test_split, run_seed=0)

    class _Args:
        dataset_profile = "amplitude_n3_smoke_v1"
        seed = 0
        budget = 1
        epochs = 2

    reporting.write_all_outputs(
        output_dir=output_dir, store=store, args=_Args(),
        provenance=collect_provenance(pathlib.Path(".")),
        dataset_diagnostics={}, search_space_report=search_space_size_report(3),
        arm_outcomes=[outcome], baseline_results=[], task_name=TASK_NAME,
        n_qubits=3, feature_count=8, readout_qubit=0, max_gates=5, elapsed_seconds=1.0,
    )
    store.close()

    for path in output_dir.rglob("*"):
        if not path.is_file() or path.suffix == ".png":
            continue
        content = path.read_text(errors="ignore")
        assert fake_secret not in content, path
        assert raw_marker not in content, path
        assert "Authorization" not in content, path
        assert "Bearer " not in content, path
