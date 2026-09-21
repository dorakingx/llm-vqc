"""Integration + regression tests for `llm_vqc.mini_demo.runner`.

Every provider here is a fake, in-process stand-in -- no test makes, or
could make, a network call. Covers the correction-pass invariants:
arm-local duplicate detection with no cross-arm contamination, arm-order
independence, complete closed-loop feedback, the API-call hard cap, the
dollar-budget block, honest partial results on interruption, explicit
initialization determinism (float64/CPU), learned-vs-initial angle
export, and artifacts free of secrets/raw responses.
"""

from __future__ import annotations

import json
import time

import pytest
import torch

from llm_vqc.evaluation.seeds import train_seed_for_circuit
from llm_vqc.evaluation.store import ResultStore
from llm_vqc.evaluation.training import TrainingConfig, train_model
from llm_vqc.llm.budget import LLMApiBudget
from llm_vqc.llm.openai_provider import CallBudgetExceededError, CallCountLimitedProvider
from llm_vqc.llm.provider import LLMResponse
from llm_vqc.mini_demo import reporting
from llm_vqc.mini_demo.compact_schema import CompactArchitecture, compact_to_circuit_ir
from llm_vqc.mini_demo.init_policy import mini_demo_init_policy
from llm_vqc.mini_demo.runner import (
    DIFF_METHOD,
    arm_task_name,
    evaluate_protected_test,
    run_llm_closed_loop_arm,
    run_llm_open_loop_arm,
    run_random_arm,
)
from llm_vqc.tasks.t1_gaussian import T1GaussianPeakTask

TASK_NAME = "T1_mini_demo_test"
TASK_DESCRIPTION = "T1 1D Gaussian-peak regression"

_ARCH_A = (
    '{"layer_1_gate": "RY", "entangler": "CNOT", "entangler_direction": "0_to_1", '
    '"layer_2_gate": "RX"}'
)
_ARCH_B = (
    '{"layer_1_gate": "RZ", "entangler": "CZ", "entangler_direction": "1_to_0", '
    '"layer_2_gate": "RY"}'
)


def _training_config(epochs=2) -> TrainingConfig:
    return TrainingConfig(epochs=epochs, batch_size=16)


def _store(tmp_path, name="results.sqlite"):
    return ResultStore.open_or_create(
        tmp_path / name, run_id="run", config_json="{}",
        config_reproducibility_fields={"test": True}, git_sha=None,
        created_at="t0", allow_incompatible=True,
    )


@pytest.fixture(scope="module")
def t1_data():
    task = T1GaussianPeakTask()
    return task.build(seed=0), task.build_test(seed=0)


class _ScriptedProvider:
    """Returns one compact-schema JSON string per call, in order. Records
    every user prompt it was given (for feedback-wiring assertions)."""

    model_name = "fake-scripted-model"

    def __init__(self, responses: list[str]):
        self._responses = responses
        self.calls_made = 0
        self.prompts_seen: list[str] = []

    def complete(self, system_prompt, user_prompt, temperature):
        self.prompts_seen.append(user_prompt)
        text = self._responses[self.calls_made]
        self.calls_made += 1
        return LLMResponse(
            raw_text=text, model=self.model_name, input_tokens=10, output_tokens=10,
            estimated_cost_usd=None, latency_seconds=0.01,
        )


class _RaisingAfterNProvider:
    model_name = "fake-flaky-model"

    def __init__(self, n: int, good_response: str):
        self.n = n
        self.good_response = good_response
        self.calls_made = 0

    def complete(self, system_prompt, user_prompt, temperature):
        self.calls_made += 1
        if self.calls_made > self.n:
            raise ConnectionError("simulated network failure")
        return LLMResponse(
            raw_text=self.good_response, model=self.model_name, input_tokens=5, output_tokens=5,
            estimated_cost_usd=None, latency_seconds=0.01,
        )


def _run_random(train_val, store, seed=0, budget=2):
    return run_random_arm(
        seed, TASK_NAME, train_val, _training_config(), budget, store, time.monotonic() + 60
    )


def _run_open(train_val, store, provider, budget=None, budget_limit=2):
    return run_llm_open_loop_arm(
        0, TASK_NAME, train_val, _training_config(), budget_limit, store, provider,
        time.monotonic() + 60, TASK_DESCRIPTION, budget=budget,
    )


def _run_closed(train_val, store, provider, budget=None, budget_limit=2):
    return run_llm_closed_loop_arm(
        0, TASK_NAME, train_val, _training_config(), budget_limit, store, provider,
        time.monotonic() + 60, TASK_DESCRIPTION, budget=budget,
    )


# --- deterministic random baseline -----------------------------------------


def test_random_arm_completes_exact_budget(tmp_path, t1_data):
    train_val, _ = t1_data
    store = _store(tmp_path)
    outcome = _run_random(train_val, store)
    store.close()
    assert outcome.ledger_summary["consumed_budget"] == 2
    assert len(outcome.candidates) == 2
    assert all(c.valid for c in outcome.candidates)
    assert outcome.task_namespace == arm_task_name(TASK_NAME, "random")


def test_random_arm_is_deterministic(tmp_path, t1_data):
    train_val, _ = t1_data
    results = []
    for i in range(2):
        store = _store(tmp_path, f"det_{i}.sqlite")
        results.append(_run_random(train_val, store))
        store.close()
    assert [c.structural_hash for c in results[0].candidates] == [
        c.structural_hash for c in results[1].candidates
    ]
    assert results[0].selected_val_rmse == results[1].selected_val_rmse


# --- arm-local duplicate detection (correction pass section 3) -------------


def test_cross_arm_duplicate_is_not_contaminated(tmp_path, t1_data):
    """Open-loop proposes A; Closed-loop proposal 1 ALSO proposes A. With
    arm-local caching, Closed-loop's A must NOT be marked duplicate."""
    train_val, _ = t1_data
    store = _store(tmp_path)
    open_provider = _ScriptedProvider([_ARCH_A, _ARCH_A])
    closed_provider = _ScriptedProvider([_ARCH_A, _ARCH_B])

    open_outcome = _run_open(train_val, store, open_provider)
    closed_outcome = _run_closed(train_val, store, closed_provider)
    store.close()

    # Open-loop proposed A twice: first valid, second a WITHIN-arm duplicate.
    assert open_outcome.candidates[0].is_duplicate is False
    assert open_outcome.candidates[1].is_duplicate is True

    # Closed-loop proposal 0 is architecture A -- but a *fresh* candidate for
    # this arm, so NOT a duplicate, and it must be trained (real val RMSE).
    assert closed_outcome.candidates[0].is_duplicate is False
    assert closed_outcome.candidates[0].valid is True
    assert closed_outcome.candidates[0].final_val_rmse is not None
    assert closed_outcome.candidates[0].actually_trained is True


def test_arm_execution_order_does_not_change_closed_loop(tmp_path, t1_data):
    """Closed-loop results and prompts must be identical whether or not
    another arm ran first in the same store."""
    train_val, _ = t1_data

    # Store 1: open-loop (proposing A) runs BEFORE closed-loop.
    store1 = _store(tmp_path, "with_open.sqlite")
    _run_open(train_val, store1, _ScriptedProvider([_ARCH_A, _ARCH_A]))
    closed_provider_1 = _ScriptedProvider([_ARCH_A, _ARCH_B])
    closed_1 = _run_closed(train_val, store1, closed_provider_1)
    store1.close()

    # Store 2: closed-loop runs alone.
    store2 = _store(tmp_path, "alone.sqlite")
    closed_provider_2 = _ScriptedProvider([_ARCH_A, _ARCH_B])
    closed_2 = _run_closed(train_val, store2, closed_provider_2)
    store2.close()

    assert [c.structural_hash for c in closed_1.candidates] == [
        c.structural_hash for c in closed_2.candidates
    ]
    assert [c.is_duplicate for c in closed_1.candidates] == [
        c.is_duplicate for c in closed_2.candidates
    ]
    assert closed_1.selected_val_rmse == closed_2.selected_val_rmse
    # Prompts identical -> no other-arm information leaked into the prompt.
    assert closed_provider_1.prompts_seen == closed_provider_2.prompts_seen


def test_closed_loop_second_prompt_derives_from_its_own_first_proposal(tmp_path, t1_data):
    train_val, _ = t1_data
    store = _store(tmp_path)
    provider = _ScriptedProvider([_ARCH_A, _ARCH_B])
    _run_closed(train_val, store, provider)
    store.close()

    first_prompt, second_prompt = provider.prompts_seen
    assert "{" not in first_prompt  # first turn carries no feedback block
    feedback = json.loads(second_prompt[second_prompt.index("{"): second_prompt.rindex("}") + 1])
    # Feedback is about closed-loop proposal 0 (architecture A), with all
    # fields present including a real validation RMSE.
    assert feedback["prior_architecture"]["layer_1_gate"] == "RY"
    assert feedback["valid"] is True
    assert feedback["is_duplicate"] is False
    assert feedback["validation_rmse"] is not None
    assert feedback["circuit_depth"] is not None
    assert feedback["two_qubit_gate_count"] is not None
    assert "protected" not in second_prompt.lower()


def test_same_architecture_retrains_separately_in_different_arms(tmp_path, t1_data):
    """Up to six trainings are permitted; the same architecture may be
    trained once per arm (separate cache namespaces)."""
    train_val, _ = t1_data
    store = _store(tmp_path)
    _run_open(train_val, store, _ScriptedProvider([_ARCH_A, _ARCH_B]), budget_limit=1)
    _run_closed(train_val, store, _ScriptedProvider([_ARCH_A, _ARCH_B]), budget_limit=1)
    # A trained under open's namespace AND under closed's namespace.
    ir = compact_to_circuit_ir(
        CompactArchitecture(
            layer_1_gate="RY", entangler="CNOT", entangler_direction="0_to_1", layer_2_gate="RX"
        )
    )
    from llm_vqc.ir.canonicalize import structural_hash
    h = structural_hash(ir)
    seed = train_seed_for_circuit(0, h)
    open_w = store.get_trained_weights(arm_task_name(TASK_NAME, "llm_open_loop"), h, seed)
    closed_w = store.get_trained_weights(arm_task_name(TASK_NAME, "llm_closed_loop"), h, seed)
    store.close()
    assert open_w is not None
    assert closed_w is not None


# --- API-call hard cap + budget --------------------------------------------


def test_shared_call_cap_is_a_hard_outbound_cap(tmp_path, t1_data):
    train_val, _ = t1_data
    store = _store(tmp_path)
    inner = _ScriptedProvider([_ARCH_A, _ARCH_B] * 10)
    shared = CallCountLimitedProvider(inner, max_calls=4)

    open_outcome = _run_open(train_val, store, shared)
    closed_outcome = _run_closed(train_val, store, shared)
    store.close()

    assert inner.calls_made == 4  # exactly the cap: 2 open + 2 closed
    assert open_outcome.api_outbound_attempts == 2
    assert closed_outcome.api_outbound_attempts == 2
    assert open_outcome.api_successful_calls == 2
    with pytest.raises(CallBudgetExceededError):
        shared.complete("s", "u", 0.2)


def test_dollar_budget_blocks_before_second_call(tmp_path, t1_data):
    train_val, _ = t1_data
    store = _store(tmp_path)
    provider = _ScriptedProvider([_ARCH_A, _ARCH_B])
    budget = LLMApiBudget(cap_usd=0.05)  # affords exactly one call at 0.05 estimate
    outcome = _run_open(train_val, store, provider, budget=budget)
    store.close()
    assert provider.calls_made == 1  # second call blocked before request
    assert len(outcome.candidates) == 1
    assert "dollar_budget_exceeded" in outcome.stop_reason


# --- honest partial results on interruption --------------------------------


def test_interrupted_api_call_produces_honest_partial_results(tmp_path, t1_data):
    train_val, _ = t1_data
    store = _store(tmp_path)
    provider = _RaisingAfterNProvider(n=1, good_response=_ARCH_A)
    outcome = _run_open(train_val, store, provider)
    store.close()

    assert len(outcome.candidates) == 1  # only the successful call recorded
    assert outcome.candidates[0].valid
    assert outcome.stop_reason is not None and "api_error" in outcome.stop_reason
    assert outcome.ledger_summary["consumed_budget"] == 1  # never fabricated as 2
    assert outcome.api_successful_calls == 1
    assert outcome.api_failed_calls == 1
    assert outcome.api_outbound_attempts == 2


# --- learned weights change; architecture fixed; float64 -------------------


def test_learned_quantum_weights_change_after_training(t1_data):
    train_val, _ = t1_data
    ir = compact_to_circuit_ir(
        CompactArchitecture(
            layer_1_gate="RY", entangler="CNOT", entangler_direction="0_to_1", layer_2_gate="RX"
        )
    )
    train_seed = train_seed_for_circuit(0, "label")
    output = train_model(
        ir, train_val, _training_config(epochs=3), train_seed,
        init_policy=mini_demo_init_policy, diff_method=DIFF_METHOD,
    )
    assert output.success
    initial = torch.tensor(output.initial_circuit_weights)
    learned = torch.tensor(output.trained_circuit_weights)
    assert not torch.allclose(initial, learned)
    # And every trained tensor was float64 on CPU.
    for p in output.param_provenance:
        if p["requires_grad"]:
            assert p["dtype"] == "torch.float64"
            assert p["device"] == "cpu"


def test_architecture_fields_not_modified_by_training(t1_data):
    train_val, _ = t1_data
    compact = CompactArchitecture(
        layer_1_gate="RZ", entangler="CZ", entangler_direction="0_to_1", layer_2_gate="RY"
    )
    ir_before = compact_to_circuit_ir(compact)
    train_seed = train_seed_for_circuit(0, "arch-fixed")
    output = train_model(
        ir_before, train_val, _training_config(epochs=2), train_seed,
        init_policy=mini_demo_init_policy, diff_method=DIFF_METHOD,
    )
    assert output.success
    assert compact_to_circuit_ir(compact) == ir_before


def test_diff_method_is_backprop():
    assert DIFF_METHOD == "backprop"


# --- protected-test quarantine ---------------------------------------------


def test_protected_test_only_after_search_and_from_own_namespace(tmp_path, t1_data):
    train_val, test_split = t1_data
    store = _store(tmp_path)
    outcome = _run_random(train_val, store)
    assert outcome.protected_test_rmse is None  # not computed during search
    evaluate_protected_test(outcome, store, TASK_NAME, test_split, train_val.spec)
    store.close()
    assert outcome.protected_test_rmse is not None and outcome.protected_test_rmse > 0


# --- exported metrics + initial/learned angles -----------------------------


def test_candidate_records_export_gate_count_and_runtime(tmp_path, t1_data):
    train_val, _ = t1_data
    store = _store(tmp_path)
    outcome = _run_random(train_val, store)
    store.close()
    for c in outcome.candidates:
        assert c.total_gate_count is not None and c.total_gate_count > 0
        assert c.training_runtime_seconds is not None
        assert c.dtype_device_verified is True
        assert c.quantum_angles_changed is True
        assert c.unique_training_id is not None


def test_selected_circuits_export_initial_and_learned_angles(tmp_path, t1_data):
    train_val, test_split = t1_data
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "figures").mkdir()
    store = _store(tmp_path)
    outcome = _run_random(train_val, store)
    evaluate_protected_test(outcome, store, TASK_NAME, test_split, train_val.spec)

    class _Args:
        provider = "openai"
        budget = 2
        epochs = 2
        seed = 0
        output = str(output_dir)

    reporting.write_all_outputs(
        output_dir=output_dir, store=store, args=_Args(),
        provenance={"execution_code_sha": "abc", "execution_git_dirty": False},
        model="fake", real_calls_made=0, max_real_calls=4, elapsed_seconds=1.0,
        prompt_version="v2", arm_outcomes=[outcome], task_name=TASK_NAME,
    )
    store.close()

    selections = json.loads((output_dir / "selected_circuits.json").read_text())
    entry = selections[0]
    assert len(entry["initial_quantum_angles"]) == 4
    assert len(entry["learned_quantum_angles"]) == 4
    assert entry["quantum_angles_changed"] is True
    assert "classical_summary_initial" in entry
    assert "classical_summary_learned" in entry


# --- no secrets / raw responses in artifacts -------------------------------


def test_written_artifacts_contain_no_secrets_or_raw_responses(tmp_path, t1_data):
    train_val, _ = t1_data
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "figures").mkdir()
    store = _store(tmp_path)

    fake_secret = "sk-FAKE1234567890ABCDEFSECRET"
    raw_marker = "RAW_RESPONSE_MARKER_SHOULD_NEVER_APPEAR"
    provider = _ScriptedProvider([f"not json {raw_marker} {fake_secret}"])
    outcome = _run_open(train_val, store, provider, budget_limit=1)

    class _Args:
        provider = "openai"
        budget = 1
        epochs = 2
        seed = 0
        output = str(output_dir)

    reporting.write_all_outputs(
        output_dir=output_dir, store=store, args=_Args(),
        provenance={"execution_code_sha": "abc", "execution_git_dirty": False},
        model="fake", real_calls_made=1, max_real_calls=4, elapsed_seconds=1.0,
        prompt_version="v2", arm_outcomes=[outcome], task_name=TASK_NAME,
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
