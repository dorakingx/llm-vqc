"""Phase 4 gate tests: LLM arms (goal §17 Phase 4 gate + §19 items:
batch proposal parsing and partial invalidity, provider/cost caps before
calls, prompt contains no test metric).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from llm_vqc.bench_v2.llm_arms import (
    ARCHIVE_TOP_K,
    BATCH_SIZE,
    LLMArchiveClosedStructureArm,
    LLMArmError,
    LLMClosedJointArm,
    LLMOpenJointArm,
    LLMOpenStructureArm,
    parse_candidate_batch,
)
from llm_vqc.bench_v2.llm_providers import (
    MockJointBatchProvider,
    MockLayeredBatchProvider,
    build_real_provider,
    is_mock_provider,
)
from llm_vqc.bench_v2.runner import JointSearchRunner, StructureSearchRunner
from llm_vqc.bench_v2.space import SpaceProfile
from llm_vqc.bench_v2.track_a_evaluator import bench_v2_run_seed
from llm_vqc.evaluation.store import ResultStore
from llm_vqc.free_amplitude.openai_provider import RealRunPreflightError
from llm_vqc.free_amplitude.training import FreeAmplitudeTrainingConfig
from llm_vqc.llm.provider import LLMResponse
from llm_vqc.tasks.signal_suite import SignalProfile, SignalSuiteTask

REPO = Path(__file__).resolve().parents[1]
SPACE3 = SpaceProfile("scalable_layered_v1", 3)
SMALL_CONFIG = FreeAmplitudeTrainingConfig(epochs=1, batch_size=16)


def _train_val():
    task = SignalSuiteTask(
        SignalProfile(family="gauss_peak", n_qubits=3, n_train=12, n_val=12, n_test=12)
    )
    train_val, _ = task.build(1000)
    return task, train_val


class ScriptedBatchProvider:
    model_name = "scripted-batch"

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = 0

    def complete(self, system_prompt, user_prompt, temperature) -> LLMResponse:
        text = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return LLMResponse(
            raw_text=text, model=self.model_name, input_tokens=10, output_tokens=10,
            estimated_cost_usd=0.0, latency_seconds=0.0,
        )


# ---------------------------------------------------------------------------
# Batch parsing
# ---------------------------------------------------------------------------


def test_parse_batch_normal_and_fenced_and_single():
    batch = {"candidates": [{"operations": []}, {"operations": [1]}]}
    assert len(parse_candidate_batch(json.dumps(batch))) == 2
    fenced = "```json\n" + json.dumps(batch) + "\n```"
    assert len(parse_candidate_batch(fenced)) == 2
    single = {"n_qubits": 3, "operations": []}
    assert parse_candidate_batch(json.dumps(single)) == [single]


def test_parse_batch_rejects_garbage():
    assert parse_candidate_batch("not json at all") is None
    assert parse_candidate_batch(json.dumps({"candidates": "nope"})) is None
    assert parse_candidate_batch(json.dumps({"something": 1})) is None


# ---------------------------------------------------------------------------
# Prompt hygiene: no test information can reach a prompt
# ---------------------------------------------------------------------------


def test_prompt_builders_have_no_test_parameters():
    """AST-level: no function in llm_prompts accepts anything test-named,
    and the module imports no test-data path."""
    tree = ast.parse((REPO / "llm_vqc/bench_v2/llm_prompts.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for arg in node.args.args:
                assert "test" not in arg.arg.lower()
        if isinstance(node, ast.ImportFrom) and node.module:
            assert "test_gate" not in node.module
            assert "final_test" not in node.module


def test_llm_arms_module_never_imports_test_paths():
    tree = ast.parse((REPO / "llm_vqc/bench_v2/llm_arms.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for fragment in ("test_gate", "final_test", "build_test"):
                assert fragment not in node.module


# ---------------------------------------------------------------------------
# Cost-cap preflight
# ---------------------------------------------------------------------------


def test_real_provider_requires_explicit_cap():
    with pytest.raises(RealRunPreflightError):
        build_real_provider("structure", env={"OPENAI_API_KEY": "sk-x", "OPENAI_MODEL": "m"})
    with pytest.raises(RealRunPreflightError):
        build_real_provider(
            "structure",
            env={"OPENAI_API_KEY": "sk-x", "OPENAI_MODEL": "m", "LLM_API_BUDGET_USD": "0"},
        )
    with pytest.raises(RealRunPreflightError):
        build_real_provider(
            "joint", env={"OPENAI_MODEL": "m", "LLM_API_BUDGET_USD": "5"}
        )


# ---------------------------------------------------------------------------
# End-to-end with offline mocks (both tracks, open + closed)
# ---------------------------------------------------------------------------


def test_llm_open_structure_end_to_end(tmp_path):
    task, train_val = _train_val()
    store = ResultStore(tmp_path / "s.sqlite")
    calls_logged = []
    arm = LLMOpenStructureArm(
        MockLayeredBatchProvider(1, SPACE3, BATCH_SIZE), SPACE3, budget_limit=4,
        call_sink=calls_logged.append,
    )
    result = StructureSearchRunner(
        arm=arm, space=SPACE3, task_name=task.spec.name,
        val_metric_name=task.val_metric_name, train_val=train_val,
        training_config=SMALL_CONFIG, budget_limit=4,
        run_seed=bench_v2_run_seed(task.spec.name, 1000, 2000),
        result_store=store, run_id="llm_open0",
    ).run()
    assert result.ledger_summary["num_unique"] == 4
    assert result.selected_hash is not None
    assert calls_logged, "every LLM call must be logged"
    state = arm.deserialize_state(store.load_run_state("llm_open0"))
    usage = arm.llm_usage_summary(state)
    assert usage["mock"] is True
    assert usage["successful_calls"] >= 1
    # Batch separation: 4 unique evaluations needed at most 2 batch calls
    # of 3 candidates (LLM-call budget != evaluation budget).
    assert state.llm_calls_made <= 3


def test_llm_closed_structure_archive_bounded_and_in_prompt(tmp_path):
    task, train_val = _train_val()
    store = ResultStore(tmp_path / "s.sqlite")
    provider = MockLayeredBatchProvider(2, SPACE3, BATCH_SIZE)
    arm = LLMArchiveClosedStructureArm(provider, SPACE3, budget_limit=5)
    StructureSearchRunner(
        arm=arm, space=SPACE3, task_name=task.spec.name,
        val_metric_name=task.val_metric_name, train_val=train_val,
        training_config=SMALL_CONFIG, budget_limit=5,
        run_seed=bench_v2_run_seed(task.spec.name, 1000, 2000),
        result_store=store, run_id="llm_closed0",
    ).run()
    state = arm.deserialize_state(store.load_run_state("llm_closed0"))
    assert 1 <= len(state.archive) <= ARCHIVE_TOP_K
    metrics = [e.val_metric for e in state.archive]
    assert metrics == sorted(metrics)
    prompt = arm._user_prompt(state)
    assert "val_metric" in prompt and "archive" in prompt.lower()
    assert "test" not in prompt.lower()  # protected-test hygiene


@pytest.mark.parametrize(
    "arm_cls", [LLMOpenJointArm, LLMClosedJointArm]
)
def test_llm_joint_arms_end_to_end(tmp_path, arm_cls):
    task, train_val = _train_val()
    store = ResultStore(tmp_path / "s.sqlite")
    arm = arm_cls(MockJointBatchProvider(3, 3, BATCH_SIZE), 3, budget_limit=3)
    result = JointSearchRunner(
        arm=arm, n_qubits=3, task_name=task.spec.name, train_val=train_val,
        budget_limit=3, run_seed=bench_v2_run_seed(task.spec.name, 1000, 2000),
        result_store=store, run_id=f"{arm.name}0",
    ).run()
    assert result.ledger_summary["num_unique"] == 3
    assert result.selected_hash is not None
    state = arm.deserialize_state(store.load_run_state(f"{arm.name}0"))
    assert state.best_candidate is not None  # joint selection needs theta


def test_partial_invalidity_is_recorded_not_repaired(tmp_path):
    """A batch with one valid and one invalid candidate: the invalid one
    is evaluated INVALID (no budget consumed, issue recorded), the valid
    one proceeds."""
    task, train_val = _train_val()
    store = ResultStore(tmp_path / "s.sqlite")
    valid_ops = [{"type": "rot", "gates": ["RY"], "wires": "all"}]
    batch = {"candidates": [
        {"operations": [{"type": "rot", "gates": ["BAD_GATE"], "wires": "all"}]},
        {"operations": valid_ops},
    ]}
    fallback = {"candidates": [{"operations": [
        {"type": "rot", "gates": ["RX"], "wires": [0]},
    ]}]}
    provider = ScriptedBatchProvider([json.dumps(batch), json.dumps(fallback)])
    arm = LLMOpenStructureArm(provider, SPACE3, budget_limit=2)
    result = StructureSearchRunner(
        arm=arm, space=SPACE3, task_name=task.spec.name,
        val_metric_name=task.val_metric_name, train_val=train_val,
        training_config=SMALL_CONFIG, budget_limit=2,
        run_seed=bench_v2_run_seed(task.spec.name, 1000, 2000),
        result_store=store, run_id="partial0",
    ).run()
    assert result.ledger_summary["num_invalid"] == 1
    assert result.ledger_summary["num_unique"] == 2


def test_parse_failure_streak_fails_loudly(tmp_path):
    task, train_val = _train_val()
    store = ResultStore(tmp_path / "s.sqlite")
    provider = ScriptedBatchProvider(["garbage"] * 10)
    arm = LLMOpenStructureArm(provider, SPACE3, budget_limit=2)
    runner = StructureSearchRunner(
        arm=arm, space=SPACE3, task_name=task.spec.name,
        val_metric_name=task.val_metric_name, train_val=train_val,
        training_config=SMALL_CONFIG, budget_limit=2,
        run_seed=bench_v2_run_seed(task.spec.name, 1000, 2000),
        result_store=store, run_id="fail0",
    )
    with pytest.raises(LLMArmError):
        runner.run()


def test_llm_resume_replays_deterministically(tmp_path):
    """Same mock provider (pure function of prompt), interrupted vs
    uninterrupted: identical selection."""
    task, train_val = _train_val()

    def make_runner(store, budget):
        arm = LLMOpenStructureArm(
            MockLayeredBatchProvider(9, SPACE3, BATCH_SIZE), SPACE3, budget_limit=4
        )
        return StructureSearchRunner(
            arm=arm, space=SPACE3, task_name=task.spec.name,
            val_metric_name=task.val_metric_name, train_val=train_val,
            training_config=SMALL_CONFIG, budget_limit=budget,
            run_seed=bench_v2_run_seed(task.spec.name, 1000, 2000),
            result_store=store, run_id="llmresume",
        )

    store_full = ResultStore(tmp_path / "full.sqlite")
    full = make_runner(store_full, 4).run()

    store_part = ResultStore(tmp_path / "part.sqlite")
    make_runner(store_part, 2).run()
    resumed = make_runner(store_part, 4).run()

    assert resumed.selected_hash == full.selected_hash
    assert resumed.ledger_summary["num_unique"] == full.ledger_summary["num_unique"]


def test_is_mock_provider_detection():
    assert is_mock_provider(MockLayeredBatchProvider(1, SPACE3, 3)) is True
    assert is_mock_provider(ScriptedBatchProvider(["x"])) is False
