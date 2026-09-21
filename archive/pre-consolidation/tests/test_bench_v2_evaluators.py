"""Phase 2 gate tests: dual-track evaluators (goal §17 Phase 2 gate).

Locks in: (1) identical architecture -> identical Track-A result across
arms (cache hit AND cold recomputation); (2) Track B structurally
optimizer-free; (3) protected test unreachable from evaluator/search
modules (AST import audit); (4) layered-space validation semantics;
(5) reference ansätze validity; (6) theta-seed purity.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from llm_vqc.bench_v2.space import (
    REFERENCE_ARMS,
    SCALABLE_QUBITS,
    SpaceProfile,
    max_ops_for,
    reference_ir,
    reference_operations,
    validate_layered_structure,
)
from llm_vqc.bench_v2.track_a_evaluator import (
    bench_v2_run_seed,
    evaluate_structure_candidate,
)
from llm_vqc.free_amplitude.training import FreeAmplitudeTrainingConfig
from llm_vqc.ir.budget import BudgetLedger
from llm_vqc.tasks.signal_suite import SignalProfile, SignalSuiteTask

REPO = Path(__file__).resolve().parents[1]

SMALL_CONFIG = FreeAmplitudeTrainingConfig(epochs=2, batch_size=8)


def _task_and_data(n_qubits: int = 3):
    task = SignalSuiteTask(
        SignalProfile(family="gauss_peak", n_qubits=n_qubits, n_train=16, n_val=16, n_test=16)
    )
    train_val, _ = task.build(1000)
    return task, train_val


def _layered_proposal():
    return {
        "operations": [
            {"type": "rot", "gates": ["RY"], "wires": "all"},
            {"type": "entangle", "pattern": "line", "gate": "CNOT", "wires": "all"},
        ]
    }


# ---------------------------------------------------------------------------
# (1) Identical architecture -> identical result across arms
# ---------------------------------------------------------------------------


def test_same_architecture_identical_result_across_arms(tmp_path):
    from llm_vqc.evaluation.store import ResultStore

    task, train_val = _task_and_data()
    space = SpaceProfile("scalable_layered_v1", 3)
    run_seed = bench_v2_run_seed(task.spec.name, data_seed=1000, search_seed=2000)

    store = ResultStore(tmp_path / "cache.sqlite")
    results = {}
    for arm in ("armA", "armB"):
        ledger = BudgetLedger()
        results[arm] = evaluate_structure_candidate(
            _layered_proposal(), space, task.spec.name, task.val_metric_name,
            run_seed, train_val, SMALL_CONFIG, f"{arm}:0", ledger, cache=store,
        )
    assert results["armA"].val_metric_value is not None
    # Second arm hits the shared cache: identical value, marked duplicate.
    assert results["armB"].val_metric_value == results["armA"].val_metric_value
    assert results["armB"].is_duplicate is True
    assert results["armB"].train_seed == results["armA"].train_seed


def test_same_architecture_identical_result_cold_recomputation(tmp_path):
    """Determinism without the cache: two cold evaluations agree exactly."""
    task, train_val = _task_and_data()
    space = SpaceProfile("scalable_layered_v1", 3)
    run_seed = bench_v2_run_seed(task.spec.name, 1000, 2000)
    values = []
    for _ in range(2):
        result = evaluate_structure_candidate(
            _layered_proposal(), space, task.spec.name, task.val_metric_name,
            run_seed, train_val, SMALL_CONFIG, "cold:0", BudgetLedger(), cache=None,
        )
        assert result.training_outcome.value == "success"
        values.append(result.val_metric_value)
    assert values[0] == values[1]


def test_different_search_seed_changes_theta_seed():
    task, _ = _task_and_data()
    seed_a = bench_v2_run_seed(task.spec.name, 1000, 2000)
    seed_b = bench_v2_run_seed(task.spec.name, 1000, 2001)
    assert seed_a != seed_b
    # Pure function: same inputs -> same output.
    assert seed_a == bench_v2_run_seed(task.spec.name, 1000, 2000)


# ---------------------------------------------------------------------------
# (2) Track B structurally optimizer-free
# ---------------------------------------------------------------------------


def _module_ast(rel_path: str) -> ast.Module:
    return ast.parse((REPO / rel_path).read_text())


def _imported_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def test_track_b_module_never_imports_optimizer():
    names = _imported_names(_module_ast("llm_vqc/bench_v2/track_b_evaluator.py"))
    assert not any("torch.optim" in n for n in names)
    # And its one evaluation dependency is the preserved no-optimizer module.
    assert any("free_amplitude.main_eval" in n for n in names)


# ---------------------------------------------------------------------------
# (3) Protected-test quarantine by import audit
# ---------------------------------------------------------------------------

QUARANTINED_MODULES = [
    "llm_vqc/bench_v2/track_a_evaluator.py",
    "llm_vqc/bench_v2/track_b_evaluator.py",
    "llm_vqc/bench_v2/space.py",
    "llm_vqc/bench_v2/init_policy.py",
    "llm_vqc/bench_v2/arm_registry.py",
]

FORBIDDEN_IMPORT_FRAGMENTS = ("test_gate", "final_test", "build_test")


@pytest.mark.parametrize("module_path", QUARANTINED_MODULES)
def test_search_side_modules_cannot_reach_test_data(module_path):
    names = _imported_names(_module_ast(module_path))
    for fragment in FORBIDDEN_IMPORT_FRAGMENTS:
        assert not any(fragment in n for n in names), (
            f"{module_path} imports {fragment!r} — protected-test quarantine broken"
        )


def test_test_gate_definition_of_protected_metric_present():
    text = (REPO / "llm_vqc/bench_v2/test_gate.py").read_text()
    assert "validation-selected protected-test" in text


# ---------------------------------------------------------------------------
# (4) Layered-space validation semantics
# ---------------------------------------------------------------------------


def test_max_ops_rule_frozen_values():
    assert [max_ops_for(n) for n in (3, 4, 5, 6, 8)] == [6, 8, 10, 12, 16]


def test_layered_validation_rejects_too_many_ops():
    space = SpaceProfile("scalable_layered_v1", 3)
    ops = [{"type": "rot", "gates": ["RY"], "wires": "all"}] * (space.max_ops + 1)
    ir, issues = validate_layered_structure(ops, space)
    assert ir is None
    assert any("max_ops" in issue.message for issue in issues)


def test_layered_validation_rejects_repeat_blocks():
    space = SpaceProfile("scalable_layered_v1", 3)
    ops = [{"type": "repeat", "times": 2,
            "body": [{"type": "rot", "gates": ["RY"], "wires": "all"}]}]
    ir, issues = validate_layered_structure(ops, space)
    assert ir is None
    assert any("repeat" in issue.message for issue in issues)


def test_layered_validation_collects_ir_issues():
    space = SpaceProfile("scalable_layered_v1", 3)
    ops = [{"type": "entangle", "pattern": "star", "gate": "CNOT", "wires": "all"}]
    ir, issues = validate_layered_structure(ops, space)  # star without center
    assert ir is None
    assert issues


def test_invalid_proposal_consumes_no_budget(tmp_path):
    task, train_val = _task_and_data()
    space = SpaceProfile("scalable_layered_v1", 3)
    ledger = BudgetLedger()
    result = evaluate_structure_candidate(
        {"operations": "nonsense"}, space, task.spec.name, task.val_metric_name,
        bench_v2_run_seed(task.spec.name, 1000, 2000), train_val, SMALL_CONFIG,
        "bad:0", ledger,
    )
    assert result.validation_outcome.value == "invalid"
    assert ledger.consumed_budget == 0
    assert ledger.num_invalid == 1


# ---------------------------------------------------------------------------
# (5) Reference ansätze
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ref_name", REFERENCE_ARMS)
def test_reference_ansatze_valid_at_all_scaling_qubits(ref_name):
    for n in SCALABLE_QUBITS:
        ir = reference_ir(ref_name, SpaceProfile("scalable_layered_v1", n))
        assert ir.n_qubits == n


def test_reference_depths_differ():
    d1 = reference_operations("ref_realamp_d1")
    d2 = reference_operations("ref_realamp_d2")
    assert len(d2) == 2 * len(d1)


# ---------------------------------------------------------------------------
# (6) Reference arm evaluation end-to-end (trains > 0 parameters)
# ---------------------------------------------------------------------------


def test_reference_evaluates_and_trains(tmp_path):
    task, train_val = _task_and_data()
    space = SpaceProfile("scalable_layered_v1", 3)
    result = evaluate_structure_candidate(
        {"operations": reference_operations("ref_realamp_d1")}, space,
        task.spec.name, task.val_metric_name,
        bench_v2_run_seed(task.spec.name, 1000, 2000), train_val,
        SMALL_CONFIG, "ref:0", BudgetLedger(),
    )
    assert result.training_outcome.value == "success"
    assert result.val_metric_value is not None
