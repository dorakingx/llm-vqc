"""Phase 3 gate tests: classical arms + bench_v2 runners (goal §17
Phase 3 gate: exact budget accounting, resume equivalence, valid
proposals, deterministic paired seeds).
"""

from __future__ import annotations

import numpy as np
import pytest

from llm_vqc.bench_v2.arms.joint_arms import (
    EvolutionaryJointArm,
    RandomJointArm,
    _mutate_complete,
)
from llm_vqc.bench_v2.arms.sampling import (
    crossover_operations,
    mutate_operations,
    sample_layered_operations,
)
from llm_vqc.bench_v2.arms.structure_arms import (
    EvolutionaryStructureArm,
    FixedReferenceArm,
    GreedyGrowthArm,
    RandomStructureArm,
)
from llm_vqc.bench_v2.runner import JointSearchRunner, StructureSearchRunner
from llm_vqc.bench_v2.space import SpaceProfile, validate_layered_structure
from llm_vqc.bench_v2.track_a_evaluator import bench_v2_run_seed
from llm_vqc.evaluation.store import ResultStore
from llm_vqc.free_amplitude.candidate_schema import validate_complete_candidate
from llm_vqc.free_amplitude.training import FreeAmplitudeTrainingConfig
from llm_vqc.tasks.signal_suite import SignalProfile, SignalSuiteTask

SMALL_CONFIG = FreeAmplitudeTrainingConfig(epochs=1, batch_size=16)
SPACE3 = SpaceProfile("scalable_layered_v1", 3)


def _train_val(n_qubits: int = 3):
    task = SignalSuiteTask(
        SignalProfile(family="gauss_peak", n_qubits=n_qubits, n_train=12, n_val=12, n_test=12)
    )
    train_val, _ = task.build(1000)
    return task, train_val


# ---------------------------------------------------------------------------
# Sampling / mutation grammar
# ---------------------------------------------------------------------------


def test_layered_sampler_always_valid_and_bounded():
    rng = np.random.default_rng(7)
    for _ in range(100):
        ops = sample_layered_operations(rng, SPACE3)
        assert 1 <= len(ops) <= SPACE3.max_ops
        ir, issues = validate_layered_structure(ops, SPACE3)
        assert ir is not None, issues


def test_layered_sampler_deterministic():
    a = sample_layered_operations(np.random.default_rng(42), SPACE3)
    b = sample_layered_operations(np.random.default_rng(42), SPACE3)
    assert a == b


def test_mutations_and_crossover_stay_valid():
    rng = np.random.default_rng(11)
    parent_a = sample_layered_operations(rng, SPACE3)
    parent_b = sample_layered_operations(rng, SPACE3)
    for _ in range(50):
        mutant = mutate_operations(rng, parent_a, SPACE3)
        assert validate_layered_structure(mutant, SPACE3)[0] is not None
        child = crossover_operations(rng, parent_a, parent_b, SPACE3)
        assert validate_layered_structure(child, SPACE3)[0] is not None


def test_joint_mutations_stay_valid_with_theta_in_range():
    rng = np.random.default_rng(13)
    from llm_vqc.free_amplitude.sampler import sample_complete_candidate

    ops = [op.model_dump() for op in sample_complete_candidate(rng, 3).operations]
    for _ in range(50):
        ops2 = _mutate_complete(rng, ops, 3)
        candidate = {"n_qubits": 3, "operations": ops2}
        assert validate_complete_candidate(candidate, expected_n_qubits=3).valid
        for op in ops2:
            if op.get("theta") is not None:
                assert -np.pi <= op["theta"] <= np.pi


# ---------------------------------------------------------------------------
# Runner: budget semantics, early stop, resume equivalence
# ---------------------------------------------------------------------------


def _structure_runner(store, arm, task, train_val, budget, run_id):
    return StructureSearchRunner(
        arm=arm, space=SPACE3, task_name=task.spec.name,
        val_metric_name=task.val_metric_name, train_val=train_val,
        training_config=SMALL_CONFIG, budget_limit=budget,
        run_seed=bench_v2_run_seed(task.spec.name, 1000, 2000),
        result_store=store, run_id=run_id,
    )


def test_runner_stops_at_exactly_budget_unique(tmp_path):
    task, train_val = _train_val()
    store = ResultStore(tmp_path / "s.sqlite")
    arm = RandomStructureArm(SPACE3)
    result = _structure_runner(store, arm, task, train_val, budget=3, run_id="r0").run()
    assert result.ledger_summary["num_unique"] == 3
    assert result.stop_reason == "budget_exhausted"
    assert result.selected_hash is not None


def test_reference_arm_early_stop(tmp_path):
    task, train_val = _train_val()
    store = ResultStore(tmp_path / "s.sqlite")
    arm = FixedReferenceArm("ref_realamp_d1", SPACE3)
    result = _structure_runner(store, arm, task, train_val, budget=5, run_id="ref0").run()
    assert result.stop_reason == "arm_exhausted"
    assert result.ledger_summary["num_unique"] == 1
    assert result.selected_hash is not None


def test_resume_equivalence(tmp_path):
    """Interrupt at budget k, resume to B: identical selection and ledger
    as an uninterrupted run with the same seeds."""
    task, train_val = _train_val()

    # Uninterrupted reference run.
    store_full = ResultStore(tmp_path / "full.sqlite")
    full = _structure_runner(
        store_full, RandomStructureArm(SPACE3), task, train_val, budget=4, run_id="run"
    ).run()

    # Interrupted-then-resumed run (fresh store, same run_id/seeds).
    store_resume = ResultStore(tmp_path / "resume.sqlite")
    _structure_runner(
        store_resume, RandomStructureArm(SPACE3), task, train_val, budget=2, run_id="run"
    ).run()  # "crash" after 2 unique evaluations
    resumed = _structure_runner(
        store_resume, RandomStructureArm(SPACE3), task, train_val, budget=4, run_id="run"
    ).run()

    assert resumed.selected_hash == full.selected_hash
    assert resumed.selected_val_metric_value == full.selected_val_metric_value
    assert resumed.ledger_summary == full.ledger_summary


def test_paired_seeds_are_arm_specific_but_deterministic(tmp_path):
    task, train_val = _train_val()
    run_seed = bench_v2_run_seed(task.spec.name, 1000, 2000)
    from llm_vqc.evaluation.seeds import derive_child_seed

    a1 = derive_child_seed(run_seed, "search", "random_structure")
    a2 = derive_child_seed(run_seed, "search", "evolutionary_structure")
    assert a1 != a2  # arms draw independent streams from the shared block
    assert a1 == derive_child_seed(run_seed, "search", "random_structure")


# ---------------------------------------------------------------------------
# Arm behaviour end-to-end (tiny budgets)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arm_factory",
    [
        lambda: RandomStructureArm(SPACE3),
        lambda: EvolutionaryStructureArm(SPACE3),
        lambda: GreedyGrowthArm(SPACE3),
    ],
)
def test_structure_arms_complete_and_select(tmp_path, arm_factory):
    task, train_val = _train_val()
    store = ResultStore(tmp_path / "s.sqlite")
    arm = arm_factory()
    result = _structure_runner(
        store, arm, task, train_val, budget=3, run_id=f"{arm.name}0"
    ).run()
    assert result.ledger_summary["num_unique"] == 3
    assert result.selected_hash is not None
    assert result.selected_val_metric_value is not None


@pytest.mark.parametrize(
    "arm_factory", [lambda: RandomJointArm(3), lambda: EvolutionaryJointArm(3)]
)
def test_joint_arms_complete_and_select(tmp_path, arm_factory):
    task, train_val = _train_val()
    store = ResultStore(tmp_path / "s.sqlite")
    arm = arm_factory()
    runner = JointSearchRunner(
        arm=arm, n_qubits=3, task_name=task.spec.name, train_val=train_val,
        budget_limit=3, run_seed=bench_v2_run_seed(task.spec.name, 1000, 2000),
        result_store=store, run_id=f"{arm.name}0",
    )
    result = runner.run()
    assert result.ledger_summary["num_unique"] == 3
    assert result.selected_hash is not None


def test_evolutionary_structure_uses_population_after_seeding(tmp_path):
    """After mu evaluated individuals exist, offspring derive from parents
    (state.population is non-empty and bounded)."""
    task, train_val = _train_val()
    store = ResultStore(tmp_path / "s.sqlite")
    arm = EvolutionaryStructureArm(SPACE3)
    _structure_runner(store, arm, task, train_val, budget=6, run_id="evo0").run()
    state = arm.deserialize_state(store.load_run_state("evo0"))
    assert len(state.population) >= 4
    metrics = [ind.metric for ind in state.population]
    assert metrics == sorted(metrics)  # kept sorted best-first (lower better)


def test_arm_registry_has_all_classical_arms():
    import llm_vqc.bench_v2.arms  # noqa: F401  (registers on import)
    from llm_vqc.bench_v2.arm_registry import ARM_FACTORIES

    for name in (
        "random_structure", "evolutionary_structure", "greedy_growth",
        "ref_realamp_d1", "ref_realamp_d2", "ref_strongent_d1", "ref_strongent_d2",
        "random_joint", "evolutionary_joint",
    ):
        assert name in ARM_FACTORIES
