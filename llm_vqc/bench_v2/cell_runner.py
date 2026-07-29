"""One-cell orchestration: data -> search -> selection -> robustness ->
protected test gate -> durable manifest (protocol §10/§14).

`run_cell` is idempotent and resumable: an existing complete manifest
short-circuits (never re-runs the test gate); a partial store resumes
via the ledger/arm-state checkpoints. The protected test gate fires
exactly once per completed replicate, strictly after selection is
frozen; its timestamp and count land in the manifest for contract C09.

This is the ONLY search-side module allowed to import
`llm_vqc.bench_v2.test_gate` and the tasks' `build_test` — the search
loop it delegates to (runner/evaluators/arms) is import-audited to have
no such path, so the gate is reachable only after `run()` returned and
selection is frozen.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from llm_vqc.bench_v2 import arms as _arms  # noqa: F401  (registers classical arms)
from llm_vqc.bench_v2 import llm_arms as _llm_arms  # noqa: F401  (registers LLM arms)
from llm_vqc.bench_v2.arm_registry import get_arm_factory
from llm_vqc.bench_v2.init_policy import make_bench_v2_init_policy
from llm_vqc.bench_v2.llm_arms import _BaseLLMArm
from llm_vqc.bench_v2.manifests import cell_id as make_cell_id
from llm_vqc.bench_v2.manifests import read_cell_manifest, write_cell_manifest
from llm_vqc.bench_v2.resources import (
    COUPLING_PROFILES,
    logical_resource_summary,
    transpiled_resource_summary,
)
from llm_vqc.bench_v2.runner import JointSearchRunner, StructureSearchRunner
from llm_vqc.bench_v2.space import (
    READOUT_QUBIT,
    SpaceProfile,
)
from llm_vqc.bench_v2.test_gate import evaluate_selected_on_test
from llm_vqc.bench_v2.track_a_evaluator import bench_v2_run_seed
from llm_vqc.bench_v2.track_b_evaluator import bench_v2_joint_run_seed
from llm_vqc.evaluation.seeds import derive_child_seed
from llm_vqc.evaluation.store import ResultStore, compute_config_compat_hash
from llm_vqc.free_amplitude.candidate_schema import (
    CompleteCandidateProposal,
    complete_candidate_to_ir_and_theta,
)
from llm_vqc.free_amplitude.init_policy import free_amplitude_train_seed
from llm_vqc.free_amplitude.training import (
    FreeAmplitudeTrainingConfig,
    train_fixed_readout_model,
)
from llm_vqc.ir.schema import CircuitIR
from llm_vqc.tasks.signal_suite import get_legacy_gauss_peak_task, get_signal_task

DATA_SEED_BASE = 1000    # frozen (protocol_v2.yaml: statistics)
SEARCH_SEED_BASE = 2000
ROBUSTNESS_REINIT_SEEDS = 5

STRUCTURE_ARMS = {
    "random_structure", "evolutionary_structure", "greedy_growth",
    "llm_open_structure", "llm_archive_closed_structure",
    "ref_realamp_d1", "ref_realamp_d2", "ref_strongent_d1", "ref_strongent_d2",
}
JOINT_ARMS = {"random_joint", "evolutionary_joint", "llm_open_joint", "llm_closed_joint"}


class CellRunnerError(Exception):
    pass


def _git_sha() -> tuple[str | None, bool | None]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
            ).stdout.strip()
        )
        return sha, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, None


def _resolve_task(task_key: str, n_qubits: int):
    """`gauss_peak_legacy` -> preserved legacy task (E0/E1); otherwise a
    signal_suite profile. Returns (task, is_classification, val_metric)."""
    if task_key == "gauss_peak_legacy":
        if n_qubits != 3:
            raise CellRunnerError("gauss_peak_legacy is frozen at n_qubits=3")
        task = get_legacy_gauss_peak_task()
        return task, False, "rmse"
    task = get_signal_task(task_key, n_qubits)
    return task, task.is_classification, task.val_metric_name


@dataclass
class CellSpec:
    experiment: str
    task_key: str
    n_qubits: int
    arm_name: str
    replicate: int
    budget_unique: int
    training_config: FreeAmplitudeTrainingConfig
    space_name: str = "scalable_layered_v1"
    max_gates_joint: int = 5

    @property
    def data_seed(self) -> int:
        return DATA_SEED_BASE + self.replicate

    @property
    def search_seed(self) -> int:
        return SEARCH_SEED_BASE + self.replicate

    @property
    def cell_id(self) -> str:
        return make_cell_id(self.task_key, self.n_qubits, self.arm_name, self.replicate)


def _build_arm(spec: CellSpec, provider_factory):
    factory = get_arm_factory(spec.arm_name)
    if spec.arm_name in {"llm_open_structure", "llm_archive_closed_structure"}:
        provider = provider_factory(spec, "structure")
        profile = SpaceProfile(spec.space_name, spec.n_qubits)
        return factory(provider, profile, spec.budget_unique)
    if spec.arm_name in {"llm_open_joint", "llm_closed_joint"}:
        provider = provider_factory(spec, "joint")
        return factory(provider, spec.n_qubits, spec.budget_unique)
    if spec.arm_name in {"random_joint", "evolutionary_joint"}:
        return factory(spec.n_qubits)
    if spec.arm_name.startswith("ref_"):
        return factory(spec.arm_name, SpaceProfile(spec.space_name, spec.n_qubits))
    return factory(SpaceProfile(spec.space_name, spec.n_qubits))


def _selected_structure_details(
    store: ResultStore, task_name: str, selected_hash: str, run_seed: int,
    config_version: str,
) -> tuple[CircuitIR, list[float], dict]:
    train_seed = free_amplitude_train_seed(run_seed, selected_hash, config_version)
    blob = store.get_trained_weights(task_name, selected_hash, train_seed)
    if blob is None:
        raise CellRunnerError(
            f"selected candidate {selected_hash} has no cached trained weights"
        )
    from llm_vqc.bench_v2.space import layered_ir_dict

    operations = blob.get("operations")
    if operations is None:
        raise CellRunnerError("selected candidate blob lacks operations")
    if blob.get("space_profile") == "compact_free_v1":
        # Track A never runs on the compact profile in the frozen matrix
        # (E2/E3 are scalable_layered_v1; compact is Track B territory).
        raise CellRunnerError("compact_free_v1 structure cells are not part of bench_v2")
    ir = CircuitIR.model_validate(layered_ir_dict(blob["n_qubits"], operations))
    return ir, list(blob["learned_angles"] or []), blob


def _selected_joint_details(best_candidate: dict) -> tuple[CircuitIR, list[float]]:
    proposal = CompleteCandidateProposal.model_validate(best_candidate)
    ir, theta = complete_candidate_to_ir_and_theta(proposal, READOUT_QUBIT)
    return ir, list(theta)


def _robustness_reeval(
    ir: CircuitIR, task, train_val, training_config, base_train_seed: int,
    max_quantum_parameters: int,
) -> list[float | None]:
    """Predeclared post-selection robustness: retrain the SELECTED
    architecture under extra theta-init seeds; report the val-metric
    spread. Never used for selection (selection already happened)."""
    values: list[float | None] = []
    policy = make_bench_v2_init_policy(max_quantum_parameters)
    for i in range(ROBUSTNESS_REINIT_SEEDS):
        seed = derive_child_seed(base_train_seed, "robustness_reinit", str(i))
        output = train_fixed_readout_model(
            ir, READOUT_QUBIT, train_val, training_config, seed, init_policy=policy
        )
        values.append(output.final_val_metric if output.success else None)
    return values


def run_cell(
    spec: CellSpec,
    runs_root: Path,
    provider_factory=None,
    skip_if_complete: bool = True,
) -> dict:
    """Execute one cell to completion; returns the manifest dict."""
    existing = read_cell_manifest(runs_root, spec.experiment, spec.cell_id)
    if skip_if_complete and existing is not None and existing.get("status") == "complete":
        return existing

    is_structure = spec.arm_name in STRUCTURE_ARMS
    if not is_structure and spec.arm_name not in JOINT_ARMS:
        raise CellRunnerError(f"unknown arm {spec.arm_name!r}")

    task, is_classification, val_metric_name = _resolve_task(spec.task_key, spec.n_qubits)
    train_val, _diag = task.build(spec.data_seed)
    task_name = task.spec.name

    run_seed = (
        bench_v2_run_seed(task_name, spec.data_seed, spec.search_seed)
        if is_structure
        else bench_v2_joint_run_seed(task_name, spec.data_seed, spec.search_seed)
    )
    git_sha, git_dirty = _git_sha()

    cell_dir = runs_root / spec.experiment / spec.cell_id
    cell_dir.mkdir(parents=True, exist_ok=True)
    config_fields = {
        "experiment": spec.experiment, "task": task_name, "n_qubits": spec.n_qubits,
        "arm": spec.arm_name, "replicate": spec.replicate,
        "budget_unique": spec.budget_unique,
        "training_config": spec.training_config.model_dump() if is_structure else None,
        "data_seed": spec.data_seed, "search_seed": spec.search_seed,
    }
    store = ResultStore.open_or_create(
        cell_dir / "results.sqlite", spec.cell_id,
        json.dumps(config_fields, sort_keys=True), config_fields,
        git_sha, datetime.now(UTC).isoformat(),
    )

    arm = _build_arm(spec, provider_factory)
    if isinstance(arm, _BaseLLMArm):
        arm.call_sink = lambda record: store.append_llm_call(spec.cell_id, record)

    if is_structure:
        space = SpaceProfile(spec.space_name, spec.n_qubits)
        runner = StructureSearchRunner(
            arm=arm, space=space, task_name=task_name, val_metric_name=val_metric_name,
            train_val=train_val, training_config=spec.training_config,
            budget_limit=spec.budget_unique, run_seed=run_seed,
            result_store=store, run_id=spec.cell_id,
        )
    else:
        runner = JointSearchRunner(
            arm=arm, n_qubits=spec.n_qubits, task_name=task_name, train_val=train_val,
            budget_limit=spec.budget_unique, run_seed=run_seed,
            result_store=store, run_id=spec.cell_id,
        )

    run_result = runner.run()
    if run_result.selected_hash is None:
        raise CellRunnerError(
            f"cell {spec.cell_id}: no candidate was ever successfully evaluated"
        )

    # ---- selection frozen; details + robustness --------------------------
    state = arm.deserialize_state(store.load_run_state(spec.cell_id))
    robustness: list[float | None] | None = None
    if is_structure:
        ir, theta, _blob = _selected_structure_details(
            store, task_name, run_result.selected_hash, run_seed,
            spec.training_config.config_version,
        )
        base_train_seed = free_amplitude_train_seed(
            run_seed, run_result.selected_hash, spec.training_config.config_version
        )
        space = SpaceProfile(spec.space_name, spec.n_qubits)
        robustness = _robustness_reeval(
            ir, task, train_val, spec.training_config, base_train_seed,
            space.max_quantum_parameters,
        )
    else:
        if state.best_candidate is None:
            raise CellRunnerError(f"cell {spec.cell_id}: joint arm lost its best candidate")
        ir, theta = _selected_joint_details(state.best_candidate)

    resources = {
        "logical": logical_resource_summary(ir),
        "transpiled": {
            profile: transpiled_resource_summary(ir, profile)
            for profile in COUPLING_PROFILES
        },
    }

    # ---- protected test gate (exactly once, after selection) -------------
    test_split, _test_diag = task.build_test(spec.data_seed)
    test_metrics = evaluate_selected_on_test(
        ir, theta, test_split, is_classification, readout_qubit=READOUT_QUBIT
    )
    test_gate = {
        "evaluation_count": 1,
        "timestamp": test_metrics.pop("test_gate_timestamp"),
        "metrics": test_metrics,
    }

    llm_usage = arm.llm_usage_summary(state) if isinstance(arm, _BaseLLMArm) else None

    manifest_path = write_cell_manifest(
        runs_root, spec.experiment,
        task=spec.task_key, n_qubits=spec.n_qubits, arm=spec.arm_name,
        replicate=spec.replicate, data_seed=spec.data_seed,
        search_seed=spec.search_seed, budget_unique=spec.budget_unique,
        ledger_summary=run_result.ledger_summary,
        selected={
            "structural_hash": run_result.selected_hash,
            "val_metric_name": val_metric_name,
            "val_metric_value": run_result.selected_val_metric_value,
            "theta_len": len(theta),
            "robustness_reinit_val_metrics": robustness,
        },
        selection_timestamp=run_result.selection_timestamp,
        test_gate=test_gate,
        config_hash=compute_config_compat_hash(config_fields),
        git_sha=git_sha, git_dirty=git_dirty,
        store_path=str(cell_dir / "results.sqlite"),
        llm=llm_usage,
        extra={"stop_reason": run_result.stop_reason, "resources": resources},
    )
    store.close()
    return json.loads(manifest_path.read_text())
