"""Cell-runner tests: end-to-end cell execution, test-gate discipline,
idempotent skip, and manifest completeness for the checker."""

from __future__ import annotations

import json

import pytest

from llm_vqc.bench_v2.cell_runner import (
    CellRunnerError,
    CellSpec,
    run_cell,
)
from llm_vqc.bench_v2.llm_providers import MockJointBatchProvider, MockLayeredBatchProvider
from llm_vqc.bench_v2.space import SpaceProfile
from llm_vqc.free_amplitude.training import FreeAmplitudeTrainingConfig

TINY = FreeAmplitudeTrainingConfig(epochs=1, batch_size=16)


def _spec(arm: str, task: str = "gauss_peak", n: int = 3, budget: int = 2) -> CellSpec:
    return CellSpec(
        experiment="TEST", task_key=task, n_qubits=n, arm_name=arm,
        replicate=0, budget_unique=budget, training_config=TINY,
    )


def _provider_factory(spec: CellSpec, track: str):
    if track == "structure":
        return MockLayeredBatchProvider(1, SpaceProfile(spec.space_name, spec.n_qubits), 3)
    return MockJointBatchProvider(1, spec.n_qubits, 3)


def test_structure_cell_end_to_end(tmp_path):
    manifest = run_cell(_spec("random_structure"), tmp_path)
    assert manifest["status"] == "complete"
    assert manifest["budget_unique"] == 2
    assert manifest["ledger"]["num_unique"] == 2
    assert manifest["data_seed"] == 1000 and manifest["search_seed"] == 2000
    assert manifest["test_gate"]["evaluation_count"] == 1
    assert manifest["test_gate"]["timestamp"] >= manifest["selection_timestamp"]
    assert "test_rmse" in manifest["test_gate"]["metrics"]
    assert manifest["selected"]["robustness_reinit_val_metrics"] is not None
    assert len(manifest["selected"]["robustness_reinit_val_metrics"]) == 5
    assert manifest["resources"]["logical"]["logical_gate_count"] >= 1
    assert set(manifest["resources"]["transpiled"]) == {"line", "ring", "all_to_all"}
    assert manifest["llm"] is None


def test_joint_cell_end_to_end(tmp_path):
    manifest = run_cell(_spec("random_joint", task="gauss_peak_legacy"), tmp_path)
    assert manifest["status"] == "complete"
    assert manifest["test_gate"]["evaluation_count"] == 1
    # Joint track: no robustness retraining (no optimizer exists).
    assert manifest["selected"]["robustness_reinit_val_metrics"] is None
    assert manifest["selected"]["theta_len"] >= 1


def test_llm_cell_manifest_reports_mock(tmp_path):
    manifest = run_cell(
        _spec("llm_open_structure"), tmp_path, provider_factory=_provider_factory
    )
    assert manifest["llm"]["mock"] is True
    assert manifest["llm"]["successful_calls"] >= 1
    assert manifest["llm"]["prompt_version"] == "bench_v2_prompt_v2"
    assert manifest["llm"]["model_snapshot"] == "mock-layered-batch-v1"


def test_cell_skip_if_complete(tmp_path):
    first = run_cell(_spec("random_structure"), tmp_path)
    second = run_cell(_spec("random_structure"), tmp_path)
    # Identical manifest returned; test gate NOT re-run (same timestamp).
    assert second["test_gate"]["timestamp"] == first["test_gate"]["timestamp"]
    assert second["written_at"] == first["written_at"]


def test_classification_cell_uses_auc(tmp_path):
    manifest = run_cell(
        _spec("random_structure", task="peak_count", n=5, budget=2), tmp_path
    )
    assert "test_auc" in manifest["test_gate"]["metrics"]
    assert manifest["selected"]["val_metric_name"] == "brier"


def test_unknown_arm_rejected(tmp_path):
    with pytest.raises(CellRunnerError):
        run_cell(_spec("no_such_arm"), tmp_path)


def test_legacy_task_pinned_to_n3(tmp_path):
    with pytest.raises(CellRunnerError):
        run_cell(_spec("random_joint", task="gauss_peak_legacy", n=5), tmp_path)


def test_manifest_json_is_checker_shaped(tmp_path):
    run_cell(_spec("random_structure"), tmp_path)
    path = tmp_path / "TEST" / "cells" / "gauss_peak_3q_random_structure_r0.json"
    manifest = json.loads(path.read_text())
    for key in (
        "cell_id", "experiment", "task", "n_qubits", "arm", "replicate",
        "data_seed", "search_seed", "budget_unique", "ledger", "selected",
        "selection_timestamp", "test_gate", "config_hash", "git_sha",
        "store_path", "status",
    ):
        assert key in manifest
