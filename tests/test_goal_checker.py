"""Goal-checker behaviour tests (goal §19: the checker must REJECT when a
required cell or artifact is removed; §18: exit non-zero until complete).
Runs the real ContractChecker against a miniature repo tree.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "check_goal_completion", REPO / "scripts" / "check_goal_completion.py"
)
_module = importlib.util.module_from_spec(_spec)
sys.modules["check_goal_completion"] = _module
_spec.loader.exec_module(_module)
ContractChecker = _module.ContractChecker


MINI_PROTOCOL = """
experiments:
  E1:
    task_n_pairs: [[taskx, 3]]
    arms: [arm_a]
    budget_unique: 2
    replicates: 2
statistics:
  data_seed_base: 1000
  search_seed_base: 2000
"""

MINI_CONTRACT = """
contract_version: mini
protocol: configs/mini_protocol.yaml
criteria:
  - id: M1_matrix
    check: matrix_complete
    experiments: [E1]
"""


def _write_manifest(root: Path, replicate: int, status: str = "complete") -> Path:
    cells = root / "runs" / "bench_v2" / "E1" / "cells"
    cells.mkdir(parents=True, exist_ok=True)
    path = cells / f"taskx_3q_arm_a_r{replicate}.json"
    path.write_text(json.dumps({
        "cell_id": f"taskx_3q_arm_a_r{replicate}", "status": status,
        "budget_unique": 2, "data_seed": 1000 + replicate,
        "search_seed": 2000 + replicate,
    }))
    return path


def _mini_repo(tmp_path: Path) -> Path:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "mini_protocol.yaml").write_text(MINI_PROTOCOL)
    (tmp_path / "GOAL_CONTRACT.yaml").write_text(MINI_CONTRACT)
    return tmp_path


def test_checker_passes_when_all_cells_present(tmp_path):
    root = _mini_repo(tmp_path)
    _write_manifest(root, 0)
    _write_manifest(root, 1)
    checker = ContractChecker(root, skip_commands=True)
    result = checker.check_matrix_complete(checker.contract["criteria"][0])
    assert result.passed, result.details


def test_checker_rejects_when_cell_removed(tmp_path):
    root = _mini_repo(tmp_path)
    _write_manifest(root, 0)
    kept = _write_manifest(root, 1)
    checker = ContractChecker(root, skip_commands=True)
    assert checker.check_matrix_complete(checker.contract["criteria"][0]).passed

    kept.unlink()  # remove one required cell -> must FAIL
    result = checker.check_matrix_complete(checker.contract["criteria"][0])
    assert not result.passed
    assert any("missing cell manifest" in d for d in result.details)


def test_checker_rejects_interrupted_cell(tmp_path):
    root = _mini_repo(tmp_path)
    _write_manifest(root, 0)
    _write_manifest(root, 1, status="running")
    checker = ContractChecker(root, skip_commands=True)
    result = checker.check_matrix_complete(checker.contract["criteria"][0])
    assert not result.passed
    assert any("interrupted" in d for d in result.details)


def test_checker_rejects_wrong_budget_or_broken_pairing(tmp_path):
    root = _mini_repo(tmp_path)
    _write_manifest(root, 0)
    path = _write_manifest(root, 1)
    manifest = json.loads(path.read_text())
    manifest["budget_unique"] = 1  # protocol says 2
    manifest["data_seed"] = 9999   # pairing broken
    path.write_text(json.dumps(manifest))
    checker = ContractChecker(root, skip_commands=True)
    result = checker.check_matrix_complete(checker.contract["criteria"][0])
    assert not result.passed
    assert any("budget" in d for d in result.details)
    assert any("pairing broken" in d for d in result.details)


def test_real_contract_currently_fails_overall():
    """The real repo checker must exit non-zero until the matrix is truly
    complete — completion can never be declared early."""
    checker = ContractChecker(REPO, skip_commands=True)
    assert checker.run() != 0
