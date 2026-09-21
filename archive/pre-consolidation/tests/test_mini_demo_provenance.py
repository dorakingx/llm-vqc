"""Tests for execution-provenance collection
(`llm_vqc.mini_demo.provenance`, correction pass section 9): record the
REAL HEAD sha, a dirty flag, and library versions -- never the base sha as
though it were the implementation sha.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from llm_vqc.mini_demo.provenance import collect_provenance


def _init_repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.dev"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "tester"], cwd=tmp_path, check=True)
    (tmp_path / "a.txt").write_text("hello")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)
    return tmp_path


def test_clean_worktree_reports_not_dirty_and_real_sha(tmp_path):
    repo = _init_repo(tmp_path)
    prov = collect_provenance(repo)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    assert prov["execution_code_sha"] == head
    assert prov["execution_git_dirty"] is False


def test_dirty_worktree_reports_dirty(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "a.txt").write_text("modified")  # uncommitted change
    prov = collect_provenance(repo)
    assert prov["execution_git_dirty"] is True


def test_provenance_records_python_and_library_versions(tmp_path):
    repo = _init_repo(tmp_path)
    prov = collect_provenance(repo)
    assert prov["python_version"]
    libs = prov["library_versions"]
    for name in ("torch", "pennylane", "openai", "numpy", "qiskit"):
        assert name in libs


def test_unavailable_git_is_treated_as_dirty(tmp_path):
    # A non-git directory: cannot prove clean, so must not claim clean.
    prov = collect_provenance(tmp_path)
    assert prov["execution_git_dirty"] is True
    assert prov["execution_code_sha"] is None
