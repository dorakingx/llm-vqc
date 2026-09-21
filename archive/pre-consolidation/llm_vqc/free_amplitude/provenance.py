"""Execution-provenance collection: the real `HEAD` SHA, a dirty flag, and
Python/library versions -- never the base/starting commit misattributed as
the implementation commit.
"""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path


def _git(repo_root: Path, *args: str) -> str | None:
    try:
        r = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True, text=True, check=True,
        )
        return r.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _lib_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
        return getattr(module, "__version__", None)
    except Exception:
        return None


def collect_provenance(repo_root: Path) -> dict:
    """Never raises -- missing pieces are `None`/`True` (safe default),
    not a guess. `execution_git_dirty` defaults to `True` when git status
    cannot be determined at all (never claim clean without proof)."""
    head = _git(repo_root, "rev-parse", "HEAD")
    status = _git(repo_root, "status", "--porcelain")
    dirty = True if status is None else bool(status.strip())
    return {
        "execution_code_sha": head,
        "execution_git_dirty": dirty,
        "python_version": platform.python_version(),
        "library_versions": {
            "torch": _lib_version("torch"),
            "pennylane": _lib_version("pennylane"),
            "openai": _lib_version("openai"),
            "numpy": _lib_version("numpy"),
            "qiskit": _lib_version("qiskit"),
        },
    }
