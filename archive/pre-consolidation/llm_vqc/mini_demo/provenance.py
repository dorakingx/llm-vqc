"""Execution-provenance collection for the mini demo (correction pass,
section 9).

The earlier real run recorded the *base* commit SHA as though it were the
implementation SHA, even though the worktree contained uncommitted
implementation changes. This module records the ACTUAL `HEAD` at run time
plus whether the worktree was dirty, so an artifact can never again
misattribute which code produced it. The intended workflow is:

  1. commit + push the corrected implementation (HEAD becomes that SHA,
     worktree clean);
  2. run the real experiment from that clean SHA (this module records it
     with `execution_git_dirty=false`);
  3. commit + push the sanitized artifacts in a second commit.
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
    """Gather execution provenance: the real HEAD SHA, dirty flag, Python
    and key-library versions. Never raises -- missing pieces are recorded
    as `None`/`unknown` rather than aborting output generation."""
    head = _git(repo_root, "rev-parse", "HEAD")
    status = _git(repo_root, "status", "--porcelain")
    # `status is None` means git was unavailable; treat unknown as dirty=True
    # (the safe assumption -- never claim clean when we cannot prove it).
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
