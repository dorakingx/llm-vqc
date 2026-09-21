"""Regression test for the secret-cleanup hardening (correction pass
section 1): local `.env` variants and backups must be git-ignored, while
the checked-in `.env.example` template stays tracked."""

from __future__ import annotations

import subprocess

import pytest


def _is_ignored(path: str) -> bool:
    """True if git would ignore `path` (run from the repo working tree)."""
    result = subprocess.run(
        ["git", "check-ignore", "-q", path],
        capture_output=True,
    )
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    pytest.skip("git check-ignore unavailable in this environment")


@pytest.mark.parametrize(
    "path",
    [".env", ".env.local", ".env.bak.20260723192416", ".env.production", ".env.anything"],
)
def test_env_variants_and_backups_are_ignored(path):
    assert _is_ignored(path), f"{path} must be git-ignored"


def test_env_example_is_not_ignored():
    assert not _is_ignored(".env.example"), ".env.example must remain tracked"
