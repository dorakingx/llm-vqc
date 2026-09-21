"""Tests for the mini demo's preflight/budget/retry policy
(`llm_vqc.mini_demo.runner.preflight_checks`, correction pass section 8):
a positive `LLM_API_BUDGET_USD` and an explicit `OPENAI_MODEL` are required
BEFORE any provider is constructed or any request issued; the OpenAI SDK
client disables its own retries.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from llm_vqc.mini_demo.runner import PreflightError, preflight_checks

_FULL_ENV = {
    "OPENAI_API_KEY": "sk-not-a-real-key-just-for-tests",
    "OPENAI_MODEL": "gpt-x",
    "LLM_API_BUDGET_USD": "2.00",
}


def test_full_env_passes_and_returns_config():
    cfg = preflight_checks(_FULL_ENV)
    assert cfg.api_key == _FULL_ENV["OPENAI_API_KEY"]
    assert cfg.model == "gpt-x"
    assert cfg.budget.cap_usd == 2.00


def test_missing_api_key_blocks():
    env = {**_FULL_ENV}
    del env["OPENAI_API_KEY"]
    with pytest.raises(PreflightError, match="OPENAI_API_KEY"):
        preflight_checks(env)


def test_missing_model_blocks_no_silent_default():
    env = {**_FULL_ENV}
    del env["OPENAI_MODEL"]
    with pytest.raises(PreflightError, match="OPENAI_MODEL"):
        preflight_checks(env)


def test_empty_model_blocks():
    with pytest.raises(PreflightError, match="OPENAI_MODEL"):
        preflight_checks({**_FULL_ENV, "OPENAI_MODEL": "   "})


def test_missing_budget_blocks():
    env = {**_FULL_ENV}
    del env["LLM_API_BUDGET_USD"]
    with pytest.raises(PreflightError, match="LLM_API_BUDGET_USD"):
        preflight_checks(env)


def test_zero_budget_blocks():
    with pytest.raises(PreflightError, match="LLM_API_BUDGET_USD"):
        preflight_checks({**_FULL_ENV, "LLM_API_BUDGET_USD": "0"})


def test_negative_budget_blocks():
    with pytest.raises(PreflightError, match="LLM_API_BUDGET_USD"):
        preflight_checks({**_FULL_ENV, "LLM_API_BUDGET_USD": "-5"})


def test_unparseable_budget_blocks():
    with pytest.raises(PreflightError, match="LLM_API_BUDGET_USD"):
        preflight_checks({**_FULL_ENV, "LLM_API_BUDGET_USD": "not-a-number"})


def test_provider_constructs_openai_client_with_zero_retries():
    """Static check: the compact provider must build its SDK client with
    max_retries=0 (so the four-call cap is a four-outbound-request cap)."""
    source = pathlib.Path("llm_vqc/mini_demo/provider.py").read_text()
    tree = ast.parse(source)
    found_zero_retries = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "OpenAI":
            for kw in node.keywords:
                if kw.arg == "max_retries":
                    assert isinstance(kw.value, ast.Constant) and kw.value.value == 0
                    found_zero_retries = True
    assert found_zero_retries, "OpenAI(...) must be constructed with max_retries=0"


def test_provider_has_no_retry_loop_over_the_request():
    """The provider issues exactly one outbound request per complete() -- no
    retry loop. `outbound_attempts` must therefore equal the call count; we
    assert the source contains no `for ... range(... retries ...)` loop."""
    source = pathlib.Path("llm_vqc/mini_demo/provider.py").read_text()
    assert "max_retries" not in source or "OpenAI(" in source
    assert "for attempt in range" not in source
