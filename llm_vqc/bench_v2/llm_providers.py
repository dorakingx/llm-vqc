"""Providers for the bench_v2 LLM arms.

Offline mocks live here (importable everywhere, zero cost, deterministic
as pure functions of the prompt text so resume replays identically). The
REAL provider stack is constructed only by `build_real_structure_provider`
/ `build_real_joint_provider`, which import the SDK-touching module
lazily and run the full preflight (`OPENAI_API_KEY` + explicit
`OPENAI_MODEL` + positive `LLM_API_BUDGET_USD` via the sanctioned
`LLMApiBudget.from_env`) before any client exists — no paid call can
happen by import side effect (contract C08 mock-vs-real accounting relies
on `is_mock_provider`).
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from llm_vqc.bench_v2.arms.sampling import sample_layered_operations
from llm_vqc.bench_v2.space import SpaceProfile
from llm_vqc.llm.provider import LLMResponse

MOCK_MODEL_PREFIX = "mock-"
PER_CELL_CALL_CAP = 40


def is_mock_provider(provider) -> bool:
    return str(getattr(provider, "model_name", "")).startswith(MOCK_MODEL_PREFIX)


def _stable_rng(*parts: str) -> np.random.Generator:
    digest = hashlib.sha256("\x1f".join(parts).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


class MockLayeredBatchProvider:
    """Offline structure-track mock: emits a batch of valid layered
    candidates drawn by the SAME frozen sampling grammar Random uses."""

    model_name = "mock-layered-batch-v1"

    def __init__(self, seed: int, profile: SpaceProfile, batch_size: int) -> None:
        self.seed = seed
        self.profile = profile
        self.batch_size = batch_size

    def complete(self, system_prompt: str, user_prompt: str, temperature: float) -> LLMResponse:
        rng = _stable_rng(str(self.seed), system_prompt, user_prompt, str(temperature))
        candidates = [
            {"operations": sample_layered_operations(rng, self.profile)}
            for _ in range(self.batch_size)
        ]
        text = json.dumps({"candidates": candidates})
        return LLMResponse(
            raw_text=text, model=self.model_name,
            input_tokens=len(system_prompt) // 4, output_tokens=len(text) // 4,
            estimated_cost_usd=0.0, latency_seconds=0.0,
        )


class MockJointBatchProvider:
    """Offline joint-track mock: batch of valid complete candidates via
    the preserved sampler."""

    model_name = "mock-joint-batch-v1"

    def __init__(self, seed: int, n_qubits: int, batch_size: int, max_gates: int = 5) -> None:
        self.seed = seed
        self.n_qubits = n_qubits
        self.batch_size = batch_size
        self.max_gates = max_gates

    def complete(self, system_prompt: str, user_prompt: str, temperature: float) -> LLMResponse:
        from llm_vqc.free_amplitude.sampler import sample_complete_candidate

        rng = _stable_rng(str(self.seed), system_prompt, user_prompt, str(temperature))
        candidates = [
            sample_complete_candidate(rng, self.n_qubits, self.max_gates).model_dump()
            for _ in range(self.batch_size)
        ]
        text = json.dumps({"candidates": candidates})
        return LLMResponse(
            raw_text=text, model=self.model_name,
            input_tokens=len(system_prompt) // 4, output_tokens=len(text) // 4,
            estimated_cost_usd=0.0, latency_seconds=0.0,
        )


#: Strict Structured-Outputs schemas for BATCH proposals (protocol §7:
#: 2-4 candidates per call; frozen at 3 in protocol_v2.yaml). Count and
#: wire-bound rules remain prompt-stated and validator-enforced, exactly
#: like the preserved single-candidate schema.

_LAYERED_OP_SCHEMA: dict = {
    "anyOf": [
        {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["rot"]},
                "gates": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["RX", "RY", "RZ", "H"]},
                },
                "wires": {
                    "anyOf": [
                        {"type": "string", "enum": ["all"]},
                        {"type": "array", "items": {"type": "integer"}},
                    ]
                },
            },
            "required": ["type", "gates", "wires"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["entangle"]},
                "pattern": {
                    "type": "string",
                    "enum": ["line", "ring", "star", "pairs", "all_to_all"],
                },
                "gate": {
                    "type": "string", "enum": ["CNOT", "CZ", "CRX", "CRY", "CRZ"],
                },
                "wires": {
                    "anyOf": [
                        {"type": "string", "enum": ["all"]},
                        {"type": "array", "items": {"type": "integer"}},
                    ]
                },
                "center": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                "pairs": {
                    "anyOf": [
                        {
                            "type": "array",
                            "items": {"type": "array", "items": {"type": "integer"}},
                        },
                        {"type": "null"},
                    ]
                },
            },
            "required": ["type", "pattern", "gate", "wires", "center", "pairs"],
            "additionalProperties": False,
        },
    ]
}

STRUCTURE_BATCH_STRICT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "operations": {"type": "array", "items": _LAYERED_OP_SCHEMA},
                },
                "required": ["operations"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["candidates"],
    "additionalProperties": False,
}


def _joint_batch_schema() -> dict:
    from llm_vqc.free_amplitude.openai_provider import COMPLETE_CANDIDATE_STRICT_SCHEMA

    return {
        "type": "object",
        "properties": {
            "candidates": {
                "type": "array",
                "items": COMPLETE_CANDIDATE_STRICT_SCHEMA,
            }
        },
        "required": ["candidates"],
        "additionalProperties": False,
    }


def build_real_provider(track: str, env: dict | None = None):
    """Preflight + construct the budget/retry-hardened real OpenAI
    provider stack with the track's strict batch response schema. Raises
    RealRunPreflightError before any client exists if the environment
    lacks the key, the explicit model id, or a positive USD cap."""
    import os

    from llm_vqc.free_amplitude.openai_provider import (
        BudgetEnforcedProvider,
        CompleteCandidateOpenAIProvider,
        RetryingProvider,
        preflight_real_run,
    )

    if track == "structure":
        schema, schema_name = STRUCTURE_BATCH_STRICT_SCHEMA, "structure_batch"
    elif track == "joint":
        schema, schema_name = _joint_batch_schema(), "joint_batch"
    else:
        raise ValueError(f"unknown track {track!r}")

    from llm_vqc.llm.openai_provider import CallCountLimitedProvider

    config = preflight_real_run(env if env is not None else dict(os.environ))
    inner = CompleteCandidateOpenAIProvider(
        api_key=config.api_key, model=config.model,
        response_schema=schema, response_schema_name=schema_name,
    )
    budgeted = BudgetEnforcedProvider(inner, config.budget)
    # Hard per-cell call cap: a 24-unique-budget cell at 3 candidates per
    # batch needs ~8-12 calls; 40 is generous headroom while making a
    # pathological loop impossible. Retries sit OUTSIDE both caps so every
    # retry is individually budget-checked and counted.
    capped = CallCountLimitedProvider(budgeted, max_calls=PER_CELL_CALL_CAP)
    return RetryingProvider(capped), config
