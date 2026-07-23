"""Offline, mock/scripted LLM providers for the free-gate experiment.

**No real or billed API call is made by anything in this module.** This
implementation pass uses only `ScriptedFreeGateProvider` (tests: exact,
pre-authored responses in order) and `MockFreeGateProvider` (the smoke
run: deterministic, seed-derived, offline, zero-cost -- the free-gate
analog of `llm_vqc.llm.provider.MockLLMProvider`). A real
Structured-Outputs-constrained provider (mirroring `llm_vqc.mini_demo.
provider.CompactSchemaOpenAIProvider`) is a small, separate addition for a
future, explicitly-authorized real-API pass; it is intentionally NOT
implemented here so this module cannot make a paid call by import side
effect alone.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from llm_vqc.free_amplitude.sampler import sample_free_gate_proposal
from llm_vqc.llm.provider import LLMResponse


def _stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


class ScriptedFreeGateProvider:
    """Returns one pre-authored JSON string per call, in order -- for
    tests that need to control exactly what the "LLM" proposes."""

    model_name = "scripted-free-gate-provider"

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.calls_made = 0

    def complete(self, system_prompt: str, user_prompt: str, temperature: float) -> LLMResponse:
        text = self._responses[self.calls_made]
        self.calls_made += 1
        return LLMResponse(
            raw_text=text, model=self.model_name,
            input_tokens=len(system_prompt) // 4, output_tokens=len(text) // 4,
            estimated_cost_usd=0.0, latency_seconds=0.0,
        )


class MockFreeGateProvider:
    """Deterministic, offline, zero-cost stand-in -- a pure function of
    `(seed, system_prompt, user_prompt, temperature)`, not an internal call
    counter (same rationale as `llm_vqc.llm.provider.MockLLMProvider`: a
    counter would break resume determinism). Samples a valid free-gate
    proposal via the identical `llm_vqc.free_amplitude.sampler` Random
    search uses, so the mock "guesses" the way a schema-constrained LLM
    might, without importing any real model or making any request."""

    model_name = "mock-free-gate-provider-v1"

    def __init__(self, seed: int, n_qubits: int, max_gates: int = 5) -> None:
        self.seed = seed
        self.n_qubits = n_qubits
        self.max_gates = max_gates

    def complete(self, system_prompt: str, user_prompt: str, temperature: float) -> LLMResponse:
        stable = _stable_seed(str(self.seed), system_prompt, user_prompt, str(temperature))
        rng = np.random.default_rng(stable)
        proposal = sample_free_gate_proposal(rng, self.n_qubits, self.max_gates)
        raw_text = json.dumps(proposal.model_dump())
        return LLMResponse(
            raw_text=raw_text, model=self.model_name,
            input_tokens=len(system_prompt), output_tokens=len(raw_text),
            estimated_cost_usd=0.0, latency_seconds=0.0,
        )
