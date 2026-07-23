"""Real, network-calling OpenAI provider for MAIN-MODE complete candidates
(structure + theta), plus the preflight/budget guardrails a real run must
pass before any request is issued.

This is the ONLY module in `llm_vqc.free_amplitude` that imports the
OpenAI SDK, and it is imported by nothing else in the package -- the
mock/scripted paths (`provider.py`) and all tests stay structurally
offline. Guardrails mirror the corrected `llm_vqc.mini_demo` real-API
policy:

  - `preflight_real_run` requires `OPENAI_API_KEY`, an explicit
    `OPENAI_MODEL` (no silent default), and a positive
    `LLM_API_BUDGET_USD` (via the sanctioned `LLMApiBudget.from_env`) --
    all BEFORE any client is constructed;
  - the SDK client is built with `max_retries=0` and a hard per-request
    timeout, and `complete()` has no retry loop, so one logical call is
    exactly one billable outbound request;
  - `BudgetEnforcedProvider` charges a conservative per-call estimate
    against the dollar cap BEFORE each request (the chat completions API
    never returns a real dollar figure; the estimate is never presented
    as an actual cost);
  - callers additionally wrap this in `llm_vqc.llm.openai_provider.
    CallCountLimitedProvider` for a hard call-count cap.

Cost/latency provenance: token counts and latency are recorded on the
returned `LLMResponse` and flow into `candidate_trace.csv`; raw prompts
and raw responses are never persisted to any committed artifact.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass

from openai import OpenAI

from llm_vqc.llm.budget import LLMApiBudget
from llm_vqc.llm.provider import LLMProvider, LLMResponse

DEFAULT_MAX_OUTPUT_TOKENS = 300
DEFAULT_REQUEST_TIMEOUT_S = 45.0
#: Conservative per-call charge against the dollar cap -- far above the
#: true cost of a ~1500-token mini-model call, so the cap can only bind
#: early, never silently overrun.
COST_ESTIMATE_PER_CALL_USD = 0.05

#: Strict Structured-Outputs schema for a complete candidate. Kept to the
#: keyword subset OpenAI strict mode reliably supports (type / properties /
#: required / additionalProperties / enum / anyOf / items); count and
#: wire-bound rules are stated in the prompt and enforced by
#: `validate_complete_candidate` -- a violating response becomes an
#: ordinary INVALID proposal, never a crash.
COMPLETE_CANDIDATE_STRICT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "n_qubits": {"type": "integer"},
        "operations": {
            "type": "array",
            "items": {
                "anyOf": [
                    {
                        "type": "object",
                        "properties": {
                            "gate": {"type": "string", "enum": ["H"]},
                            "wires": {"type": "array", "items": {"type": "integer"}},
                        },
                        "required": ["gate", "wires"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "gate": {"type": "string", "enum": ["RX", "RY", "RZ"]},
                            "wires": {"type": "array", "items": {"type": "integer"}},
                            "theta": {"type": "number"},
                        },
                        "required": ["gate", "wires", "theta"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "gate": {"type": "string", "enum": ["CRX", "CRY", "CRZ"]},
                            "wires": {"type": "array", "items": {"type": "integer"}},
                            "theta": {"type": "number"},
                        },
                        "required": ["gate", "wires", "theta"],
                        "additionalProperties": False,
                    },
                ]
            },
        },
    },
    "required": ["n_qubits", "operations"],
    "additionalProperties": False,
}


class RealRunPreflightError(Exception):
    """Raised when a required environment precondition for a real run is
    missing -- blocks execution before any provider request."""


@dataclass
class RealRunConfig:
    api_key: str
    model: str
    budget: LLMApiBudget


def preflight_real_run(env: Mapping[str, str]) -> RealRunConfig:
    """Validate the environment for a real billed run. Constructs no
    client, issues no request, never prints or returns the key elsewhere."""
    api_key = (env.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RealRunPreflightError("OPENAI_API_KEY is not set")
    model = (env.get("OPENAI_MODEL") or "").strip()
    if not model:
        raise RealRunPreflightError(
            "OPENAI_MODEL is not set (a real run refuses to silently default to a model id)"
        )
    budget = LLMApiBudget.from_env(dict(env))
    if budget is None:
        raise RealRunPreflightError(
            "a positive LLM_API_BUDGET_USD must be set (external paid API calls are "
            "blocked without an explicit hard cap)"
        )
    return RealRunConfig(api_key=api_key, model=model, budget=budget)


class CompleteCandidateOpenAIProvider:
    """One `openai.OpenAI` client with `max_retries=0`; `complete()` makes
    exactly one real, billed chat-completion request, constrained to
    `COMPLETE_CANDIDATE_STRICT_SCHEMA` via strict Structured Outputs."""

    def __init__(
        self,
        api_key: str,
        model: str,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
        min_seconds_between_calls: float = 3.0,
    ) -> None:
        self.model_name = model
        self.max_output_tokens = max_output_tokens
        self.request_timeout_s = request_timeout_s
        # Client-side pacing: back-to-back requests at 25+ calls tripped a
        # rate limit in the seeds=5 first attempt; spacing requests keeps a
        # sustained run under typical RPM windows instead of relying on
        # retries after the fact.
        self.min_seconds_between_calls = min_seconds_between_calls
        self._last_call_at = 0.0
        self.outbound_attempts = 0
        self._client = OpenAI(api_key=api_key, timeout=request_timeout_s, max_retries=0)

    def complete(self, system_prompt: str, user_prompt: str, temperature: float) -> LLMResponse:
        pace_wait = self.min_seconds_between_calls - (time.monotonic() - self._last_call_at)
        if pace_wait > 0:
            time.sleep(pace_wait)
        self._last_call_at = time.monotonic()
        start = time.monotonic()
        self.outbound_attempts += 1
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_completion_tokens=self.max_output_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "complete_candidate",
                    "strict": True,
                    "schema": COMPLETE_CANDIDATE_STRICT_SCHEMA,
                },
            },
            timeout=self.request_timeout_s,
        )
        latency = time.monotonic() - start

        raw_text = response.choices[0].message.content or ""
        usage = response.usage
        return LLMResponse(
            raw_text=raw_text,
            model=response.model or self.model_name,
            input_tokens=int(usage.prompt_tokens) if usage is not None else 0,
            output_tokens=int(usage.completion_tokens) if usage is not None else 0,
            estimated_cost_usd=None,  # the API returns no dollar figure; never fabricated
            latency_seconds=latency,
        )


class BudgetEnforcedProvider:
    """Wraps any provider; checks the dollar cap BEFORE every call and
    records the conservative estimate after. `LLMBudgetExceededError`
    propagates to the runner, which records an honest stop_reason."""

    def __init__(
        self, inner: LLMProvider, budget: LLMApiBudget,
        cost_estimate_per_call_usd: float = COST_ESTIMATE_PER_CALL_USD,
    ) -> None:
        self.model_name = inner.model_name
        self._inner = inner
        self._budget = budget
        self._estimate = cost_estimate_per_call_usd

    def complete(self, system_prompt: str, user_prompt: str, temperature: float) -> LLMResponse:
        self._budget.check_can_afford(self._estimate)
        response = self._inner.complete(system_prompt, user_prompt, temperature)
        self._budget.record_spend(
            response.estimated_cost_usd
            if response.estimated_cost_usd is not None
            else self._estimate
        )
        return response


#: Manual-retry policy (user-requested): up to two retries per logical
#: call on a TRANSIENT failure. Rate-limit errors get a LONG linear
#: backoff (an RPM/TPM window needs tens of seconds to clear -- a 2s nap
#: does nothing, as the seeds=5 first attempt demonstrated); other
#: transients get a short one. Cap violations (call-count /
#: dollar-budget) are never retried -- they are limits, not transients.
DEFAULT_MAX_MANUAL_RETRIES = 2
DEFAULT_RETRY_BACKOFF_SECONDS = 3.0
DEFAULT_RATE_LIMIT_BACKOFF_SECONDS = 15.0


def _is_rate_limit_error(exc: Exception) -> bool:
    name = type(exc).__name__
    return "RateLimit" in name or "429" in str(exc)


class RetryingProvider:
    """Wraps a provider chain and retries a failed logical call up to
    `max_manual_retries` times.

    Placement matters: this sits OUTSIDE the budget/call-count wrappers
    (`RetryingProvider(CallCountLimited(BudgetEnforced(real)))`), so every
    retry attempt is itself individually budget-checked and counted
    against the hard call cap -- a retry can never bypass either limit.
    `CallBudgetExceededError` / `LLMBudgetExceededError` propagate
    immediately without a retry. The SDK client itself keeps
    `max_retries=0`, so all retrying is visible at this one layer and
    `retried_logical_calls` is an exact audit count."""

    def __init__(
        self,
        inner: LLMProvider,
        max_manual_retries: int = DEFAULT_MAX_MANUAL_RETRIES,
        backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
        rate_limit_backoff_seconds: float = DEFAULT_RATE_LIMIT_BACKOFF_SECONDS,
    ) -> None:
        self.model_name = inner.model_name
        self._inner = inner
        self.max_manual_retries = max_manual_retries
        self.backoff_seconds = backoff_seconds
        self.rate_limit_backoff_seconds = rate_limit_backoff_seconds
        self.retried_logical_calls = 0

    def complete(self, system_prompt: str, user_prompt: str, temperature: float) -> LLMResponse:
        from llm_vqc.llm.budget import LLMBudgetExceededError
        from llm_vqc.llm.openai_provider import CallBudgetExceededError

        last_exc: Exception | None = None
        for attempt in range(self.max_manual_retries + 1):
            try:
                return self._inner.complete(system_prompt, user_prompt, temperature)
            except (CallBudgetExceededError, LLMBudgetExceededError):
                raise  # caps are limits, not transients -- never retried
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_manual_retries:
                    self.retried_logical_calls += 1
                    if _is_rate_limit_error(exc):
                        # Linear ramp: 15s, then 30s -- enough for an RPM
                        # window to clear before the next attempt.
                        time.sleep(self.rate_limit_backoff_seconds * (attempt + 1))
                    else:
                        time.sleep(self.backoff_seconds)
        assert last_exc is not None
        raise last_exc
