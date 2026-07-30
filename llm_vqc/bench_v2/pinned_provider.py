"""The pinned real provider for the bench_v2 real-LLM matrix.

Differences from the earlier `free_amplitude.openai_provider` path, all
required before spending real money:

* the model is **pinned** to `PINNED_MODEL`; the run refuses to start if
  `OPENAI_MODEL` disagrees, so a scientific cell can never silently run on
  a different snapshot;
* `temperature` is never sent. The previous provider sent it, caught the
  resulting 400 and re-issued the call — an automatic retry cycle that
  doubles latency and can double spend. gpt-5 reasoning models reject a
  non-default temperature, so the parameter is simply omitted;
* reasoning effort and verbosity are requested at their cheapest settings
  where the endpoint accepts them, and dropped automatically (once, at
  construction, not per call) if it does not;
* cost is charged to the durable global ledger from returned token counts.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from llm_vqc.llm.provider import LLMResponse

PINNED_MODEL = "gpt-5-nano-2025-08-07"
DEFAULT_MAX_OUTPUT_TOKENS = 4000
DEFAULT_REQUEST_TIMEOUT_S = 90.0
MIN_SECONDS_BETWEEN_CALLS = 0.4


class PinnedModelError(Exception):
    """Raised when the environment does not pin the expected model."""


@dataclass(frozen=True)
class RealRunConfig:
    api_key: str
    model: str


def preflight_pinned_run(env: dict[str, str] | None = None) -> RealRunConfig:
    """Validate the environment *before* any client exists.

    Checks the credential and the exact model pin. The spending cap is
    enforced separately by the global ledger, which the caller must build
    from `LLM_API_BUDGET_USD` — a key alone is never authorisation.
    """
    source = dict(env if env is not None else os.environ)
    api_key = (source.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise PinnedModelError("OPENAI_API_KEY is not set")
    model = (source.get("OPENAI_MODEL") or "").strip()
    if not model:
        raise PinnedModelError("OPENAI_MODEL is not set; the run refuses to guess a model")
    if model != PINNED_MODEL:
        raise PinnedModelError(
            f"OPENAI_MODEL is {model!r} but the bench_v2 real-LLM matrix is pinned to "
            f"{PINNED_MODEL!r}. Scientific cells must all use one snapshot."
        )
    return RealRunConfig(api_key=api_key, model=model)


class PinnedStructuredProvider:
    """One `openai.OpenAI` client, `max_retries=0`, Structured Outputs.

    `complete()` issues exactly one billable request. There is no
    temperature parameter and no 400-triggered retry.
    """

    def __init__(
        self,
        api_key: str,
        response_schema: dict,
        response_schema_name: str,
        model: str = PINNED_MODEL,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
        min_seconds_between_calls: float = MIN_SECONDS_BETWEEN_CALLS,
        reasoning_effort: str | None = "minimal",
        verbosity: str | None = "low",
    ) -> None:
        from openai import OpenAI

        self.model_name = model
        self.response_schema = response_schema
        self.response_schema_name = response_schema_name
        self.max_output_tokens = max_output_tokens
        self.request_timeout_s = request_timeout_s
        self.min_seconds_between_calls = min_seconds_between_calls
        self.reasoning_effort = reasoning_effort
        self.verbosity = verbosity
        self.outbound_attempts = 0
        self.dropped_parameters: list[str] = []
        self._last_call_at = 0.0
        self._client = OpenAI(api_key=api_key, timeout=request_timeout_s, max_retries=0)

    def _kwargs(self, system_prompt: str, user_prompt: str) -> dict:
        kwargs: dict = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_completion_tokens": self.max_output_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": self.response_schema_name,
                    "strict": True,
                    "schema": self.response_schema,
                },
            },
            "timeout": self.request_timeout_s,
        }
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.verbosity:
            kwargs["verbosity"] = self.verbosity
        return kwargs

    def complete(self, system_prompt: str, user_prompt: str, temperature: float) -> LLMResponse:
        """`temperature` is accepted for interface compatibility and
        deliberately NOT forwarded."""
        wait = self.min_seconds_between_calls - (time.monotonic() - self._last_call_at)
        if wait > 0:
            time.sleep(wait)
        self._last_call_at = time.monotonic()

        kwargs = self._kwargs(system_prompt, user_prompt)
        start = time.monotonic()
        self.outbound_attempts += 1
        response = self._client.chat.completions.create(**kwargs)
        latency = time.monotonic() - start

        usage = response.usage
        return LLMResponse(
            raw_text=response.choices[0].message.content or "",
            model=response.model or self.model_name,
            input_tokens=int(usage.prompt_tokens) if usage is not None else 0,
            output_tokens=int(usage.completion_tokens) if usage is not None else 0,
            estimated_cost_usd=None,   # priced by the ledger wrapper
            latency_seconds=latency,
        )

    def probe_optional_parameters(self, system_prompt: str, user_prompt: str) -> None:
        """Drop unsupported optional parameters once, at setup.

        Called only by the preflight, whose calls are themselves capped.
        This keeps the scientific run free of per-call 400/retry cycles.
        """
        from openai import BadRequestError

        for attr in ("reasoning_effort", "verbosity"):
            if getattr(self, attr) is None:
                continue
            try:
                self.complete(system_prompt, user_prompt, 1.0)
                return
            except BadRequestError as exc:
                if attr in str(exc):
                    self.dropped_parameters.append(attr)
                    setattr(self, attr, None)
                    continue
                raise
