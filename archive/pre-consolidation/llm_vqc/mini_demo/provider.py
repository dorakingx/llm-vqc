"""A real, network-calling OpenAI provider constrained to the compact
architecture JSON schema via Structured Outputs (`response_format`).

Parallel to `llm_vqc.llm.openai_provider.OpenAIProvider` (same
`LLMProvider`-shaped `complete()` interface, same "never fabricate cost,
never print/log the key" discipline) but request-shaped for this demo's
compact schema instead of full `CircuitIR`.

**Correction pass (section 8): zero retries.** The client is built with
`max_retries=0` and the provider itself makes exactly one outbound request
per `complete()` call -- no retry loop. For a four-proposal demo this makes
the four-logical-call cap simultaneously a four-*outbound-request* cap
(zero is within the "at most one retry" allowance and removes any
ambiguity about how many billable requests a run can issue). A network or
API error propagates to the caller, which records the failed logical call
honestly rather than silently re-issuing it.

Cost is always reported as `None` -- the OpenAI chat completions API never
returns a dollar figure, and this provider does not estimate one.
"""

from __future__ import annotations

import time

from openai import OpenAI

from llm_vqc.llm.provider import LLMResponse
from llm_vqc.mini_demo.compact_schema import COMPACT_ARCHITECTURE_JSON_SCHEMA

DEFAULT_MAX_OUTPUT_TOKENS = 200
DEFAULT_REQUEST_TIMEOUT_S = 45.0


class CompactSchemaOpenAIProvider:
    """Wraps one `openai.OpenAI` client built with `max_retries=0`.
    `complete()` makes exactly one real, billed chat-completion request per
    call (no retries), constrained to `COMPACT_ARCHITECTURE_JSON_SCHEMA` via
    strict Structured Outputs, with a hard per-request timeout.

    `outbound_attempts` counts every outbound request issued (== number of
    `complete()` calls, since there is no retry loop), so a caller can prove
    the outbound-request count never exceeds the logical-call cap."""

    def __init__(
        self,
        api_key: str,
        model: str,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
    ) -> None:
        self.model_name = model
        self.max_output_tokens = max_output_tokens
        self.request_timeout_s = request_timeout_s
        self.outbound_attempts = 0
        # max_retries=0: the OpenAI SDK will NOT transparently retry on its
        # own, so one complete() == exactly one billable outbound request.
        self._client = OpenAI(api_key=api_key, timeout=request_timeout_s, max_retries=0)

    def complete(self, system_prompt: str, user_prompt: str, temperature: float) -> LLMResponse:
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
                    "name": "compact_architecture",
                    "strict": True,
                    "schema": COMPACT_ARCHITECTURE_JSON_SCHEMA,
                },
            },
            timeout=self.request_timeout_s,
        )
        latency = time.monotonic() - start

        raw_text = response.choices[0].message.content or ""
        usage = response.usage
        input_tokens = int(usage.prompt_tokens) if usage is not None else 0
        output_tokens = int(usage.completion_tokens) if usage is not None else 0

        return LLMResponse(
            raw_text=raw_text,
            model=response.model or self.model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=None,
            latency_seconds=latency,
        )
