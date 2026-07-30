"""Provider wrapper that charges the durable global ledger.

Order of operations for one request:

1. reserve a pessimistic estimate against the cumulative cap;
2. issue the request only if the reservation succeeded;
3. settle the reservation with the real cost from the returned token
   counts;
4. release the reservation if the request never produced a response, so a
   provider error does not permanently consume budget.

`LedgerCapExceeded` propagates out of step 1 untouched: the runner turns
it into a clean stop, and it must never be swallowed by a retry wrapper.
"""

from __future__ import annotations

from llm_vqc.llm.global_ledger import GlobalSpendLedger, LedgerCapExceeded
from llm_vqc.llm.pricing import estimate_call_cost, price_for
from llm_vqc.llm.provider import LLMResponse

#: Token expectations for one bench_v2 proposal call, used only to size
#: the pre-request reservation. Measured against the real prompts: the
#: structure system prompt is ~400 tokens, the closed-loop archive adds a
#: few hundred, and a 3-candidate batch reply is a few hundred more.
EXPECTED_INPUT_TOKENS = 1200
EXPECTED_OUTPUT_TOKENS = 900


class LedgerEnforcedProvider:
    """Wraps a provider so every call is priced and capped globally."""

    def __init__(
        self,
        inner,
        ledger: GlobalSpendLedger,
        cell_id: str | None = None,
        expected_input_tokens: int = EXPECTED_INPUT_TOKENS,
        expected_output_tokens: int = EXPECTED_OUTPUT_TOKENS,
    ) -> None:
        self.model_name = inner.model_name
        self._inner = inner
        self._ledger = ledger
        self._cell_id = cell_id
        # Fail fast if the pinned model has no committed price.
        self._price = price_for(self.model_name)
        self._estimate = estimate_call_cost(
            self.model_name, expected_input_tokens, expected_output_tokens
        )
        self.settled_usd = 0.0
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def complete(self, system_prompt: str, user_prompt: str, temperature: float) -> LLMResponse:
        entry_id = self._ledger.reserve(self._estimate, self._cell_id, self.model_name)
        try:
            response = self._inner.complete(system_prompt, user_prompt, temperature)
        except Exception:
            self._ledger.release(entry_id)
            raise

        actual = self._price.cost_usd(response.input_tokens, response.output_tokens)
        self._ledger.settle(entry_id, actual, response.input_tokens, response.output_tokens)
        self.settled_usd += actual
        self.calls += 1
        self.input_tokens += response.input_tokens
        self.output_tokens += response.output_tokens
        # Report the true cost on the response so call records carry it.
        return LLMResponse(
            raw_text=response.raw_text,
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            estimated_cost_usd=actual,
            latency_seconds=response.latency_seconds,
        )

    def usage(self) -> dict:
        return {
            "model": self.model_name,
            "price_manifest": self._price.manifest_version,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "actual_usd": round(self.settled_usd, 6),
        }


__all__ = ["LedgerEnforcedProvider", "LedgerCapExceeded"]
