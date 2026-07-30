"""Real and mock providers for the mini_5gate_v1 LLM arms.

The real one reuses the pinned, ledger-charged stack built for bench_v2:
the model snapshot is fixed, cost is reserved before each request and
settled from the returned token counts, and a request that would exceed
the cumulative cap is refused before it is issued.
"""

from __future__ import annotations

from llm_vqc.mini5.prompts import RESPONSE_SCHEMA


def build_mini5_provider(spend_ledger, cell_id: str | None = None):
    """One provider per cell, all sharing the one cumulative ledger."""
    from llm_vqc.bench_v2.pinned_provider import (
        PinnedStructuredProvider,
        preflight_pinned_run,
    )
    from llm_vqc.llm.ledger_provider import LedgerEnforcedProvider

    config = preflight_pinned_run()
    provider = PinnedStructuredProvider(
        api_key=config.api_key,
        response_schema=RESPONSE_SCHEMA,
        response_schema_name="mini5_circuit",
        model=config.model,
    )
    if spend_ledger is None:
        return provider
    # One request carries a ~350-token prompt and ~120 tokens of JSON, far
    # below the bench_v2 batch defaults; reserving those instead would
    # over-reserve tenfold and could refuse a run that fits the cap.
    return LedgerEnforcedProvider(
        provider, spend_ledger, cell_id=cell_id,
        expected_input_tokens=450, expected_output_tokens=200,
    )
