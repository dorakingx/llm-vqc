"""The cap must be cumulative across cells, retries and process restarts.

The bug this replaces: `LLMApiBudget` was rebuilt per cell from the
environment and started each cell at `spent_usd = 0`, so
`LLM_API_BUDGET_USD` behaved as a per-cell allowance and a 220-cell
matrix could spend 220x the authorised amount. These tests pin the fixed
behaviour, including the restart case, which is what makes the ledger
durable rather than in-memory.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from llm_vqc.llm.global_ledger import GlobalSpendLedger, LedgerCapExceeded
from llm_vqc.llm.ledger_provider import LedgerEnforcedProvider
from llm_vqc.llm.pricing import UnknownModelPrice, estimate_call_cost, price_for
from llm_vqc.llm.provider import LLMResponse

MODEL = "gpt-5-nano-2025-08-07"


class FakeProvider:
    """Returns fixed token counts so cost is exactly predictable."""

    def __init__(self, model: str = MODEL, in_tok: int = 1000, out_tok: int = 1000) -> None:
        self.model_name = model
        self.in_tok = in_tok
        self.out_tok = out_tok
        self.calls = 0

    def complete(self, system_prompt, user_prompt, temperature):
        self.calls += 1
        return LLMResponse(
            raw_text="{}", model=self.model_name,
            input_tokens=self.in_tok, output_tokens=self.out_tok,
            estimated_cost_usd=None, latency_seconds=0.0,
        )


class ExplodingProvider:
    model_name = MODEL

    def complete(self, *a, **k):
        raise RuntimeError("network died")


# --------------------------------------------------------------- pricing --

def test_price_comes_from_the_committed_manifest():
    price = price_for(MODEL)
    assert price.input_per_1m == 0.05
    assert price.output_per_1m == 0.40
    # 1M in + 1M out = $0.05 + $0.40
    assert price.cost_usd(1_000_000, 1_000_000) == pytest.approx(0.45)


def test_unknown_model_refuses_rather_than_inventing_a_rate():
    with pytest.raises(UnknownModelPrice):
        price_for("some-model-nobody-priced")


def test_reservation_estimate_is_pessimistic():
    price = price_for(MODEL)
    exact = price.cost_usd(1200, 900)
    assert estimate_call_cost(MODEL, 1200, 900) > exact


# ---------------------------------------------------------------- ledger --

def test_two_cells_share_one_cumulative_spend(tmp_path: Path):
    """The core regression: separate cells must not each get a full cap."""
    db = tmp_path / "ledger.sqlite"
    ledger = GlobalSpendLedger(db, cap_usd=1.0)

    cell_a = LedgerEnforcedProvider(FakeProvider(), ledger, cell_id="cell_a")
    cell_b = LedgerEnforcedProvider(FakeProvider(), ledger, cell_id="cell_b")
    cell_a.complete("s", "u", 1.0)
    cell_b.complete("s", "u", 1.0)

    per_call = price_for(MODEL).cost_usd(1000, 1000)
    totals = ledger.totals()
    assert totals.calls_settled == 2
    # Cumulative, not per cell.
    assert totals.settled_usd == pytest.approx(2 * per_call)
    assert totals.input_tokens == 2000
    assert totals.output_tokens == 2000


def test_restarted_process_continues_the_same_ledger(tmp_path: Path):
    """A fresh GlobalSpendLedger on the same file resumes the running total."""
    db = tmp_path / "ledger.sqlite"
    first = GlobalSpendLedger(db, cap_usd=1.0)
    LedgerEnforcedProvider(FakeProvider(), first, cell_id="cell_a").complete("s", "u", 1.0)
    spent_before = first.totals().settled_usd
    first.close()

    resumed = GlobalSpendLedger(db, cap_usd=1.0)          # simulates a restart
    assert resumed.totals().settled_usd == pytest.approx(spent_before)
    LedgerEnforcedProvider(FakeProvider(), resumed, cell_id="cell_b").complete("s", "u", 1.0)
    assert resumed.totals().calls_settled == 2
    assert resumed.totals().settled_usd == pytest.approx(2 * spent_before)


def test_restart_in_a_separate_os_process_shares_the_total(tmp_path: Path):
    """Not just a new object — a genuinely separate interpreter."""
    db = tmp_path / "ledger.sqlite"
    ledger = GlobalSpendLedger(db, cap_usd=1.0)
    LedgerEnforcedProvider(FakeProvider(), ledger, cell_id="cell_a").complete("s", "u", 1.0)
    ledger.close()

    script = textwrap.dedent(f"""
        import json, sys
        sys.path.insert(0, {str(Path.cwd())!r})
        from llm_vqc.llm.global_ledger import GlobalSpendLedger
        led = GlobalSpendLedger({str(db)!r}, cap_usd=1.0)
        print(json.dumps(led.summary()))
    """)
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                         check=True).stdout
    summary = json.loads(out)
    assert summary["calls_settled"] == 1
    assert summary["settled_usd"] > 0


def test_cap_refuses_before_the_request_is_issued(tmp_path: Path):
    db = tmp_path / "ledger.sqlite"
    # Cap smaller than a single reservation.
    ledger = GlobalSpendLedger(db, cap_usd=1e-9)
    provider = FakeProvider()
    wrapped = LedgerEnforcedProvider(provider, ledger, cell_id="cell_a")
    with pytest.raises(LedgerCapExceeded):
        wrapped.complete("s", "u", 1.0)
    assert provider.calls == 0, "the request must not reach the provider"


def test_cap_binds_after_accumulating_across_cells(tmp_path: Path):
    db = tmp_path / "ledger.sqlite"
    per_call_estimate = estimate_call_cost(MODEL, 1200, 900)
    ledger = GlobalSpendLedger(db, cap_usd=per_call_estimate * 2.5)
    calls_made = 0
    for i in range(10):
        provider = LedgerEnforcedProvider(FakeProvider(), ledger, cell_id=f"cell_{i}")
        try:
            provider.complete("s", "u", 1.0)
            calls_made += 1
        except LedgerCapExceeded:
            break
    assert 1 <= calls_made <= 3, calls_made
    assert ledger.totals().committed_usd <= ledger.cap_usd


def test_failed_request_releases_its_reservation(tmp_path: Path):
    db = tmp_path / "ledger.sqlite"
    ledger = GlobalSpendLedger(db, cap_usd=1.0)
    wrapped = LedgerEnforcedProvider(ExplodingProvider(), ledger, cell_id="cell_a")
    with pytest.raises(RuntimeError):
        wrapped.complete("s", "u", 1.0)
    totals = ledger.totals()
    assert totals.reserved_usd == pytest.approx(0.0)
    assert totals.committed_usd == pytest.approx(0.0)


def test_settlement_uses_returned_tokens_not_a_flat_fee(tmp_path: Path):
    db = tmp_path / "ledger.sqlite"
    ledger = GlobalSpendLedger(db, cap_usd=1.0)
    small = LedgerEnforcedProvider(FakeProvider(in_tok=100, out_tok=100), ledger, "a")
    big = LedgerEnforcedProvider(FakeProvider(in_tok=10000, out_tok=10000), ledger, "b")
    small.complete("s", "u", 1.0)
    cost_small = ledger.totals().settled_usd
    big.complete("s", "u", 1.0)
    cost_big = ledger.totals().settled_usd - cost_small
    assert cost_big == pytest.approx(100 * cost_small)   # 100x tokens, 100x cost


def test_from_env_requires_a_positive_cap(tmp_path: Path):
    db = tmp_path / "ledger.sqlite"
    for env in ({}, {"LLM_API_BUDGET_USD": ""}, {"LLM_API_BUDGET_USD": "0"},
                {"LLM_API_BUDGET_USD": "-5"}, {"LLM_API_BUDGET_USD": "abc"}):
        assert GlobalSpendLedger.from_env(db, env=env) is None
    ledger = GlobalSpendLedger.from_env(db, env={"LLM_API_BUDGET_USD": "2.00"})
    assert ledger is not None and ledger.cap_usd == 2.0


def test_cap_change_is_recorded_not_silently_applied(tmp_path: Path):
    db = tmp_path / "ledger.sqlite"
    GlobalSpendLedger(db, cap_usd=1.0).close()
    ledger = GlobalSpendLedger(db, cap_usd=5.0)
    history = ledger._conn.execute(
        "SELECT value FROM ledger_meta WHERE key='cap_history'").fetchone()[0]
    assert "1.0" in history and "5.0" in history


def test_summary_reports_the_operator_facing_numbers(tmp_path: Path):
    db = tmp_path / "ledger.sqlite"
    ledger = GlobalSpendLedger(db, cap_usd=2.0)
    LedgerEnforcedProvider(FakeProvider(), ledger, "cell_a").complete("s", "u", 1.0)
    s = ledger.summary()
    for key in ("cap_usd", "settled_usd", "committed_usd", "remaining_usd",
                "calls_settled", "input_tokens", "output_tokens"):
        assert key in s
    assert s["remaining_usd"] == pytest.approx(2.0 - s["committed_usd"])
