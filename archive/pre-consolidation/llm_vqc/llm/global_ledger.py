"""Durable, process-safe global spend ledger for paid LLM calls.

The previous `LLMApiBudget` was constructed per cell from the environment
and started every cell at `spent_usd = 0`. `LLM_API_BUDGET_USD` therefore
acted as a *per-cell* allowance: a 220-cell matrix could spend 220x the
number the operator thought they had authorised. This module replaces it
with one cumulative cap shared by every cell, every retry and every
process restart.

Design:

* One SQLite file is the single source of truth. Concurrent writers are
  serialised with `BEGIN IMMEDIATE`, so a reservation is atomic even if
  several cells run in parallel.
* Money moves in two steps. `reserve()` writes a pessimistic estimate
  *before* the request leaves the process, so a crash mid-request can
  never look free. `settle()` then replaces that estimate with the real
  cost computed from the provider's returned token counts.
* `reserve()` refuses when the reservation would push cumulative spend
  past the cap. The check and the insert happen inside one transaction,
  so two processes cannot both squeeze past the limit.
* Nothing here resets or raises a cap. Raising it is an operator action:
  pass a larger value at construction, which is recorded as a new cap row
  in the audit trail.

The per-cell call guard (`CallCountLimitedProvider`) is unrelated and
stays: it bounds runaway retry loops inside one cell. It is *not* the
global cap and must never be described as one.
"""

from __future__ import annotations

import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ledger_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS entries (
    entry_id       TEXT PRIMARY KEY,
    created_at     TEXT NOT NULL,
    settled_at     TEXT,
    run_label      TEXT NOT NULL,
    cell_id        TEXT,
    model          TEXT,
    state          TEXT NOT NULL,          -- reserved | settled | released
    estimated_usd  REAL NOT NULL,
    actual_usd     REAL,
    input_tokens   INTEGER,
    output_tokens  INTEGER
);
CREATE INDEX IF NOT EXISTS entries_state ON entries(state);
"""


class LedgerCapExceeded(Exception):
    """Raised before a request that would exceed the cumulative cap."""


@dataclass(frozen=True)
class LedgerTotals:
    committed_usd: float      # settled actuals + outstanding reservations
    settled_usd: float
    reserved_usd: float
    calls_settled: int
    calls_reserved: int
    input_tokens: int
    output_tokens: int

    @property
    def remaining_usd_for(self) -> float:  # pragma: no cover - helper
        raise NotImplementedError


class GlobalSpendLedger:
    """Cumulative USD cap shared by every cell and every process."""

    def __init__(self, db_path: str | Path, cap_usd: float, run_label: str = "bench_v2") -> None:
        if cap_usd <= 0:
            raise ValueError("cap_usd must be > 0")
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.cap_usd = float(cap_usd)
        self.run_label = run_label
        self._conn = sqlite3.connect(str(self.db_path), timeout=60.0, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=60000")
        self._conn.executescript(_SCHEMA)
        self._record_cap()

    # -- construction helpers -------------------------------------------------

    @classmethod
    def from_env(
        cls,
        db_path: str | Path,
        env: dict[str, str] | None = None,
        run_label: str = "bench_v2",
    ) -> GlobalSpendLedger | None:
        """Build from `LLM_API_BUDGET_USD`, or return None when unset,
        unparseable or non-positive. There is deliberately no other way to
        supply the cap from the environment."""
        source = env if env is not None else os.environ
        raw = source.get("LLM_API_BUDGET_USD")
        if raw is None or not raw.strip():
            return None
        try:
            value = float(raw)
        except ValueError:
            return None
        if value <= 0:
            return None
        return cls(db_path, cap_usd=value, run_label=run_label)

    def _record_cap(self) -> None:
        now = datetime.now(UTC).isoformat()
        row = self._conn.execute(
            "SELECT value FROM ledger_meta WHERE key='cap_usd'"
        ).fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO ledger_meta(key, value) VALUES('cap_usd', ?)",
                (repr(self.cap_usd),),
            )
            self._conn.execute(
                "INSERT INTO ledger_meta(key, value) VALUES('cap_history', ?)",
                (f"{now}={self.cap_usd}",),
            )
        elif float(row[0]) != self.cap_usd:
            # An operator changed the cap. Record it; never silently reset.
            history = self._conn.execute(
                "SELECT value FROM ledger_meta WHERE key='cap_history'"
            ).fetchone()
            self._conn.execute(
                "UPDATE ledger_meta SET value=? WHERE key='cap_usd'",
                (repr(self.cap_usd),),
            )
            self._conn.execute(
                "UPDATE ledger_meta SET value=? WHERE key='cap_history'",
                (f"{(history[0] if history else '')};{now}={self.cap_usd}",),
            )

    # -- accounting -----------------------------------------------------------

    def totals(self) -> LedgerTotals:
        row = self._conn.execute(
            """
            SELECT
              COALESCE(SUM(CASE WHEN state='settled'  THEN actual_usd    END), 0.0),
              COALESCE(SUM(CASE WHEN state='reserved' THEN estimated_usd END), 0.0),
              COALESCE(SUM(CASE WHEN state='settled'  THEN 1 ELSE 0 END), 0),
              COALESCE(SUM(CASE WHEN state='reserved' THEN 1 ELSE 0 END), 0),
              COALESCE(SUM(input_tokens), 0),
              COALESCE(SUM(output_tokens), 0)
            FROM entries
            """
        ).fetchone()
        settled, reserved, n_settled, n_reserved, tin, tout = row
        return LedgerTotals(
            committed_usd=float(settled) + float(reserved),
            settled_usd=float(settled),
            reserved_usd=float(reserved),
            calls_settled=int(n_settled),
            calls_reserved=int(n_reserved),
            input_tokens=int(tin),
            output_tokens=int(tout),
        )

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.cap_usd - self.totals().committed_usd)

    def reserve(self, estimated_usd: float, cell_id: str | None, model: str | None) -> str:
        """Atomically claim `estimated_usd` against the cap.

        Raises `LedgerCapExceeded` *before* any request is issued when the
        claim would take cumulative spend past the cap.
        """
        if estimated_usd < 0:
            raise ValueError("estimated_usd must be >= 0")
        entry_id = uuid.uuid4().hex
        cur = self._conn.cursor()
        cur.execute("BEGIN IMMEDIATE")
        try:
            row = cur.execute(
                """
                SELECT COALESCE(SUM(CASE WHEN state='settled'  THEN actual_usd    END), 0.0)
                     + COALESCE(SUM(CASE WHEN state='reserved' THEN estimated_usd END), 0.0)
                FROM entries
                """
            ).fetchone()
            committed = float(row[0])
            if committed + estimated_usd > self.cap_usd + 1e-12:
                raise LedgerCapExceeded(
                    f"reservation ${estimated_usd:.6f} would take cumulative spend to "
                    f"${committed + estimated_usd:.6f}, past the ${self.cap_usd:.2f} cap "
                    f"(already committed ${committed:.6f} over "
                    f"{self.totals().calls_settled} settled calls)"
                )
            cur.execute(
                "INSERT INTO entries(entry_id, created_at, run_label, cell_id, model, "
                "state, estimated_usd) VALUES(?,?,?,?,?, 'reserved', ?)",
                (entry_id, datetime.now(UTC).isoformat(), self.run_label,
                 cell_id, model, float(estimated_usd)),
            )
            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise
        return entry_id

    def settle(
        self, entry_id: str, actual_usd: float, input_tokens: int, output_tokens: int
    ) -> None:
        """Replace a reservation with the real cost from returned tokens."""
        cur = self._conn.cursor()
        cur.execute("BEGIN IMMEDIATE")
        try:
            cur.execute(
                "UPDATE entries SET state='settled', actual_usd=?, input_tokens=?, "
                "output_tokens=?, settled_at=? WHERE entry_id=? AND state='reserved'",
                (float(actual_usd), int(input_tokens), int(output_tokens),
                 datetime.now(UTC).isoformat(), entry_id),
            )
            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise

    def release(self, entry_id: str) -> None:
        """Drop a reservation whose request never reached the provider."""
        cur = self._conn.cursor()
        cur.execute("BEGIN IMMEDIATE")
        try:
            cur.execute(
                "UPDATE entries SET state='released', actual_usd=0.0, settled_at=? "
                "WHERE entry_id=? AND state='reserved'",
                (datetime.now(UTC).isoformat(), entry_id),
            )
            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise

    def summary(self) -> dict:
        t = self.totals()
        return {
            "ledger_path": str(self.db_path),
            "cap_usd": self.cap_usd,
            "committed_usd": round(t.committed_usd, 6),
            "settled_usd": round(t.settled_usd, 6),
            "outstanding_reserved_usd": round(t.reserved_usd, 6),
            "remaining_usd": round(max(0.0, self.cap_usd - t.committed_usd), 6),
            "calls_settled": t.calls_settled,
            "calls_outstanding": t.calls_reserved,
            "input_tokens": t.input_tokens,
            "output_tokens": t.output_tokens,
        }

    def close(self) -> None:
        self._conn.close()
