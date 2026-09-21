"""Token-based cost computation from a dated, committed price manifest.

Replaces the previous flat `$0.05 per call` reservation, which was a
placeholder: it neither reflected what a call actually costs nor gave the
operator a real spend figure. Costs are now computed from the token
counts the provider returns, priced by `configs/bench_v2/model_prices.json`.

An unknown model raises rather than defaulting to some invented rate — a
run that cannot be priced must not be charged to a cap.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO / "configs" / "bench_v2" / "model_prices.json"


class UnknownModelPrice(Exception):
    """Raised when a model has no entry in the price manifest."""


@dataclass(frozen=True)
class ModelPrice:
    model: str
    input_per_1m: float
    output_per_1m: float
    manifest_version: str

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_per_1m / 1_000_000.0
            + output_tokens * self.output_per_1m / 1_000_000.0
        )


@lru_cache(maxsize=8)
def _load(manifest_path: str) -> dict:
    return json.loads(Path(manifest_path).read_text())


def price_for(model: str, manifest_path: Path | None = None) -> ModelPrice:
    path = Path(manifest_path) if manifest_path else DEFAULT_MANIFEST
    doc = _load(str(path))
    entry = doc["models"].get(model)
    if entry is None:
        raise UnknownModelPrice(
            f"model {model!r} is not in {path.name}; add a dated price entry "
            "before running it — the ledger refuses to charge an unpriced model"
        )
    return ModelPrice(
        model=model,
        input_per_1m=float(entry["input_per_1m"]),
        output_per_1m=float(entry["output_per_1m"]),
        manifest_version=doc["manifest_version"],
    )


def estimate_call_cost(
    model: str,
    expected_input_tokens: int,
    expected_output_tokens: int,
    safety_factor: float = 2.0,
    manifest_path: Path | None = None,
) -> float:
    """Pessimistic pre-request estimate used for the ledger reservation.

    Deliberately generous (default 2x): a reservation that is too small
    could let cumulative spend slip past the cap between reserve and
    settle, while one that is too large only makes the guard stricter.
    """
    price = price_for(model, manifest_path)
    return price.cost_usd(expected_input_tokens, expected_output_tokens) * safety_factor
