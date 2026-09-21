"""Cell completion manifests — the durable records the goal checker
(C06/C07/C08/C09/C17) verifies. One JSON per (experiment, cell), written
atomically next to the cell's SQLite store, only when the cell truly
completes; a crashed or interrupted run leaves either no manifest or a
`status != "complete"` marker, which the checker treats as incomplete.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path


def cell_id(task: str, n_qubits: int, arm: str, replicate: int) -> str:
    return f"{task}_{n_qubits}q_{arm}_r{replicate}"


def write_cell_manifest(
    runs_root: Path,
    experiment: str,
    *,
    task: str,
    n_qubits: int,
    arm: str,
    replicate: int,
    data_seed: int,
    search_seed: int,
    budget_unique: int,
    ledger_summary: dict,
    selected: dict,
    selection_timestamp: str,
    test_gate: dict,
    config_hash: str,
    git_sha: str | None,
    git_dirty: bool | None,
    store_path: str,
    llm: dict | None = None,
    status: str = "complete",
    extra: dict | None = None,
) -> Path:
    cells_dir = runs_root / experiment / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    cid = cell_id(task, n_qubits, arm, replicate)
    manifest = {
        "cell_id": cid,
        "experiment": experiment,
        "task": task,
        "n_qubits": n_qubits,
        "arm": arm,
        "replicate": replicate,
        "data_seed": data_seed,
        "search_seed": search_seed,
        "budget_unique": budget_unique,
        "ledger": ledger_summary,
        "selected": selected,
        "selection_timestamp": selection_timestamp,
        "test_gate": test_gate,
        "config_hash": config_hash,
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "store_path": store_path,
        "llm": llm,
        "status": status,
        "written_at": datetime.now(UTC).isoformat(),
    }
    if extra:
        manifest.update(extra)
    path = cells_dir / f"{cid}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    tmp.replace(path)  # atomic on POSIX
    return path


def read_cell_manifest(runs_root: Path, experiment: str, cid: str) -> dict | None:
    path = runs_root / experiment / "cells" / f"{cid}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())
