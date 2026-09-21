#!/usr/bin/env python
"""bench_v2 experiment orchestrator: enumerate the frozen protocol matrix
for one experiment and run every cell to completion, resumably.

Examples:
    # Offline smoke of every mandatory arm (mock LLM providers):
    python scripts/bench_v2/run_experiment.py --experiment E2 --smoke --mock

    # 3-replicate pilot (replicates 0..2), classical arms only:
    python scripts/bench_v2/run_experiment.py --experiment E2 --pilot \
        --arms random_structure evolutionary_structure greedy_growth

    # Full matrix (real LLM cells require OPENAI_API_KEY + OPENAI_MODEL +
    # positive LLM_API_BUDGET_USD; refused otherwise):
    python scripts/bench_v2/run_experiment.py --experiment E1

Every completed cell writes runs/bench_v2/<EID>/cells/<cell>.json; a
re-invocation skips complete cells and resumes partial ones, so an
interrupted matrix continues rather than restarting.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from llm_vqc.bench_v2.cell_runner import CellSpec, run_cell  # noqa: E402
from llm_vqc.bench_v2.llm_arms import BATCH_SIZE  # noqa: E402
from llm_vqc.bench_v2.llm_providers import (  # noqa: E402
    MockJointBatchProvider,
    MockLayeredBatchProvider,
    build_real_provider,
)
from llm_vqc.bench_v2.space import SpaceProfile  # noqa: E402
from llm_vqc.evaluation.seeds import derive_child_seed  # noqa: E402
from llm_vqc.free_amplitude.training import FreeAmplitudeTrainingConfig  # noqa: E402
from llm_vqc.llm.global_ledger import GlobalSpendLedger, LedgerCapExceeded  # noqa: E402

PROTOCOL_PATH = REPO / "configs" / "bench_v2" / "protocol_v2.yaml"
RUNS_ROOT = REPO / "runs" / "bench_v2"

#: Sizing-gate override file: written ONCE at the Phase 6 gate (before the
#: main matrix) according to the prewritten rule; absent -> provisional
#: protocol values are used.
SIZING_PATH = REPO / "configs" / "bench_v2" / "sizing_freeze.json"
#: One cumulative ledger for the whole matrix, shared by every cell.
LEDGER_PATH = RUNS_ROOT / "spend_ledger.sqlite"


def load_protocol() -> dict:
    return yaml.safe_load(PROTOCOL_PATH.read_text())


def training_config(protocol: dict, smoke: bool) -> FreeAmplitudeTrainingConfig:
    if smoke:
        return FreeAmplitudeTrainingConfig(epochs=2, batch_size=32)
    track_a = protocol["track_a"]
    epochs = int(track_a["epochs_provisional"])
    batch = int(track_a["batch_size_provisional"])
    if SIZING_PATH.is_file():
        frozen = json.loads(SIZING_PATH.read_text())
        epochs = int(frozen.get("epochs", epochs))
        batch = int(frozen.get("batch_size", batch))
    return FreeAmplitudeTrainingConfig(epochs=epochs, batch_size=batch)


def resolve_reference_for_e3(task_key: str) -> str:
    """Frozen rule: the fixed reference with the best MEDIAN validation
    metric across E2 replicates for the same task family (validation-only
    choice; protocol §10)."""
    import statistics

    cells_dir = RUNS_ROOT / "E2" / "cells"
    best_arm, best_median = None, None
    for ref in ("ref_realamp_d1", "ref_realamp_d2", "ref_strongent_d1", "ref_strongent_d2"):
        values = []
        for manifest_path in sorted(cells_dir.glob(f"{task_key}_5q_{ref}_r*.json")):
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("status") == "complete":
                value = manifest["selected"].get("val_metric_value")
                if value is not None:
                    values.append(float(value))
        if values:
            median = statistics.median(values)
            if best_median is None or median < best_median:
                best_arm, best_median = ref, median
    if best_arm is None:
        raise SystemExit(
            f"E3 needs E2 reference cells for task {task_key!r} to resolve "
            "ref_best_validation — run E2 first."
        )
    return best_arm


def enumerate_cells(
    protocol: dict, eid: str, *, smoke: bool, pilot: bool,
    arms_filter: list[str] | None, tasks_filter: list[str] | None,
) -> list[CellSpec]:
    exp = protocol["experiments"][eid]
    if "arms" not in exp or ("tasks" not in exp and "task_n_pairs" not in exp):
        raise SystemExit(f"{eid} is not a matrix experiment (use its dedicated script)")
    budget_key = "budget_unique" if "budget_unique" in exp else "budget"
    budget = int(exp[budget_key])
    replicates = int(exp["replicates"])
    if smoke:
        budget, replicates = min(budget, 2), 1
    elif pilot:
        replicates = min(replicates, 3)

    if "task_n_pairs" in exp:
        task_n = [(task, int(n)) for task, n in exp["task_n_pairs"]]
    else:
        task_n = [(task, int(n)) for task in exp["tasks"] for n in exp["n_qubits"]]

    config = training_config(protocol, smoke)
    specs: list[CellSpec] = []
    for task_key, n in task_n:
        if tasks_filter and task_key not in tasks_filter:
            continue
        for arm in exp["arms"]:
            if arms_filter and arm not in arms_filter:
                continue
            if arm == "ref_best_validation":
                # Smoke only verifies plumbing; a fixed deterministic
                # reference stands in. Real E3 resolves from completed E2
                # validation medians (validation-only rule).
                resolved_arm = (
                    "ref_realamp_d1" if smoke else resolve_reference_for_e3(task_key)
                )
            else:
                resolved_arm = arm
            for r in range(replicates):
                specs.append(CellSpec(
                    # Smoke runs live in their own namespace: a tiny-config
                    # smoke manifest must never satisfy (or block) a real
                    # matrix cell of the same coordinates.
                    experiment=f"{eid}_smoke" if smoke else eid,
                    task_key=task_key, n_qubits=n,
                    arm_name=resolved_arm, replicate=r, budget_unique=budget,
                    training_config=config,
                ))
    return specs


def make_provider_factory(mock: bool, ledger=None):
    """Real providers all share ONE cumulative ledger, so the cap spans the
    whole matrix rather than resetting per cell."""
    def factory(spec: CellSpec, track: str):
        if mock:
            seed = derive_child_seed(spec.data_seed, "mock_provider", spec.arm_name,
                                     str(spec.search_seed))
            if track == "structure":
                return MockLayeredBatchProvider(
                    seed, SpaceProfile(spec.space_name, spec.n_qubits), BATCH_SIZE
                )
            return MockJointBatchProvider(seed, spec.n_qubits, BATCH_SIZE)
        provider, _config = build_real_provider(track, ledger=ledger, cell_id=spec.cell_id)
        return provider

    return factory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=["E1", "E2", "E3"])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--mock", action="store_true",
                        help="mock LLM providers (never scientific evidence)")
    parser.add_argument("--arms", nargs="*", default=None)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--max-cells", type=int, default=None)
    args = parser.parse_args()

    protocol = load_protocol()
    specs = enumerate_cells(
        protocol, args.experiment, smoke=args.smoke, pilot=args.pilot,
        arms_filter=args.arms, tasks_filter=args.tasks,
    )
    if args.max_cells:
        specs = specs[: args.max_cells]

    ledger = None
    if not args.mock:
        ledger = GlobalSpendLedger.from_env(LEDGER_PATH, run_label="bench_v2_matrix")
        if ledger is None:
            print("REFUSING: LLM_API_BUDGET_USD is not set to a positive value; "
                  "real LLM cells need an explicit cumulative cap.")
            return 2
        print(f"[ledger] {LEDGER_PATH.name}: cap ${ledger.cap_usd:.2f}, "
              f"already committed ${ledger.totals().committed_usd:.6f}")

    provider_factory = make_provider_factory(args.mock, ledger)
    print(f"[{args.experiment}] {len(specs)} cells "
          f"(smoke={args.smoke} pilot={args.pilot} mock={args.mock})", flush=True)

    failures = []
    cap_stop = None
    for i, spec in enumerate(specs):
        if cap_stop:
            break
        start = time.perf_counter()
        try:
            manifest = run_cell(spec, RUNS_ROOT, provider_factory=provider_factory)
            elapsed = time.perf_counter() - start
            print(
                f"  [{i + 1}/{len(specs)}] {spec.cell_id}: {manifest['status']} "
                f"in {elapsed:.1f}s (test gate at {manifest['test_gate']['timestamp']})",
                flush=True,
            )
        except LedgerCapExceeded as exc:
            cap_stop = str(exc)
            print(f"  [{i + 1}/{len(specs)}] {spec.cell_id}: STOPPED — global cap reached")
            break
        except Exception as exc:
            elapsed = time.perf_counter() - start
            failures.append((spec.cell_id, f"{type(exc).__name__}: {exc}"))
            print(f"  [{i + 1}/{len(specs)}] {spec.cell_id}: FAILED after "
                  f"{elapsed:.1f}s — {type(exc).__name__}: {exc}", flush=True)

    completed = sum(1 for s in specs
                    if (RUNS_ROOT / s.experiment / "cells" / f"{s.cell_id}.json").is_file())
    print(f"[{args.experiment}] done: {completed} complete of {len(specs)} planned, "
          f"{len(failures)} failed", flush=True)
    for cell, err in failures:
        print(f"  FAILED {cell}: {err}", flush=True)
    if ledger is not None:
        s = ledger.summary()
        print(f"[ledger] calls={s['calls_settled']} in={s['input_tokens']} "
              f"out={s['output_tokens']} spent=${s['settled_usd']:.6f} "
              f"remaining=${s['remaining_usd']:.6f} of ${s['cap_usd']:.2f}", flush=True)
    if cap_stop:
        print(f"[ledger] CLEAN STOP — {cap_stop}", flush=True)
        print(f"[ledger] {len(specs) - completed} cells remain; re-run after the "
              "operator raises the cap. The cap was NOT auto-raised.", flush=True)
        return 3
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
