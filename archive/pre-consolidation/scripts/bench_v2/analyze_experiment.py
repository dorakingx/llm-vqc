#!/usr/bin/env python
"""bench_v2 analysis: regenerate every table and figure for one
experiment from durable stores by script (protocol §9, Phase 8; contract
C10/C11/C13/C14). Nothing here is hand-edited; every figure gets a
same-stem source CSV plus an entry in figure_sources.json with the
config/store hashes it was derived from.

Usage:
    python scripts/bench_v2/analyze_experiment.py --experiment E2
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

RUNS_ROOT = REPO / "runs" / "bench_v2"
OUT_ROOT = REPO / "outputs" / "bench_v2"

BOOTSTRAP_RESAMPLES = 10_000
RNG = np.random.default_rng(20260729)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_manifests(experiment: str) -> list[dict]:
    cells_dir = RUNS_ROOT / experiment / "cells"
    manifests = []
    for path in sorted(cells_dir.glob("*.json")):
        manifest = json.loads(path.read_text())
        if manifest.get("status") == "complete":
            manifests.append(manifest)
    return manifests


def primary_test_metric(manifest: dict) -> tuple[str, float]:
    metrics = manifest["test_gate"]["metrics"]
    if "test_auc" in metrics:
        return "test_auc", float(metrics["test_auc"])
    return "test_rmse", float(metrics["test_rmse"])


def _best_so_far_curve(manifest: dict) -> list[tuple[int, float]]:
    """(unique_eval_index, best_val_metric) pairs reconstructed from the
    cell store's ordered proposal events joined with its evaluations."""
    store_path = Path(manifest["store_path"])
    if not store_path.is_absolute():
        store_path = REPO / store_path
    conn = sqlite3.connect(store_path)
    events = conn.execute(
        "SELECT proposal_index, outcome, structural_hash FROM proposal_events "
        "WHERE run_id = ? ORDER BY proposal_index", (manifest["cell_id"],)
    ).fetchall()
    metric_by_hash: dict[str, float] = {}
    for (result_json, weights_json) in conn.execute(
        "SELECT result_json, trained_weights_json FROM evaluations"
    ).fetchall():
        result = json.loads(result_json)
        h = result.get("structural_hash")
        value = result.get("val_metric_value")
        if value is None and weights_json:
            value = json.loads(weights_json).get("val_rmse")
        if h is not None and value is not None:
            metric_by_hash[h] = float(value)
    conn.close()

    curve = []
    best = None
    seen: set[str] = set()
    unique_count = 0
    for _idx, outcome, structural_hash in events:
        if outcome in ("valid", "failed") and structural_hash not in seen:
            if structural_hash is not None:
                seen.add(structural_hash)
                unique_count += 1
                value = metric_by_hash.get(structural_hash)
                if value is not None and (best is None or value < best):
                    best = value
                if best is not None:
                    curve.append((unique_count, best))
    return curve


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def write_per_seed_table(experiment: str, manifests: list[dict], out_dir: Path) -> Path:
    path = out_dir / f"per_seed_{experiment}.csv"
    fields = [
        "experiment", "task", "n_qubits", "arm", "replicate", "data_seed",
        "search_seed", "budget_unique", "val_metric_name", "val_metric_value",
        "primary_test_metric", "primary_test_value", "test_rmse", "test_mae",
        "test_auc", "test_brier", "num_proposed", "num_valid", "num_invalid",
        "num_duplicate", "num_failed", "num_unique", "stop_reason",
        "logical_gate_count", "logical_depth", "two_qubit_count",
        "parameter_count", "transpiled_depth_line", "transpiled_2q_line",
        "llm_calls", "llm_tokens", "llm_cost_usd", "llm_mock", "model_snapshot",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for m in manifests:
            metric_name, metric_value = primary_test_metric(m)
            metrics = m["test_gate"]["metrics"]
            resources = m.get("resources", {})
            logical = resources.get("logical", {})
            line = resources.get("transpiled", {}).get("line", {})
            llm = m.get("llm") or {}
            writer.writerow({
                "experiment": m["experiment"], "task": m["task"],
                "n_qubits": m["n_qubits"], "arm": m["arm"],
                "replicate": m["replicate"], "data_seed": m["data_seed"],
                "search_seed": m["search_seed"],
                "budget_unique": m["budget_unique"],
                "val_metric_name": m["selected"]["val_metric_name"],
                "val_metric_value": m["selected"]["val_metric_value"],
                "primary_test_metric": metric_name,
                "primary_test_value": metric_value,
                "test_rmse": metrics.get("test_rmse"),
                "test_mae": metrics.get("test_mae"),
                "test_auc": metrics.get("test_auc"),
                "test_brier": metrics.get("test_brier"),
                "num_proposed": m["ledger"].get("num_proposed"),
                "num_valid": m["ledger"].get("num_valid"),
                "num_invalid": m["ledger"].get("num_invalid"),
                "num_duplicate": m["ledger"].get("num_duplicate"),
                "num_failed": m["ledger"].get("num_failed"),
                "num_unique": m["ledger"].get("num_unique"),
                "stop_reason": m.get("stop_reason"),
                "logical_gate_count": logical.get("logical_gate_count"),
                "logical_depth": logical.get("logical_depth"),
                "two_qubit_count": logical.get("two_qubit_count"),
                "parameter_count": logical.get("parameter_count"),
                "transpiled_depth_line": line.get("transpiled_depth"),
                "transpiled_2q_line": line.get("transpiled_two_qubit_count"),
                "llm_calls": llm.get("successful_calls"),
                "llm_tokens": (
                    (llm.get("input_tokens") or 0) + (llm.get("output_tokens") or 0)
                ) if llm else None,
                "llm_cost_usd": llm.get("estimated_cost_usd"),
                "llm_mock": llm.get("mock"),
                "model_snapshot": llm.get("model_snapshot"),
            })
    return path


def write_resource_metrics(experiment: str, manifests: list[dict], out_dir: Path) -> Path:
    """C11 columns exactly: logical_gate_count, logical_depth,
    two_qubit_count, transpiled_depth_line, transpiled_depth_ring,
    transpiled_2q_line (+ context columns)."""
    path = out_dir / "resource_metrics.csv"
    fields = [
        "task", "n_qubits", "arm", "replicate",
        "logical_gate_count", "logical_depth", "two_qubit_count",
        "controlled_rotation_count", "parameter_count",
        "transpiled_depth_line", "transpiled_depth_ring", "transpiled_depth_all_to_all",
        "transpiled_2q_line", "transpiled_2q_ring", "transpiled_2q_all_to_all",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for m in manifests:
            logical = m.get("resources", {}).get("logical", {})
            transpiled = m.get("resources", {}).get("transpiled", {})
            writer.writerow({
                "task": m["task"], "n_qubits": m["n_qubits"], "arm": m["arm"],
                "replicate": m["replicate"],
                "logical_gate_count": logical.get("logical_gate_count"),
                "logical_depth": logical.get("logical_depth"),
                "two_qubit_count": logical.get("two_qubit_count"),
                "controlled_rotation_count": logical.get("controlled_rotation_count"),
                "parameter_count": logical.get("parameter_count"),
                "transpiled_depth_line": transpiled.get("line", {}).get("transpiled_depth"),
                "transpiled_depth_ring": transpiled.get("ring", {}).get("transpiled_depth"),
                "transpiled_depth_all_to_all": (
                    transpiled.get("all_to_all", {}).get("transpiled_depth")
                ),
                "transpiled_2q_line": (
                    transpiled.get("line", {}).get("transpiled_two_qubit_count")
                ),
                "transpiled_2q_ring": (
                    transpiled.get("ring", {}).get("transpiled_two_qubit_count")
                ),
                "transpiled_2q_all_to_all": (
                    transpiled.get("all_to_all", {}).get("transpiled_two_qubit_count")
                ),
            })
    return path


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _paired_values(
    manifests: list[dict], task: str, n_qubits: int, arm_a: str, arm_b: str
) -> tuple[np.ndarray, np.ndarray]:
    by_arm: dict[str, dict[int, float]] = defaultdict(dict)
    for m in manifests:
        if m["task"] == task and m["n_qubits"] == n_qubits:
            _, value = primary_test_metric(m)
            by_arm[m["arm"]][m["replicate"]] = value
    shared = sorted(set(by_arm[arm_a]) & set(by_arm[arm_b]))
    return (
        np.array([by_arm[arm_a][r] for r in shared]),
        np.array([by_arm[arm_b][r] for r in shared]),
    )


def _permutation_p(diffs: np.ndarray, n_resamples: int = BOOTSTRAP_RESAMPLES) -> float:
    """Paired sign-flip permutation test on the mean difference."""
    observed = abs(diffs.mean())
    signs = RNG.choice([-1.0, 1.0], size=(n_resamples, len(diffs)))
    permuted = np.abs((signs * diffs).mean(axis=1))
    return float((np.sum(permuted >= observed) + 1) / (n_resamples + 1))


def _cliffs_delta_paired(diffs: np.ndarray) -> float:
    pos = np.sum(diffs > 0)
    neg = np.sum(diffs < 0)
    return float((pos - neg) / len(diffs)) if len(diffs) else float("nan")


def _hodges_lehmann(diffs: np.ndarray) -> float:
    walsh = [
        (diffs[i] + diffs[j]) / 2.0
        for i in range(len(diffs))
        for j in range(i, len(diffs))
    ]
    return float(np.median(walsh)) if walsh else float("nan")


def _bootstrap_ci(values: np.ndarray) -> tuple[float, float]:
    idx = RNG.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))
    medians = np.median(values[idx], axis=1)
    return float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


def write_stats_tables(experiment: str, manifests: list[dict], out_dir: Path) -> None:
    combos = sorted({(m["task"], m["n_qubits"]) for m in manifests})
    test_rows, effect_rows, summary_rows = [], [], []
    for task, n in combos:
        arms = sorted({m["arm"] for m in manifests if m["task"] == task and m["n_qubits"] == n})
        for arm in arms:
            values = np.array([
                primary_test_metric(m)[1] for m in manifests
                if m["task"] == task and m["n_qubits"] == n and m["arm"] == arm
            ])
            lo, hi = _bootstrap_ci(values) if len(values) > 1 else (np.nan, np.nan)
            summary_rows.append({
                "task": task, "n_qubits": n, "arm": arm, "n_replicates": len(values),
                "median": float(np.median(values)), "iqr": float(
                    np.percentile(values, 75) - np.percentile(values, 25)
                ),
                "mean": float(values.mean()), "std": float(values.std(ddof=1))
                if len(values) > 1 else 0.0,
                "bootstrap_ci95_low": lo, "bootstrap_ci95_high": hi,
            })
        family_rows = []
        for arm_a, arm_b in itertools.combinations(arms, 2):
            a, b = _paired_values(manifests, task, n, arm_a, arm_b)
            if len(a) < 3:
                continue
            diffs = a - b
            if np.allclose(diffs, 0):
                wilcoxon_p = 1.0
            else:
                wilcoxon_p = float(stats.wilcoxon(a, b, zero_method="wilcox").pvalue)
            perm_p = _permutation_p(diffs)
            family_rows.append({
                "task": task, "n_qubits": n, "arm_a": arm_a, "arm_b": arm_b,
                "n_pairs": len(a), "mean_diff_a_minus_b": float(diffs.mean()),
                "wilcoxon_p": wilcoxon_p, "permutation_p": perm_p,
            })
            effect_rows.append({
                "task": task, "n_qubits": n, "arm_a": arm_a, "arm_b": arm_b,
                "hodges_lehmann_shift": _hodges_lehmann(diffs),
                "cliffs_delta_paired": _cliffs_delta_paired(diffs),
                "median_a": float(np.median(a)), "median_b": float(np.median(b)),
            })
        # Holm correction within the (task, n) family on permutation p.
        ordered = sorted(family_rows, key=lambda r: r["permutation_p"])
        m_count = len(ordered)
        running_max = 0.0
        for rank, row in enumerate(ordered):
            adjusted = min(1.0, (m_count - rank) * row["permutation_p"])
            running_max = max(running_max, adjusted)
            row["permutation_p_holm"] = running_max
        test_rows.extend(family_rows)

    for name, rows in (
        ("stats_paired_tests.csv", test_rows),
        ("stats_effect_sizes.csv", effect_rows),
        ("stats_arm_summaries.csv", summary_rows),
    ):
        path = out_dir / name
        if rows:
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)


# ---------------------------------------------------------------------------
# Figures (each with a same-stem source CSV)
# ---------------------------------------------------------------------------


def _save_fig_with_source(fig, out_dir: Path, stem: str, source_rows: list[dict]) -> None:
    fig.savefig(out_dir / f"{stem}.png", dpi=160, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)
    if source_rows:
        with (out_dir / f"{stem}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(source_rows[0].keys()))
            writer.writeheader()
            writer.writerows(source_rows)


def figure_per_seed_strip(experiment, manifests, out_dir) -> None:
    combos = sorted({(m["task"], m["n_qubits"]) for m in manifests})
    for task, n in combos:
        subset = [m for m in manifests if m["task"] == task and m["n_qubits"] == n]
        arms = sorted({m["arm"] for m in subset})
        metric_name = primary_test_metric(subset[0])[0]
        fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(arms)), 4))
        rows = []
        by_replicate: dict[int, dict[str, float]] = defaultdict(dict)
        for i, arm in enumerate(arms):
            values = []
            for m in subset:
                if m["arm"] == arm:
                    _, v = primary_test_metric(m)
                    values.append((m["replicate"], v))
                    by_replicate[m["replicate"]][arm] = v
                    rows.append({"task": task, "n_qubits": n, "arm": arm,
                                 "replicate": m["replicate"], metric_name: v})
            xs = np.full(len(values), i) + RNG.uniform(-0.08, 0.08, len(values))
            ax.scatter(xs, [v for _, v in values], s=22, alpha=0.85, zorder=3)
            if values:
                ax.hlines(np.median([v for _, v in values]), i - 0.25, i + 0.25,
                          color="black", linewidth=2, zorder=4)
        for _, arm_values in sorted(by_replicate.items()):
            if len(arm_values) == len(arms):
                ax.plot(range(len(arms)), [arm_values[a] for a in arms],
                        color="gray", alpha=0.25, linewidth=0.8, zorder=1)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels(arms, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(f"validation-selected protected-test {metric_name[5:]}")
        ax.set_title(f"{experiment} {task} n={n}: per-seed protected-test results")
        _save_fig_with_source(fig, out_dir, f"fig_per_seed_{task}_{n}q", rows)


def figure_anytime_curves(experiment, manifests, out_dir) -> None:
    combos = sorted({(m["task"], m["n_qubits"]) for m in manifests})
    for task, n in combos:
        subset = [m for m in manifests if m["task"] == task and m["n_qubits"] == n]
        arms = sorted({m["arm"] for m in subset if not m["arm"].startswith("ref_")})
        fig, ax = plt.subplots(figsize=(6.5, 4))
        rows = []
        for arm in arms:
            curves = [
                dict(_best_so_far_curve(m)) for m in subset if m["arm"] == arm
            ]
            if not curves:
                continue
            budget = max(max(c) for c in curves if c)
            grid = range(1, budget + 1)
            matrix = []
            for curve in curves:
                best = None
                series = []
                for k in grid:
                    if k in curve:
                        best = curve[k]
                    series.append(best if best is not None else np.nan)
                matrix.append(series)
            arr = np.array(matrix, dtype=float)
            median = np.nanmedian(arr, axis=0)
            lo = np.nanpercentile(arr, 25, axis=0)
            hi = np.nanpercentile(arr, 75, axis=0)
            ax.plot(list(grid), median, label=arm, linewidth=1.6)
            ax.fill_between(list(grid), lo, hi, alpha=0.15)
            for k, med, low, high in zip(grid, median, lo, hi, strict=True):
                rows.append({"task": task, "n_qubits": n, "arm": arm,
                             "unique_evaluations": k, "median_best_val": med,
                             "q25": low, "q75": high})
        ax.set_xlabel("unique candidate evaluations")
        ax.set_ylabel("best-so-far validation metric")
        ax.set_title(f"{experiment} {task} n={n}: anytime search performance")
        ax.legend(fontsize=7)
        _save_fig_with_source(fig, out_dir, f"fig_anytime_{task}_{n}q", rows)


def figure_pareto(experiment, manifests, out_dir) -> None:
    combos = sorted({(m["task"], m["n_qubits"]) for m in manifests})
    for task, n in combos:
        subset = [m for m in manifests if m["task"] == task and m["n_qubits"] == n]
        fig, ax = plt.subplots(figsize=(6, 4))
        rows = []
        for m in subset:
            metric_name, value = primary_test_metric(m)
            two_q = (
                m.get("resources", {}).get("transpiled", {}).get("line", {})
                .get("transpiled_two_qubit_count")
            )
            if two_q is None:
                continue
            rows.append({"task": task, "n_qubits": n, "arm": m["arm"],
                         "replicate": m["replicate"], "transpiled_2q_line": two_q,
                         metric_name: value})
        arms = sorted({r["arm"] for r in rows})
        for arm in arms:
            xs = [r["transpiled_2q_line"] for r in rows if r["arm"] == arm]
            ys = [list(r.values())[-1] for r in rows if r["arm"] == arm]
            ax.scatter(xs, ys, s=24, alpha=0.8, label=arm)
        ax.set_xlabel("transpiled 2-qubit count (line coupling)")
        metric_name = primary_test_metric(subset[0])[0]
        ax.set_ylabel(f"protected-test {metric_name[5:]}")
        ax.set_title(f"{experiment} {task} n={n}: test-vs-resource view")
        ax.legend(fontsize=6)
        _save_fig_with_source(fig, out_dir, f"fig_pareto_{task}_{n}q", rows)


def figure_outcome_distribution(experiment, manifests, out_dir) -> None:
    arms = sorted({m["arm"] for m in manifests})
    rows = []
    fig, ax = plt.subplots(figsize=(max(6, 1.0 * len(arms)), 4))
    bottoms = np.zeros(len(arms))
    for outcome in ("num_valid", "num_invalid", "num_duplicate", "num_failed"):
        values = []
        for arm in arms:
            total = sum(
                m["ledger"].get(outcome, 0) for m in manifests if m["arm"] == arm
            )
            values.append(total)
            rows.append({"arm": arm, "outcome": outcome, "count": total})
        ax.bar(range(len(arms)), values, bottom=bottoms, label=outcome[4:])
        bottoms += np.array(values, dtype=float)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(arms, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("proposals")
    ax.set_title(f"{experiment}: proposal outcome distribution by arm")
    ax.legend(fontsize=7)
    _save_fig_with_source(fig, out_dir, "fig_outcomes_by_arm", rows)


def write_cost_table(experiment, manifests, out_dir) -> None:
    rows = []
    for m in manifests:
        llm = m.get("llm") or {}
        rows.append({
            "cell_id": m["cell_id"], "arm": m["arm"], "task": m["task"],
            "n_qubits": m["n_qubits"], "replicate": m["replicate"],
            "llm_calls": llm.get("successful_calls", 0),
            "input_tokens": llm.get("input_tokens", 0),
            "output_tokens": llm.get("output_tokens", 0),
            "estimated_cost_usd": llm.get("estimated_cost_usd", 0.0),
            "mock": llm.get("mock"),
        })
    path = out_dir / "cost_table.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_figure_sources(experiment, manifests, out_dir) -> None:
    protocol_hash = hashlib.sha256(
        (REPO / "configs" / "bench_v2" / "protocol_v2.yaml").read_bytes()
    ).hexdigest()
    store_hashes = {}
    for m in manifests[:0] or manifests:
        store_path = Path(m["store_path"])
        if not store_path.is_absolute():
            store_path = REPO / store_path
        if store_path.is_file():
            store_hashes[m["cell_id"]] = hashlib.sha256(
                store_path.read_bytes()
            ).hexdigest()[:16]
    sources = {
        "experiment": experiment,
        "protocol_sha256": protocol_hash,
        "generated_by": "scripts/bench_v2/analyze_experiment.py",
        "figures": sorted(p.name for p in out_dir.glob("fig_*.png")),
        "figure_source_rule": "every fig_<stem>.png/.svg has fig_<stem>.csv beside it",
        "store_sha256_prefixes": store_hashes,
    }
    (out_dir / "figure_sources.json").write_text(json.dumps(sources, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()

    manifests = load_manifests(args.experiment)
    if not manifests:
        print(f"no complete cells for {args.experiment}")
        return 1
    out_dir = OUT_ROOT / args.experiment
    out_dir.mkdir(parents=True, exist_ok=True)

    write_per_seed_table(args.experiment, manifests, out_dir)
    write_resource_metrics(args.experiment, manifests, out_dir)
    write_stats_tables(args.experiment, manifests, out_dir)
    figure_per_seed_strip(args.experiment, manifests, out_dir)
    figure_anytime_curves(args.experiment, manifests, out_dir)
    figure_pareto(args.experiment, manifests, out_dir)
    figure_outcome_distribution(args.experiment, manifests, out_dir)
    write_cost_table(args.experiment, manifests, out_dir)
    write_figure_sources(args.experiment, manifests, out_dir)
    print(f"[{args.experiment}] analyzed {len(manifests)} cells -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
