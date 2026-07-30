#!/usr/bin/env python
"""Build every figure + source table for the bench_v2 weekly deck.

All numbers come from durable stores / committed analysis artifacts;
nothing is typed in by hand. Each figure <stem>.png/.svg has its source
rows at data/<stem>.csv. Re-run to refresh (e.g. after E3 progresses).
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

FIG_DIR = HERE / "figures"
DATA_DIR = HERE / "data"
FIG_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# Midnight-Executive-adjacent palette, consistent arm colors everywhere.
NAVY = "#1E2761"
ICE = "#CADCFC"
ARM_COLORS = {
    "random_structure": "#4C72B0", "evolutionary_structure": "#DD8452",
    "greedy_growth": "#55A868", "ref_realamp_d1": "#8C8C8C",
    "ref_realamp_d2": "#B3B3B3", "ref_strongent_d1": "#5F5F5F",
    "ref_strongent_d2": "#9A9A9A", "random_joint": "#4C72B0",
    "evolutionary_joint": "#DD8452",
    "llm_open_structure": "#C44E52", "llm_archive_closed_structure": "#8172B3",
}
ARM_SHORT = {
    "random_structure": "random", "evolutionary_structure": "evolutionary",
    "greedy_growth": "greedy", "ref_realamp_d1": "RealAmp d1",
    "ref_realamp_d2": "RealAmp d2", "ref_strongent_d1": "StrongEnt d1",
    "ref_strongent_d2": "StrongEnt d2", "random_joint": "random_joint",
    "evolutionary_joint": "evo_joint",
}
plt.rcParams.update({
    "figure.dpi": 130, "font.size": 9, "axes.titlesize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})

#: Width each figure occupies on its slide, in inches (must match
#: generate_deck.js). Resolution is allocated per figure from this, so a
#: panel shown 4.2" wide is not rendered at the same pixel density as one
#: shown 12.2" wide — same bytes, sharper where it is actually seen.
DISPLAY_WIDTH_IN = {
    "fig_tasks": 12.2, "fig_e1_per_seed": 8.35, "fig_e2_per_seed": 7.85,
    "fig_e2_anytime": 7.85, "fig_e3_scaling": 4.3, "fig_e2_resources": 8.1,
    "fig_e2_diagnostics": 4.15,
}
#: Rendered pixels per displayed inch. 120 ≈ crisp for a 1600 px-wide
#: rendering of the 13.33" slide. Lowered via BENCH_V2_PPI when the deck
#: has to be small enough to upload to Google Drive in one payload.
TARGET_PPI = float(os.environ.get("BENCH_V2_PPI", "120"))

RNG = np.random.default_rng(7)


def _optimize_png(path: Path) -> None:
    """Flatten onto white and palette-quantize.

    These are line/scatter plots with a few hundred distinct colours, so a
    192-colour adaptive palette is visually indistinguishable at slide
    scale while cutting the file ~5x. That matters because the deck is
    uploaded to Google Drive as a single base64 payload. The .svg beside
    each .png remains the lossless vector original.
    """
    from PIL import Image

    with Image.open(path) as img:
        rgba = img.convert("RGBA")
        flat = Image.new("RGB", rgba.size, (255, 255, 255))
        flat.paste(rgba, mask=rgba.split()[3])
        flat.quantize(colors=192, method=Image.Quantize.MEDIANCUT).save(
            path, optimize=True
        )


def _save(fig, stem: str, rows: list[dict]) -> None:
    display_w = DISPLAY_WIDTH_IN.get(stem)
    dpi = (
        TARGET_PPI * display_w / fig.get_size_inches()[0]
        if display_w else plt.rcParams["figure.dpi"]
    )
    fig.savefig(FIG_DIR / f"{stem}.png", bbox_inches="tight", dpi=dpi)
    fig.savefig(FIG_DIR / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)
    _optimize_png(FIG_DIR / f"{stem}.png")
    if rows:
        with (DATA_DIR / f"{stem}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


# ---------------------------------------------------------------- tasks --

def fig_tasks() -> None:
    from llm_vqc.tasks.signal_suite import SignalProfile, SignalSuiteTask

    families = [
        ("gauss_peak", "T1: Gaussian peak location\n(target: peak position mu)"),
        ("sin_freq", "T2: sinusoid frequency\n(target: normalized frequency)"),
        ("change_point", "T3: change-point location\n(target: normalized change point)"),
        ("peak_count", "T4: single vs double peak\n(binary classification)"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.4))
    rows = []
    x = np.linspace(0, 1, 32)
    for ax, (family, title) in zip(axes, families, strict=True):
        task = SignalSuiteTask(SignalProfile(
            family=family, n_qubits=5, n_train=8, n_val=8, n_test=8,
        ))
        train_val, _ = task.build(1000)
        for i in range(3):
            y = train_val.train.features[i]
            target = train_val.train.targets[i]
            ax.plot(x, y, linewidth=1.3, alpha=0.9)
            for j, value in enumerate(y):
                rows.append({"family": family, "example": i, "grid_index": j,
                             "x": round(float(x[j]), 6), "y": round(float(value), 6),
                             "target": round(float(target), 6)})
        ax.set_title(title, fontsize=8.5)
        ax.set_xticks([0, 0.5, 1])
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("signal value (32 = 2^5 points)", fontsize=8)
    fig.suptitle("signal_suite_v1: 2^n-point signals for amplitude encoding "
                 "(3 examples each, data seed 1000)", fontsize=10, y=1.08)
    _save(fig, "fig_tasks", rows)


# ------------------------------------------------------------------ E1 --

def fig_e1_per_seed() -> None:
    per_seed = _read_csv(REPO / "outputs" / "bench_v2" / "E1" / "per_seed_E1.csv")
    combos = [("gauss_peak_legacy", "3"), ("sin_freq", "3"), ("sin_freq", "5")]
    arms = ["random_joint", "evolutionary_joint"]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.9), sharey=False)
    rows = []
    for ax, (task, n) in zip(axes, combos, strict=True):
        by_rep: dict[int, dict[str, float]] = defaultdict(dict)
        for row in per_seed:
            if row["task"] == task and row["n_qubits"] == n:
                by_rep[int(row["replicate"])][row["arm"]] = float(row["primary_test_value"])
                rows.append({"task": task, "n_qubits": n, "arm": row["arm"],
                             "replicate": row["replicate"],
                             "test_rmse": row["primary_test_value"]})
        for _rep, values in sorted(by_rep.items()):
            if len(values) == 2:
                ax.plot([0, 1], [values[arms[0]], values[arms[1]]],
                        color="gray", alpha=0.35, linewidth=0.8, zorder=1)
        for i, arm in enumerate(arms):
            vals = [by_rep[r][arm] for r in sorted(by_rep) if arm in by_rep[r]]
            xs = np.full(len(vals), i) + RNG.uniform(-0.05, 0.05, len(vals))
            ax.scatter(xs, vals, s=26, color=ARM_COLORS[arm], zorder=3)
            ax.hlines(np.median(vals), i - 0.2, i + 0.2, color="black",
                      linewidth=2, zorder=4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([ARM_SHORT[a] for a in arms], fontsize=8)
        ax.set_title(f"{task} n={n}", fontsize=9)
        ax.set_xlim(-0.5, 1.5)
    axes[0].set_ylabel("validation-selected\nprotected-test RMSE", fontsize=8)
    fig.suptitle("E1 (Track B, verbatim theta, B=16 unique): 10 paired replicates, "
                 "individual seeds + medians", fontsize=10, y=1.05)
    _save(fig, "fig_e1_per_seed", rows)


# ------------------------------------------------------------------ E2 --

E2_ARMS = ["random_structure", "evolutionary_structure", "greedy_growth",
           "ref_realamp_d1", "ref_realamp_d2", "ref_strongent_d1", "ref_strongent_d2"]


def fig_e2_per_seed() -> None:
    per_seed = _read_csv(REPO / "outputs" / "bench_v2" / "E2" / "per_seed_E2.csv")
    tasks = ["gauss_peak", "sin_freq", "change_point", "peak_count"]
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.1))
    rows = []
    for ax, task in zip(axes, tasks, strict=True):
        higher_better = task == "peak_count"
        for i, arm in enumerate(E2_ARMS):
            vals = [float(r["primary_test_value"]) for r in per_seed
                    if r["task"] == task and r["arm"] == arm]
            for r in per_seed:
                if r["task"] == task and r["arm"] == arm:
                    rows.append({"task": task, "arm": arm, "replicate": r["replicate"],
                                 "metric": r["primary_test_metric"],
                                 "value": r["primary_test_value"]})
            xs = np.full(len(vals), i) + RNG.uniform(-0.10, 0.10, len(vals))
            ax.scatter(xs, vals, s=14, color=ARM_COLORS[arm], alpha=0.85, zorder=3)
            ax.hlines(np.median(vals), i - 0.3, i + 0.3, color="black",
                      linewidth=1.8, zorder=4)
        metric = "test AUROC (higher better)" if higher_better else "test RMSE (lower better)"
        ax.set_title(f"{task}\n{metric}", fontsize=8.5)
        ax.set_xticks(range(len(E2_ARMS)))
        ax.set_xticklabels([ARM_SHORT[a] for a in E2_ARMS], rotation=55,
                           ha="right", fontsize=6.5)
    axes[0].set_ylabel("validation-selected\nprotected-test metric", fontsize=8)
    fig.suptitle("E2 classical arms + references (Track A, n=5, B=24 unique, 10 paired "
                 "replicates) — LLM arms pending budget authorization",
                 fontsize=10, y=1.03)
    _save(fig, "fig_e2_per_seed", rows)


def fig_e2_anytime() -> None:
    tasks = ["gauss_peak", "sin_freq", "change_point", "peak_count"]
    search_arms = ["random_structure", "evolutionary_structure", "greedy_growth"]
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 2.7), sharex=True)
    rows = []
    for ax, task in zip(axes, tasks, strict=True):
        source = _read_csv(REPO / "outputs" / "bench_v2" / "E2" / f"fig_anytime_{task}_5q.csv")
        for arm in search_arms:
            arm_rows = [r for r in source if r["arm"] == arm]
            xs = [int(r["unique_evaluations"]) for r in arm_rows]
            med = [float(r["median_best_val"]) for r in arm_rows]
            q25 = [float(r["q25"]) for r in arm_rows]
            q75 = [float(r["q75"]) for r in arm_rows]
            ax.plot(xs, med, color=ARM_COLORS[arm], linewidth=1.5,
                    label=ARM_SHORT[arm])
            ax.fill_between(xs, q25, q75, color=ARM_COLORS[arm], alpha=0.14)
            rows.extend({"task": task, **r} for r in arm_rows)
        ax.set_title(task, fontsize=9)
        ax.set_xlabel("unique evaluations", fontsize=7.5)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("best-so-far\nvalidation metric", fontsize=8)
    axes[0].legend(fontsize=6.5, frameon=False)
    fig.suptitle("E2 anytime curves (median + IQR over 10 paired replicates; "
                 "validation metric, lower is better)", fontsize=10, y=1.06)
    _save(fig, "fig_e2_anytime", rows)


def fig_e3_scaling() -> None:
    """E3 qubit scaling: per-seed points + median lines per arm."""
    per_seed = _read_csv(REPO / "outputs" / "bench_v2" / "E3" / "per_seed_E3.csv")
    arms = ["random_structure", "evolutionary_structure", "ref_strongent_d2"]
    tasks = [("gauss_peak", "T1 Gaussian peak"), ("sin_freq", "T2 sinusoid frequency")]
    ns = [3, 4, 6, 8]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    rows = []
    for ax, (task, label) in zip(axes, tasks, strict=True):
        for arm in arms:
            medians = []
            for n in ns:
                vals = [float(r["primary_test_value"]) for r in per_seed
                        if r["task"] == task and int(r["n_qubits"]) == n
                        and r["arm"] == arm]
                for r in per_seed:
                    if (r["task"] == task and int(r["n_qubits"]) == n
                            and r["arm"] == arm):
                        rows.append({"task": task, "n_qubits": n, "arm": arm,
                                     "replicate": r["replicate"],
                                     "test_rmse": r["primary_test_value"]})
                medians.append(np.median(vals) if vals else np.nan)
                if vals:
                    jitter = RNG.uniform(-0.09, 0.09, len(vals))
                    ax.scatter(np.full(len(vals), n) + jitter, vals, s=9,
                               color=ARM_COLORS[arm], alpha=0.55, zorder=2)
            ax.plot(ns, medians, color=ARM_COLORS[arm], linewidth=1.8,
                    marker="o", markersize=4, zorder=3,
                    label="StrongEnt d2 (ref)" if arm.startswith("ref_")
                    else ARM_SHORT[arm])
        ax.set_title(label, fontsize=8.5)
        ax.set_xticks(ns)
        ax.set_xlabel("qubits", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("protected-test RMSE", fontsize=8)
    axes[0].legend(fontsize=6.5, frameon=False)
    fig.suptitle("E3 scaling (B=16, 5 paired replicates per cell)",
                 fontsize=9.5, y=1.04)
    _save(fig, "fig_e3_scaling", rows)


def fig_e2_resources() -> None:
    resource = _read_csv(REPO / "outputs" / "bench_v2" / "E2" / "resource_metrics.csv")
    panels = [
        ("parameter_count", "trainable parameters"),
        ("two_qubit_count", "logical 2-qubit gates"),
        ("transpiled_2q_line", "transpiled 2q count\n(line coupling, seed 7)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
    rows = []
    for ax, (column, label) in zip(axes, panels, strict=True):
        for i, arm in enumerate(E2_ARMS):
            vals = [float(r[column]) for r in resource if r["arm"] == arm and r[column]]
            xs = np.full(len(vals), i) + RNG.uniform(-0.12, 0.12, len(vals))
            ax.scatter(xs, vals, s=12, color=ARM_COLORS[arm], alpha=0.8, zorder=3)
            ax.hlines(np.median(vals), i - 0.3, i + 0.3, color="black",
                      linewidth=1.6, zorder=4)
            for value in vals:
                rows.append({"panel": column, "arm": arm, "value": value})
        ax.set_title(label, fontsize=8.5)
        ax.set_xticks(range(len(E2_ARMS)))
        ax.set_xticklabels([ARM_SHORT[a] for a in E2_ARMS], rotation=55,
                           ha="right", fontsize=6.5)
    fig.suptitle("E2 selected circuits: resource footprint by arm (all tasks pooled, "
                 "individual circuits + medians)", fontsize=10, y=1.03)
    _save(fig, "fig_e2_resources", rows)


def fig_e2_diagnostics() -> None:
    """Expressibility KL + Meyer-Wallach Q for the best-validation selected
    circuit of every (task, arm) in E2 — descriptive only."""
    from llm_vqc.bench_v2.cell_runner import _resolve_task
    from llm_vqc.bench_v2.space import layered_ir_dict
    from llm_vqc.bench_v2.track_a_evaluator import bench_v2_run_seed
    from llm_vqc.evaluation.store import ResultStore
    from llm_vqc.free_amplitude.expressibility import intrinsic_diagnostics
    from llm_vqc.free_amplitude.init_policy import free_amplitude_train_seed
    from llm_vqc.free_amplitude.training import FREE_AMPLITUDE_TRAINING_CONFIG_VERSION
    from llm_vqc.ir.schema import CircuitIR

    cells = REPO / "runs" / "bench_v2" / "E2" / "cells"
    best: dict[tuple, dict] = {}
    for path in sorted(cells.glob("*.json")):
        m = json.loads(path.read_text())
        if m.get("status") != "complete":
            continue
        key = (m["task"], m["arm"])
        value = m["selected"].get("val_metric_value")
        if value is not None and (
            key not in best or value < best[key]["selected"]["val_metric_value"]
        ):
            best[key] = m

    rows = []
    for (task_key, arm), m in sorted(best.items()):
        task, _c, _v = _resolve_task(m["task"], m["n_qubits"])
        run_seed = bench_v2_run_seed(task.spec.name, m["data_seed"], m["search_seed"])
        train_seed = free_amplitude_train_seed(
            run_seed, m["selected"]["structural_hash"],
            FREE_AMPLITUDE_TRAINING_CONFIG_VERSION,
        )
        store_path = REPO / m["store_path"] if not Path(m["store_path"]).is_absolute() \
            else Path(m["store_path"])
        store = ResultStore(store_path)
        blob = store.get_trained_weights(
            task.spec.name, m["selected"]["structural_hash"], train_seed
        )
        store.close()
        if blob is None:
            continue
        ir = CircuitIR.model_validate(layered_ir_dict(blob["n_qubits"], blob["operations"]))
        diag = intrinsic_diagnostics(ir, m["selected"]["structural_hash"])
        rows.append({
            "task": task_key, "arm": arm, "source_cell": m["cell_id"],
            "expressibility_kl": round(diag.expressibility_kl, 6),
            "meyer_wallach_q_mean": round(diag.entanglement_capability.mean, 6),
            "parameter_count": diag.quantum_parameter_count,
            "test_metric": m["test_gate"]["metrics"].get(
                "test_auc", m["test_gate"]["metrics"].get("test_rmse")
            ),
        })

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    seen = set()
    for row in rows:
        arm = row["arm"]
        label = ARM_SHORT[arm] if arm not in seen else None
        seen.add(arm)
        ax.scatter(row["expressibility_kl"], row["meyer_wallach_q_mean"],
                   s=30 + 6 * row["parameter_count"], color=ARM_COLORS[arm],
                   alpha=0.8, label=label)
    ax.set_xlabel("expressibility KL vs Haar (lower = more expressible)", fontsize=8.5)
    ax.set_ylabel("Meyer-Wallach Q (mean)", fontsize=8.5)
    ax.legend(fontsize=6.5, frameon=False, loc="best")
    ax.set_title("E2 best-validation selected circuits: descriptive diagnostics\n"
                 "(marker size ~ parameter count; NOT assumed performance proxies)",
                 fontsize=9)
    _save(fig, "fig_e2_diagnostics", rows)


# ------------------------------------------------------- status tables --

def status_tables() -> None:
    runs_root = REPO / "runs" / "bench_v2"
    rows = []
    matrix = {
        "E1": {"classical": 60, "llm": 60},
        "E2": {"classical": 280, "llm": 80},
        "E3": {"classical": 120, "llm": 80},
    }
    for eid, expected in matrix.items():
        cells = runs_root / eid / "cells"
        status = Counter()
        if cells.is_dir():
            for path in cells.glob("*.json"):
                m = json.loads(path.read_text())
                key = "llm" if "llm" in m["arm"] else "classical"
                status[(key, m.get("status"))] += 1
        for kind in ("classical", "llm"):
            done = status.get((kind, "complete"), 0)
            rows.append({
                "experiment": eid, "cells": kind, "expected": expected[kind],
                "complete": done, "failed": status.get((kind, "failed"), 0),
                "pending": expected[kind] - done - status.get((kind, "failed"), 0),
            })
    e0 = json.loads((REPO / "outputs" / "bench_v2" / "E0" / "COMPLETE.json").read_text())
    rows.append({"experiment": "E0", "cells": "replication", "expected": 1,
                 "complete": 1 if e0.get("match") else 0, "failed": 0, "pending": 0})
    for eid in ("E4", "E5"):
        marker = REPO / "outputs" / "bench_v2" / eid / "COMPLETE.json"
        rows.append({"experiment": eid, "cells": "dependent", "expected": 1,
                     "complete": 1 if marker.is_file() else 0, "failed": 0,
                     "pending": 0 if marker.is_file() else 1})
    with (DATA_DIR / "matrix_status.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    report = json.loads(
        (REPO / "outputs" / "bench_v2" / "E0" / "replication_report.json").read_text()
    )
    e0_rows = [{
        "arm": c["arm"], "validation_match": c["validation_match"],
        "test_match": c["test_match"],
        "recomputed_test_rmse_seed_lo": min(c["recomputed_test"]),
        "recomputed_test_rmse_seed_hi": max(c["recomputed_test"]),
    } for c in report["summary_checks"]]
    with (DATA_DIR / "e0_replication_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(e0_rows[0].keys()))
        writer.writeheader()
        writer.writerows(e0_rows)


def copy_stat_sources() -> None:
    import shutil
    for eid in ("E1", "E2"):
        for name in ("stats_arm_summaries.csv", "stats_effect_sizes.csv",
                     "stats_paired_tests.csv", f"per_seed_{eid}.csv"):
            src = REPO / "outputs" / "bench_v2" / eid / name
            if src.is_file():
                shutil.copy(src, DATA_DIR / f"{eid}_{name}")


def main() -> None:
    fig_tasks()
    fig_e1_per_seed()
    fig_e2_per_seed()
    fig_e2_anytime()
    fig_e3_scaling()
    fig_e2_resources()
    fig_e2_diagnostics()
    status_tables()
    copy_stat_sources()
    print(f"figures -> {FIG_DIR}\ndata -> {DATA_DIR}")


if __name__ == "__main__":
    main()
