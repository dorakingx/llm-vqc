#!/usr/bin/env python
"""Figures for the mini_5gate_v1 deck.

Every mark is explained inside the figure, because a reader looking at
the exported PNG has no speaker notes. Colour is the only channel that
carries method identity, and it is identical in every panel.

RMSE values are comparable BETWEEN ARMS WITHIN A PANEL only. Different
tasks regress different quantities on different scales, so comparing the
height of a T1 bar with a T3 bar is meaningless; the panels are
deliberately given independent y-axes and the caption says so.
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "outputs" / "mini5" / "results.json"
FIG = REPO / "docs" / "presentation" / "mini5" / "figures"
#: outputs/ is gitignored (durable stores stay out of the repo), so the
#: numbers behind every figure are exported here, inside the deck package,
#: where a reader can check them without the store.
DATA = REPO / "docs" / "presentation" / "mini5" / "data"

# Okabe-Ito derived, colourblind safe. One colour per method, everywhere.
COLOR = {
    "random": "#7F7F7F",
    "evolutionary": "#E69F00",
    "greedy": "#009E73",
    "llm_open": "#0072B2",
    "llm_closed": "#CC79A7",
}
LABEL = {
    "random": "Random",
    "evolutionary": "Evolutionary",
    "greedy": "Greedy",
    "llm_open": "LLM open-loop",
    "llm_closed": "LLM closed-loop",
}
ORDER = ["random", "evolutionary", "greedy", "llm_open", "llm_closed"]
INK = "#1A1A2E"
MUTED = "#4A4A5E"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#5A5A6E", "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "figure.facecolor": "white",
})


def load() -> dict:
    if not RESULTS.is_file():
        sys.exit(f"missing {RESULTS} - run scripts/mini5/run_mini5.py first")
    return json.loads(RESULTS.read_text())


def by_task(doc: dict) -> dict:
    out: dict[tuple, dict[str, list]] = {}
    for cell in doc["cells"]:
        if cell["status"] != "complete":
            continue
        key = (cell["family"], cell["n_qubits"])
        out.setdefault(key, {}).setdefault(cell["arm"], []).append(cell)
    return out


def fig_main(doc: dict) -> None:
    """Per-task test RMSE, one dot per seed plus the median."""
    grouped = by_task(doc)
    tasks = [(t["family"], t["n_qubits"], t["label"]) for t in doc["tasks"]]
    fig, axes = plt.subplots(1, len(tasks), figsize=(15.0, 4.4))
    rng = np.random.default_rng(7)

    for ax, (family, n, label) in zip(axes, tasks, strict=True):
        cells = grouped.get((family, n), {})
        for i, arm in enumerate(ORDER):
            values = [c["test_rmse"] for c in cells.get(arm, [])]
            if not values:
                continue
            jitter = rng.uniform(-0.16, 0.16, size=len(values))
            ax.scatter(np.full(len(values), i) + jitter, values, s=34,
                       color=COLOR[arm], alpha=0.75, zorder=3, linewidths=0)
            median = st.median(values)
            ax.plot([i - 0.32, i + 0.32], [median, median], color=INK,
                    linewidth=2.6, zorder=4, solid_capstyle="butt")
        ax.set_title(f"{label}   ({n} qubits)", loc="left", fontsize=13.5)
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels([LABEL[a] for a in ORDER], rotation=38, ha="right",
                           fontsize=11)
        ax.grid(axis="y", color="#E6E8EE", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Test RMSE\n(lower is better)", fontsize=13)

    handles = [plt.Line2D([], [], marker="o", linestyle="none", color=COLOR[a],
                          markersize=8, label=LABEL[a]) for a in ORDER]
    handles.append(plt.Line2D([], [], marker="o", linestyle="none", color=MUTED,
                              markersize=7, label="one dot = one of 10 seeds"))
    handles.append(plt.Line2D([], [], color=INK, linewidth=2.6,
                              label="thick bar = median of the 10 seeds"))
    fig.legend(handles=handles, frameon=False, fontsize=11.5, ncol=7,
               loc="lower center", bbox_to_anchor=(0.5, -0.10))
    fig.text(0.5, -0.16,
             "Each panel has its own y-axis: RMSE is comparable between arms "
             "within a panel, never between panels, because the tasks regress "
             "different quantities.",
             ha="center", fontsize=11.5, color=MUTED)
    fig.tight_layout()
    _save(fig, "fig_main")


def fig_t4_check(doc: dict) -> None:
    """T4 only: RMSE hides discrimination that AUROC reveals."""
    grouped = by_task(doc)
    key = next(((t["family"], t["n_qubits"]) for t in doc["tasks"]
                if t["family"] == "peak_count"), None)
    if key is None or key not in grouped:
        return
    cells = grouped[key]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    rng = np.random.default_rng(11)

    for ax, (field, title, better) in zip(
        axes,
        [("test_rmse", "Test RMSE  (lower is better)", "lower"),
         ("test_auroc", "Test AUROC  (higher is better)", "higher")],
        strict=True,
    ):
        for i, arm in enumerate(ORDER):
            values = [c[field] for c in cells.get(arm, []) if c.get(field) is not None]
            if not values:
                continue
            ax.scatter(np.full(len(values), i) + rng.uniform(-0.16, 0.16, len(values)),
                       values, s=34, color=COLOR[arm], alpha=0.75, zorder=3,
                       linewidths=0)
            median = st.median(values)
            ax.plot([i - 0.32, i + 0.32], [median, median], color=INK,
                    linewidth=2.6, zorder=4)
        ax.set_title(title, loc="left", fontsize=13)
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels([LABEL[a] for a in ORDER], rotation=38, ha="right",
                           fontsize=10.5)
        ax.grid(axis="y", color="#E6E8EE", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        if better == "higher":
            ax.axhline(0.5, color="#B3261E", linewidth=1.2, linestyle="--", zorder=2)
            ax.text(0.02, 0.505, "0.5 = chance", transform=ax.get_yaxis_transform(),
                    fontsize=10.5, color="#B3261E", va="bottom")

    fig.suptitle("T4 is binary classification: RMSE against a 0/1 label is "
                 "sqrt(Brier), a proper score, but it mixes calibration with "
                 "discrimination.", fontsize=11.5, y=1.04, color=MUTED)
    fig.tight_layout()
    _save(fig, "fig_t4_check")


def fig_anytime(doc: dict) -> None:
    """Best-so-far validation RMSE against the unique-candidate budget."""
    grouped = by_task(doc)
    tasks = [(t["family"], t["n_qubits"], t["label"]) for t in doc["tasks"]]
    fig, axes = plt.subplots(1, len(tasks), figsize=(15.0, 3.9))

    for ax, (family, n, label) in zip(axes, tasks, strict=True):
        cells = grouped.get((family, n), {})
        for arm in ORDER:
            curves = [c["best_so_far"] for c in cells.get(arm, []) if c.get("best_so_far")]
            if not curves:
                continue
            width = min(len(c) for c in curves)
            stacked = np.array([c[:width] for c in curves])
            ax.plot(range(1, width + 1), np.median(stacked, axis=0),
                    color=COLOR[arm], linewidth=2.2, marker="o", markersize=4)
        ax.set_title(f"{label}   ({n} qubits)", loc="left", fontsize=13)
        ax.set_xlabel("unique candidates evaluated", fontsize=12)
        ax.grid(color="#E6E8EE", linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("best-so-far\nvalidation RMSE", fontsize=12.5)

    handles = [plt.Line2D([], [], color=COLOR[a], linewidth=2.2, marker="o",
                          markersize=4, label=LABEL[a]) for a in ORDER]
    handles.append(plt.Line2D([], [], color="none",
                              label="line = median over the 10 seeds"))
    fig.legend(handles=handles, frameon=False, fontsize=11.5, ncol=6,
               loc="lower center", bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    _save(fig, "fig_anytime")


def _save(fig, stem: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{stem}.png", bbox_inches="tight", dpi=170)
    plt.close(fig)
    print(f"  wrote figures/{stem}.png")


def summary(doc: dict) -> None:
    """Printed table the deck's numbers are read from."""
    grouped = by_task(doc)
    print("\nmedian test RMSE over 10 seeds (lower is better)")
    header = f"  {'task':22s}" + "".join(f"{LABEL[a]:>16s}" for a in ORDER)
    print(header)
    for t in doc["tasks"]:
        cells = grouped.get((t["family"], t["n_qubits"]), {})
        row = f"  {t['label']} (n={t['n_qubits']})".ljust(24)
        for arm in ORDER:
            values = [c["test_rmse"] for c in cells.get(arm, [])]
            row += f"{st.median(values):>16.4f}" if values else f"{'-':>16s}"
        print(row)


def export_data(doc: dict) -> None:
    """One row per cell: what the figures are drawn from."""
    import csv

    DATA.mkdir(parents=True, exist_ok=True)
    fields = ["family", "n_qubits", "arm", "seed", "status", "n_evaluated",
              "selected_val_rmse", "test_rmse", "test_auroc", "stop_reason",
              "selected_gate_count", "api_calls", "proposals", "duplicates",
              "invalid"]
    with (DATA / "per_cell.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for cell in doc["cells"]:
            tel = cell.get("telemetry", {})
            writer.writerow({
                **{k: cell.get(k) for k in fields if k in cell},
                "api_calls": tel.get("api_calls"),
                "proposals": tel.get("proposals"),
                "duplicates": tel.get("duplicates"),
                "invalid": tel.get("invalid"),
            })
    meta = {k: doc[k] for k in
            ("space_version", "exact_gates", "budget_unique", "seeds", "tasks",
             "arms", "wall_clock_seconds") if k in doc}
    meta["spend"] = doc.get("spend")
    (DATA / "run_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"  wrote data/per_cell.csv ({len(doc['cells'])} rows) and "
          "data/run_metadata.json")


def main() -> int:
    doc = load()
    print(f"mini5: {len(doc['cells'])} cells, "
          f"{doc['wall_clock_seconds'] / 60:.1f} min wall clock")
    fig_main(doc)
    fig_t4_check(doc)
    fig_anytime(doc)
    export_data(doc)
    summary(doc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
