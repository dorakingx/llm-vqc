#!/usr/bin/env python
"""Why the 2026-07-24 pilot saw an LLM advantage that the fixed-length
run does not: the LLM was choosing circuit SIZE, not better structure.

Three conditions, same code, same model, same tasks, 10 seeds each:
  fixed 5 gates B=8   - size removed as a decision
  variable 1-5 B=8    - size restored, generous budget
  variable 1-5 B=4    - size restored, the pilot's budget
"""

from __future__ import annotations

import collections
import json
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "mini5"
FIG = REPO / "docs" / "presentation" / "mini5" / "figures"
DATA = REPO / "docs" / "presentation" / "mini5" / "data"

CONDITIONS = [
    ("results.json", "fixed 5 gates\nB = 8"),
    ("results_variable.json", "variable 1-5 gates\nB = 8"),
    ("results_pilotlike.json", "variable 1-5 gates\nB = 4  (pilot setting)"),
]
COLOR = {"random": "#7F7F7F", "evolutionary": "#E69F00", "greedy": "#009E73",
         "llm_open": "#0072B2", "llm_closed": "#CC79A7"}
LABEL = {"random": "Random", "evolutionary": "Evolutionary", "greedy": "Greedy",
         "llm_open": "LLM open-loop", "llm_closed": "LLM closed-loop"}
ORDER = list(COLOR)
INK, MUTED = "#1A1A2E", "#4A4A5E"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#5A5A6E", "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "figure.facecolor": "white",
})


def load(name: str):
    path = OUT / name
    if not path.is_file():
        sys.exit(f"missing {path}")
    return json.loads(path.read_text())["cells"]


def main() -> int:
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.3),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.25]})

    # panels 1-3: T1 test RMSE per condition
    for ax, (name, title) in zip(axes[:3], CONDITIONS, strict=True):
        cells = [c for c in load(name) if c["family"] == "gauss_peak"]
        for i, arm in enumerate(ORDER):
            values = [c["test_rmse"] for c in cells if c["arm"] == arm]
            if not values:
                continue
            ax.scatter(np.full(len(values), i)
                       + np.random.default_rng(3).uniform(-0.15, 0.15, len(values)),
                       values, s=26, color=COLOR[arm], alpha=0.7, linewidths=0, zorder=3)
            median = st.median(values)
            ax.plot([i - 0.3, i + 0.3], [median, median], color=INK, lw=2.4, zorder=4)
        ax.set_title(title, fontsize=12, loc="left")
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels([LABEL[a] for a in ORDER], rotation=40, ha="right", fontsize=10)
        ax.set_ylim(0.10, 0.30)
        ax.grid(axis="y", color="#E6E8EE", lw=0.8, zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("T1 test RMSE\n(lower is better)", fontsize=12)

    # panel 4: the mechanism - chosen circuit size when size is free
    ax = axes[3]
    cells = load("results_variable.json")
    width = 0.15
    for i, arm in enumerate(ORDER):
        counts = collections.Counter(c["selected_gate_count"] for c in cells
                                     if c["arm"] == arm)
        total = sum(counts.values()) or 1
        xs = np.arange(1, 6) + (i - 2) * width
        ax.bar(xs, [counts.get(g, 0) / total for g in range(1, 6)],
               width=width, color=COLOR[arm], label=LABEL[arm])
    ax.set_title("Which size each method picks\nwhen size is free (1-5)",
                 fontsize=12, loc="left")
    ax.set_xlabel("gates in the selected circuit", fontsize=11.5)
    ax.set_ylabel("fraction of the 40 cells", fontsize=11.5)
    ax.set_xticks(range(1, 6))
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    ax.grid(axis="y", color="#E6E8EE", lw=0.8, zorder=0)
    ax.set_axisbelow(True)

    fig.text(0.5, -0.10,
             "Left three panels: one dot = one of 10 seeds, thick bar = median. "
             "Only Random degrades as circuit size becomes a free choice, because "
             "the LLM almost always picks the maximum size while Random draws it "
             "uniformly. The pilot rewarded that single decision.",
             ha="center", fontsize=11, color=MUTED, wrap=True)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig_size_confound.png", bbox_inches="tight", dpi=170)
    plt.close(fig)
    print("  wrote figures/fig_size_confound.png")

    DATA.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, title in CONDITIONS:
        cells = load(name)
        summary[title.replace("\n", " ")] = {
            "median_test_rmse": {
                fam: {a: st.median([c["test_rmse"] for c in cells
                                    if c["family"] == fam and c["arm"] == a])
                      for a in ORDER}
                for fam in ("gauss_peak", "sin_freq", "change_point", "peak_count")
            },
            "selected_gate_count_median": {
                a: st.median([c["selected_gate_count"] for c in cells if c["arm"] == a])
                for a in ORDER
            },
        }
    (DATA / "condition_comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("  wrote data/condition_comparison.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
