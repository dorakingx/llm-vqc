#!/usr/bin/env python
"""Figures for the 30-seed, 16-gate, task-aware run.

Two of them carry the whole story: what every method scored, and why the
hybrid arm failed. The second exists because the hypothesis it tested was
refuted, and a refuted hypothesis is the part of a study most worth
drawing.
"""

from __future__ import annotations

import glob
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "mini5"
FIG = REPO / "docs" / "presentation" / "mini5" / "figures"

ORDER = ["reference", "random", "evolutionary",
         "llm_open_ctx", "llm_closed_ctx", "llm_hybrid_ctx"]
LABEL = {"reference": "Fixed ansatz\n(no search)", "random": "Random",
         "evolutionary": "Evolutionary", "llm_open_ctx": "LLM\nopen-loop",
         "llm_closed_ctx": "LLM\nclosed-loop", "llm_hybrid_ctx": "LLM hybrid\n(random start)"}
COLOR = {"reference": "#9AA0A6", "random": "#4A4A5E", "evolutionary": "#E69F00",
         "llm_open_ctx": "#0072B2", "llm_closed_ctx": "#CC79A7",
         "llm_hybrid_ctx": "#009E73"}
TASKS = [("gauss_peak", "T1 peak position"), ("sin_freq", "T2 frequency"),
         ("change_point", "T3 change point"), ("peak_count", "T4 one vs two")]
INK, MUTED = "#1A1A2E", "#55606E"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#5A5A6E", "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "figure.facecolor": "white",
})


def load():
    cells = {}
    for pat in ("results_cls30.json", "results_llm_s*.json", "results_hyb_s*.json"):
        for path in sorted(glob.glob(str(OUT / pat))):
            for c in json.loads(Path(path).read_text())["cells"]:
                if c["status"] == "complete":
                    cells[(c["family"], c["arm"], c["seed"])] = c
    return cells


def _save(fig, stem):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{stem}.png", bbox_inches="tight", dpi=170, facecolor="white")
    plt.close(fig)
    print(f"  wrote figures/{stem}.png")


def fig_main(cells):
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.6))
    rng = np.random.default_rng(5)
    for ax, (fam, label) in zip(axes, TASKS, strict=True):
        for i, arm in enumerate(ORDER):
            v = [c["test_rmse"] for k, c in cells.items() if k[0] == fam and k[1] == arm]
            if not v:
                continue
            ax.scatter(np.full(len(v), i) + rng.uniform(-0.17, 0.17, len(v)), v,
                       s=20, color=COLOR[arm], alpha=0.65, linewidths=0, zorder=3)
            m = st.median(v)
            ax.plot([i - 0.33, i + 0.33], [m, m], color=INK, lw=2.6, zorder=4)
        ax.set_title(label, loc="left", fontsize=13)
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels([LABEL[a] for a in ORDER], rotation=45, ha="right",
                           fontsize=9.5)
        ax.grid(axis="y", color="#E6E8EE", lw=0.8, zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Test RMSE\n(lower is better)", fontsize=12.5)
    handles = [plt.Line2D([], [], marker="o", linestyle="none", color=MUTED,
                          markersize=7, label="one dot = one of 30 seeds"),
               plt.Line2D([], [], color=INK, lw=2.6, label="thick bar = median")]
    fig.legend(handles=handles, frameon=False, fontsize=11.5, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, -0.13))
    fig.text(0.5, -0.19,
             "Every panel has its own y-axis. RMSE compares arms inside a "
             "panel, never across panels.", ha="center", fontsize=11, color=MUTED)
    fig.tight_layout()
    _save(fig, "fig_v2_main")


def fig_mechanism(cells):
    """Start vs improvement: the hypothesis and its refutation."""
    arms = ["random", "llm_open_ctx", "llm_hybrid_ctx"]
    names = ["Random", "LLM\n(pure)", "LLM hybrid\n(random start)"]
    firsts, lasts, gains = [], [], []
    for a in arms:
        f = [c["best_so_far"][0] for k, c in cells.items()
             if k[1] == a and c.get("best_so_far") and len(c["best_so_far"]) >= 8]
        lst = [c["best_so_far"][-1] for k, c in cells.items()
               if k[1] == a and c.get("best_so_far") and len(c["best_so_far"]) >= 8]
        firsts.append(st.median(f))
        lasts.append(st.median(lst))
        gains.append(100 * st.median([(x - y) / x for x, y in zip(f, lst, strict=True)]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    x = np.arange(len(arms))
    ax1.bar(x - 0.19, firsts, 0.38, label="first candidate", color="#C6CBD6")
    ax1.bar(x + 0.19, lasts, 0.38, label="best of 8", color=[COLOR[a] for a in arms])
    for i, (f, l) in enumerate(zip(firsts, lasts, strict=True)):
        ax1.text(i - 0.19, f + 0.004, f"{f:.3f}", ha="center", fontsize=10)
        ax1.text(i + 0.19, l + 0.004, f"{l:.3f}", ha="center", fontsize=10, weight="bold")
    ax1.set_xticks(x); ax1.set_xticklabels(names, fontsize=11)
    ax1.set_ylabel("validation RMSE (median)", fontsize=12)
    ax1.set_title("The hybrid does start better - and ends best", loc="left", fontsize=12.5)
    ax1.legend(frameon=False, fontsize=10.5)
    ax1.grid(axis="y", color="#E6E8EE", lw=0.8); ax1.set_axisbelow(True)

    bars = ax2.bar(x, gains, 0.5, color=[COLOR[a] for a in arms])
    for i, g in enumerate(gains):
        ax2.text(i, g + 0.4, f"{g:.0f}%", ha="center", fontsize=12, weight="bold")
    ax2.axhline(gains[0], color="#B3261E", ls="--", lw=1.3)
    ax2.text(2.42, gains[0] + 0.3, "Random", color="#B3261E", fontsize=10.5, ha="right")
    ax2.set_xticks(x); ax2.set_xticklabels(names, fontsize=11)
    ax2.set_ylabel("improvement, first to eighth (%)", fontsize=12)
    ax2.set_title("...but its improvement rate collapses", loc="left", fontsize=12.5)
    ax2.grid(axis="y", color="#E6E8EE", lw=0.8); ax2.set_axisbelow(True)

    fig.text(0.5, -0.07,
             "The pure LLM improved 17% where random improved 9%, which looked "
             "like refinement skill. Give it a good start and that rate falls to "
             "7%, below random. The 17% was the room a bad start leaves, not an ability.",
             ha="center", fontsize=11.5, color="#B3261E", wrap=True)
    fig.tight_layout()
    _save(fig, "fig_v2_mechanism")
    _ = bars


if __name__ == "__main__":
    c = load()
    print(f"loaded {len(c)} cells")
    fig_main(c)
    fig_mechanism(c)
