"""Grid figures for the redesigned robustness deck.

For each Hamiltonian family, a 3 x 3 grid (rows = qubit count 4/6/8,
columns = budget B = 4/8/16). Each cell shows exactly six categories, in
this order:

  1 Random   2 Greedy   3 LLM-Open (Model A)   4 LLM-Closed (Model A)
  5 LLM-Open (Model B)   6 LLM-Closed (Model B)

Model A is the reference model of the study, Model B the alternative.
Cells that were not run are drawn as such; nothing is interpolated.

Inputs : outputs/qae_robustness/<condition>/selected_results.csv
Outputs: outputs/qae_robustness/figures/grid_<family>.{png,svg}
         outputs/qae_robustness/figures/baseline_cell.{png,svg}
         outputs/qae_robustness/figures/forest_defined.{png,svg}
         outputs/qae_robustness/figures/compliance_by_arm.{png,svg}
         outputs/qae_robustness/grid_cells.json  (what each cell contains)
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from llm_vqc.experiments.qae_robustness.conditions import (
    ALTERNATIVE_MODEL,
    CONDITIONS,
    REFERENCE_MODEL,
)

ROOT = Path("outputs/qae_robustness")
FIGURES = ROOT / "figures"

QUBITS = (4, 6, 8)
BUDGETS = (4, 8, 16)
FAMILY_TITLE = {
    "TFIM": "Transverse-field Ising chain   H = -Σ Zᵢ Zᵢ₊₁ - h Σ Xᵢ,  h ∈ [0.2, 2.0]",
    "XXZ": "XXZ Heisenberg chain   H = Σ (Xᵢ Xᵢ₊₁ + Yᵢ Yᵢ₊₁ + Δ Zᵢ Zᵢ₊₁),  Δ ∈ [0.2, 2.0]",
}

# The six categories, in the required order. Model A keeps the study's
# method colours with filled markers; Model B uses the same hue with a
# hollow marker so the split is visible even in greyscale.
CATEGORIES = [
    ("Random", "Random", None, "#9aa0a6", "o", True),
    ("Greedy", "Greedy", None, "#f9ab00", "s", True),
    ("LLM-Open", "LLM-Open\nModel A", "A", "#1a73e8", "^", True),
    ("LLM-Closed", "LLM-Closed\nModel A", "A", "#d93025", "D", True),
    ("LLM-Open", "LLM-Open\nModel B", "B", "#1a73e8", "^", False),
    ("LLM-Closed", "LLM-Closed\nModel B", "B", "#d93025", "D", False),
]
SHORT = ["Random", "Greedy", "Open A", "Closed A", "Open B", "Closed B"]
MODEL_OF = {"A": REFERENCE_MODEL, "B": ALTERNATIVE_MODEL}
YLIM = (0.35, 1.03)
YLABEL = "Held-out test trash fidelity  $F_{trash}$\n(higher is better)"


def _boot_ci(values, rng):
    boots = np.array([rng.choice(values, size=len(values), replace=True).mean()
                      for _ in range(10_000)])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def load_cells() -> dict:
    """cells[(family, n, B)][category index] -> per-seed test fidelity (sorted by seed)."""
    cells: dict = {}
    for condition in CONDITIONS:
        selected = pd.read_csv(ROOT / condition.key / "selected_results.csv")
        key = (condition.family, condition.n_qubits, condition.budget)
        cell = cells.setdefault(key, {})
        model_tag = "A" if condition.model == REFERENCE_MODEL else "B"
        for index, (method, _label, model, *_rest) in enumerate(CATEGORIES):
            if model is not None and model != model_tag:
                continue
            values = (selected[selected["method"] == method]
                      .sort_values("seed")["test_fid"].to_numpy())
            if index in cell and not np.array_equal(cell[index], values):
                raise RuntimeError(f"conflicting data for {key} category {index}")
            cell[index] = values
    return cells


def draw_cell(ax, cell: dict | None, rng, *, show_means: bool) -> None:
    if not cell:
        ax.set_facecolor("#f1f3f4")
        ax.text(0.5, 0.55, "not run", ha="center", va="center", fontsize=10,
                color="#5f6368", fontweight="bold", transform=ax.transAxes)
        ax.text(0.5, 0.38, "outside the one-factor-at-a-time design",
                ha="center", va="center", fontsize=6.8, color="#5f6368",
                transform=ax.transAxes)
        ax.set_xticks(range(len(CATEGORIES)))
        return
    jitter = np.random.default_rng(1)
    for index, (_m, _label, _model, color, marker, filled) in enumerate(CATEGORIES):
        values = cell.get(index)
        if values is None:
            ax.text(index, YLIM[0] + 0.11, "not\nrun", ha="center", va="bottom",
                    fontsize=7, color="#9aa0a6")
            continue
        x = index + jitter.uniform(-0.16, 0.16, size=len(values))
        ax.scatter(x, values, s=9, color=color, alpha=0.32 if filled else 0.5,
                   edgecolors="none" if filled else color, linewidths=0.4,
                   facecolors=color if filled else "none", zorder=1)
        mean = values.mean()
        low, high = _boot_ci(values, rng)
        ax.errorbar(index, mean, yerr=[[mean - low], [high - mean]], fmt=marker,
                    color=color, markersize=5.5, capsize=2.5, lw=1.3, zorder=3,
                    markerfacecolor=color if filled else "white",
                    markeredgecolor=color if not filled else "black",
                    markeredgewidth=1.1 if not filled else 0.5)
        if show_means:
            ax.annotate(f"{mean:.3f}", (index, high), textcoords="offset points",
                        xytext=(0, 2.5), ha="center", fontsize=7, color=color,
                        fontweight="bold")
    ax.set_xticks(range(len(CATEGORIES)))


def fig_grid(family: str, cells: dict, name: str) -> dict:
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(len(QUBITS), len(BUDGETS), figsize=(13.2, 6.5),
                             sharex=True, sharey=True)
    coverage = {}
    for r, n in enumerate(QUBITS):
        for c, budget in enumerate(BUDGETS):
            ax = axes[r, c]
            cell = cells.get((family, n, budget))
            coverage[f"{n}q_B{budget}"] = sorted(cell) if cell else []
            draw_cell(ax, cell, rng, show_means=True)
            ax.set_ylim(*YLIM)
            ax.set_xlim(-0.6, len(CATEGORIES) - 0.4)
            ax.grid(axis="y", alpha=0.22)
            ax.tick_params(axis="both", labelsize=8)
            if r == 0:
                ax.set_title(f"budget B = {budget}", fontsize=10.5, fontweight="bold",
                             pad=6)
            if c == 0:
                ax.set_ylabel(f"{n} qubits\n({n // 2} latent + {n // 2} trash)",
                              fontsize=9.5, fontweight="bold")
            if r == len(QUBITS) - 1:
                ax.set_xticklabels(SHORT, fontsize=8)
            if (n, budget) == (4, 8):
                for spine in ax.spines.values():
                    spine.set_edgecolor("#202124")
                    spine.set_linewidth(1.6)
                ax.text(0.985, 0.04, "previous condition", ha="right", va="bottom",
                        fontsize=7.5, color="#202124", fontweight="bold",
                        transform=ax.transAxes)
    fig.text(0.006, 0.5, YLABEL, rotation=90, va="center", ha="left", fontsize=9.5)
    handles = [
        plt.Line2D([], [], color=color, marker=marker, linestyle="",
                   markersize=6, markerfacecolor=color if filled else "white",
                   markeredgecolor=color if not filled else "black",
                   markeredgewidth=1.2 if not filled else 0.5,
                   label=label.replace("\n", " "))
        for _m, label, _model, color, marker, filled in CATEGORIES
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8.2,
               frameon=False, bbox_to_anchor=(0.5, -0.005), handletextpad=0.4,
               columnspacing=1.6)
    fig.suptitle(FAMILY_TITLE[family], fontsize=11.5, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0.025, 0.045, 1, 0.965))
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(FIGURES / f"{name}.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIGURES / f"{name}.png")
    return coverage


def fig_baseline(cells: dict, name: str) -> None:
    """The previous condition alone, four categories, same encoding as Model A."""
    cell = cells[("TFIM", 4, 8)]
    rng = np.random.default_rng(0)
    jitter = np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    for index in range(4):
        _m, label, _model, color, marker, _filled = CATEGORIES[index]
        values = cell[index]
        x = index + jitter.uniform(-0.13, 0.13, size=len(values))
        ax.scatter(x, values, s=26, color=color, alpha=0.5, edgecolors="white",
                   linewidths=0.6, zorder=2)
        mean = values.mean()
        low, high = _boot_ci(values, rng)
        ax.errorbar(index, mean, yerr=[[mean - low], [high - mean]], fmt=marker,
                    color=color, markersize=9, capsize=6, lw=2.2, zorder=3,
                    markeredgecolor="black", markeredgewidth=0.7)
        ax.annotate(f"{mean:.4f}", (index, high), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=10, fontweight="bold",
                    color=color)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["Random", "Greedy", "LLM-Open\n(Model A)", "LLM-Closed\n(Model A)"],
                       fontsize=9.5)
    ax.set_ylabel(YLABEL, fontsize=9.5)
    ax.set_ylim(0.62, 1.01)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("Previous condition: 4 qubits · Ising chain · B = 8 · 12 paired seeds",
                 fontsize=10.5, fontweight="bold")
    ax.text(0.995, 0.02, "dots = seeds · marker = mean · bar = bootstrap 95% CI",
            transform=ax.transAxes, ha="right", fontsize=8, color="#444")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(FIGURES / f"{name}.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIGURES / f"{name}.png")


CONTRASTS = [
    ("LLM-Open_minus_Random", "LLM-Open − Random", "#1a73e8"),
    ("LLM-Closed_minus_LLM-Open", "LLM-Closed − LLM-Open", "#d93025"),
    ("LLM-Closed_minus_Random", "LLM-Closed − Random", "#8430ce"),
    ("Greedy_minus_Random", "Greedy − Random", "#f9ab00"),
]
ROWS = [
    ("reference", "4 qubits · Ising · B = 8 · Model A  (previous)"),
    ("budget_b4", "4 qubits · Ising · B = 4 · Model A"),
    ("budget_b16", "4 qubits · Ising · B = 16 · Model A"),
    ("qubits_n6", "6 qubits · Ising · B = 8 · Model A"),
    ("qubits_n8", "8 qubits · Ising · B = 8 · Model A"),
    ("hamiltonian_xxz", "4 qubits · XXZ · B = 8 · Model A"),
    ("model_alt", "4 qubits · Ising · B = 8 · Model B"),
]


def fig_forest(name: str) -> None:
    stats = {key: json.loads((ROOT / key / "paired_stats.json").read_text())
             for key, _label in ROWS}
    fig, axes = plt.subplots(1, len(CONTRASTS), figsize=(13.2, 3.7), sharey=True)
    ys = np.arange(len(ROWS))[::-1]
    for ax, (contrast, label, color) in zip(axes, CONTRASTS, strict=True):
        reference = stats["reference"].get(contrast, {}).get("mean_paired_gain")
        if reference is not None:
            ax.axvline(reference, color="#666", lw=1.1, ls="--", zorder=1)
        for y, (key, _label) in zip(ys, ROWS, strict=True):
            entry = stats[key].get(contrast, {})
            mean = entry.get("mean_paired_gain", np.nan)
            ci = entry.get("bootstrap_95ci", [np.nan, np.nan])
            significant = (entry.get("wilcoxon_exact_two_sided_p", 1.0) or 1.0) < 0.05
            ax.errorbar(mean, y, xerr=[[mean - ci[0]], [ci[1] - mean]], fmt="o",
                        color=color, markersize=8 if key != "reference" else 10,
                        capsize=4, lw=1.9, zorder=3,
                        markerfacecolor=color if significant else "white",
                        markeredgecolor=color, markeredgewidth=1.6)
        ax.axvline(0, color="black", lw=1.0, zorder=2)
        ax.set_title(label, fontsize=10.5, fontweight="bold", color=color)
        ax.grid(axis="x", alpha=0.25)
        ax.set_xlabel("paired difference in $F_{trash}$", fontsize=8.5)
        ax.tick_params(axis="x", labelsize=8)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([label for _k, label in ROWS], fontsize=8.5)
    axes[0].set_ylim(-0.7, len(ROWS) - 0.3)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(FIGURES / f"{name}.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIGURES / f"{name}.png")


def fig_compliance(name: str) -> None:
    rows = []
    for key, label in ROWS:
        quality = json.loads((ROOT / key / "proposal_quality.json").read_text())
        entry = {"label": label.replace("  (previous)", "\n(previous)")}
        for method in ("LLM-Open", "LLM-Closed"):
            stats = quality.get(method, {})
            evaluations = stats.get("evaluations", 0)
            entry[method] = (1 - stats.get("fallback_evaluations", 0) / evaluations
                             if evaluations else None)
        rows.append(entry)
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ys = np.arange(len(rows))[::-1]
    height = 0.38
    for offset, method, color in ((height / 2, "LLM-Open", "#1a73e8"),
                                  (-height / 2, "LLM-Closed", "#d93025")):
        values = [r[method] or 0 for r in rows]
        bars = ax.barh(ys + offset, values, height=height, color=color, label=method,
                       edgecolor="white", linewidth=0.8)
        for bar, value in zip(bars, values, strict=True):
            ax.annotate(f"{value * 100:.0f}%", (value, bar.get_y() + bar.get_height() / 2),
                        textcoords="offset points", xytext=(4, 0), va="center",
                        fontsize=7.5, fontweight="bold", color=color)
    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=7.6)
    ax.set_xlabel("Share of evaluated LLM candidates that satisfied\nthe gate-count "
                  "contract (higher is better)", fontsize=8)
    ax.set_xlim(0, 1.22)
    ax.set_xticks(np.arange(0, 1.01, 0.25))
    ax.tick_params(axis="x", labelsize=7.5)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.95)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(FIGURES / f"{name}.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIGURES / f"{name}.png")


def main() -> None:
    cells = load_cells()
    coverage = {
        family: fig_grid(family, cells, f"grid_{family.lower()}")
        for family in ("TFIM", "XXZ")
    }
    fig_baseline(cells, "baseline_cell")
    fig_forest("forest_defined")
    fig_compliance("compliance_by_arm")
    (ROOT / "grid_cells.json").write_text(json.dumps({
        "categories": [label.replace("\n", " ") for _m, label, *_ in CATEGORIES],
        "models": MODEL_OF,
        "coverage": coverage,
        "note": "category indices present per cell; an empty list means the cell "
                "was not run (one-factor-at-a-time design)",
    }, indent=2))
    print(json.dumps(coverage, indent=1))


if __name__ == "__main__":
    main()
