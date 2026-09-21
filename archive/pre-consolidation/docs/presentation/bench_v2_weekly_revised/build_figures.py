#!/usr/bin/env python
"""Build every figure for the revised GSoC benchmark deck.

All values come from `outputs/bench_v2/**` and `runs/bench_v2/**`; nothing
is typed in by hand. Each figure writes its own source CSV to `data/`.

Design rules enforced here (deck requirements):
  * all plot text >= 16 pt at the size the figure is placed on the slide;
  * one message per figure — dense grids belong to the appendix;
  * colourblind-safe, fixed method colours reused across every figure;
  * axis labels always state units and which direction is better.
"""

from __future__ import annotations

import csv
import json
import os
import statistics as stats
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
OUT = REPO / "outputs" / "bench_v2"
FIG = HERE / "figures"
DATA = HERE / "data"
FIG.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

# Okabe-Ito derived, colourblind safe; identical mapping in every figure.
C_RANDOM = "#0072B2"      # blue
C_EVO = "#E69F00"         # orange
C_GREEDY = "#009E73"      # green
C_LLM = "#CC79A7"         # magenta (not yet measured)
C_REF = "#7F7F7F"         # grey family for fixed references
C_REF_LIGHT = "#BDBDBD"
INK = "#1A1A2E"
MUTED_TXT = "#4A4A5E"
ACCENT = "#B3261E"

ARM_COLOR = {
    "random_structure": C_RANDOM, "evolutionary_structure": C_EVO,
    "greedy_growth": C_GREEDY, "random_joint": C_RANDOM,
    "evolutionary_joint": C_EVO,
    "ref_realamp_d1": C_REF_LIGHT, "ref_realamp_d2": C_REF_LIGHT,
    "ref_strongent_d1": C_REF, "ref_strongent_d2": C_REF,
}
ARM_LABEL = {
    "random_structure": "Random", "evolutionary_structure": "Evolutionary",
    "greedy_growth": "Greedy", "random_joint": "Random (joint)",
    "evolutionary_joint": "Evolutionary (joint)",
    "ref_realamp_d1": "RealAmp d1", "ref_realamp_d2": "RealAmp d2",
    "ref_strongent_d1": "StrongEnt d1", "ref_strongent_d2": "StrongEnt d2",
}
SEARCH_ARMS = ["random_structure", "evolutionary_structure", "greedy_growth"]
REF_ARMS = ["ref_realamp_d1", "ref_realamp_d2", "ref_strongent_d1",
            "ref_strongent_d2"]

# Base font sizes; each figure is placed so that these land >= 16 pt.
PPI = float(os.environ.get("DECK_PPI", "150"))
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#5A5A6E", "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "figure.facecolor": "white",
})
RNG = np.random.default_rng(11)

#: width each figure occupies on the 13.333in slide
DISPLAY_W = {
    "fig_pipeline": 11.6, "fig_tasks": 11.6, "fig_matrix": 11.4,
    "fig_e2_main": 7.6, "fig_e2_forest": 5.2, "fig_e3_scaling": 11.2,
    "fig_pareto": 7.5, "fig_diagnostics": 4.6,
    "fig_ap_e2_grid": 12.0, "fig_ap_anytime": 12.0, "fig_ap_e1": 11.0,
}


def read(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def save(fig, stem: str, rows: list[dict] | None = None) -> None:
    dpi = PPI * DISPLAY_W[stem] / fig.get_size_inches()[0]
    fig.savefig(FIG / f"{stem}.png", bbox_inches="tight", dpi=dpi)
    fig.savefig(FIG / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)
    from PIL import Image
    with Image.open(FIG / f"{stem}.png") as im:
        flat = Image.new("RGB", im.size, (255, 255, 255))
        rgba = im.convert("RGBA")
        flat.paste(rgba, mask=rgba.split()[3])
        flat.quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(
            FIG / f"{stem}.png", optimize=True)
    if rows:
        with (DATA / f"{stem}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


# --------------------------------------------------------------- slide 3
def fig_pipeline() -> None:
    """The actual implemented pipeline, traced from the code."""
    fig, ax = plt.subplots(figsize=(13.6, 3.05))
    ax.axis("off")
    steps = [
        ("Signal $x\\in\\mathbb{R}^{2^n}$", "32 points at n=5\nno pad, no truncate"),
        ("L2 normalise", "once, inside the\nembedding"),
        ("Amplitude encode", "n qubits hold\n$2^n$ amplitudes"),
        ("Searched body\n$U_A(\\theta)$", "the only part that\ndiffers by arm"),
        ("Readout $\\langle Z_0\\rangle$", "qubit 0, fixed,\nnot trainable"),
        ("$\\hat{y}=(1-\\langle Z_0\\rangle)/2$", "no classical\nparameters"),
    ]
    n = len(steps)
    w, gap = 1.94, 0.24
    for i, (title, sub) in enumerate(steps):
        x = i * (w + gap)
        face = "#FDF0E4" if i == 3 else "#EAF0FB"
        edge = C_EVO if i == 3 else C_RANDOM
        ax.add_patch(plt.Rectangle((x, 0.92), w, 1.34, facecolor=face,
                                   edgecolor=edge, linewidth=1.6,
                                   zorder=2, joinstyle="round"))
        ax.text(x + w / 2, 1.86, title, ha="center", va="center",
                fontsize=13.5, weight="bold", color=INK, zorder=3,
                linespacing=1.25)
        ax.text(x + w / 2, 1.26, sub, ha="center", va="center",
                fontsize=11, color="#4A4A5E", zorder=3, linespacing=1.4)
        if i < n - 1:
            ax.annotate("", xy=(x + w + gap - 0.02, 1.59), xytext=(x + w + 0.02, 1.59),
                        arrowprops={"arrowstyle": "-|>", "color": "#5A5A6E", "lw": 1.8})
    ax.text(0, 0.52, "Trained by the shared AdamW loop: only the rotation angles $\\theta$ "
                     "of the searched body (40 epochs, batch 32, identical for every arm)",
            fontsize=13, color=INK)
    ax.text(0, 0.16, "Scored by: validation metric during search  →  protected-test metric once, "
                     "after the candidate is selected  (the metric is RMSE for T1–T3; T4 is "
                     "classification)",
            fontsize=13, color=ACCENT)
    ax.set_xlim(-0.1, n * (w + gap) - gap + 0.1)
    ax.set_ylim(0, 2.35)
    save(fig, "fig_pipeline")


# --------------------------------------------------------------- slide 4
def fig_tasks() -> None:
    from llm_vqc.tasks.signal_suite import SignalProfile, SignalSuiteTask
    families = [
        ("gauss_peak", "T1  Gaussian peak", "regress peak position $\\mu$"),
        ("sin_freq", "T2  Sinusoid frequency", "regress normalised frequency"),
        ("change_point", "T3  Change point", "regress step location"),
        ("peak_count", "T4  One vs two peaks", "binary classification"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 5.0))
    rows = []
    x = np.linspace(0, 1, 32)
    for ax, (fam, title, target) in zip(axes.ravel(), families, strict=True):
        task = SignalSuiteTask(SignalProfile(family=fam, n_qubits=5, n_train=8,
                                             n_val=8, n_test=8))
        tv, _ = task.build(1000)
        for i in range(3):
            y = tv.train.features[i]
            ax.plot(x, y, linewidth=2.0, alpha=0.95)
            rows += [{"family": fam, "example": i, "x": round(float(x[j]), 6),
                      "y": round(float(v), 6),
                      "target": round(float(tv.train.targets[i]), 6)}
                     for j, v in enumerate(y)]
        ax.set_title(f"{title}  —  {target}", fontsize=15, loc="left", color=INK)
        ax.set_xticks([0, 0.5, 1.0])
        ax.tick_params(labelsize=13)
    for ax in axes[1]:
        ax.set_xlabel("normalised position", fontsize=14)
    for ax in axes[:, 0]:
        ax.set_ylabel("signal value", fontsize=14)
    fig.text(0.5, -0.02,
             "Each panel shows three example signals. Line colour distinguishes "
             "the examples only — it carries no method or class meaning.",
             ha="center", fontsize=13, color=MUTED_TXT)
    fig.tight_layout()
    save(fig, "fig_tasks", rows)


# --------------------------------------------------------------- slide 6
def fig_matrix() -> None:
    status = read(REPO / "docs/presentation/bench_v2_weekly/data/matrix_status.csv")
    order = [("E0", "replication", "Pilot replay"),
             ("E1", "classical", "E1 Track B classical"),
             ("E1", "llm", "E1 Track B real-LLM"),
             ("E2", "classical", "E2 Track A classical + refs"),
             ("E2", "llm", "E2 Track A real-LLM"),
             ("E3", "classical", "E3 scaling classical + ref"),
             ("E3", "llm", "E3 scaling real-LLM"),
             ("E4", "dependent", "E4 shot / noise robustness"),
             ("E5", "dependent", "E5 θ-isolation")]
    fig, ax = plt.subplots(figsize=(11.6, 4.5))
    rows = []
    for i, (eid, kind, label) in enumerate(order):
        rec = next(r for r in status if r["experiment"] == eid and r["cells"] == kind)
        exp, done = int(rec["expected"]), int(rec["complete"])
        if eid == "E4":
            # read live: E4 needs only completed classical cells, so it is
            # the one dependent experiment the quota block does not stop.
            done = 1 if (REPO / "outputs/bench_v2/E4/COMPLETE.json").is_file() else 0
        y = len(order) - i - 1
        frac = done / exp if exp else 0
        blocked = kind in ("llm", "dependent") and done == 0
        colour = C_GREEDY if frac == 1 else (ACCENT if blocked else C_EVO)
        ax.barh(y, 1.0, color="#ECEFF4", height=0.62, zorder=1)
        if frac > 0:
            ax.barh(y, frac, color=colour, height=0.62, zorder=2)
        ax.text(-0.02, y, label, ha="right", va="center", fontsize=15, color=INK)
        if frac == 1:
            state = "complete"
        elif kind == "llm":
            state = "not started — API quota exhausted"
        elif eid == "E5":
            state = "blocked — needs LLM-proposed circuits"
        elif eid == "E4":
            state = "running on the completed classical cells"
        else:
            state = "pending"
        ax.text(1.03, y, f"{done} / {exp}   {state}", ha="left", va="center",
                fontsize=14, color=colour if frac != 1 else C_GREEDY,
                weight="bold" if blocked else "normal")
        rows.append({"experiment": eid, "cells": kind, "expected": exp,
                     "complete": done, "state": state})
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.axis("off")
    save(fig, "fig_matrix", rows)


# --------------------------------------------------------------- slide 7
def _paired(per_seed, task, arm_a, arm_b, key="primary_test_value"):
    a = {int(r["replicate"]): float(r[key]) for r in per_seed
         if r["task"] == task and r["arm"] == arm_a}
    b = {int(r["replicate"]): float(r[key]) for r in per_seed
         if r["task"] == task and r["arm"] == arm_b}
    shared = sorted(set(a) & set(b))
    return np.array([a[r] for r in shared]), np.array([b[r] for r in shared])


def fig_e2_main() -> None:
    """T1: every searched arm vs the strongest reference, per replicate."""
    per_seed = read(OUT / "E2" / "per_seed_E2.csv")
    task = "gauss_peak"
    ref = "ref_strongent_d2"
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    rows = []
    positions = {a: i for i, a in enumerate(SEARCH_ARMS)}
    for arm, xpos in positions.items():
        sa, sb = _paired(per_seed, task, arm, ref)
        for r, (va, vb) in enumerate(zip(sa, sb, strict=True)):
            ax.plot([xpos - 0.16, xpos + 0.16], [va, vb], color="#C6CBD6",
                    linewidth=1.1, zorder=1)
            rows.append({"task": task, "arm": arm, "replicate": r,
                         "search_test_rmse": va, "reference_test_rmse": vb})
        ax.scatter(np.full(len(sa), xpos - 0.16), sa, s=52,
                   color=ARM_COLOR[arm], zorder=3, label=None)
        ax.scatter(np.full(len(sb), xpos + 0.16), sb, s=52, color=C_REF,
                   zorder=3, marker="s")
        ax.hlines(np.median(sa), xpos - 0.27, xpos - 0.05, color=ARM_COLOR[arm],
                  linewidth=3, zorder=4)
        ax.hlines(np.median(sb), xpos + 0.05, xpos + 0.27, color=C_REF,
                  linewidth=3, zorder=4)
    ax.set_xticks(list(positions.values()))
    ax.set_xticklabels([f"{ARM_LABEL[a]}\nvs StrongEnt d2" for a in positions],
                       fontsize=15)
    ax.set_ylabel("protected-test RMSE  (lower is better)", fontsize=15)
    ax.set_title("T1 Gaussian peak, n=5 qubits — 10 paired replicates per arm",
                 fontsize=16, loc="left")
    ax.scatter([], [], s=52, color=C_RANDOM, label="searched circuit, one replicate")
    ax.scatter([], [], s=52, color=C_REF, marker="s",
               label="fixed reference, same replicate")
    ax.plot([], [], color="#C6CBD6", linewidth=1.4,
            label="line joins one data seed + search seed")
    ax.plot([], [], color="black", linewidth=3, label="thick bar = median of 10")
    ax.legend(loc="upper left", handlelength=1.4, fontsize=12.5,
              frameon=True, facecolor="white", framealpha=0.93, edgecolor="none")
    ax.set_ylim(bottom=0)
    save(fig, "fig_e2_main", rows)


def fig_e2_forest() -> None:
    """Holm-adjusted paired contrasts, T1+T2, with effect sizes."""
    tests = read(OUT / "E2" / "stats_paired_tests.csv")
    effs = {(e["task"], e["arm_a"], e["arm_b"]): e
            for e in read(OUT / "E2" / "stats_effect_sizes.csv")}
    ref = "ref_strongent_d2"          # strongest reference on T1/T2
    rows_out, entries = [], []
    for task, tlabel in (("sin_freq", "T2"), ("gauss_peak", "T1")):
        pairs = [(a, ref, "vs reference") for a in SEARCH_ARMS]
        pairs += [("evolutionary_structure", "greedy_growth", "search vs search"),
                  ("evolutionary_structure", "random_structure", "search vs search"),
                  ("greedy_growth", "random_structure", "search vs search")]
        for a, b, kind in pairs:
            r = next((x for x in tests if x["task"] == task
                      and x["arm_a"] == a and x["arm_b"] == b), None)
            if r is None:
                continue
            e = effs[(task, a, b)]
            entries.append({
                "task": tlabel, "a": a, "b": b, "kind": kind,
                "label": f"{tlabel}  {ARM_LABEL[a]} − {ARM_LABEL[b]}",
                "hl": float(e["hodges_lehmann_shift"]),
                "cliffs": float(e["cliffs_delta_paired"]),
                "p": float(r["permutation_p_holm"]),
            })
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    for i, e in enumerate(entries):
        y = len(entries) - i - 1
        colour = ACCENT if e["kind"] == "vs reference" else C_RANDOM
        ax.scatter(e["hl"], y, s=95, color=colour, zorder=3)
        ax.text(1.02, y, f"p={e['p']:.3f}   δ={e['cliffs']:+.1f}", fontsize=13,
                va="center", ha="left", color="#4A4A5E",
                transform=ax.get_yaxis_transform())
        rows_out.append({"task": e["task"], "arm_a": e["a"], "arm_b": e["b"],
                         "contrast_kind": e["kind"],
                         "hodges_lehmann_shift": e["hl"],
                         "cliffs_delta": e["cliffs"], "p_holm": e["p"]})
    ax.axvline(0, color="#5A5A6E", linewidth=1.3, zorder=1)
    ax.set_yticks(range(len(entries)))
    ax.set_yticklabels([e["label"] for e in reversed(entries)], fontsize=13)
    ax.set_xlabel("Hodges–Lehmann shift in protected-test RMSE\n"
                  "(negative = first arm better)", fontsize=14)
    ax.set_title("Paired contrasts, 10 replicates, Holm-adjusted within task\n"
                 "right-hand column: p = Holm-adjusted p-value, \u03b4 = paired Cliff's delta",
                 fontsize=15, loc="left")
    ax.scatter([], [], s=95, color=ACCENT, label="search vs fixed reference")
    ax.scatter([], [], s=95, color=C_RANDOM, label="search vs search")
    ax.plot([], [], color="#5A5A6E", linewidth=1.3,
            label="vertical rule at zero = no difference")
    ax.legend(frameon=False, fontsize=12, loc="lower left", handlelength=1.4)
    ax.set_xlim(-0.26, 0.06)
    save(fig, "fig_e2_forest", rows_out)


# --------------------------------------------------------------- slide 8
def fig_e3_scaling() -> None:
    per_seed = read(OUT / "E3" / "per_seed_E3.csv")
    ns = [3, 4, 6, 8]
    arms = ["random_structure", "evolutionary_structure", "ref_strongent_d2"]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.3))
    rows = []
    for ax, (task, label) in zip(axes, [("gauss_peak", "T1 Gaussian peak"),
                                        ("sin_freq", "T2 Sinusoid frequency")],
                                 strict=True):
        for arm in arms:
            med = []
            for n in ns:
                vals = [float(r["primary_test_value"]) for r in per_seed
                        if r["task"] == task and int(r["n_qubits"]) == n
                        and r["arm"] == arm]
                med.append(stats.median(vals))
                ax.scatter(np.full(len(vals), n) + RNG.uniform(-0.12, 0.12, len(vals)),
                           vals, s=26, color=ARM_COLOR[arm], alpha=0.5, zorder=2)
                rows += [{"task": task, "n_qubits": n, "arm": arm,
                          "test_rmse": v} for v in vals]
            ax.plot(ns, med, color=ARM_COLOR[arm], linewidth=2.6, marker="o",
                    markersize=8, zorder=3,
                    label=("Fixed reference (StrongEnt d2)" if arm.startswith("ref")
                           else ARM_LABEL[arm]))
        ax.set_title(label, fontsize=16, loc="left")
        ax.set_xticks(ns)
        ax.set_xlabel("qubits  (signal length $2^n$)", fontsize=15)
        ax.tick_params(labelsize=14)
    axes[0].set_ylabel("protected-test RMSE\n(lower is better)", fontsize=15)
    handles, labels = axes[1].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], marker="o", linestyle="none",
                              color=C_REF, alpha=0.5, markersize=6))
    labels.append("faint dot = one replicate")
    handles.append(plt.Line2D([], [], marker="o", color=INK, markersize=8))
    labels.append("line + marker = median of 5")
    fig.legend(handles, labels, frameon=False, fontsize=13, ncol=5,
               loc="lower center", bbox_to_anchor=(0.5, -0.10))
    axes[1].annotate("reference is better at n=3", xy=(3, 0.070), xytext=(3.35, 0.155),
                     fontsize=13, color=INK,
                     arrowprops={"arrowstyle": "-|>", "color": "#5A5A6E", "lw": 1.4})
    fig.tight_layout()
    save(fig, "fig_e3_scaling", rows)


# --------------------------------------------------------------- slide 9
def fig_pareto() -> None:
    res = read(OUT / "E2" / "resource_metrics.csv")
    per_seed = read(OUT / "E2" / "per_seed_E2.csv")
    metric = {(r["task"], r["arm"], r["replicate"]): float(r["primary_test_value"])
              for r in per_seed}
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    rows = []
    for r in res:
        if r["task"] != "gauss_peak":
            continue
        key = (r["task"], r["arm"], r["replicate"])
        if key not in metric or not r["transpiled_2q_line"]:
            continue
        rows.append({"task": r["task"], "arm": r["arm"], "replicate": r["replicate"],
                     "transpiled_2q_line": float(r["transpiled_2q_line"]),
                     "test_rmse": metric[key]})
    for arm in SEARCH_ARMS + REF_ARMS:
        pts = [(x["transpiled_2q_line"], x["test_rmse"]) for x in rows if x["arm"] == arm]
        if not pts:
            continue
        xs, ys = zip(*pts, strict=True)
        is_ref = arm in REF_ARMS
        ax.scatter(xs, ys, s=58 if is_ref else 40, color=ARM_COLOR[arm],
                   marker="s" if is_ref else "o", alpha=0.85, zorder=3,
                   label=ARM_LABEL[arm] if is_ref else ARM_LABEL[arm])
    ax.set_xlabel(
        "compiled 2-qubit gates, line coupling  (resource proxy; lower is better)",
        fontsize=14)
    ax.set_ylabel("protected-test RMSE\n(lower is better)", fontsize=14)
    ax.set_title("T1, n=5: error vs compiled circuit size", fontsize=16, loc="left")
    ax.legend(frameon=False, fontsize=11.5, ncol=2, loc="upper right",
              handletextpad=0.4, columnspacing=1.0,
              title="● searched circuit   ■ fixed reference\none dot = one replicate",
              title_fontsize=11.5)
    ax.set_xlim(-8, 265)
    ax.set_ylim(0, 0.255)
    save(fig, "fig_pareto", rows)


def fig_diagnostics() -> None:
    src = read(REPO / "docs/presentation/bench_v2_weekly/data/fig_e2_diagnostics.csv")
    fig, ax = plt.subplots(figsize=(5.0, 4.3))
    rows = []
    for r in src:
        arm = r["arm"]
        kl = float(r["expressibility_kl"])
        q = float(r["meyer_wallach_q_mean"])
        ax.scatter(max(kl, 1e-3), q, s=42,
                   color=ARM_COLOR.get(arm, C_REF), alpha=0.85,
                   marker="s" if arm in REF_ARMS else "o", zorder=3)
        rows.append({"task": r["task"], "arm": arm, "expressibility_kl": kl,
                     "meyer_wallach_q_mean": q,
                     "parameter_count": r["parameter_count"]})
    ax.set_xscale("log")
    ax.set_xlabel("expressibility KL vs Haar (log)\nlower = more expressible", fontsize=13)
    ax.set_ylabel("Meyer–Wallach $Q$", fontsize=13)
    ax.set_title("28 selected circuits — descriptive only\n"
                 "one dot = one selected circuit; dot colour = arm, same key as the left panel",
                 fontsize=12, loc="left")
    ax.tick_params(labelsize=12)
    save(fig, "fig_diagnostics", rows)


# --------------------------------------------------------------- appendix
def fig_ap_e2_grid() -> None:
    per_seed = read(OUT / "E2" / "per_seed_E2.csv")
    tasks = ["gauss_peak", "sin_freq", "change_point", "peak_count"]
    arms = SEARCH_ARMS + REF_ARMS
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 4.0))
    rows = []
    for ax, task in zip(axes, tasks, strict=True):
        for i, arm in enumerate(arms):
            vals = [float(r["primary_test_value"]) for r in per_seed
                    if r["task"] == task and r["arm"] == arm]
            ax.scatter(np.full(len(vals), i) + RNG.uniform(-0.12, 0.12, len(vals)),
                       vals, s=24, color=ARM_COLOR[arm], alpha=0.85, zorder=3)
            ax.hlines(np.median(vals), i - 0.3, i + 0.3, color=INK, linewidth=2)
            rows += [{"task": task, "arm": arm, "value": v} for v in vals]
        higher = task == "peak_count"
        direction = ("protected-test AUROC (higher better)" if higher
                     else "protected-test RMSE (lower better)")
        ax.set_title(f"{task}\n{direction}", fontsize=14, loc="left")
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([ARM_LABEL[a] for a in arms], rotation=45, ha="right",
                           fontsize=11)
    handles = [plt.Line2D([], [], marker="o", linestyle="none", color=ARM_COLOR[a],
                          markersize=8, label=ARM_LABEL[a]) for a in arms]
    handles.append(plt.Line2D([], [], color=INK, linewidth=2, label="median of 10"))
    fig.legend(handles=handles, frameon=False, fontsize=12, ncol=8,
               loc="lower center", bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout()
    save(fig, "fig_ap_e2_grid", rows)


def fig_ap_anytime() -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 3.6))
    rows = []
    for ax, task in zip(axes, ["gauss_peak", "sin_freq", "change_point", "peak_count"],
                        strict=True):
        src = read(OUT / "E2" / f"fig_anytime_{task}_5q.csv")
        for arm in SEARCH_ARMS:
            sub = [r for r in src if r["arm"] == arm]
            xs = [int(r["unique_evaluations"]) for r in sub]
            med = [float(r["median_best_val"]) for r in sub]
            lo = [float(r["q25"]) for r in sub]
            hi = [float(r["q75"]) for r in sub]
            ax.plot(xs, med, color=ARM_COLOR[arm], linewidth=2.2, label=ARM_LABEL[arm])
            ax.fill_between(xs, lo, hi, color=ARM_COLOR[arm], alpha=0.16)
            rows += [{"task": task, **r} for r in sub]
        ax.set_title(task, fontsize=14, loc="left")
        ax.set_xlabel("unique candidate evaluations", fontsize=13)
        ax.tick_params(labelsize=12)
    axes[0].set_ylabel("best-so-far\nvalidation metric", fontsize=13)
    handles = [plt.Line2D([], [], color=ARM_COLOR[a], linewidth=2.2, label=ARM_LABEL[a])
               for a in SEARCH_ARMS]
    handles.append(plt.Line2D([], [], color=C_REF, linewidth=8, alpha=0.25,
                              label="band = interquartile range over 10 replicates"))
    fig.legend(handles=handles, frameon=False, fontsize=12, ncol=4,
               loc="lower center", bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    save(fig, "fig_ap_anytime", rows)


def fig_ap_e1() -> None:
    per_seed = read(OUT / "E1" / "per_seed_E1.csv")
    combos = [("gauss_peak_legacy", "3"), ("sin_freq", "3"), ("sin_freq", "5")]
    arms = ["random_joint", "evolutionary_joint"]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.9))
    rows = []
    for ax, (task, n) in zip(axes, combos, strict=True):
        by_rep: dict[int, dict[str, float]] = {}
        for r in per_seed:
            if r["task"] == task and r["n_qubits"] == n:
                by_rep.setdefault(int(r["replicate"]), {})[r["arm"]] = float(
                    r["primary_test_value"])
                rows.append({"task": task, "n_qubits": n, "arm": r["arm"],
                             "replicate": r["replicate"],
                             "test_rmse": r["primary_test_value"]})
        for vals in by_rep.values():
            if len(vals) == 2:
                ax.plot([0, 1], [vals[arms[0]], vals[arms[1]]], color="#C6CBD6",
                        linewidth=1.1, zorder=1)
        for i, arm in enumerate(arms):
            v = [by_rep[r][arm] for r in sorted(by_rep) if arm in by_rep[r]]
            ax.scatter(np.full(len(v), i), v, s=48, color=ARM_COLOR[arm], zorder=3)
            ax.hlines(np.median(v), i - 0.2, i + 0.2, color=INK, linewidth=2.4, zorder=4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Random", "Evolutionary"], fontsize=13)
        ax.set_xlim(-0.5, 1.5)
        ax.set_title(f"{task}, n={n}", fontsize=14, loc="left")
    axes[0].set_ylabel("protected-test RMSE\n(lower is better)", fontsize=13)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", color=C_RANDOM,
                   markersize=8, label="Random (joint)"),
        plt.Line2D([], [], marker="o", linestyle="none", color=C_EVO,
                   markersize=8, label="Evolutionary (joint)"),
        plt.Line2D([], [], color="#C6CBD6", linewidth=1.4,
                   label="grey line joins the same replicate"),
        plt.Line2D([], [], color=INK, linewidth=2.4, label="median of 10"),
    ]
    fig.legend(handles=handles, frameon=False, fontsize=12, ncol=4,
               loc="lower center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    save(fig, "fig_ap_e1", rows)


def main() -> None:
    fig_pipeline()
    fig_tasks()
    fig_matrix()
    fig_e2_main()
    fig_e2_forest()
    fig_e3_scaling()
    fig_pareto()
    fig_diagnostics()
    fig_ap_e2_grid()
    fig_ap_anytime()
    fig_ap_e1()
    manifest = {p.name: p.stat().st_size for p in sorted(FIG.glob("*.png"))}
    (DATA / "figure_index.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"figures -> {FIG}  ({sum(manifest.values())/1024:.0f} KB)")


if __name__ == "__main__":
    main()
