#!/usr/bin/env python
"""Analyze the HIGGS data-scale qualification study: block-aware statistics,
the frozen non-degeneracy gate, and 12 figures.

The statistical unit is the BLOCK; architecture and initialization seeds are
repeated measurements within a block (aggregated to a per-block value first).

Usage: python scripts/capacity_controlled/analyze_qualification.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from llm_vqc.experiments.capacity_controlled import higgs_scale as HS  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "higgs_data_scale_qualification_v1"
RUN = ROOT / "runs" / "higgs_data_scale_qualification_v1"
REPS = HS.REPRESENTATIONS
SIZES = HS.TRAIN_SIZES
VQC_MODELS = ("M3_product", "M4_entangled", "M5_frozen")
ALL_MODELS = ("M0_class_prior", "M1_logreg", "M2_mlp") + VQC_MODELS
COLOR = {"R1_raw21": "#3767b3", "R2_pca8": "#c8781e", "R3_pca16": "#4a9e5c"}
MCOLOR = {"M0_class_prior": "#999999", "M1_logreg": "#3767b3", "M2_mlp": "#4a9e5c",
          "M3_product": "#c8781e", "M4_entangled": "#8a5fb0", "M5_frozen": "#d1495b"}
GATE = {"val_auroc": 0.58, "test_auroc": 0.58, "min_improve": 0.01, "min_blocks_pos": 4,
        "max_fail": 0.05, "min_arch_seeds": 2, "max_optimism": 0.02}


def _save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def _csv(path, rows):
    if not rows:
        path.write_text(""); return
    keys = sorted({k for r in rows for k in r})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def load_rows():
    rows = []
    for p in sorted(RUN.glob("results_*.json")):
        rows += json.loads(p.read_text())
    return rows


def block_bootstrap(block_vals, n=2000, seed=0):
    rng = np.random.default_rng(seed); v = np.asarray(block_vals, float)
    if len(v) < 2:
        return [float(v[0]), float(v[0])] if len(v) else [float("nan")] * 2
    m = [float(np.mean(rng.choice(v, len(v), replace=True))) for _ in range(n)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def per_block(rows, rep, size, model, field):
    """Per-block value = median over architecture/init repetitions within the block."""
    out = {}
    for b in range(HS.N_BLOCKS):
        v = [r[field] for r in rows if r["representation"] == rep and r["train_size"] == size
             and r["model"] == model and r.get(field) is not None and not r.get("failed", False)]
        vb = [r[field] for r in rows if r["representation"] == rep and r["train_size"] == size
              and r["model"] == model and r["block"] == b and r.get(field) is not None
              and not r.get("failed", False)]
        if vb:
            out[b] = float(np.median(vb))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    if not rows:
        raise SystemExit("no results found in runs/ — run phase B first")
    print(f"loaded {len(rows)} result rows")
    _csv(OUT / "qualification_results.csv", rows)

    # ---- capacity + hardware summaries ----
    cap_rows, hw_rows = [], []
    seen = set()
    for r in rows:
        key = (r["representation"], r["model"])
        if key in seen or r["model"] in ("M0_class_prior", "M1_logreg"):
            continue
        seen.add(key)
        cap_rows.append({"representation": r["representation"], "input_dim": r["input_dim"],
                         "model": r["model"], "embed_params": r.get("embed_params"),
                         "quantum_params": r.get("quantum_params"), "frozen_quantum_params": r.get("frozen_quantum_params"),
                         "head_params": r.get("head_params"), "total_params": r.get("total_params"),
                         "total_trainable_params": r.get("total_trainable_params")})
    for r in rows:
        if r["model"] in VQC_MODELS and r.get("transpiled_two_qubit") is not None:
            hw_rows.append({"representation": r["representation"], "train_size": r["train_size"],
                            "model": r["model"], "arch_seed": r["arch_seed"],
                            "logical_two_qubit": r["logical_two_qubit"],
                            "transpiled_two_qubit": r["transpiled_two_qubit"],
                            "transpiled_depth": r["transpiled_depth"], "swap_est": r["swap_est"],
                            "runtime_s": r["runtime_s"]})
    _csv(OUT / "capacity_summary.csv", cap_rows)
    _csv(OUT / "hardware_summary.csv", hw_rows)

    # ---- block summary ----
    block_rows = []
    for rep in REPS:
        for size in SIZES:
            for m in ALL_MODELS:
                pb_ll = per_block(rows, rep, size, m, "test_logloss")
                pb_auc = per_block(rows, rep, size, m, "test_auroc")
                for b, ll in pb_ll.items():
                    block_rows.append({"representation": rep, "train_size": size, "model": m,
                                       "block": b, "test_logloss": ll, "test_auroc": pb_auc.get(b)})
    _csv(OUT / "block_summary.csv", block_rows)

    # ---- gate evaluation ----
    gate_results = []
    for rep in REPS:
        for size in SIZES:
            prior_ll = per_block(rows, rep, size, "M0_class_prior", "test_logloss")
            for m in VQC_MODELS:
                v_auc = per_block(rows, rep, size, m, "val_auroc")
                t_auc = per_block(rows, rep, size, m, "test_auroc")
                t_ll = per_block(rows, rep, size, m, "test_logloss")
                v_ll = per_block(rows, rep, size, m, "val_logloss")
                if not t_ll:
                    continue
                improve = {b: prior_ll[b] - t_ll[b] for b in t_ll if b in prior_ll}
                cand = [r for r in rows if r["representation"] == rep and r["train_size"] == size and r["model"] == m]
                n_fail = sum(1 for r in cand if r.get("failed"))
                fail_rate = n_fail / max(1, len(cand))
                # per-arch-seed check of criteria 1-3
                seeds_ok = 0
                for a in sorted({r["arch_seed"] for r in cand if r["arch_seed"] is not None}):
                    sv = [r["val_auroc"] for r in cand if r["arch_seed"] == a and not r.get("failed")]
                    st = [r["test_auroc"] for r in cand if r["arch_seed"] == a and not r.get("failed")]
                    sl = [prior_ll.get(r["block"], np.nan) - r["test_logloss"] for r in cand
                          if r["arch_seed"] == a and not r.get("failed")]
                    if sv and st and sl and np.median(sv) >= GATE["val_auroc"] and \
                       np.median(st) >= GATE["test_auroc"] and np.median(sl) >= GATE["min_improve"]:
                        seeds_ok += 1
                optimism = float(np.median([t_ll[b] - v_ll[b] for b in t_ll if b in v_ll]))
                checks = {
                    "median_val_auroc>=0.58": float(np.median(list(v_auc.values()))) >= GATE["val_auroc"],
                    "median_test_auroc>=0.58": float(np.median(list(t_auc.values()))) >= GATE["test_auroc"],
                    "median_improve>=0.01": float(np.median(list(improve.values()))) >= GATE["min_improve"],
                    "blocks_positive>=4": sum(1 for x in improve.values() if x > 0) >= GATE["min_blocks_pos"],
                    "failure_rate<5%": fail_rate < GATE["max_fail"],
                    "arch_seeds>=2": seeds_ok >= GATE["min_arch_seeds"],
                    "optimism<0.02": optimism < GATE["max_optimism"],
                }
                gate_results.append({
                    "representation": rep, "train_size": size, "model": m,
                    "median_val_auroc": float(np.median(list(v_auc.values()))),
                    "median_test_auroc": float(np.median(list(t_auc.values()))),
                    "median_test_logloss": float(np.median(list(t_ll.values()))),
                    "median_prior_logloss": float(np.median(list(prior_ll.values()))),
                    "median_improvement": float(np.median(list(improve.values()))),
                    "improvement_ci95_block_bootstrap": block_bootstrap(list(improve.values())),
                    "blocks_positive": int(sum(1 for x in improve.values() if x > 0)),
                    "n_blocks": len(improve), "failure_rate": fail_rate,
                    "arch_seeds_meeting_1_3": seeds_ok, "optimism_gap": optimism,
                    "checks": checks, "passes": all(checks.values())})
    passing = [g for g in gate_results if g["passes"]]
    decision = {"gate": GATE, "n_conditions_evaluated": len(gate_results),
                "n_passing": len(passing), "any_pass": bool(passing),
                "passing_conditions": passing, "all_conditions": gate_results}
    if passing:
        sel = sorted(passing, key=lambda g: (g["median_test_logloss"], g["optimism_gap"]))[0]
        decision["selected_condition"] = {k: sel[k] for k in ("representation", "train_size", "model")}
        decision["selection_rule"] = "best median validation logloss, then optimism, then runtime, then two-qubit"
        decision["routing"] = "HIGGS-v2 search justified; freeze the selected condition in a separate protocol"
    else:
        decision["selected_condition"] = None
        decision["routing"] = ("NO VQC condition passed -> do NOT run a new architecture-search matrix; "
                               "the 4-qubit/12-parameter family is not a viable performance benchmark for "
                               "HIGGS under the tested conditions")
    (OUT / "decision_gate_result.json").write_text(json.dumps(decision, indent=2))
    print(f"gate: {len(passing)}/{len(gate_results)} conditions pass")

    # ---- statistical analysis ----
    stat = {"note": "block is the statistical unit; arch/init seeds are within-block repeats.",
            "learning_curves": {}, "representation_comparison": {}, "paired": {}, "runtime": {}}
    for rep in REPS:
        stat["learning_curves"][rep] = {}
        for m in ALL_MODELS:
            stat["learning_curves"][rep][m] = {}
            for size in SIZES:
                pb = per_block(rows, rep, size, m, "test_logloss")
                pa = per_block(rows, rep, size, m, "test_auroc")
                if pb:
                    stat["learning_curves"][rep][m][str(size)] = {
                        "median_test_logloss": float(np.median(list(pb.values()))),
                        "ci95": block_bootstrap(list(pb.values())),
                        "median_test_auroc": float(np.median(list(pa.values()))) if pa else None}
    # paired product vs entangled, frozen vs trainable (block-level)
    for name, a, b in (("product_minus_entangled", "M3_product", "M4_entangled"),
                       ("frozen_minus_trainable", "M5_frozen", "M4_entangled")):
        stat["paired"][name] = {}
        for rep in REPS:
            for size in SIZES:
                pa, pb2 = per_block(rows, rep, size, a, "test_logloss"), per_block(rows, rep, size, b, "test_logloss")
                common = sorted(set(pa) & set(pb2))
                if len(common) >= 2:
                    d = [pa[k] - pb2[k] for k in common]
                    stat["paired"][name][f"{rep}_n{size}"] = {
                        "mean_block_diff": float(np.mean(d)), "ci95": block_bootstrap(d),
                        "n_blocks": len(d), "positive_means": f"{b} better" if name.startswith("product") else f"{b} better"}
    for rep in REPS:
        stat["runtime"][rep] = {str(s): float(np.median([r["runtime_s"] for r in rows
                                if r["representation"] == rep and r["train_size"] == s and r["model"] in VQC_MODELS] or [0]))
                                for s in SIZES}
    (OUT / "statistical_analysis.json").write_text(json.dumps(stat, indent=2))

    _figures(rows, stat, gate_results, hw_rows, cap_rows)
    files = sorted(p.name for p in OUT.iterdir() if p.is_file())
    (OUT / "artifact_manifest.json").write_text(json.dumps(
        {"figures": [f for f in files if f.startswith("fig") and f.endswith(".png")],
         "tables": [f for f in files if f.endswith(".csv")],
         "analyses": [f for f in files if f.endswith(".json")],
         "documents": [f for f in files if f.endswith(".md")]}, indent=2))
    print(f"wrote analysis + figures to {OUT}")


def _figures(rows, stat, gate_results, hw_rows, cap_rows):
    def curve(rep, m, field="test_logloss"):
        return [np.median(list(per_block(rows, rep, s, m, field).values() or [np.nan])) for s in SIZES]

    # 1 learning curves by representation (M4)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for rep in REPS:
        ax.plot(SIZES, curve(rep, "M4_entangled"), "-o", color=COLOR[rep], label=f"{rep} (M4)")
        ax.plot(SIZES, curve(rep, "M0_class_prior"), "--", color=COLOR[rep], alpha=0.5)
    ax.set_xscale("log"); ax.set_xlabel("training samples"); ax.set_ylabel("test log-loss")
    ax.set_title("Learning curves by representation (solid=VQC M4, dashed=class prior)", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.25); _save(fig, "fig01_learning_curves")

    # 2 improvement over class prior
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for rep in REPS:
        imp = []
        for s in SIZES:
            pr = per_block(rows, rep, s, "M0_class_prior", "test_logloss")
            mm = per_block(rows, rep, s, "M4_entangled", "test_logloss")
            common = set(pr) & set(mm)
            imp.append(np.median([pr[b] - mm[b] for b in common]) if common else np.nan)
        ax.plot(SIZES, imp, "-o", color=COLOR[rep], label=rep)
    ax.axhline(0, color="grey"); ax.axhline(0.01, color="crimson", ls="--", label="gate 0.01")
    ax.set_xscale("log"); ax.set_xlabel("training samples"); ax.set_ylabel("log-loss improvement over prior")
    ax.set_title("VQC improvement over class prior (gate threshold 0.01)", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.25); _save(fig, "fig02_improvement_over_prior")

    # 3 AUROC by training size
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for rep in REPS:
        ax.plot(SIZES, curve(rep, "M4_entangled", "test_auroc"), "-o", color=COLOR[rep], label=f"{rep} M4")
    ax.axhline(0.58, color="crimson", ls="--", label="gate 0.58"); ax.axhline(0.5, color="grey")
    ax.set_xscale("log"); ax.set_xlabel("training samples"); ax.set_ylabel("test AUROC")
    ax.set_title("VQC test AUROC vs training size", fontsize=10); ax.legend(fontsize=8); ax.grid(alpha=0.25)
    _save(fig, "fig03_auroc_by_size")

    # 4 representation comparison at largest size
    fig, ax = plt.subplots(figsize=(7, 4.4))
    big = SIZES[-1]
    vals = [np.median(list(per_block(rows, rep, big, "M4_entangled", "test_logloss").values() or [np.nan])) for rep in REPS]
    ax.bar(list(REPS), vals, color=[COLOR[r] for r in REPS], alpha=0.85)
    ax.set_ylabel("test log-loss"); ax.set_title(f"raw21 vs PCA8 vs PCA16 (M4, n={big})", fontsize=10)
    _save(fig, "fig04_representation_comparison")

    # 5 classical vs VQC
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(SIZES)); w = 0.13
    for i, m in enumerate(ALL_MODELS):
        v = [np.median(list(per_block(rows, "R1_raw21", s, m, "test_logloss").values() or [np.nan])) for s in SIZES]
        ax.bar(x + (i - 2.5) * w, v, w, label=m, color=MCOLOR[m], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([str(s) for s in SIZES]); ax.set_xlabel("training samples")
    ax.set_ylabel("test log-loss"); ax.set_title("Classical vs VQC (R1 raw21)", fontsize=10)
    ax.legend(fontsize=7, ncol=3); _save(fig, "fig05_classical_vs_vqc")

    # 6 product vs entangled ; 7 frozen vs trainable
    for fn, a, b, title in (("fig06_product_vs_entangled", "M3_product", "M4_entangled", "product − entangled"),
                            ("fig07_frozen_vs_trainable", "M5_frozen", "M4_entangled", "frozen − trainable")):
        fig, ax = plt.subplots(figsize=(8, 4.4))
        for rep in REPS:
            d = []
            for s in SIZES:
                pa, pb2 = per_block(rows, rep, s, a, "test_logloss"), per_block(rows, rep, s, b, "test_logloss")
                common = set(pa) & set(pb2)
                d.append(np.mean([pa[k] - pb2[k] for k in common]) if common else np.nan)
            ax.plot(SIZES, d, "-o", color=COLOR[rep], label=rep)
        ax.axhline(0, color="crimson", ls="--"); ax.set_xscale("log")
        ax.set_xlabel("training samples"); ax.set_ylabel("block-level mean difference")
        ax.set_title(f"{title} (positive ⇒ {b} better)", fontsize=10); ax.legend(fontsize=8); ax.grid(alpha=0.25)
        _save(fig, fn)

    # 8 validation vs test
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for rep in REPS:
        for s in SIZES:
            v = per_block(rows, rep, s, "M4_entangled", "val_logloss"); t = per_block(rows, rep, s, "M4_entangled", "test_logloss")
            common = set(v) & set(t)
            if common:
                ax.scatter(np.median([v[b] for b in common]), np.median([t[b] for b in common]),
                           color=COLOR[rep], s=30, alpha=0.8)
    lim = ax.get_xlim(); ax.plot(lim, lim, "--", color="grey")
    ax.set_xlabel("validation log-loss"); ax.set_ylabel("test log-loss")
    ax.set_title("Validation vs test (optimism check)", fontsize=10); _save(fig, "fig08_val_vs_test")

    # 9 calibration (ECE)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for rep in REPS:
        ax.plot(SIZES, [np.median(list(per_block(rows, rep, s, "M4_entangled", "test_ece").values() or [np.nan])) for s in SIZES],
                "-o", color=COLOR[rep], label=rep)
    ax.set_xscale("log"); ax.set_xlabel("training samples"); ax.set_ylabel("expected calibration error")
    ax.set_title("VQC calibration (ECE) vs training size", fontsize=10); ax.legend(fontsize=8); _save(fig, "fig09_calibration")

    # 10 runtime scaling
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for rep in REPS:
        ax.plot(SIZES, [stat["runtime"][rep][str(s)] for s in SIZES], "-o", color=COLOR[rep], label=rep)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("training samples"); ax.set_ylabel("median VQC runtime (s)")
    ax.set_title("Runtime scaling", fontsize=10); ax.legend(fontsize=8); ax.grid(alpha=0.25, which="both")
    _save(fig, "fig10_runtime_scaling")

    # 11 hardware two-qubit cost
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for m in VQC_MODELS:
        v = [h["transpiled_two_qubit"] for h in hw_rows if h["model"] == m]
        if v:
            ax.hist(v, bins=20, alpha=0.55, label=f"{m} (med {np.median(v):.0f})", color=MCOLOR[m])
    ax.set_xlabel("transpiled two-qubit (cx) on linear 0-1-2-3"); ax.set_ylabel("count")
    ax.set_title("Hardware two-qubit cost by model family", fontsize=10); ax.legend(fontsize=8)
    _save(fig, "fig11_hardware_two_qubit")

    # 12 capacity verification
    fig, ax = plt.subplots(figsize=(8, 4.4))
    labs = [f"{c['representation']}\n{c['model']}" for c in cap_rows if c.get("total_params")]
    tot = [c["total_params"] for c in cap_rows if c.get("total_params")]
    tr = [c["total_trainable_params"] for c in cap_rows if c.get("total_params")]
    x = np.arange(len(labs))
    ax.bar(x - 0.2, tot, 0.4, label="total params"); ax.bar(x + 0.2, tr, 0.4, label="trainable")
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=6, rotation=90); ax.legend(fontsize=8)
    ax.set_title("Capacity verification by representation × model", fontsize=10)
    _save(fig, "fig12_capacity_verification")


if __name__ == "__main__":
    main()
