#!/usr/bin/env python
"""Analyze the HIGGS-v1 matrix with BLOCK-LEVEL hierarchical inference, hardware-
aware selection + Pareto, selection gain, classification diagnostics, power, and
22 figures. All from durable per-block stores + run summary. No raw data published.

Usage: python scripts/capacity_controlled/analyze_higgs.py --tag pilot
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from llm_vqc.evaluation.model import HybridQNNModel  # noqa: E402
from llm_vqc.evaluation.seeds import train_seed_for_circuit  # noqa: E402
from llm_vqc.evaluation.store import ResultStore  # noqa: E402
from llm_vqc.experiments.capacity_controlled import higgs_hardware as HW  # noqa: E402
from llm_vqc.experiments.capacity_controlled import higgs_task as HT  # noqa: E402
from llm_vqc.experiments.capacity_controlled import space as S  # noqa: E402
from llm_vqc.ir.budget import ProposalOutcome  # noqa: E402
from llm_vqc.ir.schema import CircuitIR  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "capacity_controlled_higgs_v1"
TASK = "HIGGS"
ARMS = ("controlled_random", "controlled_evolutionary", "controlled_greedy")
LABEL = {"controlled_random": "Random", "controlled_evolutionary": "Evolutionary", "controlled_greedy": "Greedy"}
COLOR = {"controlled_random": "#3767b3", "controlled_evolutionary": "#c8781e", "controlled_greedy": "#8a5fb0"}
CONSUMING = (ProposalOutcome.VALID, ProposalOutcome.DUPLICATE, ProposalOutcome.FAILED)
DELTA = HT.EQUIVALENCE_MARGIN            # 0.01
DELTAS = (0.005, 0.01, 0.02)
CAP = HT.TOTAL_TRAINABLE_PARAMS          # 53
PARETO_REF = (1.0, 40.0)
_HW_CACHE: dict[str, dict] = {}


def _save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def _csv(path, rows):
    if not rows:
        path.write_text(""); return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def tost(d, delta, alpha=0.05):
    d = np.asarray(d, float); n = len(d); mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    if se == 0:
        return {"mean_diff": mean, "ci90": [mean, mean], "equivalent": bool(abs(mean) < delta)}
    df = n - 1; tcrit = float(stats.t.ppf(0.95, df))
    ci = [mean - tcrit * se, mean + tcrit * se]
    return {"mean_diff": mean, "ci90": ci, "equivalent": bool(ci[0] > -delta and ci[1] < delta)}


def block_bootstrap(block_contrasts, n=2000, seed=0):
    rng = np.random.default_rng(seed); v = np.asarray(block_contrasts, float)
    m = [float(np.mean(rng.choice(v, len(v), replace=True))) for _ in range(n)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def cliffs_delta(a, b):
    gt = sum(1 for x, y in itertools.product(a, b) if x > y)
    lt = sum(1 for x, y in itertools.product(a, b) if x < y)
    return (gt - lt) / (len(a) * len(b)) if a and b else 0.0


def _hw(hash_, ir):
    if hash_ not in _HW_CACHE:
        _HW_CACHE[hash_] = HW.transpile_metrics(ir)
    return _HW_CACHE[hash_]


def anytime(store, run_id, seed):
    best, seen, cum, npr, nuq = None, set(), 0.0, 0, 0
    px, ux, tx, ys = [], [], [], []
    for e in store.iter_proposal_events(run_id):
        if e.outcome in CONSUMING:
            npr += 1
            if e.structural_hash:
                ts = train_seed_for_circuit(seed, e.structural_hash)
                c = store.get_cached(TASK, e.structural_hash, ts)
                if e.structural_hash not in seen:
                    seen.add(e.structural_hash); nuq += 1
                    if c is not None:
                        cum += float(c.wall_clock_seconds or 0.0)
                if c and c.val_metric_value is not None and (best is None or c.val_metric_value < best):
                    best = c.val_metric_value
            px.append(npr); ux.append(nuq); tx.append(cum); ys.append(best)
    return {"proposals": px, "unique": ux, "walltime": tx, "best_val": ys}


def run_candidates(store, run_id, seed):
    """Ordered list of (hash, val_logloss) for consuming proposals."""
    out = []
    for e in store.iter_proposal_events(run_id):
        if e.outcome in CONSUMING and e.structural_hash:
            ts = train_seed_for_circuit(seed, e.structural_hash)
            c = store.get_cached(TASK, e.structural_hash, ts)
            if c and c.val_metric_value is not None:
                out.append((e.structural_hash, c.val_metric_value, ts))
    return out


def test_logloss_of(store, h, ts, test):
    c = store.get_cached(TASK, h, ts); w = store.get_trained_weights(TASK, h, ts)
    ir = CircuitIR.model_validate_json(c.circuit_canonical_json)
    return HT.evaluate_all_test_metrics(ir, w["classical_state"], test)["logloss"], ir


def hardware_aware_pick(cands, store, test):
    """Best val -> retain within 0.01 -> fewest transpiled 2q -> tie depth -> tie hash."""
    if not cands:
        return None
    best_val = min(v for _, v, _ in cands)
    retained = [(h, v, ts) for h, v, ts in cands if v <= best_val + 0.01]
    scored = []
    for h, v, ts in retained:
        c = store.get_cached(TASK, h, ts)
        ir = CircuitIR.model_validate_json(c.circuit_canonical_json)
        m = _hw(h, ir)
        scored.append((m["transpiled_two_qubit_cx"], m["transpiled_depth"], h, v, ts, m))
    scored.sort(key=lambda x: (x[0], x[1], x[2]))
    return scored[0]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="pilot")
    ap.add_argument("--run-dir", default=str(ROOT / "runs" / "capacity_controlled_higgs_v1"))
    args = ap.parse_args(); run_dir = Path(args.run_dir); OUT.mkdir(parents=True, exist_ok=True)
    summary = json.loads((run_dir / f"{args.tag}_summary.json").read_text())
    blocks, seeds = summary["blocks"], summary["seeds"]
    stores = {b: ResultStore(run_dir / f"block_{b}" / "results.sqlite") for b in blocks}
    tests = {}
    assignments = HT.build_block_assignments()
    for b in blocks:
        _, tests[b] = HT.build_block(b, assignments)

    # candidate_results + capacity verify
    cand, capset = [], set()
    for b in blocks:
        for e in stores[b].all_evaluations(TASK):
            if not e.circuit_canonical_json or not e.circuit_cost:
                continue
            ir = CircuitIR.model_validate_json(e.circuit_canonical_json)
            m = HybridQNNModel(ir, raw_feature_dim=HT.RAW_FEATURE_DIM, head_out_dim=1)
            total = int(sum(p.numel() for p in m.parameters())); capset.add(total)
            hw = _hw(e.structural_hash, ir); c = e.circuit_cost
            cand.append({"block": b, "structural_hash": (e.structural_hash or "")[:12],
                         "depth": c.depth, "two_qubit_gate_count": c.two_qubit_gate_count,
                         "total_trainable_params": total, "val_logloss": e.val_metric_value,
                         "transpiled_two_qubit": hw["transpiled_two_qubit_cx"], "transpiled_depth": hw["transpiled_depth"],
                         "wall_clock_seconds": round(e.wall_clock_seconds, 4)})
    if capset != {CAP}:
        raise RuntimeError(f"HIGGS CAPACITY NOT UNIFORM: {sorted(capset)} != {{{CAP}}}")
    print(f"[verify] all {len(cand)} candidates have exactly {CAP} params")
    _csv(OUT / "candidate_results.csv", cand)

    # run_summary + resource + anytime + test cells + hardware-aware
    run_rows, res_rows, curves = [], [], {}
    test_cell = {a: {} for a in ARMS}; val_cell = {a: {} for a in ARMS}
    conf = {a: {"tn": 0, "fp": 0, "fn": 0, "tp": 0} for a in ARMS}
    hw_rows = []
    for r in summary["runs"]:
        a, b, seed = r["arm"], r["block"], r["seed"]; cell = (b, seed)
        t = r["test"]; test_cell[a][cell] = t["logloss"] if t else None
        val_cell[a][cell] = r["selected_val_logloss"]; led = r["ledger_summary"]
        if t:
            for k in conf[a]:
                conf[a][k] += t["confusion"][k]
        run_rows.append({"run_id": r["run_id"], "arm": a, "block": b, "seed": seed,
                         "consumed_budget": led["consumed_budget"], "num_invalid": led["num_invalid"],
                         "num_duplicate": led["num_duplicate"], "num_failed": led["num_failed"],
                         "num_unique": led["num_unique"], "selected_val_logloss": r["selected_val_logloss"],
                         "test_logloss": t["logloss"] if t else None, "test_auc": t["auc"] if t else None,
                         "test_accuracy": t["accuracy"] if t else None, "test_brier": t["brier"] if t else None})
        cur = anytime(stores[b], r["run_id"], seed); curves[r["run_id"]] = cur
        res_rows.append({"run_id": r["run_id"], "arm": a, "block": b, "seed": seed,
                         "proposals_consumed": led["consumed_budget"], "unique_trainings": led["num_unique"],
                         "duplicates": led["num_duplicate"], "invalid": led["num_invalid"], "failed": led["num_failed"],
                         "cumulative_walltime_s": round(cur["walltime"][-1], 3) if cur["walltime"] else 0.0})
        # hardware-aware selection
        cands = run_candidates(stores[b], r["run_id"], seed)
        perf_ll = t["logloss"] if t else None
        perf_hw = _hw(r["selected_structural_hash"], CircuitIR.model_validate_json(
            stores[b].get_cached(TASK, r["selected_structural_hash"], train_seed_for_circuit(seed, r["selected_structural_hash"])).circuit_canonical_json)) if r["selected_structural_hash"] else None
        hwpick = hardware_aware_pick(cands, stores[b], tests[b])
        hw_ll = None; hw_2q = None
        if hwpick:
            hw_2q = hwpick[0]
            hw_ll, _ = test_logloss_of(stores[b], hwpick[2], hwpick[4], tests[b])
        hw_rows.append({"run_id": r["run_id"], "arm": a, "block": b, "seed": seed,
                        "perf_selected_test_logloss": perf_ll,
                        "perf_selected_transpiled_2q": perf_hw["transpiled_two_qubit_cx"] if perf_hw else None,
                        "hw_selected_test_logloss": hw_ll, "hw_selected_transpiled_2q": hw_2q})
    _csv(OUT / "run_summary.csv", run_rows); (OUT / "run_summary.json").write_text(json.dumps(
        {"runs": run_rows, "block_assignments": summary.get("block_assignments", [])}, indent=2))
    _csv(OUT / "resource_accounting.csv", res_rows)
    _csv(OUT / "hardware_cost_summary.csv", hw_rows)

    # block_summary
    block_rows = []
    for b in blocks:
        for a in ARMS:
            tv = [test_cell[a][(b, s)] for s in seeds if test_cell[a].get((b, s)) is not None]
            block_rows.append({"block": b, "arm": a, "n": len(tv), "test_logloss_median": float(np.median(tv)),
                               "test_logloss_mean": float(np.mean(tv))})
    _csv(OUT / "block_summary.csv", block_rows)

    # aux: classical, ablations
    aux = summary["aux"]; cls_rows, c7_rows, ft_rows, pe_rows = [], [], [], []
    for b in blocks:
        for seed in seeds:
            for x in aux[str(b)]["classical"][str(seed)]:
                cls_rows.append({"block": b, "seed": seed, "baseline": x["baseline"],
                                 "total_params": x.get("total_params"), "test_logloss": x["logloss"],
                                 "test_auc": x["auc"], "test_accuracy": x["accuracy"]})
                if x["baseline"] == "C7_fair_nas":
                    c7_rows.append({"block": b, "seed": seed, "selected": json.dumps(x.get("selected", {})),
                                    "total_params": x["total_params"], "val_logloss": x.get("val_logloss"),
                                    "test_logloss": x["logloss"], "n_candidates": x.get("n_candidates")})
        for row in aux[str(b)]["freeze_train"]:
            if row["trainable_logloss"] is not None and row["frozen_logloss"] is not None:
                ft_rows.append({**row, "diff_frozen_minus_trainable": row["frozen_logloss"] - row["trainable_logloss"]})
        for row in aux[str(b)]["product_entangled"]:
            if row["entangled_logloss"] is not None and row["product_logloss"] is not None:
                pe_rows.append({**row, "diff_product_minus_entangled": row["product_logloss"] - row["entangled_logloss"]})
    _csv(OUT / "classical_baseline_summary.csv", cls_rows)
    _csv(OUT / "classical_search_summary.csv", c7_rows)
    _csv(OUT / "paired_freeze_train_results.csv", ft_rows)
    _csv(OUT / "paired_entanglement_results.csv", pe_rows)

    # selection gain
    sel_rows = []
    fixed_dist = [r["trainable_logloss"] for r in ft_rows]
    for r in summary["runs"]:
        if r["arm"] != "controlled_random":
            continue
        b, seed = r["block"], r["seed"]; cands = run_candidates(stores[b], r["run_id"], seed)
        for B in (1, 5, 10, 25):
            sub = cands[:B]
            if sub:
                bh, bv, bts = min(sub, key=lambda x: x[1])
                ll, _ = test_logloss_of(stores[b], bh, bts, tests[b])
                sel_rows.append({"block": b, "seed": seed, "B": B, "selected_test_logloss": ll})
    _csv(OUT / "selection_gain_summary.csv", sel_rows)
    for st in stores.values():
        st.close()

    # ---- statistics (block-level) ----
    def pooled(cm, a):
        return [cm[a][(b, s)] for b in blocks for s in seeds if cm[a].get((b, s)) is not None]

    def block_contrasts(a, bb):
        out = []
        for b in blocks:
            d = [test_cell[a][(b, s)] - test_cell[bb][(b, s)] for s in seeds
                 if test_cell[a].get((b, s)) is not None and test_cell[bb].get((b, s)) is not None]
            if d:
                out.append(float(np.mean(d)))
        return out

    stat = {"note": f"HIGGS-v1 {len(blocks)} blocks x {len(seeds)} seeds; primary metric log-loss; "
                    f"BLOCK is the inference unit; every candidate {CAP} params.",
            "per_arm_test_logloss": {a: {"n_cells": len(pooled(test_cell, a)),
                                         "median": float(np.median(pooled(test_cell, a))),
                                         "mean": float(np.mean(pooled(test_cell, a))),
                                         "sd": float(np.std(pooled(test_cell, a), ddof=1))} for a in ARMS}}
    # equivalence
    equiv = {"primary_metric": "logloss", "margin": DELTA, "inference_unit": "block", "pairs": {}}
    for a, bb in itertools.combinations(ARMS, 2):
        bc = block_contrasts(a, bb)
        equiv["pairs"][f"{a}__vs__{bb}"] = {
            "n_blocks": len(bc), "block_contrast_mean": float(np.mean(bc)), "block_contrast_sd": float(np.std(bc, ddof=1)),
            "tost_by_delta": {f"{dl}": tost(bc, dl) for dl in DELTAS},
            "block_bootstrap_ci95": block_bootstrap(bc)}
    equiv["all_pairs_equivalent_at_primary"] = bool(all(
        equiv["pairs"][k]["tost_by_delta"][f"{DELTA}"]["equivalent"] for k in equiv["pairs"]))
    (OUT / "equivalence_analysis.json").write_text(json.dumps(equiv, indent=2))

    # KW/MWU on block medians (secondary)
    block_med = {a: [np.median([test_cell[a][(b, s)] for s in seeds if test_cell[a].get((b, s)) is not None]) for b in blocks] for a in ARMS}
    H, p = stats.kruskal(*[block_med[a] for a in ARMS])
    stat["kruskal_wallis_block_medians"] = {"H": float(H), "p_value": float(p)}

    # ablations (block+arch clustered)
    def cluster_block(rows, unit_key, diff_key):
        by_arch, by_block = {}, {}
        for r in rows:
            by_arch.setdefault(r[unit_key], []).append(r[diff_key])
            by_block.setdefault(r["block"], []).append(r[diff_key])
        arch_means = [float(np.mean(v)) for v in by_arch.values()]
        block_means = [float(np.mean(v)) for v in by_block.values()]
        # hierarchical bootstrap: blocks then archs
        rng = np.random.default_rng(0); cellmap = {}
        for r in rows:
            cellmap.setdefault((r["block"], r[unit_key]), []).append(r[diff_key])
        bl = sorted({r["block"] for r in rows}); ar = sorted({r[unit_key] for r in rows})
        boot = []
        for _ in range(3000):
            vals = []
            for b in rng.choice(bl, len(bl), True):
                for u in rng.choice(ar, len(ar), True):
                    cell = cellmap.get((b, u))
                    if cell:
                        vals.append(cell[rng.integers(len(cell))])
            if vals:
                boot.append(float(np.mean(vals)))
        ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        return {"n_pairs": len(rows), "n_blocks": len(block_means), "n_architectures": len(arch_means),
                "n_seeds": len({r["seed"] for r in rows}), "overall_mean": float(np.mean([r[diff_key] for r in rows])),
                "arch_averaged_mean": float(np.mean(arch_means)), "block_averaged_mean": float(np.mean(block_means)),
                "hierarchical_bootstrap_ci95": ci, "cohen_dz_block": float(np.mean(block_means) / (np.std(block_means, ddof=1) + 1e-12)),
                "ci_includes_zero": bool(ci[0] <= 0 <= ci[1]), "frac_positive": float(np.mean(np.asarray([r[diff_key] for r in rows]) > 0))}
    stat["ablation_freeze_minus_train"] = cluster_block(ft_rows, "arch", "diff_frozen_minus_trainable")
    stat["ablation_freeze_minus_train"]["positive_means"] = "trainable better than frozen"
    stat["ablation_product_minus_entangled"] = cluster_block(pe_rows, "pair", "diff_product_minus_entangled")
    stat["ablation_product_minus_entangled"]["positive_means"] = "entangled better than product"

    # classical + selection + hardware/pareto
    def bmed(name):
        v = [r["test_logloss"] for r in cls_rows if r["baseline"] == name]
        return float(np.median(v)) if v else None
    stat["baselines_test_logloss_median"] = {n: bmed(n) for n in
        ("C0_majority", "C1_logreg", "C2_linear_svm_cal", "C3_rbf_svm_cal", "C4_hist_gb", "C5_random_forest", "C6_fixed_mlp", "C7_fair_nas")}
    stat["baselines_test_logloss_median"]["QS_best_arm"] = min(stat["per_arm_test_logloss"][a]["median"] for a in ARMS)
    stat["selection_gain"] = {"fixed_arch_expected_test_logloss": float(np.median(fixed_dist)) if fixed_dist else None,
                              "best_of_B_median": {str(B): (lambda v: float(np.median(v)) if v else None)(
                                  [r["selected_test_logloss"] for r in sel_rows if r["B"] == B]) for B in (1, 5, 10, 25)}}
    stat["hardware"] = _pareto_stats(hw_rows)
    (OUT / "statistical_analysis.json").write_text(json.dumps(stat, indent=2))
    (OUT / "pareto_summary.json").write_text(json.dumps(stat["hardware"], indent=2))

    # power (block-level)
    obs_sd = float(np.median([equiv["pairs"][k]["block_contrast_sd"] for k in equiv["pairs"]]))
    _power(obs_sd, DELTA, len(blocks))

    _figures(summary, blocks, seeds, test_cell, val_cell, block_rows, curves, equiv, stat,
             cls_rows, ft_rows, pe_rows, sel_rows, fixed_dist, cand, run_rows, conf, hw_rows, run_dir, tests)
    _manifests(args.tag, summary, blocks, seeds)
    print(f"wrote HIGGS-v1 machine-readable results + 22 figures to {OUT}")


def _pareto_stats(hw_rows):
    def front(points):
        pts = sorted(points)  # (logloss, 2q) minimize both
        front = []; best_2q = float("inf")
        for ll, q in pts:
            if q < best_2q:
                front.append((ll, q)); best_2q = q
        return front

    def hypervolume(fr):
        fr = sorted(fr); hv = 0.0; prev_ll = 0.0
        for ll, q in fr:
            hv += max(0.0, PARETO_REF[0] - ll) * max(0.0, PARETO_REF[1] - q) * 0 + 0
        # simple 2D dominated area vs reference (staircase)
        hv = 0.0; last_q = PARETO_REF[1]
        for ll, q in sorted(fr, key=lambda x: x[0]):
            hv += max(0.0, PARETO_REF[0] - ll) * max(0.0, last_q - q)
            last_q = min(last_q, q)
        return hv
    res = {"reference_point": {"logloss": PARETO_REF[0], "transpiled_two_qubit": PARETO_REF[1]}, "per_arm": {}}
    for a in ARMS:
        pts = [(r["hw_selected_test_logloss"], r["hw_selected_transpiled_2q"]) for r in hw_rows
               if r["arm"] == a and r["hw_selected_test_logloss"] is not None and r["hw_selected_transpiled_2q"] is not None]
        pts += [(r["perf_selected_test_logloss"], r["perf_selected_transpiled_2q"]) for r in hw_rows
                if r["arm"] == a and r["perf_selected_test_logloss"] is not None and r["perf_selected_transpiled_2q"] is not None]
        fr = front(pts)
        res["per_arm"][a] = {"n_points": len(pts), "pareto_front": fr, "hypervolume": hypervolume(fr)}
    return res


def _power(sd, delta, n, out=OUT):
    rng = np.random.default_rng(1); res = {}
    for true in (0.0, 0.005, 0.01, 0.02):
        eq = 0
        for _ in range(10000):
            d = rng.normal(true, sd, n); m = d.mean(); se = d.std(ddof=1) / np.sqrt(n); df = n - 1
            if max(stats.t.cdf((m - delta) / se, df), stats.t.sf((m + delta) / se, df)) < 0.05:
                eq += 1
        res[f"{true:.3f}"] = {"P_equivalence": eq / 10000}
    (out / "power_analysis.json").write_text(json.dumps(
        {"note": "block-level power using observed block-contrast SD", "observed_block_contrast_sd": sd,
         "n_blocks": n, "margin": delta, "scenarios": res}, indent=2))


def _grouped(ax, named, title, yl="test log-loss (lower better)"):
    labs = [n for n, _ in named]; meds = [np.median(v) if v else np.nan for _, v in named]
    x = np.arange(len(labs)); ax.bar(x, meds, 0.62, color="#3767b3", alpha=0.8)
    for i, (_, v) in enumerate(named):
        if v:
            ax.scatter([i] * len(v), v, color="black", s=8, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(labs, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(yl); ax.set_title(title, fontsize=10); ax.grid(axis="y", alpha=0.25)


def _paired_hist(fn, diffs, title):
    diffs = [d for d in diffs if d is not None]
    fig, ax = plt.subplots(figsize=(7.5, 4.2)); ax.hist(diffs, bins=30, color="#3767b3", alpha=0.8)
    ax.axvline(0, color="crimson", ls="--", lw=1.5); ax.axvline(np.median(diffs), color="black", lw=1.5, label=f"median={np.median(diffs):+.4f}")
    ax.set_xlabel("paired difference"); ax.set_ylabel("count")
    ax.set_title(f"{title}\nn_pairs={len(diffs)}, frac>0={np.mean(np.asarray(diffs) > 0):.2f}", fontsize=9)
    ax.legend(fontsize=9); _save(fig, fn)


def _figures(summary, blocks, seeds, test_cell, val_cell, block_rows, curves, equiv, stat,
             cls_rows, ft_rows, pe_rows, sel_rows, fixed_dist, cand, run_rows, conf, hw_rows, run_dir, tests):
    def pooled(cm, a):
        return [cm[a][(b, s)] for b in blocks for s in seeds if cm[a].get((b, s)) is not None]

    # 1 qualification (from json)
    try:
        q = json.loads((OUT / "task_qualification.json").read_text())["attempts"][0]["models"]
        fig, ax = plt.subplots(figsize=(7, 4)); names = [m["model"] for m in q]
        ax.bar(names, [m["val_auroc"] for m in q], color="#3767b3", alpha=0.8)
        ax.axhline(0.65, color="crimson", ls="--"); ax.axhline(0.92, color="crimson", ls="--")
        ax.set_ylabel("dev-val AUROC"); ax.set_title("HIGGS qualification: AUROC (band = pass range)", fontsize=10)
        ax.tick_params(axis="x", rotation=20); _save(fig, "fig01_qualification")
    except Exception:  # noqa: BLE001
        pass

    # 2 class balance / split integrity
    fig, ax = plt.subplots(figsize=(7, 4))
    bals = [np.mean(tests[b].targets) for b in blocks]
    ax.bar([f"b{b}" for b in blocks], bals, color="#4a9e5c", alpha=0.8); ax.axhline(0.5, color="grey", ls="--")
    ax.set_ylim(0, 1); ax.set_ylabel("protected-test positive fraction")
    ax.set_title("Per-block class balance (10 disjoint blocks, 2000 test each)", fontsize=10); _save(fig, "fig02_class_balance")

    # 3 test logloss by arm & block
    fig, ax = plt.subplots(figsize=(10, 4.6)); x = np.arange(len(ARMS)); wd = 0.08
    for j, b in enumerate(blocks):
        for i, a in enumerate(ARMS):
            v = [test_cell[a][(b, s)] for s in seeds if test_cell[a].get((b, s)) is not None]
            ax.bar(x[i] + (j - 4.5) * wd, np.median(v), wd, color=COLOR[a], alpha=0.5, edgecolor="none")
    ax.set_xticks(x); ax.set_xticklabels([LABEL[a] for a in ARMS]); ax.set_ylabel("test log-loss")
    ax.set_title("Protected-test log-loss by arm and block (median; 10 blocks/arm)", fontsize=10)
    ax.grid(axis="y", alpha=0.25); _save(fig, "fig03_test_logloss_by_arm_block")

    # 4 equivalence intervals
    fig, ax = plt.subplots(figsize=(8.5, 4)); labs, mn, lo, hi = [], [], [], []
    for k, pr in equiv["pairs"].items():
        t = pr["tost_by_delta"][f"{DELTA}"]; labs.append(k.replace("controlled_", "").replace("__vs__", " vs "))
        mn.append(t["mean_diff"]); lo.append(t["ci90"][0]); hi.append(t["ci90"][1])
    y = np.arange(len(labs)); ax.errorbar(mn, y, xerr=[np.array(mn) - np.array(lo), np.array(hi) - np.array(mn)], fmt="o", color="#3767b3", capsize=4)
    ax.axvline(0, color="grey"); ax.axvline(-DELTA, color="crimson", ls="--"); ax.axvline(DELTA, color="crimson", ls="--")
    ax.axvspan(-DELTA, DELTA, color="green", alpha=0.08); ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=9)
    ax.set_xlabel("block-level mean paired test-logloss diff (90% CI)")
    ax.set_title(f"Block-level equivalence: 90% CIs vs margin ±{DELTA}", fontsize=10); _save(fig, "fig04_equivalence_intervals")

    # 5 block variability
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for a in ARMS:
        meds = [np.median([test_cell[a][(b, s)] for s in seeds if test_cell[a].get((b, s)) is not None]) for b in blocks]
        ax.plot(blocks, meds, "-o", color=COLOR[a], label=LABEL[a])
    ax.set_xlabel("block"); ax.set_ylabel("median test log-loss"); ax.set_xticks(blocks)
    ax.set_title("Block-to-block variability", fontsize=10); ax.legend(fontsize=9); ax.grid(alpha=0.25); _save(fig, "fig05_block_variability")

    # 6,7,8 anytime
    for axis, fn, xl in (("proposals", "fig06_anytime_proposals", "proposals"),
                         ("unique", "fig07_anytime_unique", "unique trainings"),
                         ("walltime", "fig08_anytime_walltime", "cumulative wall-clock (s)")):
        fig, ax = plt.subplots(figsize=(8.5, 4.4))
        for a in ARMS:
            for rid, cur in curves.items():
                if not rid.startswith(a):
                    continue
                xs = [x for x, yv in zip(cur[axis], cur["best_val"]) if yv is not None]; ys = [yv for yv in cur["best_val"] if yv is not None]
                if xs:
                    ax.plot(xs, ys, color=COLOR[a], alpha=0.2, lw=0.7)
        for a in ARMS:
            ax.plot([], [], color=COLOR[a], label=LABEL[a])
        ax.set_xlabel(xl); ax.set_ylabel("best-so-far val log-loss"); ax.legend(fontsize=9)
        ax.set_title(f"Anytime: best val log-loss vs {xl}", fontsize=10); ax.grid(alpha=0.25); _save(fig, fn)

    # 9 classical baselines
    fig, ax = plt.subplots(figsize=(10, 4.6))
    names = [("QS best arm", pooled(test_cell, min(ARMS, key=lambda a: np.median(pooled(test_cell, a)))))]
    for n in ("C0_majority", "C1_logreg", "C2_linear_svm_cal", "C3_rbf_svm_cal", "C4_hist_gb", "C5_random_forest", "C6_fixed_mlp", "C7_fair_nas"):
        names.append((n.replace("_", " "), [r["test_logloss"] for r in cls_rows if r["baseline"] == n]))
    _grouped(ax, names, "Searched VQC vs classical baselines (test log-loss)"); _save(fig, "fig09_classical_baselines")

    # 10 fair NAS vs VQC
    fig, ax = plt.subplots(figsize=(7, 4.4))
    names = [(LABEL[a], pooled(test_cell, a)) for a in ARMS] + [("C7 fair NAS", [r["test_logloss"] for r in cls_rows if r["baseline"] == "C7_fair_nas"])]
    _grouped(ax, names, "VQC search vs fair classical NAS"); _save(fig, "fig10_vqc_vs_fair_nas")

    # 11,12 paired ablation effects
    _paired_hist("fig11_freeze_vs_train", [r["diff_frozen_minus_trainable"] for r in ft_rows], "frozen − trainable test log-loss (positive ⇒ training helps)")
    _paired_hist("fig12_product_vs_entangled", [r["diff_product_minus_entangled"] for r in pe_rows], "product − entangled test log-loss (positive ⇒ entanglement helps)")

    # 13 selection gain
    fig, ax = plt.subplots(figsize=(7.5, 4.4)); Bs = (1, 5, 10, 25); meds = []
    for B in Bs:
        v = [r["selected_test_logloss"] for r in sel_rows if r["B"] == B and r["selected_test_logloss"] is not None]
        meds.append(np.median(v)); ax.scatter([B] * len(v), v, color="#3767b3", alpha=0.25, s=10)
    ax.plot(Bs, meds, "-o", color="#3767b3", label="best-of-B median")
    if fixed_dist:
        ax.axhline(np.median(fixed_dist), color="crimson", ls="--", label="expected fixed arch")
    ax.set_xlabel("selection budget B"); ax.set_ylabel("protected-test log-loss"); ax.legend(fontsize=9)
    ax.set_title("Validation-selection gain (block-aggregated)", fontsize=10); ax.grid(alpha=0.25); _save(fig, "fig13_selection_gain")

    # 14 capacity verification
    fig, ax = plt.subplots(figsize=(7.5, 4)); tot = [c["total_trainable_params"] for c in cand]
    ax.plot(range(len(tot)), tot, ".", color="#4a9e5c", ms=2); ax.axhline(CAP, color="crimson", ls="--"); ax.set_ylim(CAP - 20, CAP + 20)
    ax.set_xlabel("candidate index"); ax.set_ylabel("total trainable params")
    ax.set_title(f"Capacity verification: all {len(tot)} candidates = {CAP} (unique {sorted(set(tot))})", fontsize=10); _save(fig, "fig14_capacity_verification")

    # 15 rates
    fig, ax = plt.subplots(figsize=(7.5, 4.4)); x = np.arange(len(ARMS)); wd = 0.25
    for j, (key, lab) in enumerate((("num_invalid", "invalid"), ("num_duplicate", "duplicate"), ("num_failed", "failed"))):
        rates = [sum(r[key] for r in run_rows if r["arm"] == a) / max(1, sum(r["consumed_budget"] + r["num_invalid"] for r in run_rows if r["arm"] == a)) for a in ARMS]
        ax.bar(x + (j - 1) * wd, rates, wd, label=lab)
    ax.set_xticks(x); ax.set_xticklabels([LABEL[a] for a in ARMS]); ax.legend(fontsize=9); ax.set_ylabel("fraction")
    ax.set_title("Invalid / duplicate / failed rates", fontsize=10); ax.grid(axis="y", alpha=0.25); _save(fig, "fig15_proposal_rates")

    # 16 runtime
    fig, ax = plt.subplots(figsize=(7.5, 4)); rt = [c["wall_clock_seconds"] for c in cand if c["wall_clock_seconds"]]
    ax.hist(rt, bins=40, color="#3767b3", alpha=0.8); ax.set_xlabel("train s/candidate"); ax.set_ylabel("count")
    ax.set_title(f"Runtime per candidate (n={len(rt)}, median={np.median(rt):.2f}s)", fontsize=10); _save(fig, "fig16_runtime")

    # 17 AUROC by arm
    fig, ax = plt.subplots(figsize=(7, 4.2))
    names = [(LABEL[a], [r["test_auc"] for r in run_rows if r["arm"] == a and r["test_auc"] is not None]) for a in ARMS]
    _grouped(ax, names, "Protected-test AUROC by arm (higher better)", yl="test AUROC"); _save(fig, "fig17_auroc")

    # 18 calibration (best run)
    best = min((r for r in summary["runs"] if r["test"] is not None), key=lambda r: r["test"]["logloss"])
    st = ResultStore(run_dir / f"block_{best['block']}" / "results.sqlite")
    h = best["selected_structural_hash"]; ts = train_seed_for_circuit(best["seed"], h)
    c = st.get_cached(TASK, h, ts); w = st.get_trained_weights(TASK, h, ts); st.close()
    ir = CircuitIR.model_validate_json(c.circuit_canonical_json)
    md = HT.evaluate_all_test_metrics(ir, w["classical_state"], tests[best["block"]], with_curves=True)
    fig, ax = plt.subplots(figsize=(5, 5)); cal = md["calibration"]
    ax.plot([c2["conf"] for c2 in cal], [c2["acc"] for c2 in cal], "-o", color="#c8781e"); ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("mean predicted prob"); ax.set_ylabel("empirical accuracy"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(f"Calibration — best run {best['run_id']} (Brier={md['brier']:.3f})", fontsize=9); _save(fig, "fig18_calibration")
    # 22 (roc for best; also used as external audit placeholder handled separately)
    fig, ax = plt.subplots(figsize=(5, 5)); ax.plot(md["roc"]["fpr"], md["roc"]["tpr"], "-", color="#3767b3"); ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title(f"ROC best run (AUC={md['auc']:.3f})", fontsize=9); _save(fig, "fig17b_roc_best")

    # 19 transpiled 2q costs
    fig, ax = plt.subplots(figsize=(7.5, 4)); ax.hist([c["transpiled_two_qubit"] for c in cand], bins=30, color="#8a5fb0", alpha=0.8)
    ax.set_xlabel("transpiled two-qubit (cx) on linear topology"); ax.set_ylabel("count")
    ax.set_title("Transpiled two-qubit cost across candidates", fontsize=10); _save(fig, "fig19_transpiled_two_qubit")

    # 20 Pareto fronts
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for a in ARMS:
        fr = stat["hardware"]["per_arm"][a]["pareto_front"]
        if fr:
            fr = sorted(fr); ax.plot([p[0] for p in fr], [p[1] for p in fr], "-o", color=COLOR[a], label=f"{LABEL[a]} (HV={stat['hardware']['per_arm'][a]['hypervolume']:.2f})")
    ax.set_xlabel("protected-test log-loss"); ax.set_ylabel("transpiled two-qubit")
    ax.set_title("Performance–cost Pareto fronts by arm", fontsize=10); ax.legend(fontsize=8); ax.grid(alpha=0.25); _save(fig, "fig20_pareto")

    # 21 perf vs hw-aware selection
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    perf = [r["perf_selected_test_logloss"] for r in hw_rows if r["perf_selected_test_logloss"] is not None]
    hw = [r["hw_selected_test_logloss"] for r in hw_rows if r["hw_selected_test_logloss"] is not None]
    p2 = [r["perf_selected_transpiled_2q"] for r in hw_rows if r["perf_selected_transpiled_2q"] is not None]
    h2 = [r["hw_selected_transpiled_2q"] for r in hw_rows if r["hw_selected_transpiled_2q"] is not None]
    ax.bar([0, 1, 2, 3], [np.median(perf), np.median(hw), np.median(p2), np.median(h2)],
           color=["#3767b3", "#c8781e", "#3767b3", "#c8781e"], alpha=0.8)
    ax.set_xticks([0, 1, 2, 3]); ax.set_xticklabels(["perf logloss", "hw logloss", "perf 2q", "hw 2q"], fontsize=8)
    ax.set_title("Performance-selected vs hardware-aware-selected (medians)", fontsize=10); _save(fig, "fig21_perf_vs_hw")

    # 22 confusion matrices per arm
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for ax, a in zip(axes, ARMS):
        c2 = conf[a]; mat = np.array([[c2["tn"], c2["fp"]], [c2["fn"], c2["tp"]]])
        ax.imshow(mat, cmap="Blues"); ax.set_title(LABEL[a], fontsize=10)
        for (i, j), v in np.ndenumerate(mat):
            ax.text(j, i, str(v), ha="center", va="center", fontsize=10)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pred bg", "pred sig"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["true bg", "true sig"])
    fig.suptitle("Aggregate protected-test confusion (all blocks/seeds)", fontsize=11); fig.tight_layout(); _save(fig, "fig22_confusion")


def _manifests(tag, summary, blocks, seeds):
    files = sorted(p.name for p in OUT.iterdir() if p.is_file())
    (OUT / "experiment_manifest.json").write_text(json.dumps(
        {"package": "capacity_controlled_higgs_v1", "tag": tag, "git_sha": summary.get("git_sha"),
         "blocks": blocks, "seeds": seeds, "budget": summary["budget"], "arms": ARMS, "files": files,
         "reproduction": {"run": f"python scripts/capacity_controlled/run_experiment_higgs.py --blocks 0-9 --seeds {','.join(map(str, seeds))} --budget {summary['budget']} --tag {tag}",
                          "analyze": f"python scripts/capacity_controlled/analyze_higgs.py --tag {tag}",
                          "external_holdout": "python scripts/capacity_controlled/external_holdout_higgs.py"},
         "integrity": "no LLM, no amplitude, block-level inference, protected test + external holdout quarantined, uniform 53-param capacity verified."}, indent=2))
    (OUT / "artifact_manifest.json").write_text(json.dumps(
        {"figures": [f for f in files if f.startswith("fig") and f.endswith(".png")],
         "tables": [f for f in files if f.endswith(".csv")], "analyses": [f for f in files if f.endswith(".json")]}, indent=2))
    (OUT / "experiment_config.json").write_text(json.dumps(
        {"space": "HIGGS_capacity_controlled_v1", "total_trainable_params": CAP, "blocks": blocks, "seeds": seeds,
         "budget": summary["budget"], "arms": list(ARMS), "selection_metric": "logloss",
         "primary_test_metric": "logloss", "equivalence_margin": DELTA, "git_sha": summary.get("git_sha")}, indent=2))


if __name__ == "__main__":
    main()
