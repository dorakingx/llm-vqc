#!/usr/bin/env python
"""Post-hoc cluster-aware reanalysis of the T2-v1 paired ablations.

The T2-v1 primary paired stats treated the n=300 (20 archs/pairs x 5 splits x 3
seeds) differences as independent. They are NOT: architectures, data splits, and
initialization conditions repeat. This script reanalyzes the *already-stored* T2
paired differences with proper clustering (no simulation rerun) and asks whether
the T2 conclusions survive. It does NOT modify the T2 output package; results go
into the HIGGS-v1 package (T2_CLUSTER_SENSITIVITY.md + t2_cluster_sensitivity.json).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
T2 = ROOT / "outputs" / "capacity_controlled_t2_v1"
OUT = ROOT / "outputs" / "capacity_controlled_higgs_v1"


def _load(path, unit_col, diff_col):
    rows = []
    for r in csv.DictReader(path.open()):
        if r[diff_col] in ("", "None"):
            continue
        rows.append({"unit": int(r[unit_col]), "split": int(r["split"]),
                     "seed": int(r["seed"]), "diff": float(r[diff_col])})
    return rows


def hierarchical_bootstrap(rows, n=4000, seed=0):
    """Resample splits (clusters), then units (arch/pair) within, then seeds."""
    rng = np.random.default_rng(seed)
    splits = sorted({r["split"] for r in rows})
    units = sorted({r["unit"] for r in rows})
    by = {}
    for r in rows:
        by.setdefault((r["split"], r["unit"]), []).append(r["diff"])
    means = []
    for _ in range(n):
        rs_splits = rng.choice(splits, len(splits), replace=True)
        rs_units = rng.choice(units, len(units), replace=True)
        vals = []
        for s in rs_splits:
            for u in rs_units:
                cell = by.get((s, u))
                if cell:
                    vals.append(cell[rng.integers(len(cell))])
        if vals:
            means.append(float(np.mean(vals)))
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))], float(np.mean(means))


def averaged_effect(rows, by_key):
    groups = {}
    for r in rows:
        groups.setdefault(r[by_key], []).append(r["diff"])
    per = {k: float(np.mean(v)) for k, v in groups.items()}
    vals = list(per.values())
    n = len(vals)
    mean = float(np.mean(vals)); sd = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 1 else 0.0
    # t CI
    from scipy import stats
    ci = [mean - stats.t.ppf(0.975, n - 1) * se, mean + stats.t.ppf(0.975, n - 1) * se] if n > 1 else [mean, mean]
    p = float(stats.ttest_1samp(vals, 0.0).pvalue) if n > 1 and sd > 0 else 1.0
    frac_pos = float(np.mean(np.asarray(vals) > 0))
    return {"n_clusters": n, "mean_of_cluster_means": mean, "ci95": ci, "t_p_value": p,
            "frac_clusters_positive": frac_pos}


def naive(rows):
    from scipy import stats
    d = np.array([r["diff"] for r in rows])
    w = stats.wilcoxon(d) if any(d != 0) else None
    return {"n_obs": len(d), "mean": float(d.mean()), "median": float(np.median(d)),
            "naive_wilcoxon_p_treating_iid": float(w.pvalue) if w else None,
            "frac_positive": float(np.mean(d > 0))}


def analyze(rows, label, positive_means):
    boot_ci, boot_mean = hierarchical_bootstrap(rows)
    return {"contrast": label, "positive_means": positive_means,
            "naive_iid": naive(rows),
            "architecture_averaged": averaged_effect(rows, "unit"),
            "split_averaged": averaged_effect(rows, "split"),
            "hierarchical_bootstrap_split_then_unit": {"ci95_mean_diff": boot_ci, "boot_mean": boot_mean},
            "conclusion_survives_clustering": bool(boot_ci[0] <= 0.0 <= boot_ci[1])}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ft = _load(T2 / "paired_freeze_train_results.csv", "arch", "diff_frozen_minus_trainable")
    pe = _load(T2 / "paired_entanglement_results.csv", "pair", "diff_product_minus_entangled")
    res = {
        "note": "Post-hoc cluster-aware reanalysis of stored T2-v1 paired ablations. "
                "No simulation rerun; T2 package unchanged. Clusters: data split (5) and "
                "architecture/pair (20); seeds (3) nested. statsmodels unavailable -> "
                "hierarchical bootstrap + cluster-averaged t-tests instead of mixed-effects.",
        "freeze_vs_train": analyze(ft, "frozen_minus_trainable", "positive => trainable better than frozen"),
        "product_vs_entangled": analyze(pe, "product_minus_entangled", "positive => entangled better than product"),
    }
    (OUT / "t2_cluster_sensitivity.json").write_text(json.dumps(res, indent=2))
    for k in ("freeze_vs_train", "product_vs_entangled"):
        b = res[k]
        print(f"=== {k} ===")
        print(f"  naive iid: mean={b['naive_iid']['mean']:+.5f} wilcoxon_p={b['naive_iid']['naive_wilcoxon_p_treating_iid']}")
        print(f"  arch-averaged (n=20): mean={b['architecture_averaged']['mean_of_cluster_means']:+.5f} "
              f"ci95={[round(x,5) for x in b['architecture_averaged']['ci95']]} p={b['architecture_averaged']['t_p_value']:.3f}")
        print(f"  split-averaged (n=5): mean={b['split_averaged']['mean_of_cluster_means']:+.5f} p={b['split_averaged']['t_p_value']:.3f}")
        print(f"  hier bootstrap CI95={[round(x,5) for x in b['hierarchical_bootstrap_split_then_unit']['ci95_mean_diff']]} "
              f"survives(includes 0)={b['conclusion_survives_clustering']}")
    print(f"wrote {OUT}/t2_cluster_sensitivity.json")


if __name__ == "__main__":
    main()
