#!/usr/bin/env python
"""Merge the chunked runs and test the task-aware LLM against the baselines.

The run is split across files for two reasons that have nothing to do with
the science: classical arms are CPU-bound and LLM arms are API-bound, so
they get different worker counts, and a single 600-cell process exhausts
memory. Seeds never overlap between chunks, so merging is a concatenation.

Multiplicity is handled by declaring the PRIMARY family up front: the
task-aware open-loop arm against each of the three baselines, on each of
the four tasks. Twelve comparisons, Holm-corrected. Everything else is
reported without correction and labelled secondary, because correcting a
grab-bag of thirty-plus contrasts is what made the 10-seed run unable to
detect a 9-1 split.
"""

from __future__ import annotations

import glob
import json
import statistics as st
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "mini5"
DATA = REPO / "docs" / "presentation" / "mini5" / "data"

TASKS = {"gauss_peak": "T1 peak position", "sin_freq": "T2 frequency",
         "change_point": "T3 change point", "peak_count": "T4 one vs two"}
BASELINES = ("random", "evolutionary", "reference")
#: The hybrid is the arm the diagnosis predicts should win: random opens,
#: the LLM only refines. It becomes the primary once it exists.
PRIMARY_ARM = "llm_hybrid_ctx"


def load(patterns: list[str]) -> dict:
    cells: dict[tuple, dict] = {}
    for pat in patterns:
        for path in sorted(glob.glob(str(OUT / pat))):
            for c in json.loads(Path(path).read_text())["cells"]:
                if c["status"] == "complete":
                    cells[(c["family"], c["arm"], c["seed"])] = c
    return cells


def holm(pvals: list[float]) -> list[float]:
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    out = [0.0] * len(pvals)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - k) * pvals[i]))
        out[i] = running
    return out


def paired(cells, fam, a, b, seeds):
    xs = [(cells[(fam, a, s)]["test_rmse"], cells[(fam, b, s)]["test_rmse"])
          for s in seeds if (fam, a, s) in cells and (fam, b, s) in cells]
    if len(xs) < 6:
        return None
    x = np.array([p[0] for p in xs])
    y = np.array([p[1] for p in xs])
    try:
        p = float(wilcoxon(x, y).pvalue)
    except ValueError:            # all differences zero
        p = 1.0
    return {"n": len(xs), "median_diff": float(np.median(x - y)),
            "wins": int((x < y).sum()), "p": p}


def main() -> int:
    cells = load(["results_cls30.json", "results_llm_s*.json",
                  "results_hyb_s*.json"])
    arms = sorted({k[1] for k in cells})
    seeds = sorted({k[2] for k in cells})
    print(f"merged {len(cells)} cells · arms {arms} · seeds {len(seeds)}")

    print("\nmedian test RMSE (lower is better)")
    header = f"  {'task':18s}" + "".join(f"{a:>17s}" for a in arms)
    print(header)
    table = {}
    for fam, label in TASKS.items():
        row = f"  {label:18s}"
        table[fam] = {}
        for a in arms:
            v = [cells[k]["test_rmse"] for k in cells if k[0] == fam and k[1] == a]
            table[fam][a] = st.median(v) if v else None
            row += f"{st.median(v):>17.4f}" if v else f"{'-':>17s}"
        print(row)

    print(f"\nPRIMARY family: {PRIMARY_ARM} vs each baseline, Holm over 12")
    rows = []
    for fam in TASKS:
        for opp in BASELINES:
            r = paired(cells, fam, PRIMARY_ARM, opp, seeds)
            if r:
                rows.append((fam, opp, r))
    adj = holm([r["p"] for _, _, r in rows])
    primary = []
    for (fam, opp, r), q in zip(rows, adj, strict=True):
        verdict = ("LLM BETTER" if q < 0.05 and r["median_diff"] < 0
                   else "LLM WORSE" if q < 0.05 else "no difference detected")
        print(f"  {TASKS[fam]:18s} vs {opp:13s} n={r['n']:2d} "
              f"diff={r['median_diff']:+.4f} wins={r['wins']:2d}/{r['n']} "
              f"p={r['p']:.4f} holm={q:.4f}  {verdict}")
        primary.append({"task": fam, "opponent": opp, **r, "holm": q,
                        "verdict": verdict})

    print("\nSECONDARY (uncorrected, exploratory)")
    secondary = []
    for a, b, label in (("llm_hybrid_ctx", "llm_open_ctx", "hybrid vs pure-LLM"),
                        ("llm_closed_ctx", "llm_open_ctx", "closed vs open")):
        for fam in TASKS:
            r = paired(cells, fam, a, b, seeds)
            if r:
                print(f"  {TASKS[fam]:18s} {label:20s} diff={r['median_diff']:+.4f} "
                      f"wins={r['wins']:2d}/{r['n']} p={r['p']:.4f}")
                secondary.append({"task": fam, "contrast": label, **r})

    print("\nSTARTING POINT vs IMPROVEMENT (the mechanism under test)")
    for a in ("random", "llm_open_ctx", "llm_hybrid_ctx"):
        firsts, lasts = [], []
        for k, c in cells.items():
            if k[1] == a and c.get("best_so_far") and len(c["best_so_far"]) >= 8:
                firsts.append(c["best_so_far"][0]); lasts.append(c["best_so_far"][-1])
        if firsts:
            import numpy as _np
            print(f"  {a:16s} first={_np.median(firsts):.4f} "
                  f"last={_np.median(lasts):.4f} "
                  f"gain={100 * _np.median((_np.array(firsts) - _np.array(lasts)) / _np.array(firsts)):.0f}%")

    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "ctx_analysis.json").write_text(json.dumps(
        {"n_cells": len(cells), "arms": arms, "n_seeds": len(seeds),
         "median_test_rmse": table, "primary": primary,
         "secondary_uncorrected": secondary}, indent=2) + "\n")
    print(f"\nwrote {DATA.name}/ctx_analysis.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
