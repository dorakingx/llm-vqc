"""Analysis for docs/research/QAE_PRIOR_BASELINES_PROTOCOL.md (no API calls).

    python scripts/qae/analyze_prior_baselines.py

Reads outputs/qae_prior_baselines/{candidate_results,matched_warm}.csv and the
committed robustness-study tables; writes summary.json, contrasts.csv,
budget_curves.csv and the paper figures under paper/figures/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs/qae_prior_baselines"
FIG = ROOT / "paper/figures"

PRIOR_CONDITIONS = {"reference": 8, "hamiltonian_xxz": 8, "qubits_n6": 8, "qubits_n8": 8}
ALL_CONDITIONS = {"reference": 8, "budget_b4": 4, "budget_b16": 16, "qubits_n6": 8,
                  "qubits_n8": 8, "hamiltonian_xxz": 8, "model_alt": 8}
LABEL = {"reference": "$n{=}4$, TFIM, $B{=}8$", "budget_b4": "$B{=}4$",
         "budget_b16": "$B{=}16$", "qubits_n6": "$n{=}6$", "qubits_n8": "$n{=}8$",
         "hamiltonian_xxz": "XXZ chain", "model_alt": "alternative model"}
# Conditions whose non-LLM arms are identical to the n=4 TFIM pools.
POOL_SOURCE = {"budget_b4": "reference", "budget_b16": "reference", "model_alt": "reference"}


def study_dir(key: str) -> Path:
    return ROOT / ("outputs/qae_tfim_neutral_v5" if key == "reference"
                   else f"outputs/qae_robustness/{key}")


def selected(frame: pd.DataFrame, by: str = "seed") -> pd.Series:
    """Lowest validation loss == highest validation fidelity."""
    return frame.loc[frame.groupby(by).val_fid.idxmax()].set_index(by).test_fid


def paired(a: pd.Series, b: pd.Series, rng_seed: int = 0) -> dict:
    d = (a - b).dropna().to_numpy()
    rng = np.random.default_rng(rng_seed)
    boots = rng.choice(d, size=(10_000, len(d)), replace=True).mean(axis=1)
    sd = d.std(ddof=1)
    p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    return {"mean_gain": float(d.mean()),
            "ci_lo": float(np.percentile(boots, 2.5)),
            "ci_hi": float(np.percentile(boots, 97.5)),
            "dz": float(d.mean() / sd) if sd > 0 else float("nan"),
            "wins": int((d > 0).sum()), "n": int(len(d)), "p": p}


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj.tolist()


def committed_selected(key: str) -> pd.DataFrame:
    s = pd.read_csv(study_dir(key) / "selected_results.csv")
    return s.pivot(index="seed", columns="method", values="test_fid")


def prior_arm_selected(cand: pd.DataFrame, key: str, arm: str, budget: int) -> pd.Series:
    src = POOL_SOURCE.get(key, key)
    x = cand[(cand.condition == src) & (cand.arm == arm)]
    if arm != "Hand":
        x = x[x.order <= budget]
    return selected(x)


def best_of_b_curve(cand: pd.DataFrame, key: str, arm: str, budgets, draws=2000, seed=0):
    """Expected selected test F over random B-subsets of each seed's pool."""
    x = cand[(cand.condition == key) & (cand.arm == arm)]
    rng = np.random.default_rng(seed)
    out = {}
    for b in budgets:
        per_seed = []
        for _, g in x.groupby("seed"):
            v, t = g.val_fid.to_numpy(), g.test_fid.to_numpy()
            if b > len(v):
                continue
            idx = np.array([rng.choice(len(v), b, replace=False) for _ in range(draws)])
            pick = idx[np.arange(draws), v[idx].argmax(axis=1)]
            per_seed.append(t[pick].mean())
        if per_seed:
            out[b] = float(np.mean(per_seed))
    return out


def tail_mass(values: np.ndarray, tau: float) -> float:
    return float(np.mean(values >= tau))


def equivalent_budget(q: float, p: float, budget: int) -> float:
    """B_eq = B ln(1-q)/ln(1-p): uniform budget matching the LLM's success odds."""
    if p <= 0:
        return float("inf")
    if q >= 1:
        return float("inf")
    return budget * np.log1p(-q) / np.log1p(-p)


def llm_candidates(key: str) -> pd.DataFrame:
    c = pd.read_csv(study_dir(key) / "candidate_results.csv")
    return c[c.method.isin(["LLM-Open", "LLM-Closed"])]


def main() -> None:
    cand = pd.read_csv(OUT / "candidate_results.csv")
    summary: dict = {"prior": {}, "matched_warm": {}, "warm_vs_random": {},
                     "open_pool_concentration": {}, "tail": {}}
    rows = []

    # --- primary: prior-matched baselines --------------------------------
    primary = []
    for key, budget in PRIOR_CONDITIONS.items():
        sel = committed_selected(key)
        arms = {a: prior_arm_selected(cand, key, a, budget)
                for a in ["Uniform", "RY", "NN", "RY+NN", "LLM-shuffled", "Hand"]}
        means = {a: float(v.mean()) for a, v in arms.items()}
        means.update({m: float(sel[m].mean()) for m in sel.columns})
        summary["prior"][key] = {"means": means}
        for name, a, b in [("RY+NN - Uniform", arms["RY+NN"], arms["Uniform"]),
                           ("LLM-Open - RY+NN", sel["LLM-Open"], arms["RY+NN"]),
                           ("LLM-Closed - RY+NN", sel["LLM-Closed"], arms["RY+NN"])]:
            st = paired(a, b)
            primary.append((key, name, st))
        for name, a, b in [("RY - Uniform", arms["RY"], arms["Uniform"]),
                           ("NN - Uniform", arms["NN"], arms["Uniform"]),
                           ("LLM-Open - LLM-shuffled", sel["LLM-Open"], arms["LLM-shuffled"]),
                           ("LLM-Open - Hand", sel["LLM-Open"], arms["Hand"]),
                           ("LLM-Closed - Hand", sel["LLM-Closed"], arms["Hand"])]:
            rows.append({"family": "secondary", "condition": key, "contrast": name,
                         **paired(a, b), "p_holm": None})
    adj = holm([st["p"] for _, _, st in primary])
    for (key, name, st), p_adj in zip(primary, adj, strict=True):
        rows.append({"family": "primary", "condition": key, "contrast": name, **st,
                     "p_holm": p_adj})

    # --- addendum A: matched-warm feedback ablation ----------------------
    mw_path = OUT / "matched_warm.csv"
    if mw_path.exists():
        mw = pd.read_csv(mw_path)
        mw_rows = []
        for key, budget in ALL_CONDITIONS.items():
            x = mw[mw.condition == key]
            w = budget // 2
            closed = selected(x[x.arm == "LLM-Closed"])
            warm = x[(x.arm == "LLM-Closed") & (x.phase == "explore")]
            fresh = selected(pd.concat([warm, x[x.arm == "LLM-Fresh"]]))
            sel = committed_selected(key)
            st = paired(closed, fresh)
            mw_rows.append((key, st))
            summary["matched_warm"][key] = {
                "closed_reeval": float(closed.mean()), "fresh": float(fresh.mean()),
                "closed_committed": float(sel["LLM-Closed"].mean()),
                "open_committed": float(sel["LLM-Open"].mean()),
                "warm_only": float(selected(warm).mean()), "w": w}
            rows.append({"family": "secondary", "condition": key,
                         "contrast": "LLM-Fresh - Random", **paired(fresh, sel["Random"]),
                         "p_holm": None})
        adj = holm([st["p"] for _, st in mw_rows])
        for (key, st), p_adj in zip(mw_rows, adj, strict=True):
            rows.append({"family": "matched_warm", "condition": key,
                         "contrast": "LLM-Closed - LLM-Fresh", **st, "p_holm": p_adj})

    # --- warm-only (per-seed generation) vs Random at B/2 -----------------
    wr = []
    for key, budget in ALL_CONDITIONS.items():
        c = pd.read_csv(study_dir(key) / "candidate_results.csv")
        w = budget // 2
        warm = selected(c[(c.method == "LLM-Closed") & (c.order <= w)])
        rnd = selected(c[(c.method == "Random") & (c.order <= w)])
        st = paired(warm, rnd)
        wr.append((key, st))
        summary["warm_vs_random"][key] = {
            "warm": float(warm.mean()), "random": float(rnd.mean()), "w": w}
        # LLM-Open selected-candidate concentration
        s = pd.read_csv(study_dir(key) / "selected_results.csv")
        vc = s[s.method == "LLM-Open"].candidate.value_counts()
        summary["open_pool_concentration"][key] = {
            "distinct_selected": int(len(vc)), "modal_count": int(vc.iloc[0]),
            "pool_size": int(c[c.method == "LLM-Open"].candidate.nunique())}
    adj = holm([st["p"] for _, st in wr])
    for (key, st), p_adj in zip(wr, adj, strict=True):
        rows.append({"family": "warm_vs_random", "condition": key,
                     "contrast": "LLM-warm - Random (B/2)", **st, "p_holm": p_adj})

    # --- tail masses and equivalent uniform budget ------------------------
    for key in PRIOR_CONDITIONS:
        llm = llm_candidates(key)
        warm = llm[(llm.method == "LLM-Closed") & (llm.phase == "explore")].test_fid.to_numpy()
        refine = llm[(llm.method == "LLM-Closed") & (llm.phase == "refine")].test_fid.to_numpy()
        uni = cand[(cand.condition == key) & (cand.arm == "Uniform")].test_fid.to_numpy()
        prior = cand[(cand.condition == key) & (cand.arm == "RY+NN")].test_fid.to_numpy()
        hand = cand[(cand.condition == key) & (cand.arm == "Hand")].test_fid.to_numpy()
        entry = {}
        for tau in (0.80, 0.90, 0.95):
            q, p, r, pr = (tail_mass(warm, tau), tail_mass(uni, tau),
                           tail_mass(refine, tau), tail_mass(prior, tau))
            entry[str(tau)] = {"q_llm_warm": q, "r_llm_refine": r, "p_uniform": p,
                               "p_ry_nn": pr, "n_warm": int(len(warm)), "n_uniform": int(len(uni)),
                               "B_eq_uniform_for_B8": equivalent_budget(q, p, 8),
                               "B_eq_rynn_for_B8": equivalent_budget(q, pr, 8)}
        entry["candidate_means"] = {"llm_warm": float(warm.mean()),
                                    "llm_refine": float(refine.mean()),
                                    "uniform": float(uni.mean()), "ry_nn": float(prior.mean()),
                                    "hand": float(hand.mean())}
        summary["tail"][key] = entry

    contrasts = pd.DataFrame(rows)
    contrasts.to_csv(OUT / "contrasts.csv", index=False)

    # --- budget curves -----------------------------------------------------
    curves = []
    for key in PRIOR_CONDITIONS:
        for arm in ["Uniform", "RY", "NN", "RY+NN", "LLM-shuffled"]:
            for b, v in best_of_b_curve(cand, key, arm, [1, 2, 4, 8, 16, 32, 64]).items():
                curves.append({"condition": key, "arm": arm, "B": b, "mean_selected_test": v})
    pd.DataFrame(curves).to_csv(OUT / "budget_curves.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    make_figures(cand, summary, pd.DataFrame(curves))
    print(contrasts.to_string())


def make_figures(cand, summary, curves) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    colors = {"Uniform": "#8c8c8c", "RY": "#c9a227", "NN": "#5b8fd1", "RY+NN": "#2a9d8f",
              "LLM-shuffled": "#b07aa1", "LLM-Open": "#d1495b", "LLM-Closed": "#7a1f2b",
              "Hand": "#444444"}
    titles = {"reference": "$n=4$ TFIM", "hamiltonian_xxz": "$n=4$ XXZ",
              "qubits_n6": "$n=6$ TFIM", "qubits_n8": "$n=8$ TFIM"}
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.3), sharey=False)
    for ax, key in zip(axes, PRIOR_CONDITIONS, strict=True):
        cv = curves[curves.condition == key]
        for arm in ["Uniform", "RY", "NN", "RY+NN", "LLM-shuffled"]:
            c = cv[cv.arm == arm]
            ax.plot(c.B, c.mean_selected_test, marker="o", ms=2.5, lw=1.2,
                    color=colors[arm], label=arm)
        m = summary["prior"][key]["means"]
        ax.scatter([8], [m["LLM-Open"]], marker="*", s=60, color=colors["LLM-Open"],
                   zorder=5, label="LLM-Open ($B$=8)")
        ax.scatter([8], [m["LLM-Closed"]], marker="D", s=22, color=colors["LLM-Closed"],
                   zorder=5, label="LLM-Closed ($B$=8)")
        ax.axhline(m["Hand"], color=colors["Hand"], ls=":", lw=1, label="Hand templates ($B$=2)")
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 4, 16, 64])
        ax.set_xticklabels(["1", "4", "16", "64"])
        ax.set_title(titles[key])
        ax.set_xlabel("budget $B$")
        ax.grid(alpha=0.3, lw=0.5)
    axes[0].set_ylabel("selected test $F_{\\rm trash}$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.12), fontsize=7.5)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(FIG / "fig_budget_curves.pdf", bbox_inches="tight")
    plt.close(fig)

    # Closed - Open (committed, shared pool) vs Closed - Fresh (matched warm)
    contr = pd.read_csv(OUT / "contrasts.csv")
    if (contr.contrast == "LLM-Closed - LLM-Fresh").any():
        keys = list(ALL_CONDITIONS)
        fig, ax = plt.subplots(figsize=(3.5, 2.6))
        for i, key in enumerate(keys):
            y = len(keys) - 1 - i
            sel = committed_selected(key)
            co = paired(sel["LLM-Closed"], sel["LLM-Open"])
            cf = contr[(contr.condition == key)
                      & (contr.contrast == "LLM-Closed - LLM-Fresh")].iloc[0]
            err_o = [[co["mean_gain"] - co["ci_lo"]], [co["ci_hi"] - co["mean_gain"]]]
            ax.errorbar(co["mean_gain"], y + 0.17, xerr=err_o,
                        fmt="o", ms=3.5, color="#d1495b", lw=1, capsize=2,
                        label="Closed $-$ Open (shared pool)" if i == 0 else None)
            err_f = [[cf.mean_gain - cf.ci_lo], [cf.ci_hi - cf.mean_gain]]
            ax.errorbar(cf.mean_gain, y - 0.17, xerr=err_f,
                        fmt="s", ms=3.5, color="#2a9d8f", lw=1, capsize=2,
                        label="Closed $-$ Fresh (matched warm)" if i == 0 else None)
        ax.axvline(0, color="k", lw=0.7)
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels([LABEL[k] for k in reversed(keys)])
        ax.set_xlabel("paired gain in selected test $F_{\\rm trash}$")
        ax.legend(frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(0, 1.0))
        ax.grid(axis="x", alpha=0.3, lw=0.5)
        fig.tight_layout()
        fig.savefig(FIG / "fig_feedback_forest.pdf", bbox_inches="tight")
        plt.close(fig)

if __name__ == "__main__":
    main()
