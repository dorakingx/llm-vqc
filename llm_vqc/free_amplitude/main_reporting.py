"""Main-mode reporting: per-candidate/per-arm/per-seed CSV+JSON, cross-seed
aggregate statistics (mean/median/std/IQR/95% bootstrap CI), the intrinsic
and task-conditioned expressibility/entanglement diagnostics, and 15
required figures -- each saved as PNG **and** SVG with its source data as
CSV -- plus a `selected_circuits/` diagram per arm/seed.

Every number is read back from the run's own `MainArmRunOutcome` records
or computed from the recorded candidates; nothing is invented. Expressibility
and entanglement are treated as DESCRIPTIVE diagnostics, never as assumed
predictors of RMSE (their scatter-vs-RMSE plots carry that caveat).
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from llm_vqc.free_amplitude.candidate_schema import (  # noqa: E402
    CompleteCandidateProposal,
    architecture_ir,
    complete_candidate_to_ir_and_theta,
)
from llm_vqc.free_amplitude.expressibility import (  # noqa: E402
    IntrinsicDiagnosticsCache,
    task_conditioned_entanglement,
)
from llm_vqc.free_amplitude.main_runner import MainArmRunOutcome  # noqa: E402

_ARM_LABELS = {
    "random": "Random",
    "llm_open_loop": "LLM Open-loop",
    "llm_closed_loop": "LLM Closed-loop",
}
_ARM_ORDER = ["random", "llm_open_loop", "llm_closed_loop"]
#: One fixed color per method across EVERY figure, so a reader can track a
#: method between slides without re-reading legends: Random = neutral gray,
#: Open-loop = blue, Closed-loop = red.
_ARM_COLORS = {
    "random": "#7f7f7f",
    "llm_open_loop": "#1f77b4",
    "llm_closed_loop": "#d62728",
}


def _apply_slide_style() -> None:
    """Slide-ready defaults: large readable fonts, bold titles, no chart
    clutter -- applied once before any figure is drawn."""
    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.labelsize": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.fontsize": 11,
        "legend.frameon": False,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "savefig.dpi": 200,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
    })


def _annotate_lower_better(ax, axis: str = "y") -> None:
    """A single glance-parseable cue that smaller values win."""
    text = "↓ lower is better" if axis == "y" else "← lower is better"
    ax.text(
        0.99, 0.98, text, transform=ax.transAxes, ha="right", va="top",
        fontsize=11, style="italic", color="#444444",
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _save_fig(fig, output_dir: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / "figures" / f"{name}.png", dpi=150)
    fig.savefig(output_dir / "figures" / f"{name}.svg")
    plt.close(fig)


def _save_source_csv(output_dir: Path, name: str, fieldnames: list[str], rows: list[dict]) -> None:
    with (output_dir / "figures" / f"{name}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _bootstrap_ci(values: list[float], n_boot: int = 2000, seed: int = 0) -> tuple:
    vals = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if len(vals) == 0:
        return (None, None)
    if len(vals) == 1:
        return (float(vals[0]), float(vals[0]))
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_boot)])
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def _agg_stats(values: list[float]) -> dict:
    vals = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if len(vals) == 0:
        return {"n": 0}
    lo, hi = _bootstrap_ci(vals.tolist())
    return {
        "n": int(len(vals)), "mean": float(np.mean(vals)), "median": float(np.median(vals)),
        "std": float(np.std(vals)), "iqr": float(np.percentile(vals, 75) - np.percentile(vals, 25)),
        "ci95_low": lo, "ci95_high": hi,
    }


def _rebuild_proposal(operations: list[dict], n_qubits: int) -> CompleteCandidateProposal:
    return CompleteCandidateProposal.model_validate(
        {"n_qubits": n_qubits, "operations": operations}
    )


def write_all_outputs(
    output_dir: Path,
    args: Any,
    provenance: dict,
    profile: Any,
    arm_outcomes: list[MainArmRunOutcome],
    dataset_diagnostics_by_seed: dict,
    search_space_report: dict,
    diagnostics_cache: IntrinsicDiagnosticsCache,
    task: Any,
    readout_qubit: int,
    elapsed_seconds: float,
    run_label: str,
) -> None:
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures" / "selected_circuits").mkdir(parents=True, exist_ok=True)
    _apply_slide_style()

    # Compute intrinsic diagnostics for every unique architecture_hash.
    arch_diag = _compute_architecture_diagnostics(
        arm_outcomes, diagnostics_cache, profile.n_qubits, readout_qubit
    )
    # Task-conditioned entanglement for each selected candidate.
    selected_task_ent = _compute_selected_task_entanglement(
        arm_outcomes, task, readout_qubit
    )

    _write_experiment_config(output_dir, args, provenance, profile, readout_qubit, run_label)
    _write_dataset_summary(output_dir, dataset_diagnostics_by_seed)
    _write_search_space_summary(output_dir, search_space_report)
    _write_candidate_trace(output_dir, arm_outcomes)
    _write_selected_circuits_json(output_dir, arm_outcomes, arch_diag, selected_task_ent)
    _write_architecture_diagnostics_csv(output_dir, arch_diag)
    _write_cross_seed_summary(output_dir, arm_outcomes)

    # Headline slide figure + the 15 required figures.
    _fig_arm_comparison_summary(output_dir, arm_outcomes)
    _fig_best_so_far_rmse(output_dir, arm_outcomes)
    _fig_validation_rmse_distribution(output_dir, arm_outcomes)
    _fig_test_rmse_by_arm(output_dir, arm_outcomes)
    _fig_prediction_vs_target(output_dir, arm_outcomes, task, readout_qubit)
    _fig_complexity_vs_rmse(output_dir, arm_outcomes)
    _fig_closed_loop_theta_trajectory(output_dir, arm_outcomes)
    _fig_proposal_outcomes(output_dir, arm_outcomes)
    _fig_expressibility_by_arm(output_dir, arm_outcomes, arch_diag)
    _fig_entangling_capability_by_arm(output_dir, arm_outcomes, arch_diag)
    _fig_expressibility_vs_rmse(output_dir, arm_outcomes, arch_diag)
    _fig_entanglement_vs_rmse(output_dir, arm_outcomes, arch_diag)
    _fig_expressibility_vs_entanglement(output_dir, arm_outcomes, arch_diag)
    _fig_selected_fidelity_histograms(output_dir, arm_outcomes, arch_diag)
    _fig_selected_entanglement_distributions(output_dir, arm_outcomes, arch_diag, selected_task_ent)
    _draw_selected_circuits(output_dir, arm_outcomes, profile.n_qubits, readout_qubit)

    _write_report_md(output_dir, arm_outcomes, run_label, elapsed_seconds)
    _write_readme(output_dir, args, profile, readout_qubit, run_label)


# --- diagnostics computation -------------------------------------------------


def _compute_architecture_diagnostics(arm_outcomes, cache, n_qubits, readout_qubit) -> dict:
    """`architecture_hash -> IntrinsicDiagnostics` for every unique
    architecture across all evaluated candidates (cached)."""
    result: dict = {}
    for outcome in arm_outcomes:
        for c in outcome.candidates:
            if c.outcome != "evaluated" or c.architecture_hash is None or c.operations is None:
                continue
            if c.architecture_hash in result:
                continue
            proposal = _rebuild_proposal(c.operations, n_qubits)
            ir = architecture_ir(proposal, readout_qubit)
            result[c.architecture_hash] = cache.get_or_compute(ir, c.architecture_hash)
    return result


def _compute_selected_task_entanglement(arm_outcomes, task, readout_qubit) -> dict:
    """`(seed, arm) -> EntanglementStats` for each arm's selected candidate,
    using that seed's actual validation inputs and the proposed theta."""
    result: dict = {}
    data_by_seed: dict = {}
    for outcome in arm_outcomes:
        if outcome.selected_operations is None:
            continue
        if outcome.seed not in data_by_seed:
            tv, _ = task.build(seed=outcome.seed)
            data_by_seed[outcome.seed] = tv
        tv = data_by_seed[outcome.seed]
        proposal = _rebuild_proposal(outcome.selected_operations, outcome.n_qubits)
        ir, theta = complete_candidate_to_ir_and_theta(proposal, readout_qubit)
        # Cap sample count for runtime; a descriptive distribution needs only
        # a representative subset of the validation inputs.
        inputs = tv.val.features[:60]
        result[(outcome.seed, outcome.arm)] = task_conditioned_entanglement(ir, theta, inputs)
    return result


# --- data files --------------------------------------------------------------


def _write_experiment_config(output_dir, args, provenance, profile, readout_qubit, run_label):
    config = {
        "run_label": run_label,
        "experiment_name": "free_amplitude_fixed_readout_v1",
        "mode": "main_joint_search",
        "scientific_status": (
            "integration/smoke demonstration only; not a statistically powered result"
        ),
        "n_qubits": profile.n_qubits,
        "feature_count": profile.feature_count,
        "readout_qubit": readout_qubit,
        "max_gates": args.max_gates,
        "budget_per_arm": args.budget,
        "seeds": args.seeds,
        "encoding": "amplitude",
        "prediction_formula": "mu_hat = (1 - <Z_readout>) / 2",
        "classical_parameter_count": 0,
        "optimizer": "none (LLM/sampler supplies theta; evaluated verbatim)",
        "dataset_profile": profile.name,
        "sigma_min": profile.sigma_min,
        "sigma_max": profile.sigma_max,
        "search_arms": ["random", "llm_open_loop", "llm_closed_loop"],
        "generated_at_utc": _now_iso(),
        **provenance,
    }
    (output_dir / "EXPERIMENT_CONFIG.json").write_text(json.dumps(config, indent=2))


def _write_dataset_summary(output_dir, dataset_diagnostics_by_seed):
    serializable = {str(k): v for k, v in dataset_diagnostics_by_seed.items()}
    (output_dir / "dataset_summary.json").write_text(json.dumps(serializable, indent=2))


def _write_search_space_summary(output_dir, search_space_report):
    (output_dir / "search_space_summary.json").write_text(json.dumps(search_space_report, indent=2))


_CANDIDATE_FIELDS = [
    "seed", "arm", "proposal_index", "proposal_id", "operations", "theta",
    "architecture_hash", "candidate_hash", "valid", "is_duplicate", "outcome",
    "train_mse", "val_rmse", "val_mae", "prediction_mean", "prediction_std",
    "prediction_min", "prediction_max", "constant_output", "circuit_depth",
    "searched_body_depth", "searched_body_gate_count", "controlled_gate_count",
    "quantum_parameter_count", "parameter_count_in_causal_cone", "fraction_in_causal_cone",
    "best_so_far_val_rmse", "api_latency_seconds", "input_tokens", "output_tokens",
]


def _write_candidate_trace(output_dir, arm_outcomes):
    with (output_dir / "candidate_trace.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CANDIDATE_FIELDS)
        writer.writeheader()
        for outcome in arm_outcomes:
            for c in outcome.candidates:
                row = asdict(c)
                row["operations"] = json.dumps(row["operations"]) if row["operations"] else None
                row["theta"] = json.dumps(row["theta"]) if row["theta"] is not None else None
                writer.writerow(row)


def _write_selected_circuits_json(output_dir, arm_outcomes, arch_diag, selected_task_ent):
    selections = []
    for outcome in arm_outcomes:
        entry = {
            "seed": outcome.seed, "arm": outcome.arm, "n_qubits": outcome.n_qubits,
            "selected_candidate_hash": outcome.selected_candidate_hash,
            "selected_architecture_hash": outcome.selected_architecture_hash,
            "selected_operations": outcome.selected_operations,
            "selected_theta": outcome.selected_theta,
            "selected_val_rmse": outcome.selected_val_rmse,
            "protected_test_rmse": outcome.protected_test_rmse,
            "protected_test_mae": outcome.protected_test_mae,
            "classical_parameter_count": 0,
        }
        diag = arch_diag.get(outcome.selected_architecture_hash)
        if diag is not None:
            entry["intrinsic_expressibility_kl"] = diag.expressibility_kl
            entry["intrinsic_entanglement_capability_mean"] = diag.entanglement_capability.mean
        tc = selected_task_ent.get((outcome.seed, outcome.arm))
        if tc is not None:
            entry["task_conditioned_entanglement_mean"] = tc.mean
            entry["task_conditioned_entanglement_median"] = tc.median
        selections.append(entry)
    (output_dir / "selected_circuits.json").write_text(json.dumps(selections, indent=2))


def _write_architecture_diagnostics_csv(output_dir, arch_diag):
    fieldnames = [
        "architecture_hash", "n_qubits", "quantum_parameter_count", "expressibility_kl",
        "entanglement_capability_mean", "entanglement_capability_median",
        "entanglement_capability_std", "entanglement_capability_min", "entanglement_capability_max",
        "diagnostic_seed", "diagnostic_state_samples", "diagnostic_fidelity_pairs",
    ]
    with (output_dir / "architecture_diagnostics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for h, d in arch_diag.items():
            writer.writerow({
                "architecture_hash": h, "n_qubits": d.n_qubits,
                "quantum_parameter_count": d.quantum_parameter_count,
                "expressibility_kl": d.expressibility_kl,
                "entanglement_capability_mean": d.entanglement_capability.mean,
                "entanglement_capability_median": d.entanglement_capability.median,
                "entanglement_capability_std": d.entanglement_capability.std,
                "entanglement_capability_min": d.entanglement_capability.min,
                "entanglement_capability_max": d.entanglement_capability.max,
                "diagnostic_seed": d.diagnostic_seed,
                "diagnostic_state_samples": d.diagnostic_state_samples,
                "diagnostic_fidelity_pairs": d.diagnostic_fidelity_pairs,
            })


def _write_cross_seed_summary(output_dir, arm_outcomes):
    summary = {}
    for arm in _ARM_ORDER:
        arm_out = [o for o in arm_outcomes if o.arm == arm]
        if not arm_out:
            continue
        selected_val = [o.selected_val_rmse for o in arm_out]
        selected_test = [o.protected_test_rmse for o in arm_out]
        summary[arm] = {
            "n_seeds": len(arm_out),
            "selected_validation_rmse": _agg_stats(selected_val),
            "selected_test_rmse": _agg_stats(selected_test),
        }
    (output_dir / "cross_seed_summary.json").write_text(json.dumps(summary, indent=2))


def _label(arm: str) -> str:
    return _ARM_LABELS.get(arm, arm)


def _evaluated(outcome: MainArmRunOutcome):
    return [c for c in outcome.candidates if c.outcome == "evaluated" and c.val_rmse is not None]


# --- figures (1-15) ----------------------------------------------------------


def _fig_arm_comparison_summary(output_dir, arm_outcomes):
    """Headline slide figure: selected validation & test RMSE per method as
    grouped bars with 95% bootstrap-CI whiskers, per-seed dots, and value
    labels -- one glance shows which method won and by how much."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    rows = []
    x = np.arange(len(_ARM_ORDER))
    width = 0.36
    for j, (metric, attr, hatch) in enumerate([
        ("Validation RMSE", "selected_val_rmse", None),
        ("Test RMSE", "protected_test_rmse", "//"),
    ]):
        means, err_lo, err_hi, colors = [], [], [], []
        for arm in _ARM_ORDER:
            vals = [
                getattr(o, attr) for o in arm_outcomes
                if o.arm == arm and getattr(o, attr) is not None
            ]
            mean = float(np.mean(vals)) if vals else 0.0
            lo, hi = _bootstrap_ci(vals) if vals else (mean, mean)
            means.append(mean)
            err_lo.append(mean - (lo if lo is not None else mean))
            err_hi.append((hi if hi is not None else mean) - mean)
            colors.append(_ARM_COLORS[arm])
            for o in arm_outcomes:
                if o.arm == arm and getattr(o, attr) is not None:
                    rows.append({"arm": arm, "seed": o.seed, "metric": metric,
                                 "value": getattr(o, attr)})
        offset = (j - 0.5) * width
        bars = ax.bar(
            x + offset, means, width, yerr=[err_lo, err_hi], capsize=5,
            color=colors, alpha=1.0 if j == 0 else 0.55, hatch=hatch,
            edgecolor="white", linewidth=0.8,
            label=metric,
        )
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=11, fontweight="bold")
        # per-seed dots over each bar
        for i, arm in enumerate(_ARM_ORDER):
            vals = [
                getattr(o, attr) for o in arm_outcomes
                if o.arm == arm and getattr(o, attr) is not None
            ]
            ax.scatter([x[i] + offset] * len(vals), vals, s=22, color="black",
                       alpha=0.55, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([_label(a) for a in _ARM_ORDER], fontsize=13, fontweight="bold")
    ax.set_ylabel("RMSE")
    ax.set_title("Method comparison: selected-candidate RMSE (mean ± 95% CI; dots = seeds)")
    _annotate_lower_better(ax)
    ax.legend(loc="upper left")
    _save_fig(fig, output_dir, "arm_comparison_summary")
    _save_source_csv(output_dir, "arm_comparison_summary",
                     ["arm", "seed", "metric", "value"], rows)


def _fig_best_so_far_rmse(output_dir, arm_outcomes):
    """1. proposal index vs best-so-far validation RMSE: one BOLD mean line
    per method (fixed method color) + thin per-seed traces behind it."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    rows = []
    for arm in _ARM_ORDER:
        color = _ARM_COLORS[arm]
        per_seed: dict[int, list] = {}
        for outcome in arm_outcomes:
            if outcome.arm != arm:
                continue
            xs = [c.proposal_index for c in outcome.candidates]
            ys = [c.best_so_far_val_rmse for c in outcome.candidates]
            if not xs:
                continue
            per_seed[outcome.seed] = ys
            ax.plot(xs, ys, color=color, alpha=0.25, lw=1.2)
            for x_val, y in zip(xs, ys, strict=True):
                rows.append({"arm": arm, "seed": outcome.seed, "proposal_index": x_val,
                             "best_so_far_val_rmse": y})
        if per_seed:
            max_len = max(len(v) for v in per_seed.values())
            mean_curve = [
                float(np.mean([v[i] for v in per_seed.values() if len(v) > i and v[i] is not None]))
                for i in range(max_len)
            ]
            ax.plot(range(max_len), mean_curve, color=color, lw=3.2, marker="o",
                    markersize=7, label=f"{_label(arm)} (mean)")
    ax.set_xlabel("proposal index")
    ax.set_ylabel("best-so-far validation RMSE")
    ax.set_title("Search progress: best-so-far validation RMSE (bold = mean over seeds)")
    _annotate_lower_better(ax)
    ax.legend()
    _save_fig(fig, output_dir, "best_so_far_rmse")
    _save_source_csv(output_dir, "best_so_far_rmse",
                     ["arm", "seed", "proposal_index", "best_so_far_val_rmse"], rows)


def _fig_validation_rmse_distribution(output_dir, arm_outcomes):
    """2. candidate validation RMSE distribution by arm -- method-colored
    boxes so the same color means the same method on every slide."""
    fig, ax = plt.subplots(figsize=(8, 5))
    rows = []
    data, labels, colors = [], [], []
    for arm in _ARM_ORDER:
        vals = [
            c.val_rmse for o in arm_outcomes if o.arm == arm for c in _evaluated(o)
        ]
        if vals:
            data.append(vals)
            labels.append(_label(arm))
            colors.append(_ARM_COLORS[arm])
            for v in vals:
                rows.append({"arm": arm, "val_rmse": v})
    if data:
        box = ax.boxplot(data, tick_labels=labels, showmeans=True, patch_artist=True)
        for patch, color in zip(box["boxes"], colors, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        for median in box["medians"]:
            median.set_color("black")
            median.set_linewidth(2)
    ax.set_ylabel("candidate validation RMSE")
    ax.set_title("All evaluated candidates: validation-RMSE distribution by method")
    _annotate_lower_better(ax)
    _save_fig(fig, output_dir, "validation_rmse_distribution")
    _save_source_csv(output_dir, "validation_rmse_distribution", ["arm", "val_rmse"], rows)


def _fig_test_rmse_by_arm(output_dir, arm_outcomes):
    """3. selected Test RMSE by arm/seed: method-colored bars (mean) with
    95% CI whiskers, value labels, and per-seed dots."""
    fig, ax = plt.subplots(figsize=(8, 5))
    rows = []
    means, err_lo, err_hi, colors = [], [], [], []
    for arm in _ARM_ORDER:
        arm_out = [o for o in arm_outcomes if o.arm == arm]
        tests = [o.protected_test_rmse for o in arm_out if o.protected_test_rmse is not None]
        for o in arm_out:
            rows.append({"arm": arm, "seed": o.seed, "test_rmse": o.protected_test_rmse})
        mean = float(np.mean(tests)) if tests else 0.0
        lo, hi = _bootstrap_ci(tests) if tests else (mean, mean)
        means.append(mean)
        err_lo.append(mean - (lo if lo is not None else mean))
        err_hi.append((hi if hi is not None else mean) - mean)
        colors.append(_ARM_COLORS[arm])
    x = np.arange(len(_ARM_ORDER))
    bars = ax.bar(x, means, 0.55, yerr=[err_lo, err_hi], capsize=6, color=colors,
                  edgecolor="white")
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=12, fontweight="bold")
    for i, arm in enumerate(_ARM_ORDER):
        vals = [
            o.protected_test_rmse for o in arm_outcomes
            if o.arm == arm and o.protected_test_rmse is not None
        ]
        ax.scatter([x[i]] * len(vals), vals, s=26, color="black", alpha=0.6, zorder=3,
                   label="individual seeds" if i == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels([_label(a) for a in _ARM_ORDER], fontsize=13, fontweight="bold")
    ax.set_ylabel("selected protected-test RMSE")
    ax.set_title("Final comparison: protected-test RMSE (mean ± 95% CI)")
    _annotate_lower_better(ax)
    ax.legend(loc="upper left")
    _save_fig(fig, output_dir, "test_rmse_by_arm")
    _save_source_csv(output_dir, "test_rmse_by_arm", ["arm", "seed", "test_rmse"], rows)


def _fig_prediction_vs_target(output_dir, arm_outcomes, task, readout_qubit):
    """4. Test target vs prediction with y=x for selected candidates."""
    from llm_vqc.free_amplitude.model import FixedReadoutQuantumModel

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    rows = []
    test_by_seed = {}
    for outcome in arm_outcomes:
        if outcome.selected_operations is None:
            continue
        if outcome.seed not in test_by_seed:
            test_by_seed[outcome.seed], _ = task.build_test(seed=outcome.seed)
        test = test_by_seed[outcome.seed]
        proposal = _rebuild_proposal(outcome.selected_operations, outcome.n_qubits)
        ir, theta = complete_candidate_to_ir_and_theta(proposal, readout_qubit)
        import torch
        model = FixedReadoutQuantumModel(ir, readout_qubit=readout_qubit)
        model.double()
        with torch.no_grad():
            model.q_layer.weights.copy_(torch.tensor(theta, dtype=torch.float64))
            x_test = torch.tensor(test.features[:200], dtype=torch.float64)
            preds = model(x_test).numpy().reshape(-1)
        targets = np.asarray(test.targets[:200], dtype=np.float64)
        ax.scatter(targets, preds, s=8, alpha=0.4, label=f"{_label(outcome.arm)} s{outcome.seed}")
        for t, p in zip(targets, preds, strict=True):
            rows.append({"arm": outcome.arm, "seed": outcome.seed, "target": float(t),
                         "prediction": float(p)})
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="y = x")
    ax.set_xlabel("test target (mu)")
    ax.set_ylabel("prediction (mu_hat)")
    ax.set_title("Test target vs prediction (selected candidates)")
    ax.legend(fontsize=6)
    _save_fig(fig, output_dir, "prediction_vs_target")
    _save_source_csv(output_dir, "prediction_vs_target",
                     ["arm", "seed", "target", "prediction"], rows)


def _fig_complexity_vs_rmse(output_dir, arm_outcomes):
    """5. validation RMSE vs body depth, gate count, controlled-gate count."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    metrics = [
        ("searched_body_depth", "body depth"),
        ("searched_body_gate_count", "gate count"),
        ("controlled_gate_count", "controlled-gate count"),
    ]
    rows = []
    for ax, (attr, xlabel) in zip(axes, metrics, strict=True):
        for arm in _ARM_ORDER:
            xs, ys = [], []
            for o in arm_outcomes:
                if o.arm != arm:
                    continue
                for c in _evaluated(o):
                    x = getattr(c, attr)
                    if isinstance(x, (int, float)):
                        xs.append(x)
                        ys.append(c.val_rmse)
                        rows.append({"arm": arm, "metric": xlabel, "x": x, "val_rmse": c.val_rmse})
            if xs:
                ax.scatter(xs, ys, alpha=0.6, s=45, color=_ARM_COLORS[arm], label=_label(arm))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("validation RMSE (lower is better)")
        ax.legend(fontsize=7)
    fig.suptitle("Circuit complexity vs validation RMSE")
    _save_fig(fig, output_dir, "complexity_vs_rmse")
    _save_source_csv(output_dir, "complexity_vs_rmse", ["arm", "metric", "x", "val_rmse"], rows)


def _fig_closed_loop_theta_trajectory(output_dir, arm_outcomes):
    """6. Closed-loop theta proposals + validation RMSE over iterations.
    Parameters of unrelated architectures are NOT connected (each connected
    trajectory is confined to one architecture_hash)."""
    fig, (ax_theta, ax_rmse) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    rows = []
    for outcome in arm_outcomes:
        if outcome.arm != "llm_closed_loop":
            continue
        # Group consecutive candidates by architecture_hash so we only
        # connect theta values that belong to the SAME architecture.
        segments: list[list] = []
        for c in outcome.candidates:
            if c.outcome != "evaluated" or c.theta is None:
                continue
            if segments and segments[-1][-1].architecture_hash == c.architecture_hash:
                segments[-1].append(c)
            else:
                segments.append([c])
            for j, th in enumerate(c.theta):
                rows.append({"seed": outcome.seed, "proposal_index": c.proposal_index,
                             "architecture_hash": c.architecture_hash, "theta_index": j,
                             "theta": th, "val_rmse": c.val_rmse})
        for seg in segments:
            if len(seg) < 2:
                # single-point architecture: plot a marker but connect nothing
                for c in seg:
                    for th in c.theta:
                        ax_theta.scatter(c.proposal_index, th, s=25,
                                         color=f"C{outcome.seed}")
                continue
            n_theta = min(len(c.theta) for c in seg)
            for ti in range(n_theta):
                ax_theta.plot([c.proposal_index for c in seg], [c.theta[ti] for c in seg],
                              marker="o", alpha=0.6, color=f"C{outcome.seed}")
        xs = [c.proposal_index for c in outcome.candidates if c.val_rmse is not None]
        ys = [c.val_rmse for c in outcome.candidates if c.val_rmse is not None]
        if xs:
            ax_rmse.plot(xs, ys, marker="s", label=f"seed {outcome.seed}")
    ax_theta.set_ylabel("proposed theta (rad)")
    ax_theta.set_title("Closed-loop theta trajectory (connected only within one architecture)")
    ax_rmse.set_ylabel("validation RMSE")
    ax_rmse.set_xlabel("proposal index")
    ax_rmse.legend(fontsize=7)
    _save_fig(fig, output_dir, "closed_loop_theta_trajectory")
    _save_source_csv(output_dir, "closed_loop_theta_trajectory",
                     ["seed", "proposal_index", "architecture_hash", "theta_index", "theta",
                      "val_rmse"], rows)


def _fig_proposal_outcomes(output_dir, arm_outcomes):
    """7. valid, invalid, duplicate, failed counts by arm."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    categories = ["valid_unique", "invalid", "duplicate", "failed"]
    counts_by_arm = {arm: {c: 0 for c in categories} for arm in _ARM_ORDER}
    for outcome in arm_outcomes:
        for c in outcome.candidates:
            if not c.valid:
                counts_by_arm[outcome.arm]["invalid"] += 1
            elif c.is_duplicate:
                counts_by_arm[outcome.arm]["duplicate"] += 1
            elif c.outcome == "failed":
                counts_by_arm[outcome.arm]["failed"] += 1
            else:
                counts_by_arm[outcome.arm]["valid_unique"] += 1
    x = np.arange(len(_ARM_ORDER))
    width = 0.2
    rows = []
    for i, cat in enumerate(categories):
        vals = [counts_by_arm[arm][cat] for arm in _ARM_ORDER]
        ax.bar(x + (i - 1.5) * width, vals, width, label=cat)
        for arm, v in zip(_ARM_ORDER, vals, strict=True):
            rows.append({"arm": arm, "category": cat, "count": v})
    ax.set_xticks(x)
    ax.set_xticklabels([_label(a) for a in _ARM_ORDER])
    ax.set_ylabel("count")
    ax.set_title("Proposal outcomes by arm")
    ax.legend(fontsize=8)
    _save_fig(fig, output_dir, "proposal_outcomes")
    _save_source_csv(output_dir, "proposal_outcomes", ["arm", "category", "count"], rows)


def _arm_arch_values(arm_outcomes, arch_diag, value_fn):
    """Collect (arm, seed, val_rmse, diagnostic_value) for selected+all
    evaluated candidates whose architecture has a computed diagnostic."""
    out = []
    for outcome in arm_outcomes:
        for c in _evaluated(outcome):
            d = arch_diag.get(c.architecture_hash)
            if d is None:
                continue
            out.append((outcome.arm, outcome.seed, c.val_rmse, value_fn(d), c))
    return out


def _fig_expressibility_by_arm(output_dir, arm_outcomes, arch_diag):
    """8. architecture Expressibility KL distribution by arm (lower = more
    expressible)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rows = []
    data, labels = [], []
    for arm in _ARM_ORDER:
        vals = [v for a, s, r, v, c in _arm_arch_values(arm_outcomes, arch_diag,
                lambda d: d.expressibility_kl) if a == arm]
        if vals:
            data.append(vals)
            labels.append(_label(arm))
            for v in vals:
                rows.append({"arm": arm, "expressibility_kl": v})
    if data:
        ax.boxplot(data, tick_labels=labels, showmeans=True)
    ax.set_ylabel("expressibility KL vs Haar")
    ax.set_title("Architecture expressibility by arm  (lower KL = more expressible)")
    _save_fig(fig, output_dir, "expressibility_by_arm")
    _save_source_csv(output_dir, "expressibility_by_arm", ["arm", "expressibility_kl"], rows)


def _fig_entangling_capability_by_arm(output_dir, arm_outcomes, arch_diag):
    """9. architecture Meyer-Wallach Q distribution by arm."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rows = []
    data, labels = [], []
    for arm in _ARM_ORDER:
        vals = [v for a, s, r, v, c in _arm_arch_values(arm_outcomes, arch_diag,
                lambda d: d.entanglement_capability.mean) if a == arm]
        if vals:
            data.append(vals)
            labels.append(_label(arm))
            for v in vals:
                rows.append({"arm": arm, "entanglement_capability_mean": v})
    if data:
        ax.boxplot(data, tick_labels=labels, showmeans=True)
    ax.set_ylabel("mean Meyer-Wallach Q")
    ax.set_title("Architecture entangling capability by arm")
    _save_fig(fig, output_dir, "entangling_capability_by_arm")
    _save_source_csv(output_dir, "entangling_capability_by_arm",
                     ["arm", "entanglement_capability_mean"], rows)


def _fig_expressibility_vs_rmse(output_dir, arm_outcomes, arch_diag):
    """10. expressibility vs validation RMSE (descriptive, not predictive)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rows = []
    for arm in _ARM_ORDER:
        pts = [(v, r) for a, s, r, v, c in _arm_arch_values(arm_outcomes, arch_diag,
               lambda d: d.expressibility_kl) if a == arm]
        if pts:
            ax.scatter(
                [p[0] for p in pts], [p[1] for p in pts], alpha=0.6, s=45,
                color=_ARM_COLORS[arm], label=_label(arm),
            )
            for v, r in pts:
                rows.append({"arm": arm, "expressibility_kl": v, "val_rmse": r})
    ax.set_xlabel("expressibility KL (lower = more expressible)")
    ax.set_ylabel("validation RMSE")
    ax.set_title("Expressibility vs validation RMSE (descriptive, not a predictor)")
    ax.legend(fontsize=7)
    _save_fig(fig, output_dir, "expressibility_vs_validation_rmse")
    _save_source_csv(output_dir, "expressibility_vs_validation_rmse",
                     ["arm", "expressibility_kl", "val_rmse"], rows)


def _fig_entanglement_vs_rmse(output_dir, arm_outcomes, arch_diag):
    """11. entanglement capability vs validation RMSE (descriptive)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rows = []
    for arm in _ARM_ORDER:
        pts = [(v, r) for a, s, r, v, c in _arm_arch_values(arm_outcomes, arch_diag,
               lambda d: d.entanglement_capability.mean) if a == arm]
        if pts:
            ax.scatter(
                [p[0] for p in pts], [p[1] for p in pts], alpha=0.6, s=45,
                color=_ARM_COLORS[arm], label=_label(arm),
            )
            for v, r in pts:
                rows.append({"arm": arm, "entanglement_capability_mean": v, "val_rmse": r})
    ax.set_xlabel("mean Meyer-Wallach Q")
    ax.set_ylabel("validation RMSE")
    ax.set_title("Entanglement capability vs validation RMSE (descriptive, not a predictor)")
    ax.legend(fontsize=7)
    _save_fig(fig, output_dir, "entanglement_vs_validation_rmse")
    _save_source_csv(output_dir, "entanglement_vs_validation_rmse",
                     ["arm", "entanglement_capability_mean", "val_rmse"], rows)


def _fig_expressibility_vs_entanglement(output_dir, arm_outcomes, arch_diag):
    """12. expressibility vs entanglement, marker size = circuit complexity
    (gate count), identifying arm/seed."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    rows = []
    for arm in _ARM_ORDER:
        for a, s, _r_unused, _v_unused, c in _arm_arch_values(
            arm_outcomes, arch_diag, lambda d: d.expressibility_kl
        ):
            if a != arm:
                continue
            d = arch_diag[c.architecture_hash]
            size = 20 + 15 * (c.searched_body_gate_count or 1)
            ax.scatter(d.expressibility_kl, d.entanglement_capability.mean, s=size, alpha=0.5,
                       color=_ARM_COLORS[arm])
            rows.append({"arm": arm, "seed": s, "expressibility_kl": d.expressibility_kl,
                         "entanglement_capability_mean": d.entanglement_capability.mean,
                         "gate_count": c.searched_body_gate_count})
    # legend proxies
    for arm in _ARM_ORDER:
        ax.scatter([], [], color=_ARM_COLORS[arm], label=_label(arm))
    ax.set_xlabel("expressibility KL (lower = more expressible)")
    ax.set_ylabel("mean Meyer-Wallach Q")
    ax.set_title("Expressibility vs entanglement (marker size = gate count)")
    ax.legend(fontsize=8)
    _save_fig(fig, output_dir, "expressibility_vs_entanglement")
    _save_source_csv(output_dir, "expressibility_vs_entanglement",
                     ["arm", "seed", "expressibility_kl", "entanglement_capability_mean",
                      "gate_count"], rows)


def _fig_selected_fidelity_histograms(output_dir, arm_outcomes, arch_diag):
    """13. selected architectures' fidelity distributions overlaid with Haar."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    rows = []
    haar_plotted = False
    for outcome in arm_outcomes:
        d = arch_diag.get(outcome.selected_architecture_hash)
        if d is None:
            continue
        centers = 0.5 * (np.array(d.bin_edges[:-1]) + np.array(d.bin_edges[1:]))
        ax.plot(centers, d.fidelity_histogram, alpha=0.6,
                label=f"{_label(outcome.arm)} s{outcome.seed}")
        if not haar_plotted:
            ax.plot(centers, d.haar_histogram, "k--", lw=1.5, label="Haar reference")
            haar_plotted = True
        for ccenter, fs, hs in zip(centers, d.fidelity_histogram, d.haar_histogram, strict=True):
            rows.append({"arm": outcome.arm, "seed": outcome.seed, "fidelity_bin_center": ccenter,
                         "sampled_prob": fs, "haar_prob": hs})
    ax.set_xlabel("state fidelity F")
    ax.set_ylabel("probability")
    ax.set_title("Selected architectures' fidelity distributions vs Haar")
    ax.legend(fontsize=7)
    _save_fig(fig, output_dir, "selected_fidelity_histograms")
    _save_source_csv(output_dir, "selected_fidelity_histograms",
                     ["arm", "seed", "fidelity_bin_center", "sampled_prob", "haar_prob"], rows)


def _fig_selected_entanglement_distributions(
    output_dir, arm_outcomes, arch_diag, selected_task_ent
):
    """14. intrinsic AND task-conditioned Q distributions for selected
    candidates (summary points: intrinsic mean/std and task-conditioned
    mean/std side by side)."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    rows = []
    xticks, xlabels = [], []
    pos = 0
    for outcome in arm_outcomes:
        d = arch_diag.get(outcome.selected_architecture_hash)
        tc = selected_task_ent.get((outcome.seed, outcome.arm))
        if d is None and tc is None:
            continue
        label = f"{_label(outcome.arm)}\ns{outcome.seed}"
        if d is not None:
            ax.errorbar(pos - 0.15, d.entanglement_capability.mean,
                        yerr=d.entanglement_capability.std, fmt="o", color="tab:blue",
                        capsize=3, label="intrinsic" if pos == 0 else None)
            rows.append({"arm": outcome.arm, "seed": outcome.seed, "kind": "intrinsic",
                         "mean": d.entanglement_capability.mean,
                         "std": d.entanglement_capability.std})
        if tc is not None:
            ax.errorbar(pos + 0.15, tc.mean, yerr=tc.std, fmt="s", color="tab:orange",
                        capsize=3, label="task-conditioned" if pos == 0 else None)
            rows.append({"arm": outcome.arm, "seed": outcome.seed, "kind": "task_conditioned",
                         "mean": tc.mean, "std": tc.std})
        xticks.append(pos)
        xlabels.append(label)
        pos += 1
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=7)
    ax.set_ylabel("Meyer-Wallach Q")
    ax.set_title("Selected candidates: intrinsic vs task-conditioned entanglement")
    ax.legend(fontsize=8)
    _save_fig(fig, output_dir, "selected_entanglement_distributions")
    _save_source_csv(output_dir, "selected_entanglement_distributions",
                     ["arm", "seed", "kind", "mean", "std"], rows)


def _draw_selected_circuits(output_dir, arm_outcomes, n_qubits, readout_qubit):
    """15. readable circuit diagram per arm/seed showing gates, wires, theta."""
    from qiskit import QuantumCircuit

    from llm_vqc.ir.compiler_qiskit import _apply_body
    from llm_vqc.ir.expand import build_program

    sub = output_dir / "figures" / "selected_circuits"
    for outcome in arm_outcomes:
        if outcome.selected_operations is None:
            continue
        proposal = _rebuild_proposal(outcome.selected_operations, outcome.n_qubits)
        ir, theta = complete_candidate_to_ir_and_theta(proposal, readout_qubit)
        base = f"{outcome.arm}_seed{outcome.seed}"
        try:
            program = build_program(ir)
            qc = QuantumCircuit(program.n_qubits)
            _apply_body(qc, program, [float(t) for t in theta])
            qc.barrier()
            fig = qc.draw("mpl", fold=-1)
            fig.suptitle(
                f"{_label(outcome.arm)} seed{outcome.seed} "
                f"(amplitude-encode then body; measure Z(q{readout_qubit}))",
                fontsize=9,
            )
            fig.savefig(sub / f"{base}.png", dpi=150, bbox_inches="tight")
            fig.savefig(sub / f"{base}.svg", bbox_inches="tight")
            plt.close(fig)
        except Exception:
            pass
        # Always also write a text rendering (robust, machine-readable).
        lines = [
            f"# {_label(outcome.arm)} seed{outcome.seed}",
            "amplitude encoding on all wires (state preparation), then:",
        ]
        for op in outcome.selected_operations:
            th = op.get("theta")
            th_str = f" theta={th:.4f}" if isinstance(th, (int, float)) else ""
            lines.append(f"  {op['gate']}({', '.join(map(str, op['wires']))}){th_str}")
        lines.append(f"measure Z on qubit {readout_qubit}; mu_hat = (1 - <Z>) / 2")
        (sub / f"{base}.txt").write_text("\n".join(lines))


# --- report + readme ---------------------------------------------------------

_FIGURES = [
    "arm_comparison_summary",
    "best_so_far_rmse", "validation_rmse_distribution", "test_rmse_by_arm",
    "prediction_vs_target", "complexity_vs_rmse", "closed_loop_theta_trajectory",
    "proposal_outcomes", "expressibility_by_arm", "entangling_capability_by_arm",
    "expressibility_vs_validation_rmse", "entanglement_vs_validation_rmse",
    "expressibility_vs_entanglement", "selected_fidelity_histograms",
    "selected_entanglement_distributions",
]


def _write_report_md(output_dir, arm_outcomes, run_label, elapsed_seconds):
    lines = [
        f"# {run_label}",
        "",
        "# Free-Amplitude Fixed-Readout -- Joint Structure-and-Theta Search",
        "",
        "**Main mode: the LLM/sampler proposes both the circuit structure AND the "
        "numerical angles; there is NO optimizer and NO classical layer "
        "(`classical_parameter_count = 0`). The prediction is "
        "`mu_hat = (1 - <Z_0>) / 2`.** This is an integration/smoke demonstration, "
        "not a scientific performance claim.",
        "",
        f"Elapsed: {elapsed_seconds:.1f}s.",
        "",
        "## Selected candidate per arm/seed",
        "",
        "| seed | arm | val RMSE | test RMSE | test MAE | q params | body depth | "
        "classical params |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for outcome in sorted(arm_outcomes, key=lambda o: (o.seed, _ARM_ORDER.index(o.arm))):
        selected = next(
            (c for c in outcome.candidates
             if c.candidate_hash == outcome.selected_candidate_hash), None
        )
        has_v = outcome.selected_val_rmse is not None
        has_t = outcome.protected_test_rmse is not None
        has_m = outcome.protected_test_mae is not None
        vr = f"{outcome.selected_val_rmse:.4f}" if has_v else "n/a"
        tr = f"{outcome.protected_test_rmse:.4f}" if has_t else "n/a"
        tm = f"{outcome.protected_test_mae:.4f}" if has_m else "n/a"
        qp = selected.quantum_parameter_count if selected else "n/a"
        bd = selected.searched_body_depth if selected else "n/a"
        lines.append(
            f"| {outcome.seed} | {_label(outcome.arm)} | {vr} | {tr} | {tm} | {qp} | {bd} | 0 |"
        )

    lines += [
        "",
        "## Cross-seed aggregates",
        "See `cross_seed_summary.json` for mean / median / std / IQR / 95% bootstrap CI "
        "of selected validation and test RMSE by arm.",
        "",
        "## Figures",
        "Every figure is saved as PNG and SVG under `figures/`, with its source data as "
        "a `.csv` of the same name.",
        "",
    ]
    for name in _FIGURES:
        lines.append(f"### {name}")
        lines.append(f"![{name}](figures/{name}.png)")
        lines.append("")
    lines += [
        "### selected_circuits",
        "Readable circuit diagram (gates, wires, theta) per arm/seed under "
        "`figures/selected_circuits/` (PNG + SVG + a robust `.txt` rendering).",
        "",
        "## Expressibility and entanglement are descriptive diagnostics",
        "Expressibility (fidelity KL vs Haar; lower = more expressible) and entanglement "
        "capability (Meyer-Wallach Q) are computed per unique architecture from |0...0> "
        "(amplitude encoding excluded), cached by `architecture_hash`. They are reported "
        "as descriptive diagnostics and are NOT assumed to predict RMSE. A single "
        "fixed-theta value is never called 'expressibility' -- expressibility is a "
        "distribution over the whole parameter space.",
        "",
        "## Amplitude-encoding limitation",
        "L2 amplitude normalization removes overall multiplicative scale: two samples "
        "differing only by a positive amplitude factor A become the same normalized "
        "quantum state before noise. Only the peak location (mu) is targeted; absolute "
        "signal magnitude is not recoverable.",
        "",
        "---",
        f"# {run_label}",
        "Mock / non-LLM demonstration; zero real API calls; not a scientific claim.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines))


def _write_readme(output_dir, args, profile, readout_qubit, run_label):
    content = f"""# free_amplitude_fixed_readout_v1 (joint structure-and-theta search)

> **{run_label}**

Main mode: the LLM/sampler proposes a COMPLETE candidate (structure + numerical
theta); the circuit is evaluated at exactly those angles with NO optimizer and
NO classical layer. Prediction: `mu_hat = (1 - <Z_{readout_qubit}>) / 2`.
`classical_parameter_count` is always 0.

## Configuration (derived from the dataset profile)
- dataset_profile: {profile.name}
- n_qubits: {profile.n_qubits}, feature_count: {profile.feature_count} (= 2^{profile.n_qubits})
- readout_qubit: {readout_qubit} (fixed). max_gates: {args.max_gates}.
- budget per arm: {args.budget}. seeds: {args.seeds}.
- Arms: Random, LLM Open-loop, LLM Closed-loop. Duplicate detection is by
  `candidate_hash` (structure + theta) and is arm-and-seed-local.

## Identity hashes
- `architecture_hash`: gates / order / wires only (keys the cached
  expressibility/entanglement diagnostics).
- `candidate_hash`: architecture + canonical theta (a different theta is a
  different candidate).

## Files
- `report.md`, `EXPERIMENT_CONFIG.json`, `dataset_summary.json`,
  `search_space_summary.json`, `cross_seed_summary.json`.
- `candidate_trace.csv`, `selected_circuits.json`, `architecture_diagnostics.csv`.
- `figures/` -- 14 aggregate figures (PNG + SVG + source CSV each) plus
  `figures/selected_circuits/` diagrams.

No raw prompts, raw model responses, full weight arrays, or credentials are
stored here.
"""
    (output_dir / "README.md").write_text(content)
