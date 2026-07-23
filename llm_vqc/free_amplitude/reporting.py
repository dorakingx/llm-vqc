"""Writes every artifact under `outputs/free_amplitude_fixed_readout_v1/`
from the run's own results only (Codex instruction section 23).

No value here is invented: every number is read back from an
`ArmRunOutcome`/`CandidateRecord`/`BaselineResult`/the trained-weight
cache, or is a static constant describing the fixed design. Raw prompts,
raw model responses, full weight arrays, and API keys are never written
here.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from llm_vqc.evaluation.store import ResultStore  # noqa: E402
from llm_vqc.free_amplitude.baselines import BaselineResult  # noqa: E402
from llm_vqc.free_amplitude.init_policy import FREE_AMPLITUDE_INIT_VERSION  # noqa: E402
from llm_vqc.free_amplitude.runner import ArmRunOutcome, arm_task_name  # noqa: E402
from llm_vqc.free_amplitude.training import FREE_AMPLITUDE_TRAINING_CONFIG_VERSION  # noqa: E402

_ARM_LABELS = {
    "random": "Random",
    "llm_open_loop": "LLM Open-loop",
    "llm_closed_loop": "LLM Closed-loop",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _find_selected_candidate(outcome: ArmRunOutcome):
    if outcome.selected_structural_hash is None:
        return None
    return next(
        (c for c in outcome.candidates if c.structural_hash == outcome.selected_structural_hash),
        None,
    )


def write_all_outputs(
    output_dir: Path,
    store: ResultStore,
    args: Any,
    provenance: dict,
    dataset_diagnostics: dict,
    search_space_report: dict,
    arm_outcomes: list[ArmRunOutcome],
    baseline_results: list[BaselineResult],
    task_name: str,
    n_qubits: int,
    feature_count: int,
    readout_qubit: int,
    max_gates: int,
    elapsed_seconds: float,
    run_label: str = "MOCK/NON-LLM AMPLITUDE-ENCODING SMOKE RUN",
) -> None:
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    _write_experiment_config(
        output_dir, args, provenance, n_qubits, feature_count, readout_qubit, max_gates, run_label
    )
    _write_dataset_summary(output_dir, dataset_diagnostics)
    _write_search_space_summary(output_dir, search_space_report)
    _write_candidate_trace_csv(output_dir, arm_outcomes)
    _write_selected_circuits(output_dir, arm_outcomes, store, task_name)
    _write_training_summary_csv(output_dir, arm_outcomes)
    _write_architecture_diagnostics_csv(output_dir, arm_outcomes)
    _write_pareto_summary_csv(output_dir, arm_outcomes)
    _write_figures(output_dir, arm_outcomes)
    _write_report_md(output_dir, arm_outcomes, baseline_results, elapsed_seconds, run_label)
    _write_readme(output_dir, args, n_qubits, feature_count, readout_qubit, max_gates, run_label)


def _write_experiment_config(
    output_dir: Path, args: Any, provenance: dict, n_qubits: int, feature_count: int,
    readout_qubit: int, max_gates: int, run_label: str,
) -> None:
    config = {
        "run_label": run_label,
        "experiment_name": "free_amplitude_fixed_readout_v1",
        "scientific_status": (
            "integration/smoke demonstration only; not a statistically powered result"
        ),
        "n_qubits": n_qubits,
        "feature_count": feature_count,
        "readout_qubit": readout_qubit,
        "max_gates": max_gates,
        "encoding": "amplitude",
        "measurement": {"observable": "Z", "wires": [readout_qubit]},
        "classical_parameter_count": 0,
        "initialization_policy": FREE_AMPLITUDE_INIT_VERSION,
        "training_config_version": FREE_AMPLITUDE_TRAINING_CONFIG_VERSION,
        "differentiation_method": "backprop",
        "duplicate_detection_scope": "arm_local",
        "dataset_profile": getattr(args, "dataset_profile", None),
        "seed": getattr(args, "seed", None),
        "budget_per_arm": getattr(args, "budget", None),
        "epochs_per_candidate": getattr(args, "epochs", None),
        "search_arms": ["random", "llm_open_loop", "llm_closed_loop"],
        "generated_at_utc": _now_iso(),
        **provenance,
    }
    (output_dir / "EXPERIMENT_CONFIG.json").write_text(json.dumps(config, indent=2))


def _write_dataset_summary(output_dir: Path, dataset_diagnostics: dict) -> None:
    (output_dir / "dataset_summary.json").write_text(json.dumps(dataset_diagnostics, indent=2))


def _write_search_space_summary(output_dir: Path, search_space_report: dict) -> None:
    (output_dir / "search_space_summary.json").write_text(json.dumps(search_space_report, indent=2))


def _write_candidate_trace_csv(output_dir: Path, arm_outcomes: list[ArmRunOutcome]) -> None:
    fieldnames = [
        "arm", "proposal_index", "proposal_id", "operations", "valid", "is_duplicate",
        "structural_hash", "training_outcome", "final_val_rmse", "circuit_depth",
        "quantum_parameter_count", "controlled_gate_count", "initial_gradient_norm",
        "final_gradient_norm", "zero_gradient_parameter_count", "parameter_count_in_causal_cone",
        "api_latency_seconds", "input_tokens", "output_tokens",
    ]
    with (output_dir / "candidate_trace.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for outcome in arm_outcomes:
            for c in outcome.candidates:
                writer.writerow(
                    {
                        "arm": c.arm, "proposal_index": c.proposal_index,
                        "proposal_id": c.proposal_id,
                        "operations": json.dumps(c.operations) if c.operations else None,
                        "valid": c.valid, "is_duplicate": c.is_duplicate,
                        "structural_hash": c.structural_hash,
                        "training_outcome": c.training_outcome,
                        "final_val_rmse": c.final_val_rmse, "circuit_depth": c.circuit_depth,
                        "quantum_parameter_count": c.quantum_parameter_count,
                        "controlled_gate_count": c.controlled_gate_count,
                        "initial_gradient_norm": c.initial_gradient_norm,
                        "final_gradient_norm": c.final_gradient_norm,
                        "zero_gradient_parameter_count": c.zero_gradient_parameter_count,
                        "parameter_count_in_causal_cone": c.parameter_count_in_causal_cone,
                        "api_latency_seconds": c.api_latency_seconds,
                        "input_tokens": c.input_tokens, "output_tokens": c.output_tokens,
                    }
                )


def _selected_weights(outcome: ArmRunOutcome, store: ResultStore, task_name: str) -> dict | None:
    if outcome.selected_structural_hash is None or outcome.selected_train_seed is None:
        return None
    namespaced = outcome.task_namespace or arm_task_name(task_name, outcome.arm)
    return store.get_trained_weights(
        namespaced, outcome.selected_structural_hash, outcome.selected_train_seed
    )


def _write_selected_circuits(
    output_dir: Path, arm_outcomes: list[ArmRunOutcome], store: ResultStore, task_name: str
) -> None:
    selections = []
    for outcome in arm_outcomes:
        entry: dict = {
            "arm": outcome.arm,
            "task_namespace": outcome.task_namespace,
            "selected_structural_hash": outcome.selected_structural_hash,
            "selected_operations": outcome.selected_operations,
            "selected_val_rmse": outcome.selected_val_rmse,
            "protected_test_rmse": outcome.protected_test_rmse,
            "protected_test_mae": outcome.protected_test_mae,
            "train_seed": outcome.selected_train_seed,
        }
        weights = _selected_weights(outcome, store, task_name)
        if weights is not None:
            entry["initial_angles"] = weights.get("initial_angles")
            entry["learned_angles"] = weights.get("learned_angles")
            entry["angle_deltas"] = weights.get("angle_deltas")
            entry["quantum_parameter_count"] = len(weights.get("learned_angles") or [])
            entry["classical_parameter_count"] = weights.get("classical_parameter_count", 0)
            entry["searched_body_gate_count"] = weights.get("searched_body_gate_count")
            entry["controlled_gate_count"] = weights.get("controlled_gate_count")
            entry["searched_body_depth"] = weights.get("searched_body_depth")
            entry["total_compiled_depth"] = weights.get("total_compiled_depth")
            entry["decomposed_state_preparation_gate_count"] = weights.get(
                "decomposed_state_preparation_gate_count"
            )
            entry["decomposed_state_preparation_depth"] = weights.get(
                "decomposed_state_preparation_depth"
            )
            entry["parameter_count_in_causal_cone"] = weights.get("parameter_count_in_causal_cone")
            entry["fraction_in_causal_cone"] = weights.get("fraction_in_causal_cone")
        selections.append(entry)
    (output_dir / "selected_circuits.json").write_text(json.dumps(selections, indent=2))


def _write_training_summary_csv(output_dir: Path, arm_outcomes: list[ArmRunOutcome]) -> None:
    fieldnames = [
        "arm", "selected_structural_hash", "selected_operations", "selected_val_rmse",
        "protected_test_rmse", "protected_test_mae", "num_proposed", "num_valid",
        "num_invalid", "num_duplicate", "num_failed", "num_unique", "consumed_budget",
        "stop_reason", "api_successful_calls", "api_failed_calls", "api_outbound_attempts",
    ]
    with (output_dir / "training_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for outcome in arm_outcomes:
            writer.writerow(
                {
                    "arm": outcome.arm,
                    "selected_structural_hash": outcome.selected_structural_hash,
                    "selected_operations": (
                        json.dumps(outcome.selected_operations)
                        if outcome.selected_operations else None
                    ),
                    "selected_val_rmse": outcome.selected_val_rmse,
                    "protected_test_rmse": outcome.protected_test_rmse,
                    "protected_test_mae": outcome.protected_test_mae,
                    **outcome.ledger_summary,
                    "stop_reason": outcome.stop_reason,
                    "api_successful_calls": outcome.api_successful_calls,
                    "api_failed_calls": outcome.api_failed_calls,
                    "api_outbound_attempts": outcome.api_outbound_attempts,
                }
            )


def _write_architecture_diagnostics_csv(
    output_dir: Path, arm_outcomes: list[ArmRunOutcome]
) -> None:
    fieldnames = [
        "arm", "proposal_id", "structural_hash", "circuit_depth", "controlled_gate_count",
        "quantum_parameter_count", "parameter_count_in_causal_cone",
        "zero_gradient_parameter_count", "initial_gradient_norm", "final_gradient_norm",
    ]
    with (output_dir / "architecture_diagnostics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for outcome in arm_outcomes:
            for c in outcome.candidates:
                if c.training_outcome != "success":
                    continue
                writer.writerow(
                    {
                        "arm": c.arm, "proposal_id": c.proposal_id,
                        "structural_hash": c.structural_hash, "circuit_depth": c.circuit_depth,
                        "controlled_gate_count": c.controlled_gate_count,
                        "quantum_parameter_count": c.quantum_parameter_count,
                        "parameter_count_in_causal_cone": c.parameter_count_in_causal_cone,
                        "zero_gradient_parameter_count": c.zero_gradient_parameter_count,
                        "initial_gradient_norm": c.initial_gradient_norm,
                        "final_gradient_norm": c.final_gradient_norm,
                    }
                )


def _write_pareto_summary_csv(output_dir: Path, arm_outcomes: list[ArmRunOutcome]) -> None:
    """Val-RMSE vs. circuit-depth Pareto frontier across all successfully
    trained candidates (both objectives lower-is-better)."""
    candidates = [
        c
        for outcome in arm_outcomes
        for c in outcome.candidates
        if c.training_outcome == "success"
        and c.final_val_rmse is not None
        and c.circuit_depth is not None
    ]
    pareto = []
    for c in candidates:
        dominated = any(
            (other.final_val_rmse <= c.final_val_rmse and other.circuit_depth <= c.circuit_depth)
            and (other.final_val_rmse < c.final_val_rmse or other.circuit_depth < c.circuit_depth)
            for other in candidates
        )
        if not dominated:
            pareto.append(c)

    fieldnames = [
        "arm", "proposal_id", "structural_hash", "final_val_rmse", "circuit_depth",
        "quantum_parameter_count",
    ]
    with (output_dir / "pareto_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in sorted(pareto, key=lambda c: c.final_val_rmse):
            writer.writerow(
                {
                    "arm": c.arm, "proposal_id": c.proposal_id,
                    "structural_hash": c.structural_hash,
                    "final_val_rmse": c.final_val_rmse, "circuit_depth": c.circuit_depth,
                    "quantum_parameter_count": c.quantum_parameter_count,
                }
            )


def _write_figures(output_dir: Path, arm_outcomes: list[ArmRunOutcome]) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    arm_names, best_values = [], []
    for outcome in arm_outcomes:
        label = _ARM_LABELS.get(outcome.arm, outcome.arm)
        values = [c.final_val_rmse for c in outcome.candidates if c.final_val_rmse is not None]
        if values:
            ax.scatter([label] * len(values), values, alpha=0.5, color="tab:gray", zorder=2)
        if outcome.selected_val_rmse is not None:
            arm_names.append(label)
            best_values.append(outcome.selected_val_rmse)
    if arm_names:
        ax.scatter(
            arm_names, best_values, color="tab:blue", zorder=3, label="selected (best)", s=80
        )
        ax.legend()
    ax.set_ylabel("Validation RMSE")
    ax.set_title("Validation RMSE by arm (free-amplitude-fixed-readout)")
    fig.tight_layout()
    fig.savefig(output_dir / "figures" / "validation_rmse_by_arm.png", dpi=150)
    fig.savefig(output_dir / "figures" / "validation_rmse_by_arm.svg")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    plotted = False
    for outcome in arm_outcomes:
        label = _ARM_LABELS.get(outcome.arm, outcome.arm)
        depths = [c.circuit_depth for c in outcome.candidates if c.training_outcome == "success"]
        rmses = [c.final_val_rmse for c in outcome.candidates if c.training_outcome == "success"]
        if depths:
            ax2.scatter(depths, rmses, label=label)
            plotted = True
    ax2.set_xlabel("Compiled circuit depth")
    ax2.set_ylabel("Validation RMSE")
    ax2.set_title("Depth vs. validation RMSE (Pareto view)")
    if plotted:
        ax2.legend()
    fig2.tight_layout()
    fig2.savefig(output_dir / "figures" / "pareto_depth_vs_rmse.png", dpi=150)
    fig2.savefig(output_dir / "figures" / "pareto_depth_vs_rmse.svg")
    plt.close(fig2)


def _write_report_md(
    output_dir: Path, arm_outcomes: list[ArmRunOutcome], baseline_results: list[BaselineResult],
    elapsed_seconds: float, run_label: str,
) -> None:
    lines = [
        f"# {run_label}",
        "",
        "# Free-Amplitude Fixed-Readout Report",
        "",
        "This is an integration/smoke demonstration. It does not establish that "
        "any search arm is superior, and is not a scientific performance claim.",
        "",
        f"Elapsed: {elapsed_seconds:.1f}s.",
        "",
        "## Baselines",
        "",
        "| name | trainable params | val RMSE | val MAE | test RMSE | test MAE |",
        "|---|---|---|---|---|---|",
    ]
    for b in baseline_results:
        test_r = f"{b.test_rmse:.4f}" if b.test_rmse is not None else "n/a"
        test_m = f"{b.test_mae:.4f}" if b.test_mae is not None else "n/a"
        lines.append(
            f"| {b.name} | {b.trainable_parameter_count} | {b.val_rmse:.4f} | {b.val_mae:.4f} | "
            f"{test_r} | {test_m} |"
        )

    lines += ["", "## Every candidate", "",
        "| arm | idx | valid | dup | val RMSE | depth | 2q gates | q params | params in cone | "
        "zero-grad params |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for outcome in arm_outcomes:
        label = _ARM_LABELS.get(outcome.arm, outcome.arm)
        for c in outcome.candidates:
            val_rmse = f"{c.final_val_rmse:.4f}" if c.final_val_rmse is not None else "n/a"
            lines.append(
                f"| {label} | {c.proposal_index} | {c.valid} | {c.is_duplicate} | {val_rmse} | "
                f"{c.circuit_depth} | {c.controlled_gate_count} | {c.quantum_parameter_count} | "
                f"{c.parameter_count_in_causal_cone} | {c.zero_gradient_parameter_count} |"
            )

    lines += ["", "## Selected architecture per arm", ""]
    for outcome in arm_outcomes:
        label = _ARM_LABELS.get(outcome.arm, outcome.arm)
        lines.append(f"### {label}")
        lines.append(f"- stop reason: {outcome.stop_reason or 'budget exhausted normally'}")
        lines.append(f"- ledger: {outcome.ledger_summary}")
        lines.append(
            f"- API: {outcome.api_successful_calls} ok / {outcome.api_failed_calls} failed / "
            f"{outcome.api_outbound_attempts} outbound attempts"
        )
        if outcome.selected_structural_hash is None:
            lines.append("- no successfully trained candidate in this arm.")
        else:
            lines.append(f"- selected operations: `{outcome.selected_operations}`")
            lines.append(f"- structural hash: `{outcome.selected_structural_hash}`")
            lines.append(f"- selected validation RMSE: {outcome.selected_val_rmse:.4f}")
            test_str = (
                f"{outcome.protected_test_rmse:.4f} (MAE {outcome.protected_test_mae:.4f})"
                if outcome.protected_test_rmse is not None else "not evaluated"
            )
            lines.append(f"- protected-test RMSE: {test_str}")
            lines.append("- classical trainable parameters: 0")
        lines.append("")

    lines += [
        "(Initial/learned quantum angles and full diagnostics are in "
        "`selected_circuits.json`, `architecture_diagnostics.csv`, and `candidate_trace.csv`.)",
        "",
        "## Amplitude-encoding limitation",
        "L2 amplitude normalization removes overall multiplicative scale: two samples "
        "differing only by a positive amplitude factor A become the same normalized "
        "quantum state before noise. This model cannot recover absolute signal magnitude; "
        "only the peak location (mu) is targeted.",
        "",
        "---",
        f"# {run_label}",
        "This is an integration/smoke demonstration, not a scientific performance claim.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines))


def _write_readme(
    output_dir: Path, args: Any, n_qubits: int, feature_count: int, readout_qubit: int,
    max_gates: int, run_label: str,
) -> None:
    content = f"""# free_amplitude_fixed_readout_v1

> **{run_label}**

Free-form parameterized-gate VQC search with amplitude encoding and a
fixed quantum readout. The model has NO classical trainable layer: amplitude
encoding in, a free 1-{max_gates} gate quantum body, `mu_hat = (1 - <Z_{readout_qubit}>) / 2`
out. `classical_parameter_count` is always exactly 0.

## Configuration
- n_qubits: {n_qubits}. feature_count: {feature_count} (= 2^{n_qubits}).
- readout_qubit: {readout_qubit} (fixed; never selected by the LLM, never trained).
- max_gates: {max_gates}.
- dataset_profile: {getattr(args, "dataset_profile", None)}.
- Search arms: Random, LLM Open-loop, LLM Closed-loop -- duplicate detection is
  arm-local (each arm has its own cache namespace).

## Files
- `report.md` -- baselines, every candidate, selected architecture per arm.
- `EXPERIMENT_CONFIG.json`, `dataset_summary.json`, `search_space_summary.json`.
- `candidate_trace.csv`, `training_summary.csv`, `architecture_diagnostics.csv`,
  `pareto_summary.csv`, `selected_circuits.json`.
- `figures/validation_rmse_by_arm.{{png,svg}}`, `figures/pareto_depth_vs_rmse.{{png,svg}}`.

No raw prompts, raw model responses, full weight arrays, or credentials are
stored here.
"""
    (output_dir / "README.md").write_text(content)
