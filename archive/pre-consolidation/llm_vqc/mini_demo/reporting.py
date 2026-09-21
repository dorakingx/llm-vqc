"""Writes every artifact under `outputs/mini_llm_api_vqc_demo_v1/` from the
run's own in-memory/`ResultStore`-persisted results only.

No value in any written file is invented: every number is read back from a
`CandidateRecord`/`ArmRunOutcome` the runner populated from a real
`EvaluationResult`/`LLMCallRecord`, from the trained-weight cache, or from
a static constant describing the demo's fixed design. Raw prompts, raw
model responses, full weight arrays, and API keys are never written here --
only parsed proposals, aggregate token/latency numbers, structural
metrics, and compact statistical summaries of the learned weights.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from llm_vqc.evaluation.store import ResultStore  # noqa: E402
from llm_vqc.mini_demo.capacity import EXPECTED_TOTAL_TRAINABLE_PARAMETERS  # noqa: E402
from llm_vqc.mini_demo.compact_schema import (  # noqa: E402
    COMPACT_ARCHITECTURE_JSON_SCHEMA,
    EXPECTED_QUANTUM_PARAMETER_COUNT,
    N_QUBITS,
)
from llm_vqc.mini_demo.init_policy import MINI_DEMO_INIT_VERSION  # noqa: E402
from llm_vqc.mini_demo.records import ArmRunOutcome  # noqa: E402
from llm_vqc.mini_demo.runner import DIFF_METHOD, arm_task_name  # noqa: E402

_ARM_LABELS = {
    "random": "Random",
    "llm_open_loop": "LLM Open-loop",
    "llm_closed_loop": "LLM Closed-loop",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _find_selected_candidate(outcome: ArmRunOutcome):
    """The candidate matching `outcome.selected_structural_hash`, or `None`.
    Guards against the `None` sentinel matching an INVALID/FAILED
    candidate's `None` structural_hash."""
    if outcome.selected_structural_hash is None:
        return None
    return next(
        (c for c in outcome.candidates if c.structural_hash == outcome.selected_structural_hash),
        None,
    )


def _stat_summary(values: list[float]) -> dict:
    """Compact statistical summary of a flat weight tensor -- never the full
    array (keeps checkpoints/large blobs out of committed artifacts)."""
    n = len(values)
    if n == 0:
        return {"count": 0}
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return {
        "count": n,
        "mean": mean,
        "std": math.sqrt(var),
        "min": min(values),
        "max": max(values),
        "l2_norm": math.sqrt(sum(v * v for v in values)),
    }


def _classical_summaries(state: dict | None) -> dict:
    """Per-classical-tensor summaries (embed/head), excluding the quantum
    angles (`q_layer.weights`), which are reported separately in full."""
    if not state:
        return {}
    return {
        name: {"shape": _shape_of(values), **_stat_summary(values)}
        for name, values in state.items()
        if name != "q_layer.weights"
    }


def _shape_of(flat: list[float]) -> list[int]:
    # The demo's classical tensors are small and 1-D-flattened in the store;
    # record the flattened length as the recorded shape dimension.
    return [len(flat)]


def write_all_outputs(
    output_dir: Path,
    store: ResultStore,
    args: Any,
    provenance: dict,
    model: str,
    real_calls_made: int,
    max_real_calls: int,
    elapsed_seconds: float,
    prompt_version: str,
    arm_outcomes: list[ArmRunOutcome],
    task_name: str = "T1_mini_demo",
    run_status: str = "corrected",
) -> None:
    _write_compact_schema_json(output_dir)
    _write_experiment_config(
        output_dir, args, provenance, model, max_real_calls, prompt_version, run_status
    )
    _write_candidate_results_csv(output_dir, arm_outcomes)
    _write_run_summary(output_dir, arm_outcomes, args, elapsed_seconds, run_status)
    llm_calls = _collect_llm_calls(store)
    _write_api_usage_summary(
        output_dir, llm_calls, arm_outcomes, model, real_calls_made, max_real_calls
    )
    _write_selected_circuits(output_dir, arm_outcomes, store, task_name)
    _write_learned_parameters_md(output_dir, args, arm_outcomes, store, task_name)
    _write_metrics_md(output_dir)
    _write_readme(output_dir, args, model, real_calls_made, max_real_calls, run_status)
    _write_figures(output_dir, arm_outcomes)
    _write_report_md(
        output_dir, arm_outcomes, args, model, real_calls_made, max_real_calls,
        elapsed_seconds, run_status,
    )
    _write_artifact_manifest(
        output_dir, args, provenance, elapsed_seconds, arm_outcomes, run_status
    )


def _collect_llm_calls(store: ResultStore) -> list[dict]:
    calls = []
    for run_id in ("llm_open_loop", "llm_closed_loop"):
        for record in store.iter_llm_calls(run_id):
            calls.append(
                {
                    "arm": run_id,
                    "proposal_id": record.proposal_id,
                    "model": record.model,
                    "input_tokens": record.input_tokens,
                    "output_tokens": record.output_tokens,
                    "total_tokens": record.input_tokens + record.output_tokens,
                    "latency_seconds": record.latency_seconds,
                    "validation_errors": record.validation_errors,
                    "created_at": record.created_at,
                }
            )
    return calls


def _write_compact_schema_json(output_dir: Path) -> None:
    (output_dir / "compact_architecture_schema.json").write_text(
        json.dumps(COMPACT_ARCHITECTURE_JSON_SCHEMA, indent=2)
    )


def _write_experiment_config(
    output_dir: Path, args: Any, provenance: dict, model: str, max_real_calls: int,
    prompt_version: str, run_status: str,
) -> None:
    config = {
        "experiment_name": "mini_llm_api_vqc_demo_v1",
        "run_status": run_status,
        "scientific_status": "integration demonstration only; not a statistically powered result",
        "task": "T1_gaussian_peak",
        "qubits": N_QUBITS,
        "quantum_parameters": EXPECTED_QUANTUM_PARAMETER_COUNT,
        "total_trainable_parameters": EXPECTED_TOTAL_TRAINABLE_PARAMETERS,
        "encoding": {"type": "angle", "gate": "RY", "wires": [0, 1]},
        "measurement": {"observable": "Z", "wires": [0, 1]},
        "search_arms": ["random", "llm_open_loop", "llm_closed_loop"],
        "budget_per_arm": args.budget,
        "seeds": 1,
        "seed_value": args.seed,
        "epochs_per_candidate": args.epochs,
        "batch_size": 16,
        "optimizer": "adamw",
        "initialization_policy": MINI_DEMO_INIT_VERSION,
        "differentiation_method": DIFF_METHOD,
        "dtype": "float64",
        "device": "cpu",
        "duplicate_detection_scope": "arm_local",
        "max_real_llm_calls": max_real_calls,
        "max_candidate_trainings": 6,
        "provider": args.provider,
        "model": model,
        "prompt_version": prompt_version,
        "generated_at_utc": _now_iso(),
        **provenance,
    }
    (output_dir / "EXPERIMENT_CONFIG.json").write_text(json.dumps(config, indent=2))


def _write_candidate_results_csv(output_dir: Path, arm_outcomes: list[ArmRunOutcome]) -> None:
    fieldnames = [
        "arm", "proposal_index", "proposal_id", "layer_1_gate", "entangler",
        "entangler_direction", "layer_2_gate", "valid", "is_duplicate", "structural_hash",
        "training_outcome", "final_train_mse", "final_val_rmse", "circuit_depth",
        "total_gate_count", "two_qubit_gate_count", "quantum_parameter_count",
        "total_trainable_parameters", "training_runtime_seconds", "cache_hit",
        "actually_trained", "unique_training_id", "dtype_device_verified",
        "quantum_angles_changed", "api_latency_seconds", "input_tokens", "output_tokens",
    ]
    with (output_dir / "candidate_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for outcome in arm_outcomes:
            for c in outcome.candidates:
                compact = c.compact or {}
                writer.writerow(
                    {
                        "arm": c.arm,
                        "proposal_index": c.proposal_index,
                        "proposal_id": c.proposal_id,
                        "layer_1_gate": compact.get("layer_1_gate"),
                        "entangler": compact.get("entangler"),
                        "entangler_direction": compact.get("entangler_direction"),
                        "layer_2_gate": compact.get("layer_2_gate"),
                        "valid": c.valid,
                        "is_duplicate": c.is_duplicate,
                        "structural_hash": c.structural_hash,
                        "training_outcome": c.training_outcome,
                        "final_train_mse": c.final_train_mse,
                        "final_val_rmse": c.final_val_rmse,
                        "circuit_depth": c.circuit_depth,
                        "total_gate_count": c.total_gate_count,
                        "two_qubit_gate_count": c.two_qubit_gate_count,
                        "quantum_parameter_count": c.quantum_parameter_count,
                        "total_trainable_parameters": c.total_trainable_parameters,
                        "training_runtime_seconds": c.training_runtime_seconds,
                        "cache_hit": c.cache_hit,
                        "actually_trained": c.actually_trained,
                        "unique_training_id": c.unique_training_id,
                        "dtype_device_verified": c.dtype_device_verified,
                        "quantum_angles_changed": c.quantum_angles_changed,
                        "api_latency_seconds": c.api_latency_seconds,
                        "input_tokens": c.input_tokens,
                        "output_tokens": c.output_tokens,
                    }
                )


def _write_run_summary(
    output_dir: Path, arm_outcomes: list[ArmRunOutcome], args: Any, elapsed_seconds: float,
    run_status: str,
) -> None:
    rows = []
    for outcome in arm_outcomes:
        rows.append(
            {
                "arm": outcome.arm,
                **outcome.ledger_summary,
                "unique_trainings": sum(1 for c in outcome.candidates if c.actually_trained),
                "stop_reason": outcome.stop_reason,
                "selected_structural_hash": outcome.selected_structural_hash,
                "selected_val_rmse": outcome.selected_val_rmse,
                "protected_test_rmse": outcome.protected_test_rmse,
            }
        )
    with (output_dir / "run_summary.csv").open("w", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "run_status": run_status,
        "seed": args.seed,
        "budget_per_arm": args.budget,
        "epochs_per_candidate": args.epochs,
        "elapsed_seconds": elapsed_seconds,
        "arms": rows,
        "note": (
            "End-to-end integration demonstration with n=1 seed; does not establish "
            "that any search arm is superior."
        ),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))


def _write_api_usage_summary(
    output_dir: Path, llm_calls: list[dict], arm_outcomes: list[ArmRunOutcome],
    model: str, real_calls_made: int, max_real_calls: int,
) -> None:
    total_input = sum(c["input_tokens"] for c in llm_calls)
    total_output = sum(c["output_tokens"] for c in llm_calls)
    latencies = [c["latency_seconds"] for c in llm_calls]
    total_latency = sum(latencies)
    successful = sum(o.api_successful_calls for o in arm_outcomes)
    failed = sum(o.api_failed_calls for o in arm_outcomes)
    attempts = sum(o.api_outbound_attempts for o in arm_outcomes)
    summary = {
        "provider": "OpenAI (real)",
        "model": model,
        "max_real_llm_calls": max_real_calls,
        "real_calls_made": real_calls_made,
        "successful_logical_calls": successful,
        "failed_logical_calls": failed,
        "outbound_request_attempts": attempts,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_tokens": total_input + total_output,
        "latency_seconds_per_successful_call": latencies,
        "total_successful_call_latency_seconds": total_latency,
        "mean_latency_seconds": (total_latency / len(latencies)) if latencies else None,
        "calls": llm_calls,
    }
    (output_dir / "api_usage_summary.json").write_text(json.dumps(summary, indent=2))


def _selected_weights(outcome: ArmRunOutcome, store: ResultStore, task_name: str) -> dict | None:
    if outcome.selected_structural_hash is None or outcome.selected_train_seed is None:
        return None
    namespaced = outcome.task_namespace or arm_task_name(task_name, outcome.arm)
    return store.get_trained_weights(
        namespaced, outcome.selected_structural_hash, outcome.selected_train_seed
    )


def _angles_changed(initial: list | None, learned: list | None, tol: float = 1e-6) -> bool | None:
    if not initial or not learned or len(initial) != len(learned):
        return None
    return any(abs(a - b) > tol for a, b in zip(initial, learned, strict=False))


def _write_selected_circuits(
    output_dir: Path, arm_outcomes: list[ArmRunOutcome], store: ResultStore, task_name: str
) -> None:
    selections = []
    for outcome in arm_outcomes:
        entry: dict = {
            "arm": outcome.arm,
            "task_namespace": outcome.task_namespace,
            "selected_structural_hash": outcome.selected_structural_hash,
            "selected_val_rmse": outcome.selected_val_rmse,
            "protected_test_rmse": outcome.protected_test_rmse,
            "train_seed": outcome.selected_train_seed,
        }
        selected_candidate = _find_selected_candidate(outcome)
        entry["compact_architecture"] = selected_candidate.compact if selected_candidate else None

        weights = _selected_weights(outcome, store, task_name)
        if weights is not None:
            initial_angles = weights.get("initial_circuit_weights")
            learned_angles = weights.get("circuit_weights")
            entry["initial_quantum_angles"] = initial_angles
            entry["learned_quantum_angles"] = learned_angles
            entry["quantum_angles_changed"] = _angles_changed(initial_angles, learned_angles)
            entry["classical_summary_initial"] = _classical_summaries(
                weights.get("initial_classical_state")
            )
            entry["classical_summary_learned"] = _classical_summaries(
                weights.get("classical_state")
            )
            entry["param_provenance"] = weights.get("param_provenance")
            entry["total_trainable_parameters"] = EXPECTED_TOTAL_TRAINABLE_PARAMETERS
        selections.append(entry)
    (output_dir / "selected_circuits.json").write_text(json.dumps(selections, indent=2))


def _write_learned_parameters_md(
    output_dir: Path, args: Any, arm_outcomes: list[ArmRunOutcome], store: ResultStore,
    task_name: str,
) -> None:
    lines = [
        "# Learned Parameters -- mini_llm_api_vqc_demo_v1",
        "",
        f"Initialization policy: `{MINI_DEMO_INIT_VERSION}`. "
        f"Differentiation method: `{DIFF_METHOD}`. dtype: float64. device: CPU.",
        "",
        "## Gradient-trained parameters "
        f"(AdamW, {args.epochs} epochs, batch size 16)",
        "",
        "### Classical embedding: `Linear(21 -> 2)` -- 42 weights + 2 biases",
        "- initialization: weights Xavier (Glorot) uniform, biases zeros.",
        "",
        "### Quantum parameters: 4 rotation angles (2 per variational layer)",
        "- initialization: `Uniform[-pi, pi]`.",
        "- optimized jointly with the classical parameters through PennyLane's",
        "  `qml.qnn.TorchLayer` with `diff_method=\"backprop\"`.",
        "",
        "### Classical output head: `Linear(2 -> 1)` -- 2 weights + 1 bias",
        "- initialization: weights Xavier uniform, bias zero.",
        "",
        "**Total: 44 + 4 + 3 = 51 trainable parameters** -- every candidate is",
        "hard-verified before training (`verify_fixed_capacity`) and again inside",
        f"`{MINI_DEMO_INIT_VERSION}` (exactly 4 quantum / 51 total, all float64, all CPU).",
        "",
        "## Not gradient-trained (chosen by the LLM / random arm, or fixed by design)",
        "- Gate families for layer 1 and layer 2 (`RX`/`RY`/`RZ`).",
        "- The optional entangler (`CNOT`/`CZ`/`NONE`) and its direction.",
        "- Layer ordering; the input-dependent RY encoding angles; the fixed",
        "  encoding (RY on wires [0,1]) and measurement (Z on wires [0,1]);",
        "  dataset values and targets.",
        "",
        "## Initial vs learned quantum angles (each selected arm)",
        "",
    ]
    for outcome in arm_outcomes:
        label = _ARM_LABELS.get(outcome.arm, outcome.arm)
        weights = _selected_weights(outcome, store, task_name)
        lines.append(f"### {label}")
        if weights is None:
            lines.append("- no successfully trained candidate selected in this arm.")
            lines.append("")
            continue
        initial = weights.get("initial_circuit_weights")
        learned = weights.get("circuit_weights")
        lines.append(f"- initial quantum angles: `{_fmt_floats(initial)}`")
        lines.append(f"- learned quantum angles: `{_fmt_floats(learned)}`")
        lines.append(f"- at least one angle changed (> 1e-6): {_angles_changed(initial, learned)}")
        lines.append("")
    (output_dir / "LEARNED_PARAMETERS.md").write_text("\n".join(lines))


def _fmt_floats(values: list | None) -> str:
    if not values:
        return "n/a"
    return "[" + ", ".join(f"{v:.6f}" for v in values) + "]"


def _write_metrics_md(output_dir: Path) -> None:
    content = """# Metrics -- mini_llm_api_vqc_demo_v1

## Training MSE
Minibatch gradient updates: `mean((mu_hat - mu)^2)`.

## Validation RMSE
Candidate comparison / architecture selection within each arm (lower is
better): `sqrt(mean((mu_hat - mu)^2))`.

## Protected-test RMSE
Computed exactly once per completed arm, for the architecture selected on
validation RMSE, after that arm's search ended. Never returned to the LLM
or to any search arm.

## Process metrics (per candidate)
- validity / duplicate status (duplicate is **arm-local** -- a circuit is a
  duplicate only relative to earlier proposals of the SAME arm);
- final training MSE, final validation RMSE, training runtime;
- circuit depth, total gate count, two-qubit gate count;
- quantum parameter count (4), total trainable parameter count (51);
- cache hit, actually-trained-vs-reused, unique training id;
- dtype/device verification result (float64 + CPU);
- whether at least one quantum angle changed after training.

## API metrics
- successful logical calls, failed logical calls, outbound request attempts;
- total input / output / total tokens;
- latency per successful call and total successful-call latency.
"""
    (output_dir / "METRICS.md").write_text(content)


def _write_readme(
    output_dir: Path, args: Any, model: str, real_calls_made: int, max_real_calls: int,
    run_status: str,
) -> None:
    content = f"""# mini_llm_api_vqc_demo_v1

A minimal, real-OpenAI-API integration demonstration: an LLM proposes a
2-qubit variational circuit's discrete structure (compact 4-field grammar),
AdamW trains its 51 continuous parameters (4 quantum rotation angles) on the
T1 Gaussian-peak-position regression task, validation RMSE selects the best
circuit per arm, and a protected test partition is scored once per arm.

**run_status:** `{run_status}`. Integration demonstration only (1 seed,
budget=2 per arm, {args.epochs} epochs per candidate, {max_real_calls}
real LLM calls). Does not establish that any search arm is superior.

## Corrected-pass guarantees
- Duplicate detection and closed-loop feedback are **arm-local** (no
  cross-arm contamination; arm execution order cannot change results).
- Closed-loop feedback always carries all six fields (architecture,
  validation RMSE-or-null, validity, duplicate status, depth-or-null,
  two-qubit-gate-count-or-null); never any protected-test field.
- Explicit initialization (`mini_demo_init_v1`: Xavier/zeros/Uniform[-pi,pi]),
  `diff_method="backprop"`, all trainable tensors float64 on CPU.
- Provider built with `max_retries=0`; a positive `LLM_API_BUDGET_USD` and
  an explicit `OPENAI_MODEL` are required before any request.

## Configuration
- Task: T1 Gaussian-peak regression (train=150, val=250, protected test=2000).
- Qubits: 2. Quantum parameters: 4. Total trainable parameters: 51.
- Provider: OpenAI (`{model}`), real API calls, hard cap {max_real_calls}.
- Seed: {args.seed}.

## Reproduction
```bash
export OPENAI_API_KEY=...          # never printed or committed
export OPENAI_MODEL=...            # required; no silent default
export LLM_API_BUDGET_USD=2.00     # required positive hard cap
python scripts/run_mini_llm_vqc_demo.py --provider openai --budget 2 \\
    --epochs {args.epochs} --seed {args.seed} \\
    --output outputs/mini_llm_api_vqc_demo_v1
```

No raw prompts, raw model responses, full weight arrays, or credentials are
stored here -- only parsed proposals, aggregate token/latency numbers, and
compact weight summaries.
"""
    (output_dir / "README.md").write_text(content)


def _write_figures(output_dir: Path, arm_outcomes: list[ArmRunOutcome]) -> None:
    _write_val_rmse_figure(output_dir, arm_outcomes)
    _write_training_curves_figure(output_dir, arm_outcomes)


def _write_val_rmse_figure(output_dir: Path, arm_outcomes: list[ArmRunOutcome]) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    arm_names, best_values = [], []
    for outcome in arm_outcomes:
        label = _ARM_LABELS.get(outcome.arm, outcome.arm)
        candidate_values = [
            c.final_val_rmse for c in outcome.candidates if c.final_val_rmse is not None
        ]
        if candidate_values:
            ax.scatter(
                [label] * len(candidate_values), candidate_values,
                alpha=0.5, color="tab:gray", zorder=2,
            )
        if outcome.selected_val_rmse is not None:
            arm_names.append(label)
            best_values.append(outcome.selected_val_rmse)
    if arm_names:
        ax.scatter(
            arm_names, best_values, color="tab:blue", zorder=3, label="selected (best)", s=80
        )
    ax.set_ylabel("Validation RMSE")
    ax.set_title("Validation RMSE by arm (mini demo, n=1 seed)")
    if arm_names:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "figures" / "validation_rmse_by_arm.png", dpi=150)
    fig.savefig(output_dir / "figures" / "validation_rmse_by_arm.svg")
    plt.close(fig)


def _write_training_curves_figure(output_dir: Path, arm_outcomes: list[ArmRunOutcome]) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    plotted = False
    for outcome in arm_outcomes:
        label = _ARM_LABELS.get(outcome.arm, outcome.arm)
        selected = _find_selected_candidate(outcome)
        if selected is not None and selected.train_loss_history:
            ax.plot(
                range(1, len(selected.train_loss_history) + 1),
                selected.train_loss_history,
                marker="o",
                label=f"{label} (selected, train MSE)",
            )
            plotted = True
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training MSE")
    ax.set_title("Training curves for each arm's selected circuit")
    if plotted:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "figures" / "training_curves.png", dpi=150)
    fig.savefig(output_dir / "figures" / "training_curves.svg")
    plt.close(fig)


def _write_report_md(
    output_dir: Path, arm_outcomes: list[ArmRunOutcome], args: Any, model: str,
    real_calls_made: int, max_real_calls: int, elapsed_seconds: float, run_status: str,
) -> None:
    lines = [
        "# Mini LLM-VQC API Demo -- Report",
        "",
        f"**run_status: {run_status}**",
        "",
        "This is an end-to-end integration demonstration with n=1 seed.",
        "It does not establish that any search arm is superior.",
        "",
        f"Provider: OpenAI (`{model}`). Real API calls made: {real_calls_made}/{max_real_calls}. "
        f"Elapsed: {elapsed_seconds:.1f}s.",
        "",
        "Duplicate detection and closed-loop feedback are arm-local; closed-loop",
        "feedback always includes validation RMSE, depth, and two-qubit-gate count",
        "(or explicit null) for the arm's own previous proposal.",
        "",
        "## Every candidate",
        "",
        "| arm | idx | layer1 | entangler | dir | layer2 | valid | dup | trained | "
        "train MSE | val RMSE | depth | gates | 2q | angles_changed | dtype_ok | "
        "latency(s) | tokens(in/out) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for outcome in arm_outcomes:
        label = _ARM_LABELS.get(outcome.arm, outcome.arm)
        for c in outcome.candidates:
            compact = c.compact or {}
            train_mse = f"{c.final_train_mse:.4f}" if c.final_train_mse is not None else "n/a"
            val_rmse = f"{c.final_val_rmse:.4f}" if c.final_val_rmse is not None else "n/a"
            latency = f"{c.api_latency_seconds:.2f}" if c.api_latency_seconds is not None else "n/a"
            tokens = (
                f"{c.input_tokens}/{c.output_tokens}" if c.input_tokens is not None else "n/a"
            )
            lines.append(
                f"| {label} | {c.proposal_index} | {compact.get('layer_1_gate', 'n/a')} | "
                f"{compact.get('entangler', 'n/a')} | "
                f"{compact.get('entangler_direction', 'n/a')} | "
                f"{compact.get('layer_2_gate', 'n/a')} | {c.valid} | {c.is_duplicate} | "
                f"{c.actually_trained} | {train_mse} | {val_rmse} | {c.circuit_depth} | "
                f"{c.total_gate_count} | {c.two_qubit_gate_count} | {c.quantum_angles_changed} | "
                f"{c.dtype_device_verified} | {latency} | {tokens} |"
            )

    lines += ["", "## Selected architecture per arm", ""]
    for outcome in arm_outcomes:
        label = _ARM_LABELS.get(outcome.arm, outcome.arm)
        selected = _find_selected_candidate(outcome)
        lines.append(f"### {label}")
        lines.append(f"- stop reason: {outcome.stop_reason or 'budget exhausted normally'}")
        lines.append(f"- ledger: {outcome.ledger_summary}")
        lines.append(
            f"- API: {outcome.api_successful_calls} ok / {outcome.api_failed_calls} failed / "
            f"{outcome.api_outbound_attempts} outbound attempts"
        )
        if selected is None:
            lines.append("- no successfully trained candidate in this arm.")
        else:
            compact = selected.compact or {}
            lines.append(f"- selected architecture: `{compact}`")
            lines.append(f"- structural hash: `{outcome.selected_structural_hash}`")
            lines.append(f"- selected validation RMSE: {outcome.selected_val_rmse:.4f}")
            test_str = (
                f"{outcome.protected_test_rmse:.4f}"
                if outcome.protected_test_rmse is not None else "not evaluated"
            )
            lines.append(f"- protected-test RMSE: {test_str}")
            lines.append(f"- total trainable parameters: {selected.total_trainable_parameters}")
        lines.append("")

    lines += [
        "(Initial and learned quantum angles, and classical-weight summaries, are in "
        "`selected_circuits.json` and `LEARNED_PARAMETERS.md`.)",
        "",
        "---",
        "This is an end-to-end integration demonstration with n=1 seed.",
        "It does not establish that any search arm is superior.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines))


def _write_artifact_manifest(
    output_dir: Path, args: Any, provenance: dict, elapsed_seconds: float,
    arm_outcomes: list[ArmRunOutcome], run_status: str,
) -> None:
    manifest = {
        "schema_version": 2,
        "publication_status": "sanitized_for_publication",
        "run_status": run_status,
        "generated_at_utc": _now_iso(),
        "source_result_store_relative_path": "runs/mini_llm_api_vqc_demo_v1/results.sqlite",
        "seed": args.seed,
        "budget_per_arm": args.budget,
        "epochs_per_candidate": args.epochs,
        "elapsed_seconds": elapsed_seconds,
        **provenance,
        "arms": [
            {
                "arm": outcome.arm,
                "task_namespace": outcome.task_namespace,
                "stop_reason": outcome.stop_reason,
                "ledger_summary": outcome.ledger_summary,
                "selected_structural_hash": outcome.selected_structural_hash,
            }
            for outcome in arm_outcomes
        ],
        "files": sorted(
            str(p.relative_to(output_dir)) for p in output_dir.rglob("*") if p.is_file()
        ),
        "excludes_by_policy": [
            "raw API prompts", "raw API responses", "API keys/headers",
            "hidden reasoning", "SQLite databases", "model checkpoints",
            "full weight arrays",
        ],
    }
    (output_dir / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2))
