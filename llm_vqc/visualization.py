"""Visualization helpers for VQC exploration results."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from llm_vqc.circuit_explorer import GateAction, gates_to_quantum_circuit

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("outputs")
EXPLORATION_TRAJECTORY_FILENAME = "exploration_trajectory.png"
BEST_CIRCUIT_FILENAME = "best_circuit.png"
BEST_CIRCUIT_TEXT_FILENAME = "best_circuit.txt"
GATE_DISTRIBUTION_FILENAME = "gate_distribution.png"

GATE_ORDER = ("H", "X", "Y", "Z", "CX", "CY", "CZ")


class VisualizationError(Exception):
    """Raised when visualization assets cannot be generated."""


def ensure_output_dir(output_dir: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    """Create the output directory if it does not exist."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_exploration_trajectory(
    exploration_result: dict[str, Any],
    output_path: Path | str | None = None,
) -> Path:
    """
    Plot unique-state discovery progress across BFS depth and iteration.

    Saves a two-panel figure:
    - Top: cumulative unique states vs exploration step (iteration)
    - Bottom: cumulative unique states vs circuit depth (search depth)
    """
    trajectory = exploration_result.get("exploration_trajectory")
    if not trajectory:
        raise VisualizationError("exploration_result is missing exploration_trajectory.")

    output_dir = ensure_output_dir(
        Path(output_path).parent if output_path else DEFAULT_OUTPUT_DIR
    )
    destination = (
        Path(output_path)
        if output_path
        else output_dir / EXPLORATION_TRAJECTORY_FILENAME
    )

    steps = [point["step"] for point in trajectory]
    coverage = [point["coverage_score"] for point in trajectory]

    depth_counts = {
        int(depth): int(count)
        for depth, count in exploration_result.get("new_states_at_depth", {}).items()
    }
    depths = sorted(depth_counts)
    cumulative_by_depth: list[int] = []
    running_total = 0
    for depth in depths:
        running_total += depth_counts[depth]
        cumulative_by_depth.append(running_total)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

    axes[0].plot(steps, coverage, color="#2563eb", linewidth=2)
    axes[0].set_title("Unique States vs Exploration Iteration")
    axes[0].set_xlabel("BFS Iteration (Discovery Step)")
    axes[0].set_ylabel("Cumulative Unique States")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(depths, cumulative_by_depth, color="#059669", linewidth=2, marker="o")
    axes[1].set_title("Unique States vs Search Depth")
    axes[1].set_xlabel("Circuit Depth (Gate Count)")
    axes[1].set_ylabel("Cumulative Unique States")
    axes[1].grid(True, alpha=0.3)

    num_qubits = exploration_result.get("num_qubits", "?")
    max_depth = exploration_result.get("max_depth", "?")
    total_unique = exploration_result.get("total_unique_states", "?")
    fig.suptitle(
        f"VQC Exploration Trajectory (N={num_qubits}, G={max_depth}, total={total_unique})",
        fontsize=12,
    )

    try:
        fig.savefig(destination, dpi=150, bbox_inches="tight")
    except OSError as exc:
        raise VisualizationError(f"Failed to save trajectory plot: {exc}") from exc
    finally:
        plt.close(fig)

    logger.info("Saved exploration trajectory plot to %s", destination)
    return destination


def plot_gate_distribution(
    exploration_result: dict[str, Any],
    output_path: Path | str | None = None,
) -> Path:
    """Plot gate-type frequency across all simplest (minimum-depth) circuits."""
    gate_distribution = exploration_result.get("gate_distribution")
    if not gate_distribution:
        raise VisualizationError("exploration_result is missing gate_distribution.")

    output_dir = ensure_output_dir(
        Path(output_path).parent if output_path else DEFAULT_OUTPUT_DIR
    )
    destination = (
        Path(output_path)
        if output_path
        else output_dir / GATE_DISTRIBUTION_FILENAME
    )

    gate_names = [gate for gate in GATE_ORDER if gate in gate_distribution]
    if not gate_names:
        gate_names = list(gate_distribution.keys())

    counts = [int(gate_distribution[gate]) for gate in gate_names]

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    bars = ax.bar(gate_names, counts, color="#7c3aed", alpha=0.85)
    ax.set_title("Gate Distribution Across Simplest Circuits")
    ax.set_xlabel("Gate Type")
    ax.set_ylabel("Total Occurrences")
    ax.grid(True, axis="y", alpha=0.3)

    for bar, count in zip(bars, counts, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    num_qubits = exploration_result.get("num_qubits", "?")
    max_depth = exploration_result.get("max_depth", "?")
    fig.suptitle(
        f"Gate Usage in Minimum-Depth Circuits (N={num_qubits}, G={max_depth})",
        fontsize=11,
    )

    try:
        fig.savefig(destination, dpi=150, bbox_inches="tight")
    except OSError as exc:
        raise VisualizationError(f"Failed to save gate distribution plot: {exc}") from exc
    finally:
        plt.close(fig)

    logger.info("Saved gate distribution plot to %s", destination)
    return destination


def save_best_circuit_visualization(
    exploration_result: dict[str, Any],
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Save Qiskit circuit diagram and ASCII fallback for the best circuit."""
    best_circuit = exploration_result.get("best_circuit")
    num_qubits = exploration_result.get("num_qubits")
    if not best_circuit or num_qubits is None:
        raise VisualizationError("exploration_result is missing best_circuit or num_qubits.")

    directory = ensure_output_dir(output_dir)
    image_path = directory / BEST_CIRCUIT_FILENAME
    text_path = directory / BEST_CIRCUIT_TEXT_FILENAME

    gate_actions = tuple(
        GateAction(gate["name"], tuple(gate["qubits"]))
        for gate in best_circuit.get("gates", [])
    )
    circuit = gates_to_quantum_circuit(gate_actions, int(num_qubits))

    try:
        figure = circuit.draw(output="mpl", fold=-1)
        figure.savefig(image_path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        logger.info("Saved best circuit diagram to %s", image_path)
    except Exception as exc:
        logger.warning("Matplotlib circuit draw failed, saving text fallback only: %s", exc)

    try:
        ascii_diagram = str(circuit.draw(output="text", fold=-1))
        header = (
            f"Best Circuit (depth={best_circuit.get('depth')}, "
            f"gates={best_circuit.get('gate_count')})\n"
            f"Sequence: {best_circuit.get('circuit_str', 'I')}\n\n"
        )
        text_path.write_text(header + ascii_diagram, encoding="utf-8")
    except OSError as exc:
        raise VisualizationError(f"Failed to save circuit text diagram: {exc}") from exc

    logger.info("Saved best circuit ASCII diagram to %s", text_path)

    saved: dict[str, Path] = {"best_circuit_text": text_path}
    if image_path.exists():
        saved["best_circuit_image"] = image_path
    return saved


def generate_exploration_visualizations(
    exploration_result: dict[str, Any],
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Generate all exploration visual artifacts without crashing on partial failures."""
    if "error" in exploration_result:
        raise VisualizationError(
            f"Cannot visualize failed exploration result: {exploration_result['error']}"
        )

    directory = ensure_output_dir(output_dir)
    outputs: dict[str, Path] = {}

    for name, generator in (
        ("exploration_trajectory", lambda: plot_exploration_trajectory(
            exploration_result, directory / EXPLORATION_TRAJECTORY_FILENAME
        )),
        ("gate_distribution", lambda: plot_gate_distribution(
            exploration_result, directory / GATE_DISTRIBUTION_FILENAME
        )),
    ):
        try:
            outputs[name] = generator()
        except VisualizationError as exc:
            logger.warning("Skipped %s: %s", name, exc)

    try:
        circuit_paths = save_best_circuit_visualization(exploration_result, directory)
        outputs.update(circuit_paths)
    except VisualizationError as exc:
        logger.warning("Skipped best circuit visualization: %s", exc)

    if not outputs:
        raise VisualizationError("No visualization artifacts could be generated.")

    return outputs
