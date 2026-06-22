"""Tests for exploration visualization helpers."""

from __future__ import annotations

from pathlib import Path

from llm_vqc.circuit_explorer import explore_circuit_space
from llm_vqc.visualization import (
    generate_exploration_visualizations,
    plot_exploration_trajectory,
    save_best_circuit_visualization,
)


def test_plot_exploration_trajectory(tmp_path: Path) -> None:
    result = explore_circuit_space(num_qubits=2, max_depth=2)
    output_path = tmp_path / "optimization_trajectory.png"

    saved_path = plot_exploration_trajectory(result, output_path)

    assert saved_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert len(result["exploration_trajectory"]) == result["total_unique_states"]


def test_save_best_circuit_visualization(tmp_path: Path) -> None:
    result = explore_circuit_space(num_qubits=2, max_depth=2)

    outputs = save_best_circuit_visualization(result, tmp_path)

    assert outputs["best_circuit_text"].exists()
    text = outputs["best_circuit_text"].read_text(encoding="utf-8")
    assert result["best_circuit"]["circuit_str"] in text
    if "best_circuit_image" in outputs:
        assert outputs["best_circuit_image"].exists()


def test_generate_exploration_visualizations(tmp_path: Path) -> None:
    result = explore_circuit_space(num_qubits=2, max_depth=2)

    outputs = generate_exploration_visualizations(result, tmp_path)

    assert outputs["optimization_trajectory"].name == "optimization_trajectory.png"
    assert outputs["best_circuit_text"].name == "best_circuit.txt"
    assert outputs["optimization_trajectory"].exists()
