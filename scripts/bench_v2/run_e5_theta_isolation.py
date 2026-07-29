#!/usr/bin/env python
"""E5 — theta-isolation ablation (protocol §10): on architectures frozen
from the E1 LLM joint cells, compare the quality of

  (1) the LLM's verbatim theta            —  1 objective evaluation,
  (2) random theta sampling               — 16 objective evaluations,
  (3) a classical optimizer (Nelder-Mead) — 16 objective evaluations,

with EXACT objective-call matching between (2) and (3) and the (1)-call
budget of the LLM condition reported explicitly. All objective values are
VALIDATION MSE on the cell's own frozen split — E5 introduces no new
protected-test evaluations at all.

Architecture source (frozen rule): every completed E1 `llm_open_joint` /
`llm_closed_joint` cell contributes its SELECTED candidate's architecture
(gates + wires, theta stripped) once.

Writes outputs/bench_v2/E5/theta_isolation.csv + COMPLETE.json.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from llm_vqc.bench_v2.cell_runner import _resolve_task  # noqa: E402
from llm_vqc.bench_v2.space import READOUT_QUBIT  # noqa: E402
from llm_vqc.evaluation.seeds import derive_child_seed  # noqa: E402
from llm_vqc.evaluation.store import ResultStore  # noqa: E402
from llm_vqc.free_amplitude.candidate_schema import (  # noqa: E402
    CompleteCandidateProposal,
    complete_candidate_to_ir_and_theta,
)
from llm_vqc.free_amplitude.model import FixedReadoutQuantumModel  # noqa: E402

RUNS_ROOT = REPO / "runs" / "bench_v2"
OUT_DIR = REPO / "outputs" / "bench_v2" / "E5"
MATCHED_BUDGET = 16  # objective calls for random AND classical conditions
LLM_ARMS = ("llm_open_joint", "llm_closed_joint")


class _ObjectiveCounter:
    """Counts every objective evaluation — the audited budget."""

    def __init__(self, model: FixedReadoutQuantumModel, features, targets) -> None:
        self.model = model
        self.x = torch.tensor(features, dtype=torch.float64)
        self.y = np.asarray(targets, dtype=np.float64)
        self.calls = 0

    def __call__(self, theta: np.ndarray) -> float:
        self.calls += 1
        with torch.no_grad():
            self.model.q_layer.weights.copy_(
                torch.tensor(np.asarray(theta, dtype=np.float64))
            )
            pred = self.model(self.x).cpu().numpy().reshape(-1)
        return float(np.mean((pred - self.y) ** 2))


def main() -> int:
    cells_dir = RUNS_ROOT / "E1" / "cells"
    manifests = [
        json.loads(p.read_text())
        for p in sorted(cells_dir.glob("*.json"))
        if json.loads(p.read_text()).get("status") == "complete"
    ] if cells_dir.is_dir() else []
    llm_manifests = [m for m in manifests if m["arm"] in LLM_ARMS]
    if not llm_manifests:
        print("E5: no completed E1 LLM cells found — run E1 first")
        return 1

    rows = []
    for manifest in llm_manifests:
        store_path = Path(manifest["store_path"])
        if not store_path.is_absolute():
            store_path = REPO / store_path
        store = ResultStore(store_path)
        state = json.loads(store.load_run_state(manifest["cell_id"]))
        store.close()
        proposal = CompleteCandidateProposal.model_validate(state["best_candidate"])
        ir, llm_theta = complete_candidate_to_ir_and_theta(proposal, READOUT_QUBIT)
        n_params = len(llm_theta)

        task, _is_cls, _vm = _resolve_task(manifest["task"], manifest["n_qubits"])
        train_val, _diag = task.build(manifest["data_seed"])
        model = FixedReadoutQuantumModel(ir, readout_qubit=READOUT_QUBIT)
        model.double()
        objective = _ObjectiveCounter(
            model, train_val.val.features, train_val.val.targets
        )

        llm_value = objective(np.asarray(llm_theta))
        llm_calls = objective.calls

        rng = np.random.default_rng(derive_child_seed(
            manifest["data_seed"], "e5_random_theta", manifest["cell_id"]
        ))
        objective.calls = 0
        if n_params:
            random_values = [
                objective(rng.uniform(-np.pi, np.pi, size=n_params))
                for _ in range(MATCHED_BUDGET)
            ]
        else:
            random_values = [objective(np.zeros(0))]
        random_calls = objective.calls
        random_best = float(min(random_values))

        objective.calls = 0
        if n_params:
            x0 = rng.uniform(-np.pi, np.pi, size=n_params)
            result = minimize(
                objective, x0, method="Nelder-Mead",
                options={"maxfev": MATCHED_BUDGET, "xatol": 1e-8, "fatol": 1e-12},
            )
            classical_best = float(result.fun)
        else:
            classical_best = float(objective(np.zeros(0)))
        classical_calls = objective.calls

        rows.append({
            "source_cell": manifest["cell_id"], "task": manifest["task"],
            "n_qubits": manifest["n_qubits"], "arm": manifest["arm"],
            "replicate": manifest["replicate"], "n_parameters": n_params,
            "llm_theta_val_mse": llm_value, "llm_objective_calls": llm_calls,
            "random_best_val_mse": random_best,
            "random_objective_calls": random_calls,
            "random_first_val_mse": float(random_values[0]),
            "classical_best_val_mse": classical_best,
            "classical_objective_calls": classical_calls,
            "matched_budget": MATCHED_BUDGET,
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "theta_isolation.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    # Budget matching: both non-LLM conditions receive the identical
    # 16-call allotment; Nelder-Mead may legitimately CONVERGE below it
    # (actual calls are recorded per row and never exceed the budget).
    call_matched = all(
        r["random_objective_calls"] == r["matched_budget"]
        and r["classical_objective_calls"] <= r["matched_budget"]
        for r in rows if r["n_parameters"]
    )
    (OUT_DIR / "COMPLETE.json").write_text(json.dumps({
        "experiment": "E5", "status": "complete", "architectures": len(rows),
        "matched_budget": MATCHED_BUDGET, "call_matched": call_matched,
        "objective": "validation MSE only; no protected-test evaluations",
    }, indent=2) + "\n")
    print(f"E5: {len(rows)} architectures, call_matched={call_matched}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
