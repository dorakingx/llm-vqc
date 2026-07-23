"""Main-mode prompt templates: the LLM proposes a COMPLETE candidate
(structure + numerical `theta`), and the closed-loop feedback carries the
arm's full compact history.

Only that arm's own evaluated history ever reaches a closed-loop prompt
(no Random/other-arm results, no protected-Test metric -- there is no Test
field in the feedback structure at all). The feedback is a single
deterministic JSON block: full per-proposal history plus the current best
and remaining budget, so the next proposal may change gate type, order,
wires, gate count, or theta with complete information.
"""

from __future__ import annotations

import json

PROMPT_VERSION = "free_amplitude_main_v1"

_MAIN_TASK_TEMPLATE = """\
Task: predict the peak location (mu) of a noisy 1D Gaussian curve encoded \
directly as a quantum amplitude vector (regression). Lower validation RMSE \
is better.

The model has NO classical layers. The {feature_count} input values (an \
exact 2^{n_qubits} = {feature_count} amplitude vector) are loaded via \
amplitude encoding onto {n_qubits} qubits, your proposed circuit runs at \
the angles YOU provide, and the prediction is read directly off qubit 0:

    mu_hat = (1 - <Z_0>) / 2

Qubit 0 is the ONLY qubit that determines the prediction. A gate that never \
causally connects to qubit 0 has zero effect on the output and wastes your \
{max_gates}-gate budget -- route information toward qubit 0.

You propose a COMPLETE candidate: an ORDERED sequence of {min_gates} to \
{max_gates} operations, EACH with its own numerical angle where required. \
The angles are NOT optimized afterward -- the circuit is evaluated at \
exactly the theta values you give, so choose them deliberately.

Gate rules:
  - H acts on one wire and has NO theta.
  - RX, RY, RZ act on one wire and REQUIRE one finite theta in [-pi, pi].
  - CRX, CRY, CRZ act on two wires [control, target] (control != target) \
and REQUIRE one finite theta in [-pi, pi].
  - At least one parameterized (non-H) gate is required.

Respond with ONLY a single JSON object of this exact shape -- no code, no \
explanation, no markdown fences, no extra fields:
{{
  "n_qubits": {n_qubits},
  "operations": [
    {{"gate": "H", "wires": [1]}},
    {{"gate": "CRY", "wires": [1, 0], "theta": 1.247}},
    {{"gate": "RZ", "wires": [2], "theta": -0.381}}
  ]
}}
"""


def build_main_system_prompt(
    n_qubits: int, feature_count: int, max_gates: int, min_gates: int = 1
) -> str:
    return _MAIN_TASK_TEMPLATE.format(
        n_qubits=n_qubits, feature_count=feature_count, max_gates=max_gates, min_gates=min_gates
    )


def build_main_open_loop_user_prompt() -> str:
    """Open-loop turn: identical every call, no history of any kind."""
    return "Propose a complete candidate (structure and theta)."


def build_main_closed_loop_first_user_prompt() -> str:
    return "This is your first proposal. Propose a complete candidate (structure and theta)."


def build_main_closed_loop_feedback_user_prompt(
    history: list[dict],
    best_operations: list[dict] | None,
    best_val_rmse: float | None,
    remaining_budget: int,
) -> str:
    """Closed-loop feedback: the arm's FULL compact history as one
    deterministic JSON block. Each `history` entry (built by the runner
    from this arm's own results only) already contains: operations+theta,
    train MSE, val RMSE/MAE, prediction std, validity/duplicate status,
    gate/controlled-gate count and body depth, readout-lightcone /
    sensitivity diagnostics. Never any protected-Test metric.
    """
    feedback = {
        "history": history,
        "best_so_far": {
            "operations": best_operations,
            "validation_rmse": round(best_val_rmse, 6) if best_val_rmse is not None else None,
        },
        "remaining_budget": remaining_budget,
    }
    feedback_json = json.dumps(feedback, sort_keys=True)
    return (
        "Your full search history so far in THIS arm (all fields; null means not "
        f"available):\n{feedback_json}\n"
        "Propose a different or improved complete candidate (lower validation RMSE is "
        "better). You may change gate type, order, wires, gate count, or theta."
    )
