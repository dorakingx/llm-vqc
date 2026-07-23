"""Fixed, versioned prompt templates for the free-gate LLM arms (Codex
instruction sections 8, 17).

Only that arm's own evaluated-candidate history ever reaches a closed-loop
prompt (section 17: "no Random Search results; no open-loop results; no
other arms' circuits; no protected-test RMSE; no unproposed candidates; no
oracle information"). The feedback block is one deterministic JSON-like
object with every field always present (`null` when not applicable) --
never mutually-exclusive prose branches that silently drop fields for a
duplicate/invalid proposal (the exact defect found and fixed in
`llm_vqc.mini_demo`'s closed-loop feedback during that experiment's
correction pass).
"""

from __future__ import annotations

import json

PROMPT_VERSION = "free_amplitude_v1"

_TASK_DESCRIPTION_TEMPLATE = """\
Task: predict the peak location (mu) of a noisy 1D Gaussian curve encoded \
directly as a quantum amplitude vector (regression). Lower validation RMSE \
is better.

The model has NO classical layers of any kind: the {feature_count} \
input values (already an exact 2^{n_qubits} = {feature_count} amplitude \
vector) are loaded via amplitude encoding onto {n_qubits} qubits, your \
proposed quantum circuit runs, and the prediction is read directly off \
qubit 0:

    mu_hat = (1 - <Z_0>) / 2

Qubit 0 is the ONLY qubit that determines the prediction. Any gate you \
place that never causally connects to qubit 0 (directly or through an \
entangling chain) has zero effect on the output and wastes your limited \
{max_gates}-gate budget -- route task-relevant information toward qubit 0.

You choose an ORDERED sequence of {min_gates} to {max_gates} gates from \
exactly these seven gate types: H, RX, RY, RZ, CRX, CRY, CRZ. \
H, RX, RY, RZ act on exactly one wire. CRX, CRY, CRZ act on exactly two \
wires, given as [control, target] (order matters -- control and target \
must differ). There is no required topology, layering, or entangler -- \
any order of any of these gates is allowed, as long as at least one \
gate is a parameterized rotation (RX, RY, RZ, CRX, CRY, or CRZ; H has no \
trainable parameter).

Respond with ONLY a single JSON object of this exact shape -- no code, no \
explanation, no markdown fences, no additional fields, and NEVER include \
angle/parameter values (the trainable angles are optimized separately by \
gradient descent, not chosen by you):
{{
  "n_qubits": {n_qubits},
  "operations": [
    {{"gate": "H", "wires": [1]}},
    {{"gate": "CRY", "wires": [1, 0]}},
    {{"gate": "RZ", "wires": [2]}}
  ]
}}
"""


def build_system_prompt(
    n_qubits: int, feature_count: int, max_gates: int, min_gates: int = 1
) -> str:
    return _TASK_DESCRIPTION_TEMPLATE.format(
        n_qubits=n_qubits, feature_count=feature_count, max_gates=max_gates, min_gates=min_gates
    )


def build_open_loop_user_prompt() -> str:
    """Open-loop turn: identical every call, no history of any kind."""
    return "Propose a circuit."


def build_closed_loop_first_user_prompt() -> str:
    return "This is your first proposal. Propose a circuit."


def build_closed_loop_feedback_user_prompt(
    prior_operations: list[dict],
    valid: bool,
    is_duplicate: bool,
    val_rmse: float | None,
    searched_body_gate_count: int | None,
    quantum_parameter_count: int | None,
    controlled_gate_count: int | None,
    compiled_depth: int | None,
    initial_gradient_norm: float | None,
    final_gradient_norm: float | None,
    zero_gradient_parameter_count: int | None,
    parameter_count_in_causal_cone: int | None,
    best_so_far_val_rmse: float | None,
    best_so_far_operations: list[dict] | None,
    remaining_budget: int,
) -> str:
    """Closed-loop feedback turn: one deterministic JSON block with every
    field always present (`null` when not applicable/not computed) --
    never a prose branch that silently omits a field. Carries ONLY this
    arm's own evaluated history: no other arm's results, no protected-test
    metric (there is no parameter here through which one could be passed).
    """
    feedback = {
        "prior_proposal": {"operations": prior_operations},
        "valid": valid,
        "is_duplicate": is_duplicate,
        "validation_rmse": round(val_rmse, 6) if val_rmse is not None else None,
        "searched_body_gate_count": searched_body_gate_count,
        "quantum_parameter_count": quantum_parameter_count,
        "controlled_gate_count": controlled_gate_count,
        "compiled_depth": compiled_depth,
        "initial_gradient_norm": (
            round(initial_gradient_norm, 8) if initial_gradient_norm is not None else None
        ),
        "final_gradient_norm": (
            round(final_gradient_norm, 8) if final_gradient_norm is not None else None
        ),
        "zero_gradient_parameter_count": zero_gradient_parameter_count,
        "parameter_count_in_q0_causal_cone": parameter_count_in_causal_cone,
        "best_so_far": {
            "validation_rmse": (
                round(best_so_far_val_rmse, 6) if best_so_far_val_rmse is not None else None
            ),
            "operations": best_so_far_operations,
        },
        "remaining_budget": remaining_budget,
    }
    feedback_json = json.dumps(feedback, sort_keys=True)
    return (
        "Feedback on your previous proposal and your own search history so far "
        f"(all fields; null means not available):\n{feedback_json}\n"
        "Propose a different or improved circuit (lower validation RMSE is better). "
        "Route parameters toward qubit 0's causal cone if the coverage above was low."
    )
