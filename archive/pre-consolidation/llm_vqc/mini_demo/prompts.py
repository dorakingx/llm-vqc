"""Fixed, versioned prompt templates for the mini demo's LLM arms.

Deliberately parallel to `llm_vqc.llm.prompts` (same "prompts are versioned
in-repo, not generated differently per call" discipline) but describing
the compact 4-field grammar (`llm_vqc.mini_demo.compact_schema`) instead
of full `CircuitIR`. Only master-plan-approved feedback fields ever reach
the closed-loop prompt: validation RMSE, circuit depth, two-qubit gate
count, and duplicate/valid status -- never a protected-test metric (there
is no such field on `SearchFeedback` to leak in the first place).
"""

from __future__ import annotations

import json

PROMPT_VERSION = "mini_demo_v2"

_TASK_AND_GRAMMAR = """\
Task: predict the peak position (mu) of a noisy 1D Gaussian curve from 21 \
sampled points (regression). Lower validation RMSE is better.

The model always has exactly 2 qubits and exactly 4 trainable quantum \
rotation angles (2 in the first variational layer, 2 in the second). You \
do not choose the qubit count, the data encoding, or the measurement -- \
those are fixed. You choose only these four discrete fields:
{
  "layer_1_gate": "RX" | "RY" | "RZ",
  "entangler": "CNOT" | "CZ" | "NONE",
  "entangler_direction": "0_to_1" | "1_to_0",
  "layer_2_gate": "RX" | "RY" | "RZ"
}
layer_1_gate and layer_2_gate each apply independently to both qubits (one \
trainable angle per qubit per layer). entangler is an optional, fixed \
(non-trainable) two-qubit gate placed between the two rotation layers; \
"NONE" means no entangling gate. entangler_direction only affects CNOT \
(CZ and NONE are direction-independent).

Respond with ONLY a single JSON object matching that exact shape -- no \
code, no explanation, no markdown fences, no additional fields.
"""


def build_system_prompt(task_description: str) -> str:
    return (
        f"You are designing a small variational quantum circuit for: {task_description}.\n"
        f"{_TASK_AND_GRAMMAR}"
    )


def build_open_loop_user_prompt() -> str:
    """Open-loop turn: no feedback from any earlier proposal, not even a
    proposal count -- every open-loop call is prompted identically."""
    return "Propose a circuit."


def build_closed_loop_first_user_prompt() -> str:
    return "This is your first proposal. Propose a circuit."


def build_closed_loop_feedback_user_prompt(
    prior_layer_1_gate: str,
    prior_entangler: str,
    prior_entangler_direction: str,
    prior_layer_2_gate: str,
    valid: bool,
    is_duplicate: bool,
    val_rmse: float | None,
    circuit_depth: int | None,
    two_qubit_gate_count: int | None,
) -> str:
    """Closed-loop feedback turn: a single deterministic JSON-like block
    reporting *all* required fields for the prior proposal, then a request
    for a different or improved architecture.

    **Correction pass (section 4): always include every field.** Earlier the
    builder used mutually-exclusive prose branches that omitted validation
    RMSE and circuit metrics whenever the prior proposal was a duplicate (or
    invalid). Now the same six fields -- architecture, validation RMSE (or
    explicit null), validity, duplicate status, circuit depth (or null),
    two-qubit-gate count (or null) -- appear unconditionally, so the LLM
    never silently loses information. Never includes any protected-test
    field: none is available to this function's caller in the first place
    (there is no test parameter here to pass one through).
    """
    prior_architecture = {
        "layer_1_gate": prior_layer_1_gate,
        "entangler": prior_entangler,
        "entangler_direction": prior_entangler_direction,
        "layer_2_gate": prior_layer_2_gate,
    }
    # `json.dumps(None) == "null"`, and floats are rounded for determinism;
    # every field is present regardless of validity/duplicate status.
    feedback = {
        "prior_architecture": prior_architecture,
        "valid": valid,
        "is_duplicate": is_duplicate,
        "validation_rmse": round(val_rmse, 6) if val_rmse is not None else None,
        "circuit_depth": circuit_depth,
        "two_qubit_gate_count": two_qubit_gate_count,
    }
    feedback_json = json.dumps(feedback, sort_keys=True)
    return (
        f"Feedback on your previous proposal (all fields; null means not "
        f"available):\n{feedback_json}\n"
        "Propose a different or improved circuit (lower validation RMSE is better)."
    )
