"""Prompts and strict schema for the `mini_5gate_v1` LLM arms.

One candidate per call (batch = 1). That is a deliberate choice, not an
economy: it makes the closed loop genuinely sequential (propose, learn,
propose) and makes the open loop exactly N independent draws, which is
the same shape as the random arm's N independent draws. Batching would
correlate candidates within a response and coarsen the feedback.

The prompt states the model contract and the grammar and nothing else.
It carries no task description, no data statistics and no test-side
information, so open-loop sees only the contract and n; closed-loop sees
the same plus validation-only feedback on its own past candidates.
"""

from __future__ import annotations

from llm_vqc.mini5.space import (
    CONTROLLED_GATES,
    EXACT_GATES,
    SINGLE_QUBIT_GATES,
)

MINI_PROMPT_VERSION = "mini_5gate_prompt_v2"

_SYSTEM = """You design one COMPLETE variational quantum circuit: the gate
sequence AND every numeric rotation angle. There is NO training step and
NO optimizer - your angles are used exactly as written, so choose them
deliberately.

Fixed model contract: a real signal of length 2**{n} is amplitude-encoded
on {n} qubits; your gate body runs; Pauli-Z is measured on qubit 0; the
prediction is mu_hat = (1 - <Z0>)/2 in [0,1]. Lower validation RMSE is
better.

Output exactly one JSON object, no prose and no markdown fences:
{{"operations": [OP, OP, OP, OP, OP]}}

HARD CONSTRAINT: the list contains EXACTLY {k} operations. Not fewer, not
more. A list of any other length is rejected and wastes your budget.

Each OP is {{"gate": G, "wires": [...], "theta": angle-or-null}}:
  G in {singles} -> wires [q], one wire;
  G in {controlled} -> wires [control, target], two DIFFERENT wires;
  theta is a number in [-3.14159, 3.14159] for every gate except H;
  theta is null for H, which takes no angle.

Wire indices are 0..{n_max}. Re-proposing a circuit with identical gates,
identical wires AND identical angles counts as a duplicate: it consumes a
call but earns no evaluation, so vary something every time."""

_OPEN_USER = """Propose circuit {index} of {budget}.

You are given no feedback about previous circuits. Rely on the model
contract and the grammar alone."""

_CLOSED_USER_FIRST = """Propose circuit {index} of {budget}.

No circuit has been evaluated yet."""

_CLOSED_USER = """Propose circuit {index} of {budget}.
{duplicate_warning}
Validation results for what you have already tried, best first. These are
validation-only numbers; no test data is involved.

{archive}

ALREADY TRIED - do NOT propose any of these again. Re-proposing one is a
duplicate: it consumes a call, earns no evaluation, and leaves you with
fewer circuits than the other methods.

{tried}

Your next circuit MUST differ from every circuit listed above in at least
one gate, one wire, or one angle. Change something deliberately: keep what
the results suggest is working and vary the rest."""

_DUPLICATE_WARNING = """
WARNING: your previous reply repeated a circuit you had already proposed.
It was discarded and earned nothing. Do not repeat it again.
"""


def system_prompt(n_qubits: int) -> str:
    return _SYSTEM.format(
        n=n_qubits,
        n_max=n_qubits - 1,
        k=EXACT_GATES,
        singles=list(SINGLE_QUBIT_GATES),
        controlled=list(CONTROLLED_GATES),
    )


def open_user_prompt(index: int, budget: int) -> str:
    return _OPEN_USER.format(index=index, budget=budget)


def _format_op(op: dict) -> str:
    wires = ",".join(str(w) for w in op["wires"])
    if op.get("theta") is None:
        return f"{op['gate']}(q{wires})"
    return f"{op['gate']}(q{wires},{float(op['theta']):+.3f})"


def closed_user_prompt(
    index: int,
    budget: int,
    archive: list[dict],
    tried: list[list[dict]] | None = None,
    last_was_duplicate: bool = False,
) -> str:
    """`archive` is validation-side only: operations plus the validation
    RMSE they achieved. `tried` is EVERY circuit already proposed,
    including ones rejected as duplicates, listed so the model can see
    exactly what not to repeat -- the v1 prompt showed only the scored
    top-8 and drew a 74% duplicate rate as the model kept re-proposing
    its own best entry."""
    if not archive and not tried:
        return _CLOSED_USER_FIRST.format(index=index, budget=budget)
    ranked = [
        f"{rank}. val_rmse={entry['val_rmse']:.4f}  "
        + " ".join(_format_op(op) for op in entry["operations"])
        for rank, entry in enumerate(sorted(archive, key=lambda e: e["val_rmse"]), start=1)
    ]
    seen = tried if tried is not None else [e["operations"] for e in archive]
    tried_lines = [
        f"- {' '.join(_format_op(op) for op in ops)}" for ops in seen
    ]
    return _CLOSED_USER.format(
        index=index,
        budget=budget,
        duplicate_warning=_DUPLICATE_WARNING if last_was_duplicate else "",
        archive="\n".join(ranked) if ranked else "(nothing scored yet)",
        tried="\n".join(tried_lines),
    )


#: Strict JSON schema. OpenAI structured outputs require every declared
#: property to appear in `required`, so `theta` is a nullable union rather
#: than an optional key - a lesson learned from the bench_v2 `center`
#: field, where an optional-looking property was silently forced into
#: every object and then rejected by the validator.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "operations": {
            "type": "array",
            "minItems": EXACT_GATES,
            "maxItems": EXACT_GATES,
            "items": {
                "type": "object",
                "properties": {
                    "gate": {
                        "type": "string",
                        "enum": list(SINGLE_QUBIT_GATES) + list(CONTROLLED_GATES),
                    },
                    "wires": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 1,
                        "maxItems": 2,
                    },
                    "theta": {"type": ["number", "null"]},
                },
                "required": ["gate", "wires", "theta"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["operations"],
    "additionalProperties": False,
}
