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

from llm_vqc.mini5.angles import ANGLE_RANGES, ANY_RANGE
from llm_vqc.mini5.space import (
    CONTROLLED_GATES,
    EXACT_GATES,
    SINGLE_QUBIT_GATES,
)

MINI_PROMPT_VERSION = "mini5_taskaware_prompt_v5"

_SYSTEM = """You design a variational quantum circuit body of EXACTLY
{k} gates on {n} qubits.

Fixed contract: a signal of length 2**{n} is amplitude-encoded on the
qubits, your gates run, Pauli-Z is measured on qubit 0, and the
prediction is mu_hat = (1 - <Z0>)/2. Lower validation RMSE is better.
Only gates whose effect reaches qubit 0 can change the prediction.
{theta_rule}
Output one JSON object, no prose and no fences:
{{"operations": [{{"gate": G, "wires": [...], {theta_key}}}, ...]}}
  G in {singles} -> wires [q], one wire;
  G in {controlled} -> wires [control, target], two DIFFERENT wires;
  wire indices are 0..{n_max}.

The list must hold EXACTLY {k} operations. Any other length is rejected
and wastes your budget. Re-proposing an identical circuit is a duplicate:
it consumes a call and earns no evaluation."""

#: Range mode. The model picks WHERE the angle comes from, never the
#: number itself - measured three times, this model cannot sample a
#: continuum and collapses onto pi/4, then onto the prompt's examples,
#: then onto typeable digit runs.
_THETA_RANGE_RULE = """
You do NOT choose angles. For each rotation you choose the RANGE its
angle is drawn from, and the harness then samples uniformly at random
inside that range. Pick the range where you believe good angles live.
"{any_label}" is the whole [-pi, pi] interval, which is exactly what the
random baseline does - choose it when you have no preference. H takes no
angle: use null.
  theta_range must be one of {ranges}
"""

#: Optimizer mode. Angles are trained, so proposing them is meaningless.
_THETA_TRAINED_RULE = """
You do NOT choose angles at all. A shared AdamW optimizer trains every
rotation angle after you propose the structure, identically for every
method, so your only job is the gate sequence and the wiring. Omit
angles entirely.
"""
#: Task descriptions, written from the generator definitions in
#: llm_vqc/tasks/signal_suite/generators.py and from nothing else. No
#: performance observation of any kind went into this text: the author has
#: seen test results for these tasks, so anything beyond what the data
#: generator does would leak them into the arm being measured.
TASK_CONTEXT = {
    "gauss_peak":
        "The signal is a single Gaussian bump on a uniform grid, with a "
        "random height, width and baseline. Predict the POSITION of its "
        "peak, normalised to [0,1].",
    "sin_freq":
        "The signal is a sinusoid on a uniform grid, with a random "
        "amplitude, phase and offset. Predict its FREQUENCY, normalised "
        "to [0,1].",
    "change_point":
        "The signal is piecewise: one level, then a step to another "
        "level, each side with a small random slope. Predict the "
        "POSITION of the step, normalised to [0,1].",
    "peak_count":
        "The signal contains either one Gaussian bump or two separated "
        "ones. Predict 1 if there are TWO bumps and 0 if there is one; "
        "the two classes are balanced.",
}

_TASK_BLOCK = """
The task this circuit is for:
  {description}
The target is the value the prediction (1 - <Z0>)/2 should match.
"""

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
one gate, one wire, or one angle.

{refine_rule}Say nothing; only emit the JSON."""

_REFINE_RANGE = """You have no gradients, so behave like a derivative-free
optimizer: keep what the scores suggest is working and change one thing at
a time - a gate, a wire, or one angle range.
"""

#: Angles are trained here, so re-proposing a shape with "different
#: angles" is the SAME candidate. Saying otherwise produced a 37-duplicate
#: cell that exhausted its proposal guard.
_REFINE_TRAINED = """Angles are trained, so they are not yours to vary:
two circuits with the same gates and wires are the SAME candidate no
matter what angles you imagine for them.

You have 16 positions. Before you answer, pick ONE position index and
either change its gate type or change its wires, then emit the whole
circuit with that single edit applied. Do not return a circuit you have
already sent - re-sending your current best is the most common way to
waste a turn here, and it earns nothing.
"""

_DUPLICATE_WARNING = """
WARNING: your previous reply repeated a circuit you had already proposed.
It was discarded and earned nothing. Do not repeat it again.
"""


def system_prompt(n_qubits: int, n_gates: int = EXACT_GATES,
                  trained_angles: bool = False) -> str:
    if trained_angles:
        rule, key = _THETA_TRAINED_RULE, '"theta_range": null'
    else:
        rule = _THETA_RANGE_RULE.format(any_label=ANY_RANGE,
                                        ranges=list(ANGLE_RANGES))
        key = '"theta_range": R'
    return _SYSTEM.format(
        n=n_qubits, n_max=n_qubits - 1, k=n_gates,
        theta_rule=rule, theta_key=key,
        singles=list(SINGLE_QUBIT_GATES),
        controlled=list(CONTROLLED_GATES),
    )


def open_user_prompt(index: int, budget: int, family: str | None = None) -> str:
    """`family` set = task-aware arm; None = the blind arm, unchanged."""
    body = _OPEN_USER.format(index=index, budget=budget)
    if family is None:
        return body
    return _TASK_BLOCK.format(description=TASK_CONTEXT[family]) + body


def _format_op(op: dict) -> str:
    """Feedback shows the RANGE the proposer chose, not the angle the
    harness happened to draw - otherwise the model would be asked to
    reason about a number it did not pick and cannot control."""
    wires = ",".join(str(w) for w in op["wires"])
    label = op.get("theta_range")
    if label is None:
        return f"{op['gate']}(q{wires})"
    return f"{op['gate']}(q{wires},{label})"


def closed_user_prompt(
    index: int,
    budget: int,
    archive: list[dict],
    tried: list[list[dict]] | None = None,
    last_was_duplicate: bool = False,
    trained_angles: bool = False,
    family: str | None = None,
) -> str:
    """`archive` is validation-side only: operations plus the validation
    RMSE they achieved. `tried` is EVERY circuit already proposed,
    including ones rejected as duplicates, listed so the model can see
    exactly what not to repeat -- the v1 prompt showed only the scored
    top-8 and drew a 74% duplicate rate as the model kept re-proposing
    its own best entry."""
    prefix = "" if family is None else _TASK_BLOCK.format(
        description=TASK_CONTEXT[family])
    if not archive and not tried:
        return prefix + _CLOSED_USER_FIRST.format(index=index, budget=budget)
    ranked = [
        f"{rank}. val_rmse={entry['val_rmse']:.4f}  "
        + " ".join(_format_op(op) for op in entry["operations"])
        for rank, entry in enumerate(sorted(archive, key=lambda e: e["val_rmse"]), start=1)
    ]
    seen = tried if tried is not None else [e["operations"] for e in archive]
    tried_lines = [
        f"- {' '.join(_format_op(op) for op in ops)}" for ops in seen
    ]
    return prefix + _CLOSED_USER.format(
        index=index,
        budget=budget,
        refine_rule=_REFINE_TRAINED if trained_angles else _REFINE_RANGE,
        duplicate_warning=_DUPLICATE_WARNING if last_was_duplicate else "",
        archive="\n".join(ranked) if ranked else "(nothing scored yet)",
        tried="\n".join(tried_lines),
    )


#: Strict JSON schema. OpenAI structured outputs require every declared
#: property to appear in `required`, so `theta_range` is a nullable union
#: rather than an optional key - the lesson from the bench_v2 `center`
#: field, which looked optional, was silently forced into every object,
#: and was then rejected by the validator.
def response_schema(n_gates: int = EXACT_GATES,
                    trained_angles: bool = False) -> dict:
    """The API itself enforces the gate count and the range menu, so a
    malformed length or an invented angle range cannot even be returned."""
    theta = ({"type": "null"} if trained_angles
             else {"type": ["string", "null"], "enum": [*ANGLE_RANGES, None]})
    return {
        "type": "object",
        "properties": {
            "operations": {
                "type": "array", "minItems": n_gates, "maxItems": n_gates,
                "items": {
                    "type": "object",
                    "properties": {
                        "gate": {"type": "string",
                                 "enum": [*SINGLE_QUBIT_GATES, *CONTROLLED_GATES]},
                        "wires": {"type": "array", "items": {"type": "integer"},
                                  "minItems": 1, "maxItems": 2},
                        "theta_range": theta,
                    },
                    "required": ["gate", "wires", "theta_range"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["operations"],
        "additionalProperties": False,
    }
