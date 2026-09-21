"""Search-space accounting and the Random-search sampler (Codex
instruction section 16).

For `n` qubits, the number of valid gate-placement *actions* at one
sequence position is:

```text
single-qubit actions:  4 * n           (H, RX, RY, RZ, each on any wire)
controlled actions:    3 * n * (n-1)   (CRX, CRY, CRZ, each ordered pair)
total:                 A(n) = 4n + 3n(n-1) = 3n^2 + n
```

`A(3) = 30`, `A(5) = 80`, matching the instruction's worked examples. This
module does NOT exhaustively enumerate or train the full space (`sum_{g=1}
^{G} A(n)^g` candidates for a max body length `G` is astronomically larger
than any usable evaluation budget) -- `search_space_size_report` computes
the count for reporting only; `sample_free_gate_proposal` draws one
uniformly-random *action* per sequence position via rejection sampling
against the same `validate_free_gate_proposal` every other proposal goes
through (so "valid by construction" and "valid because it passed
validation" are the same check, not two separate implementations that
could silently diverge).
"""

from __future__ import annotations

import numpy as np

from llm_vqc.free_amplitude.schema import (
    MAX_OPERATIONS,
    SINGLE_WIRE_GATES,
    TWO_WIRE_GATES,
    FreeGateOperation,
    FreeGateProposal,
    validate_free_gate_proposal,
)

DEFAULT_MAX_SAMPLE_ATTEMPTS = 500


class FreeGateSamplingError(Exception):
    """Raised when a valid free-gate proposal could not be sampled within
    the attempt budget -- never silently returns an invalid proposal."""


def num_placement_actions(n_qubits: int) -> int:
    """`A(n) = 4n + 3n(n-1) = 3n^2 + n` (Codex instruction section 16)."""
    return 4 * n_qubits + 3 * n_qubits * (n_qubits - 1)


def search_space_size_report(n_qubits: int, max_gates: int = MAX_OPERATIONS) -> dict:
    """`sum_{g=1}^{max_gates} A(n)^g`, reported for context only -- the
    search NEVER exhaustively enumerates or trains this space."""
    a_n = num_placement_actions(n_qubits)
    per_length = {g: a_n**g for g in range(1, max_gates + 1)}
    return {
        "n_qubits": n_qubits,
        "placement_actions_per_position": a_n,
        "max_gates": max_gates,
        "candidates_per_sequence_length": per_length,
        "total_candidates": sum(per_length.values()),
    }


def _enumerate_actions(n_qubits: int) -> list[tuple[str, tuple[int, ...]]]:
    actions: list[tuple[str, tuple[int, ...]]] = []
    for gate in SINGLE_WIRE_GATES:
        for wire in range(n_qubits):
            actions.append((gate, (wire,)))
    for gate in TWO_WIRE_GATES:
        for control in range(n_qubits):
            for target in range(n_qubits):
                if control != target:
                    actions.append((gate, (control, target)))
    return actions


def sample_free_gate_proposal(
    rng: np.random.Generator,
    n_qubits: int,
    max_gates: int = MAX_OPERATIONS,
    require_at_least_one_parameterized: bool = True,
    max_attempts: int = DEFAULT_MAX_SAMPLE_ATTEMPTS,
) -> FreeGateProposal:
    """Draw one valid free-gate proposal uniformly at random: a sequence
    length in `[1, max_gates]`, then each position drawn uniformly from the
    `A(n)` placement actions, via rejection sampling against
    `validate_free_gate_proposal` (rejects adjacent-duplicate operations,
    enforces at-least-one-parameterized, etc. -- the identical check every
    other proposal source goes through).
    """
    actions = _enumerate_actions(n_qubits)
    for _ in range(max_attempts):
        length = int(rng.integers(1, max_gates + 1))
        indices = rng.integers(0, len(actions), size=length)
        operations = [
            FreeGateOperation(gate=actions[i][0], wires=list(actions[i][1])) for i in indices
        ]
        candidate = FreeGateProposal(n_qubits=n_qubits, operations=operations)
        result = validate_free_gate_proposal(
            candidate, expected_n_qubits=n_qubits,
            require_at_least_one_parameterized=require_at_least_one_parameterized,
        )
        if result.valid:
            return candidate
    raise FreeGateSamplingError(
        f"failed to sample a valid free-gate proposal for n_qubits={n_qubits} "
        f"after {max_attempts} attempts"
    )


def sample_complete_candidate(
    rng: np.random.Generator,
    n_qubits: int,
    max_gates: int = MAX_OPERATIONS,
    max_attempts: int = DEFAULT_MAX_SAMPLE_ATTEMPTS,
):
    """Draw one valid **complete** candidate for the main mode: a random
    valid structure PLUS a random angle `theta ~ Uniform[-pi, pi]` for each
    parameterized gate (`H` gets none). Rejection-sampled against
    `validate_complete_candidate` -- the identical check every complete
    candidate goes through.
    """
    # Imported here (not at module top) to avoid a candidate_schema ->
    # sampler import cycle at load time.
    from llm_vqc.free_amplitude.candidate_schema import (
        CompleteCandidateProposal,
        CompleteGateOperation,
        validate_complete_candidate,
    )
    from llm_vqc.free_amplitude.schema import PARAMETERIZED_GATES

    actions = _enumerate_actions(n_qubits)
    for _ in range(max_attempts):
        length = int(rng.integers(1, max_gates + 1))
        indices = rng.integers(0, len(actions), size=length)
        operations = []
        for i in indices:
            gate, wires = actions[i]
            theta = (
                float(rng.uniform(-np.pi, np.pi)) if gate in PARAMETERIZED_GATES else None
            )
            operations.append(CompleteGateOperation(gate=gate, wires=list(wires), theta=theta))
        candidate = CompleteCandidateProposal(n_qubits=n_qubits, operations=operations)
        result = validate_complete_candidate(candidate, expected_n_qubits=n_qubits)
        if result.valid:
            return candidate
    raise FreeGateSamplingError(
        f"failed to sample a valid complete candidate for n_qubits={n_qubits} "
        f"after {max_attempts} attempts"
    )
