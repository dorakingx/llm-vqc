"""The five comparison arms of `mini_5gate_v1`.

Every arm implements the same contract: given a budget of B *unique*
candidates, propose circuits one at a time until B of them have been
evaluated. Duplicates and grammar violations are recorded but do not
consume budget, so no arm can buy extra evaluations by proposing badly.

Classical arms (random, evolutionary, greedy) cost no API calls at all.
Only the two LLM arms talk to a provider, and they issue exactly one
request per proposal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from llm_vqc.mini5.angles import ANGLE_RANGES, uniform_label
from llm_vqc.mini5.prompts import (
    closed_user_prompt,
    open_user_prompt,
    system_prompt,
)
from llm_vqc.mini5.space import (
    EXACT_GATES,
    grammar_issues,
    mutate_circuit,
    reference_circuit,
    sample_circuit,
)

#: Guard against a pathological proposer looping forever on duplicates or
#: malformed replies. Reaching this means the cell stops early and says so
#: rather than silently reporting a short search as a complete one.
MAX_PROPOSALS_PER_CELL = 40


@dataclass
class Proposal:
    operations: list[dict]
    source: str


@dataclass
class ArmTelemetry:
    proposals: int = 0
    invalid: int = 0
    duplicates: int = 0
    api_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    parse_failures: int = 0
    issues: list[str] = field(default_factory=list)


class BaseArm:
    name = "base"
    uses_api = False

    def __init__(self, rng: np.random.Generator, n_qubits: int, budget: int,
                 min_gates: int = EXACT_GATES, max_gates: int = EXACT_GATES) -> None:
        self.rng = rng
        self.n_qubits = n_qubits
        self.budget = budget
        self.min_gates = min_gates
        self.max_gates = max_gates
        self.telemetry = ArmTelemetry()

    def _sample(self) -> list[dict]:
        return sample_circuit(self.rng, self.n_qubits, self.min_gates, self.max_gates)

    def _mutate(self, ops: list[dict]) -> list[dict]:
        return mutate_circuit(self.rng, ops, self.n_qubits,
                              self.min_gates, self.max_gates)

    def propose(self, index: int) -> Proposal | None:
        raise NotImplementedError

    def observe(self, operations: list[dict], val_rmse: float) -> None:
        """Called only for candidates that were actually evaluated."""

    def note_rejected(self, operations: list[dict], reason: str) -> None:
        """Called when a proposal was discarded as a duplicate or as
        invalid, so an arm that can learn from it gets the chance."""


class RandomArm(BaseArm):
    """The null. In range mode it picks a bin uniformly, which is exactly
    a uniform draw over [-pi, pi] - so it is the special case the LLM arms
    can always fall back to."""

    name = "random"

    def propose(self, index: int) -> Proposal:
        ops = self._sample()
        for op in ops:
            if op.get("theta") is not None:
                op["theta_range"] = uniform_label(self.rng)
        return Proposal(ops, "random")


class ReferenceArm(BaseArm):
    """Fixed hardware-efficient ansatz. Runs no search: it proposes the
    same circuit once and stops, so it consumes one evaluation rather
    than the budget B. Its angles are drawn the same way every other arm's
    are, so the only thing that differs is the structure."""

    name = "reference"

    def propose(self, index: int) -> Proposal | None:
        if index > 1:
            return None
        ops = reference_circuit(self.n_qubits)
        for op in ops:
            op["theta_range"] = uniform_label(self.rng)
        return Proposal(ops, "fixed")


class EvolutionaryArm(BaseArm):
    """(mu + lambda) with mu=4: seed the population by sampling, then
    mutate a surviving parent. Mutation preserves the five-gate length."""

    name = "evolutionary"
    MU = 4

    def __init__(self, rng, n_qubits, budget, min_gates=EXACT_GATES,
                 max_gates=EXACT_GATES) -> None:
        super().__init__(rng, n_qubits, budget, min_gates, max_gates)
        self.population: list[tuple[float, list[dict]]] = []

    def propose(self, index: int) -> Proposal:
        if len(self.population) < self.MU:
            return Proposal(self._sample(), "seed")
        parents = sorted(self.population, key=lambda p: p[0])[: self.MU]
        _, parent = parents[int(self.rng.integers(len(parents)))]
        return Proposal(self._mutate(parent), "mutation")

    def observe(self, operations, val_rmse) -> None:
        self.population.append((val_rmse, operations))


class GreedyArm(BaseArm):
    """Hill climbing at fixed length: mutate the incumbent, keep the
    mutant only if validation improves, otherwise fall back and try
    again. This is the fixed-size analogue of greedy growth, which cannot
    apply here because the circuit length is pinned at five."""

    name = "greedy"

    def __init__(self, rng, n_qubits, budget, min_gates=EXACT_GATES,
                 max_gates=EXACT_GATES) -> None:
        super().__init__(rng, n_qubits, budget, min_gates, max_gates)
        self.incumbent: list[dict] | None = None
        self.incumbent_rmse = float("inf")
        self._last: list[dict] | None = None

    def propose(self, index: int) -> Proposal:
        if self.incumbent is None:
            self._last = self._sample()
            return Proposal(self._last, "start")
        self._last = self._mutate(self.incumbent)
        return Proposal(self._last, "climb")

    def observe(self, operations, val_rmse) -> None:
        if val_rmse < self.incumbent_rmse:
            self.incumbent, self.incumbent_rmse = operations, val_rmse


class LLMArm(BaseArm):
    """One request per proposal. `closed` decides whether validation-side
    feedback on the arm's own evaluated candidates is included."""

    uses_api = True

    def __init__(self, rng, n_qubits, budget, provider, closed: bool,
                 min_gates=EXACT_GATES, max_gates=EXACT_GATES,
                 trained_angles: bool = False, family: str | None = None) -> None:
        super().__init__(rng, n_qubits, budget, min_gates, max_gates)
        self.trained_angles = trained_angles
        #: set = this arm is told what the task is; None = blind, as before
        self.family = family
        self.provider = provider
        self.closed = closed
        self.name = "llm_closed" if closed else "llm_open"
        self.archive: list[dict] = []
        #: every circuit proposed, evaluated or not, so the prompt can
        #: name exactly what must not be repeated
        self.tried: list[list[dict]] = []
        self.last_was_duplicate = False
        self._system = system_prompt(n_qubits, min_gates, trained_angles)

    def propose(self, index: int) -> Proposal | None:
        user = (
            closed_user_prompt(index, self.budget, self.archive,
                               tried=self.tried,
                               last_was_duplicate=self.last_was_duplicate,
                               trained_angles=self.trained_angles,
                               family=self.family)
            if self.closed
            else open_user_prompt(index, self.budget, self.family)
        )
        response = self.provider.complete(self._system, user, 1.0)
        self.telemetry.api_calls += 1
        self.telemetry.input_tokens += response.input_tokens
        self.telemetry.output_tokens += response.output_tokens
        try:
            payload = json.loads(response.raw_text)
            operations = payload["operations"]
        except (json.JSONDecodeError, KeyError, TypeError):
            self.telemetry.parse_failures += 1
            return None
        if self.closed and isinstance(operations, list):
            self.tried.append(operations)
        return Proposal(operations, "llm")

    def observe(self, operations, val_rmse) -> None:
        self.last_was_duplicate = False
        if self.closed:
            self.archive.append({"operations": operations, "val_rmse": val_rmse})

    def note_rejected(self, operations, reason) -> None:
        self.last_was_duplicate = reason == "duplicate"


class HybridArm(LLMArm):
    """Random supplies the opening candidates; the LLM only improves.

    Measured motivation, not a hunch. Over 120 cells the LLM's first
    candidate was worse than random's (median validation 0.3551 vs
    0.3042) while its improvement from first to eighth was better (17% vs
    9%). Its prior about what a good circuit looks like is wrong for this
    readout, but its ability to refine is real. So take the opening from
    the sampler and spend the LLM on the part it is good at.

    Necessarily closed-loop: the LLM has to see what the random seeds
    scored, or it cannot improve on them.
    """

    def __init__(self, rng, n_qubits, budget, provider, seeds: int = 2,
                 min_gates=EXACT_GATES, max_gates=EXACT_GATES,
                 trained_angles: bool = False, family: str | None = None) -> None:
        super().__init__(rng, n_qubits, budget, provider, True,
                         min_gates, max_gates, trained_angles, family)
        self.name = "llm_hybrid_ctx"
        self.n_seeds = seeds

    def propose(self, index: int) -> Proposal | None:
        if index <= self.n_seeds:
            ops = self._sample()
            for op in ops:
                if op.get("theta") is not None:
                    op["theta_range"] = uniform_label(self.rng)
            self.tried.append(ops)
            return Proposal(ops, "random_seed")
        return super().propose(index)


def build_arm(name: str, rng, n_qubits: int, budget: int, provider=None,
              min_gates: int = EXACT_GATES, max_gates: int = EXACT_GATES,
              trained_angles: bool = False, family: str | None = None) -> BaseArm:
    g = (min_gates, max_gates)
    if name == "random":
        return RandomArm(rng, n_qubits, budget, *g)
    if name == "evolutionary":
        return EvolutionaryArm(rng, n_qubits, budget, *g)
    if name == "greedy":
        return GreedyArm(rng, n_qubits, budget, *g)
    if name == "reference":
        return ReferenceArm(rng, n_qubits, budget, *g)
    if name == "llm_open":
        return LLMArm(rng, n_qubits, budget, provider, False, *g, trained_angles)
    if name == "llm_open_ctx":
        arm = LLMArm(rng, n_qubits, budget, provider, False, *g, trained_angles, family)
        arm.name = "llm_open_ctx"
        return arm
    if name == "llm_closed":
        return LLMArm(rng, n_qubits, budget, provider, True, *g, trained_angles)
    if name == "llm_hybrid_ctx":
        return HybridArm(rng, n_qubits, budget, provider, 2, *g,
                         trained_angles, family)
    if name == "llm_closed_ctx":
        arm = LLMArm(rng, n_qubits, budget, provider, True, *g, trained_angles, family)
        arm.name = "llm_closed_ctx"
        return arm
    raise ValueError(f"unknown arm {name!r}")


ARM_NAMES = ("random", "evolutionary", "reference",
             "llm_open_ctx", "llm_closed_ctx", "llm_hybrid_ctx")
CLASSICAL_ARMS = ("random", "evolutionary", "greedy", "reference")
LLM_ARMS = ("llm_open", "llm_open_ctx", "llm_closed", "llm_closed_ctx",
            "llm_hybrid_ctx")

__all__ = [
    "ARM_NAMES", "CLASSICAL_ARMS", "LLM_ARMS", "EXACT_GATES",
    "MAX_PROPOSALS_PER_CELL", "ANGLE_RANGES", "ArmTelemetry", "BaseArm",
    "Proposal", "build_arm", "grammar_issues",
]
