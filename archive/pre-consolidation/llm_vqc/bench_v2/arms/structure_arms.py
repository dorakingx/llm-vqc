"""Track-A classical structure arms: random, evolutionary (mu+lambda),
greedy growth, and the fixed references (protocol §6).

All arms implement the preserved `SearchArm` interface and are driven by
`llm_vqc.bench_v2.runner.StructureSearchRunner` through the identical
evaluate-structure path; none can reach the evaluator, store, or test
data directly. Proposals are `{"operations": [...]}` in the shared
layered grammar; the sampling/mutation grammar is frozen in
`llm_vqc.bench_v2.arms.sampling`.

Selection semantics: every arm uses the shared `is_strictly_better`
comparison on the validation metric with lower-is-better (Track A records
val RMSE for regression and sqrt-Brier for T4 — both lower-better), and
`select_final` returns the best evaluated candidate's structural hash.

An arm may return `None` from `propose` to declare itself exhausted (the
fixed references do, after their single template); the bench_v2 runner
finalizes early and records the stop reason.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from llm_vqc.bench_v2.arms.sampling import (
    crossover_operations,
    mutate_operations,
    sample_layered_operations,
)
from llm_vqc.bench_v2.space import SpaceProfile, reference_operations
from llm_vqc.evaluation.seeds import derive_child_seed
from llm_vqc.search.arm import SearchArm
from llm_vqc.search.comparison import is_strictly_better
from llm_vqc.search.feedback import SearchFeedback

#: Frozen evolutionary constants (protocol §6): (mu + lambda) with
#: mu=4 parents, lambda=8 offspring per generation, crossover prob 0.5.
EVO_MU = 4
EVO_LAMBDA = 8
EVO_CROSSOVER_PROB = 0.5


def _best_update(
    state_best_hash: str | None, state_best_metric: float | None, feedback: SearchFeedback
) -> tuple[str | None, float | None]:
    if feedback.val_metric_value is not None and is_strictly_better(
        feedback.val_metric_value, state_best_metric, lower_is_better=True
    ):
        return feedback.structural_hash, feedback.val_metric_value
    return state_best_hash, state_best_metric


class RandomStructureState(BaseModel):
    seed: int
    n_proposed: int = 0
    best_hash: str | None = None
    best_metric: float | None = None


class RandomStructureArm(SearchArm[RandomStructureState]):
    """Uninformed control: every proposal an independent grammar draw."""

    name = "random_structure"

    def __init__(self, profile: SpaceProfile) -> None:
        self.profile = profile

    def initialize(self, seed: int) -> RandomStructureState:
        return RandomStructureState(seed=seed)

    def propose(self, state: RandomStructureState) -> dict:
        rng = np.random.default_rng(
            derive_child_seed(state.seed, "random_structure", str(state.n_proposed))
        )
        return {"operations": sample_layered_operations(rng, self.profile)}

    def update_state(self, state, proposal, feedback):
        best_hash, best_metric = _best_update(state.best_hash, state.best_metric, feedback)
        return state.model_copy(update={
            "n_proposed": state.n_proposed + 1,
            "best_hash": best_hash, "best_metric": best_metric,
        })

    def select_final(self, state) -> str | None:
        return state.best_hash

    def deserialize_state(self, raw_json: str) -> RandomStructureState:
        return RandomStructureState.model_validate_json(raw_json)


class _Individual(BaseModel):
    operations: list[dict]
    structural_hash: str | None = None
    metric: float | None = None


class EvolutionaryStructureState(BaseModel):
    seed: int
    n_proposed: int = 0
    population: list[_Individual] = Field(default_factory=list)
    pending: _Individual | None = None
    best_hash: str | None = None
    best_metric: float | None = None


class EvolutionaryStructureArm(SearchArm[EvolutionaryStructureState]):
    """(mu + lambda) evolution over op lists: keep the mu best evaluated
    individuals; offspring are crossover (p=0.5) of two uniform parents or
    a mutation of one, via the frozen operator set."""

    name = "evolutionary_structure"

    def __init__(self, profile: SpaceProfile) -> None:
        self.profile = profile

    def initialize(self, seed: int) -> EvolutionaryStructureState:
        return EvolutionaryStructureState(seed=seed)

    def _rng(self, state) -> np.random.Generator:
        return np.random.default_rng(
            derive_child_seed(state.seed, "evolutionary_structure", str(state.n_proposed))
        )

    def propose(self, state: EvolutionaryStructureState) -> dict:
        rng = self._rng(state)
        evaluated = [ind for ind in state.population if ind.metric is not None]
        if len(evaluated) < EVO_MU:  # seeding generation: random draws
            ops = sample_layered_operations(rng, self.profile)
        else:
            parents = sorted(evaluated, key=lambda ind: ind.metric)[:EVO_MU]
            if rng.random() < EVO_CROSSOVER_PROB and len(parents) >= 2:
                idx = rng.choice(len(parents), size=2, replace=False)
                ops = crossover_operations(
                    rng, parents[int(idx[0])].operations, parents[int(idx[1])].operations,
                    self.profile,
                )
            else:
                parent = parents[int(rng.integers(0, len(parents)))]
                ops = mutate_operations(rng, parent.operations, self.profile)
        # Stash the pending individual so update_state can attach feedback.
        state.pending = _Individual(operations=ops)
        return {"operations": ops}

    def update_state(self, state, proposal, feedback):
        population = list(state.population)
        pending = state.pending or _Individual(operations=proposal.get("operations", []))
        if feedback.val_metric_value is not None:
            pending.structural_hash = feedback.structural_hash
            pending.metric = feedback.val_metric_value
            population.append(pending)
            population = sorted(
                [p for p in population if p.metric is not None], key=lambda ind: ind.metric
            )[: EVO_MU + EVO_LAMBDA]
        best_hash, best_metric = _best_update(state.best_hash, state.best_metric, feedback)
        return state.model_copy(update={
            "n_proposed": state.n_proposed + 1, "population": population,
            "pending": None, "best_hash": best_hash, "best_metric": best_metric,
        })

    def select_final(self, state) -> str | None:
        return state.best_hash

    def deserialize_state(self, raw_json: str) -> EvolutionaryStructureState:
        return EvolutionaryStructureState.model_validate_json(raw_json)


class GreedyGrowthState(BaseModel):
    seed: int
    n_proposed: int = 0
    current_ops: list[dict] = Field(default_factory=list)
    current_metric: float | None = None
    pending_ops: list[dict] | None = None
    best_hash: str | None = None
    best_metric: float | None = None


class GreedyGrowthArm(SearchArm[GreedyGrowthState]):
    """Greedy operator growth: propose the incumbent op list plus one
    sampled op (or a fresh 1-op circuit when empty); adopt the extension
    as the new incumbent only if it strictly improves the validation
    metric. At the op cap, proposals switch to single-op mutations of the
    incumbent (same adopt-if-better rule)."""

    name = "greedy_growth"

    def __init__(self, profile: SpaceProfile) -> None:
        self.profile = profile

    def initialize(self, seed: int) -> GreedyGrowthState:
        return GreedyGrowthState(seed=seed)

    def propose(self, state: GreedyGrowthState) -> dict:
        rng = np.random.default_rng(
            derive_child_seed(state.seed, "greedy_growth", str(state.n_proposed))
        )
        from llm_vqc.bench_v2.arms.sampling import sample_layered_op
        from llm_vqc.bench_v2.space import validate_layered_structure

        if not state.current_ops:
            # Seed circuit: a single sampled op, validated as a 1-op list.
            ops = None
            for _ in range(50):
                candidate = [sample_layered_op(rng, self.profile.n_qubits)]
                ir, _issues = validate_layered_structure(candidate, self.profile)
                if ir is not None:
                    ops = candidate
                    break
            if ops is None:
                ops = sample_layered_operations(rng, self.profile)
        elif len(state.current_ops) < self.profile.max_ops:
            ops = [dict(op) for op in state.current_ops] + [
                sample_layered_op(rng, self.profile.n_qubits)
            ]
        else:
            ops = mutate_operations(rng, state.current_ops, self.profile)
        state.pending_ops = ops
        return {"operations": ops}

    def update_state(self, state, proposal, feedback):
        pending = state.pending_ops or proposal.get("operations", [])
        current_ops, current_metric = state.current_ops, state.current_metric
        if feedback.val_metric_value is not None and is_strictly_better(
            feedback.val_metric_value, current_metric, lower_is_better=True
        ):
            current_ops, current_metric = pending, feedback.val_metric_value
        best_hash, best_metric = _best_update(state.best_hash, state.best_metric, feedback)
        return state.model_copy(update={
            "n_proposed": state.n_proposed + 1, "current_ops": current_ops,
            "current_metric": current_metric, "pending_ops": None,
            "best_hash": best_hash, "best_metric": best_metric,
        })

    def select_final(self, state) -> str | None:
        return state.best_hash

    def deserialize_state(self, raw_json: str) -> GreedyGrowthState:
        return GreedyGrowthState.model_validate_json(raw_json)


class FixedReferenceState(BaseModel):
    seed: int
    proposed: bool = False
    best_hash: str | None = None
    best_metric: float | None = None


class FixedReferenceArm(SearchArm[FixedReferenceState]):
    """A predeclared template 'arm': proposes its reference circuit once,
    then declares exhaustion (propose -> None)."""

    def __init__(self, ref_name: str, profile: SpaceProfile) -> None:
        self.name = ref_name
        self.profile = profile

    def initialize(self, seed: int) -> FixedReferenceState:
        return FixedReferenceState(seed=seed)

    def propose(self, state: FixedReferenceState) -> dict | None:
        if state.proposed:
            return None
        return {"operations": reference_operations(self.name)}

    def update_state(self, state, proposal, feedback):
        best_hash, best_metric = _best_update(state.best_hash, state.best_metric, feedback)
        return state.model_copy(update={
            "proposed": True, "best_hash": best_hash, "best_metric": best_metric,
        })

    def select_final(self, state) -> str | None:
        return state.best_hash

    def deserialize_state(self, raw_json: str) -> FixedReferenceState:
        return FixedReferenceState.model_validate_json(raw_json)
