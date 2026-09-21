"""Track-B classical joint arms: random and the MANDATORY mixed
discrete/continuous evolutionary baseline (protocol §5, goal §8).

Candidates are complete: gates + explicit wires + verbatim theta. The
random arm reuses the preserved `sample_complete_candidate` (rejection-
sampled against the shared validator). The evolutionary arm evolves a
chromosome of (gate, wires, theta) tuples with structural mutations,
Gaussian theta perturbation (sigma frozen below, angles wrapped into
[-pi, pi]), single-point crossover, and (mu + lambda) selection by
validation RMSE — identical evaluation-budget accounting to every other
arm, because the runner routes all of them through the same
`evaluate_complete_candidate`.

Candidate identity includes theta (candidate_hash), so a theta
perturbation of an existing structure is a NEW candidate — exactly the
semantics the direct-joint track requires.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from llm_vqc.evaluation.seeds import derive_child_seed
from llm_vqc.free_amplitude.candidate_schema import validate_complete_candidate
from llm_vqc.free_amplitude.sampler import sample_complete_candidate
from llm_vqc.free_amplitude.schema import (
    ALL_GATES,
    MAX_OPERATIONS,
    PARAMETERIZED_GATES,
    SINGLE_WIRE_GATES,
)
from llm_vqc.search.arm import SearchArm
from llm_vqc.search.comparison import is_strictly_better
from llm_vqc.search.feedback import SearchFeedback

#: Frozen joint-evolution constants (protocol §5/§6).
JOINT_EVO_MU = 4
JOINT_EVO_LAMBDA = 8
JOINT_EVO_CROSSOVER_PROB = 0.5
JOINT_EVO_THETA_SIGMA = 0.3
JOINT_MUTATION_KINDS = ("theta_perturb", "gate_replace", "rewire", "add", "remove")
_MAX_MUTATION_ATTEMPTS = 200


def _wrap_angle(theta: float) -> float:
    return float((theta + np.pi) % (2 * np.pi) - np.pi)


def _sample_gate_op(rng: np.random.Generator, n_qubits: int) -> dict:
    gate = str(rng.choice(ALL_GATES))
    if gate in SINGLE_WIRE_GATES:
        wires = [int(rng.integers(0, n_qubits))]
    else:
        control, target = rng.choice(n_qubits, size=2, replace=False)
        wires = [int(control), int(target)]
    theta = float(rng.uniform(-np.pi, np.pi)) if gate in PARAMETERIZED_GATES else None
    return {"gate": gate, "wires": wires, "theta": theta}


def _mutate_complete(
    rng: np.random.Generator, operations: list[dict], n_qubits: int
) -> list[dict]:
    for _ in range(_MAX_MUTATION_ATTEMPTS):
        ops = [dict(op) for op in operations]
        applicable = [
            k for k in JOINT_MUTATION_KINDS
            if not (k == "add" and len(ops) >= MAX_OPERATIONS)
            and not (k == "remove" and len(ops) <= 1)
            and not (
                k == "theta_perturb"
                and not any(op.get("theta") is not None for op in ops)
            )
        ]
        kind = str(rng.choice(applicable))
        if kind == "theta_perturb":
            candidates = [i for i, op in enumerate(ops) if op.get("theta") is not None]
            i = int(rng.choice(candidates))
            ops[i]["theta"] = _wrap_angle(
                ops[i]["theta"] + float(rng.normal(0.0, JOINT_EVO_THETA_SIGMA))
            )
        elif kind == "gate_replace":
            i = int(rng.integers(0, len(ops)))
            ops[i] = _sample_gate_op(rng, n_qubits)
        elif kind == "rewire":
            i = int(rng.integers(0, len(ops)))
            gate = ops[i]["gate"]
            if gate in SINGLE_WIRE_GATES:
                ops[i]["wires"] = [int(rng.integers(0, n_qubits))]
            else:
                control, target = rng.choice(n_qubits, size=2, replace=False)
                ops[i]["wires"] = [int(control), int(target)]
        elif kind == "add":
            ops.insert(int(rng.integers(0, len(ops) + 1)), _sample_gate_op(rng, n_qubits))
        else:  # remove
            ops.pop(int(rng.integers(0, len(ops))))
        candidate = {"n_qubits": n_qubits, "operations": ops}
        if validate_complete_candidate(candidate, expected_n_qubits=n_qubits).valid:
            return ops
    return [op.model_dump() for op in sample_complete_candidate(rng, n_qubits).operations]


def _crossover_complete(
    rng: np.random.Generator, a: list[dict], b: list[dict], n_qubits: int
) -> list[dict]:
    for _ in range(_MAX_MUTATION_ATTEMPTS):
        cut_a = int(rng.integers(0, len(a) + 1))
        cut_b = int(rng.integers(0, len(b) + 1))
        child = [dict(op) for op in a[:cut_a]] + [dict(op) for op in b[cut_b:]]
        child = child[:MAX_OPERATIONS]
        if not child:
            continue
        candidate = {"n_qubits": n_qubits, "operations": child}
        if validate_complete_candidate(candidate, expected_n_qubits=n_qubits).valid:
            return child
    return _mutate_complete(rng, a, n_qubits)


class RandomJointState(BaseModel):
    seed: int
    n_proposed: int = 0
    best_hash: str | None = None
    best_metric: float | None = None
    best_candidate: dict | None = None


class RandomJointArm(SearchArm[RandomJointState]):
    name = "random_joint"

    def __init__(self, n_qubits: int) -> None:
        self.n_qubits = n_qubits

    def initialize(self, seed: int) -> RandomJointState:
        return RandomJointState(seed=seed)

    def propose(self, state: RandomJointState) -> dict:
        rng = np.random.default_rng(
            derive_child_seed(state.seed, "random_joint", str(state.n_proposed))
        )
        candidate = sample_complete_candidate(rng, self.n_qubits)
        return candidate.model_dump()

    def update_state(self, state, proposal, feedback: SearchFeedback):
        best_hash, best_metric = state.best_hash, state.best_metric
        best_candidate = state.best_candidate
        if feedback.val_metric_value is not None and is_strictly_better(
            feedback.val_metric_value, best_metric, lower_is_better=True
        ):
            best_hash, best_metric = feedback.structural_hash, feedback.val_metric_value
            best_candidate = proposal
        return state.model_copy(update={
            "n_proposed": state.n_proposed + 1, "best_hash": best_hash,
            "best_metric": best_metric, "best_candidate": best_candidate,
        })

    def select_final(self, state) -> str | None:
        return state.best_hash

    def deserialize_state(self, raw_json: str) -> RandomJointState:
        return RandomJointState.model_validate_json(raw_json)


class _JointIndividual(BaseModel):
    operations: list[dict]
    candidate_hash: str | None = None
    metric: float | None = None


class EvolutionaryJointState(BaseModel):
    seed: int
    n_proposed: int = 0
    population: list[_JointIndividual] = Field(default_factory=list)
    pending: _JointIndividual | None = None
    best_hash: str | None = None
    best_metric: float | None = None
    best_candidate: dict | None = None


class EvolutionaryJointArm(SearchArm[EvolutionaryJointState]):
    """(mu + lambda) over complete candidates — the mandatory classical
    mixed discrete/continuous baseline for the direct-joint claim."""

    name = "evolutionary_joint"

    def __init__(self, n_qubits: int) -> None:
        self.n_qubits = n_qubits

    def initialize(self, seed: int) -> EvolutionaryJointState:
        return EvolutionaryJointState(seed=seed)

    def propose(self, state: EvolutionaryJointState) -> dict:
        rng = np.random.default_rng(
            derive_child_seed(state.seed, "evolutionary_joint", str(state.n_proposed))
        )
        evaluated = [ind for ind in state.population if ind.metric is not None]
        if len(evaluated) < JOINT_EVO_MU:
            sampled = sample_complete_candidate(rng, self.n_qubits)
            ops = [op.model_dump() for op in sampled.operations]
        else:
            parents = sorted(evaluated, key=lambda ind: ind.metric)[:JOINT_EVO_MU]
            if rng.random() < JOINT_EVO_CROSSOVER_PROB and len(parents) >= 2:
                idx = rng.choice(len(parents), size=2, replace=False)
                ops = _crossover_complete(
                    rng, parents[int(idx[0])].operations, parents[int(idx[1])].operations,
                    self.n_qubits,
                )
            else:
                parent = parents[int(rng.integers(0, len(parents)))]
                ops = _mutate_complete(rng, parent.operations, self.n_qubits)
        state.pending = _JointIndividual(operations=ops)
        return {"n_qubits": self.n_qubits, "operations": ops}

    def update_state(self, state, proposal, feedback: SearchFeedback):
        population = list(state.population)
        pending = state.pending or _JointIndividual(operations=proposal.get("operations", []))
        if feedback.val_metric_value is not None:
            pending.candidate_hash = feedback.structural_hash
            pending.metric = feedback.val_metric_value
            population.append(pending)
            population = sorted(
                [p for p in population if p.metric is not None], key=lambda ind: ind.metric
            )[: JOINT_EVO_MU + JOINT_EVO_LAMBDA]
        best_hash, best_metric = state.best_hash, state.best_metric
        best_candidate = state.best_candidate
        if feedback.val_metric_value is not None and is_strictly_better(
            feedback.val_metric_value, best_metric, lower_is_better=True
        ):
            best_hash, best_metric = feedback.structural_hash, feedback.val_metric_value
            best_candidate = proposal
        return state.model_copy(update={
            "n_proposed": state.n_proposed + 1, "population": population, "pending": None,
            "best_hash": best_hash, "best_metric": best_metric,
            "best_candidate": best_candidate,
        })

    def select_final(self, state) -> str | None:
        return state.best_hash

    def deserialize_state(self, raw_json: str) -> EvolutionaryJointState:
        return EvolutionaryJointState.model_validate_json(raw_json)
