"""The 3-arm search loop: Random, LLM Open-loop, LLM Closed-loop -- each
budget-limited to B proposals (Codex instruction section 17-18).

**Arm-local duplicate detection.** Each arm evaluates candidates under its
own cache namespace (`task_name::arm`), so duplicate status and
closed-loop feedback are local to that arm: a circuit is a duplicate only
relative to what that SAME arm proposed earlier, and arm execution order
cannot change any arm's results or prompts. (This mirrors a defect found
and fixed in `llm_vqc.mini_demo`'s first implementation -- a single shared
cache let one arm's proposal silently "duplicate" another arm's, which
also silently changed what feedback the closed-loop LLM saw. Namespacing
by arm from the start avoids reintroducing that bug here.)

No real or billed LLM API call is made by anything in this module as
delivered -- `run_llm_open_loop_arm`/`run_llm_closed_loop_arm` accept any
object satisfying the `LLMProvider` protocol structurally, and this
implementation pass only ever passes `llm_vqc.free_amplitude.provider`'s
mock/scripted providers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from llm_vqc.evaluation.results import EvaluationResult, TrainingOutcome, ValidationOutcome
from llm_vqc.evaluation.seeds import derive_child_seed
from llm_vqc.evaluation.store import ResultStore
from llm_vqc.free_amplitude.final_test import evaluate_on_test
from llm_vqc.free_amplitude.harness import evaluate_free_gate_candidate
from llm_vqc.free_amplitude.init_policy import free_amplitude_train_seed
from llm_vqc.free_amplitude.sampler import sample_free_gate_proposal
from llm_vqc.free_amplitude.schema import (
    parse_free_gate_json,
)
from llm_vqc.free_amplitude.training import (
    FREE_AMPLITUDE_TRAINING_CONFIG_VERSION,
    FreeAmplitudeTrainingConfig,
)
from llm_vqc.ir.budget import BudgetLedger, ProposalOutcome, ProposalRecord
from llm_vqc.ir.schema import CircuitIR
from llm_vqc.tasks.base import DataSplit, TrainValData

SAFETY_MAX_PROPOSALS_PER_ARM = 25
TEMPERATURE = 0.2

ARM_RANDOM = "random"
ARM_OPEN_LOOP = "llm_open_loop"
ARM_CLOSED_LOOP = "llm_closed_loop"


def arm_task_name(base_task_name: str, arm: str) -> str:
    """Arm-scoped cache namespace -- see module docstring."""
    return f"{base_task_name}::{arm}"


@dataclass
class CandidateRecord:
    arm: str
    proposal_index: int
    proposal_id: str
    operations: list[dict] | None
    valid: bool
    is_duplicate: bool
    structural_hash: str | None
    training_outcome: str
    final_val_rmse: float | None
    circuit_depth: int | None
    quantum_parameter_count: int | None
    controlled_gate_count: int | None
    initial_gradient_norm: float | None = None
    final_gradient_norm: float | None = None
    zero_gradient_parameter_count: int | None = None
    parameter_count_in_causal_cone: int | None = None
    api_latency_seconds: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass
class ArmRunOutcome:
    arm: str
    candidates: list[CandidateRecord] = field(default_factory=list)
    ledger_summary: dict[str, int] = field(default_factory=dict)
    stop_reason: str | None = None
    selected_structural_hash: str | None = None
    selected_train_seed: int | None = None
    selected_val_rmse: float | None = None
    selected_operations: list[dict] | None = None
    protected_test_rmse: float | None = None
    protected_test_mae: float | None = None
    task_namespace: str | None = None
    api_successful_calls: int = 0
    api_failed_calls: int = 0
    api_outbound_attempts: int = 0


def _weights_extra(
    store: ResultStore, task_name: str, structural_hash: str | None, train_seed: int | None
) -> dict:
    if structural_hash is None or train_seed is None:
        return {}
    weights = store.get_trained_weights(task_name, structural_hash, train_seed)
    return weights or {}


def _record_from_result(
    arm: str, proposal_index: int, proposal_id: str, operations: list[dict] | None,
    result: EvaluationResult, store: ResultStore, namespaced_task: str,
    llm_response=None,
) -> CandidateRecord:
    cost = result.circuit_cost
    extra = _weights_extra(store, namespaced_task, result.structural_hash, result.train_seed)
    return CandidateRecord(
        arm=arm, proposal_index=proposal_index, proposal_id=proposal_id, operations=operations,
        valid=result.validation_outcome == ValidationOutcome.VALID,
        is_duplicate=result.is_duplicate, structural_hash=result.structural_hash,
        training_outcome=result.training_outcome.value, final_val_rmse=result.val_metric_value,
        circuit_depth=cost.depth if cost is not None else None,
        quantum_parameter_count=cost.parameter_count if cost is not None else None,
        controlled_gate_count=cost.two_qubit_gate_count if cost is not None else None,
        initial_gradient_norm=extra.get("initial_gradient_norm"),
        final_gradient_norm=extra.get("final_gradient_norm"),
        zero_gradient_parameter_count=extra.get("zero_gradient_parameter_count"),
        parameter_count_in_causal_cone=extra.get("parameter_count_in_causal_cone"),
        api_latency_seconds=(llm_response.latency_seconds if llm_response is not None else None),
        input_tokens=(llm_response.input_tokens if llm_response is not None else None),
        output_tokens=(llm_response.output_tokens if llm_response is not None else None),
    )


def _select_best(outcome: ArmRunOutcome) -> None:
    trained = [
        c for c in outcome.candidates
        if c.training_outcome == "success" and c.final_val_rmse is not None
    ]
    if not trained:
        return
    best = min(trained, key=lambda c: c.final_val_rmse)
    outcome.selected_structural_hash = best.structural_hash
    outcome.selected_val_rmse = best.final_val_rmse
    outcome.selected_operations = best.operations


def run_random_arm(
    run_seed: int, base_task_name: str, expected_n_qubits: int, readout_qubit: int,
    train_val: TrainValData, training_config: FreeAmplitudeTrainingConfig, budget_limit: int,
    store: ResultStore, deadline: float, max_gates: int = 5,
) -> ArmRunOutcome:
    arm = ARM_RANDOM
    namespaced_task = arm_task_name(base_task_name, arm)
    ledger = BudgetLedger.from_store(store, arm)
    rng = np.random.default_rng(derive_child_seed(run_seed, "search", arm))
    outcome = ArmRunOutcome(arm=arm, task_namespace=namespaced_task)
    proposal_index = ledger.num_proposed

    while not ledger.is_exhausted(budget_limit):
        if proposal_index >= SAFETY_MAX_PROPOSALS_PER_ARM:
            outcome.stop_reason = "safety_cap_reached"
            break
        if time.monotonic() >= deadline:
            outcome.stop_reason = "wall_clock_deadline"
            break
        proposal_id = f"{arm}:{proposal_index}"
        proposal = sample_free_gate_proposal(rng, expected_n_qubits, max_gates)
        result = evaluate_free_gate_candidate(
            proposal, namespaced_task, run_seed, expected_n_qubits, readout_qubit,
            train_val, training_config, proposal_id, ledger, cache=store,
            proposal_event_store=store, run_id=arm,
        )
        outcome.candidates.append(
            _record_from_result(
                arm, proposal_index, proposal_id,
                [op.model_dump() for op in proposal.operations], result, store, namespaced_task,
            )
        )
        proposal_index += 1

    outcome.ledger_summary = ledger.summary()
    _select_best(outcome)
    return outcome


def _llm_arm_loop(
    arm: str, run_seed: int, base_task_name: str, expected_n_qubits: int, readout_qubit: int,
    train_val: TrainValData, training_config: FreeAmplitudeTrainingConfig, budget_limit: int,
    store: ResultStore, provider, deadline: float, max_gates: int, open_loop: bool,
) -> ArmRunOutcome:
    from llm_vqc.free_amplitude.prompts import (
        build_closed_loop_feedback_user_prompt,
        build_closed_loop_first_user_prompt,
        build_open_loop_user_prompt,
        build_system_prompt,
    )

    namespaced_task = arm_task_name(base_task_name, arm)
    ledger = BudgetLedger.from_store(store, arm)
    system_prompt = build_system_prompt(
        n_qubits=expected_n_qubits, feature_count=2**expected_n_qubits, max_gates=max_gates,
    )
    outcome = ArmRunOutcome(arm=arm, task_namespace=namespaced_task)
    proposal_index = ledger.num_proposed
    prior_result: EvaluationResult | None = None
    prior_operations: list[dict] | None = None

    while not ledger.is_exhausted(budget_limit):
        if proposal_index >= SAFETY_MAX_PROPOSALS_PER_ARM:
            outcome.stop_reason = "safety_cap_reached"
            break
        if time.monotonic() >= deadline:
            outcome.stop_reason = "wall_clock_deadline"
            break

        if open_loop:
            user_prompt = build_open_loop_user_prompt()
        elif prior_result is None:
            user_prompt = build_closed_loop_first_user_prompt()
        else:
            cost = prior_result.circuit_cost
            extra = _weights_extra(
                store, namespaced_task, prior_result.structural_hash, prior_result.train_seed
            )
            user_prompt = build_closed_loop_feedback_user_prompt(
                prior_operations=prior_operations or [],
                valid=prior_result.validation_outcome == ValidationOutcome.VALID,
                is_duplicate=prior_result.is_duplicate,
                val_rmse=prior_result.val_metric_value,
                searched_body_gate_count=extra.get("searched_body_gate_count"),
                quantum_parameter_count=cost.parameter_count if cost else None,
                controlled_gate_count=cost.two_qubit_gate_count if cost else None,
                compiled_depth=cost.depth if cost else None,
                initial_gradient_norm=extra.get("initial_gradient_norm"),
                final_gradient_norm=extra.get("final_gradient_norm"),
                zero_gradient_parameter_count=extra.get("zero_gradient_parameter_count"),
                parameter_count_in_causal_cone=extra.get("parameter_count_in_causal_cone"),
                best_so_far_val_rmse=outcome.selected_val_rmse,
                best_so_far_operations=outcome.selected_operations,
                remaining_budget=ledger.remaining_budget(budget_limit),
            )

        proposal_id = f"{arm}:{proposal_index}"
        try:
            response = provider.complete(system_prompt, user_prompt, TEMPERATURE)
        except Exception as exc:
            outcome.api_outbound_attempts += 1
            outcome.api_failed_calls += 1
            outcome.stop_reason = f"api_error: {type(exc).__name__}: {exc}"
            break
        outcome.api_outbound_attempts += 1
        outcome.api_successful_calls += 1

        proposal, err = parse_free_gate_json(response.raw_text)
        if proposal is None:
            from llm_vqc.evaluation.results import (
                CompilationOutcome,
                FailureCategory,
            )
            from llm_vqc.ir.validators import ValidationIssue

            result = EvaluationResult(
                proposal_id=proposal_id, task_name=namespaced_task, run_seed=run_seed,
                structural_hash=None, circuit_canonical_json=None,
                validation_outcome=ValidationOutcome.INVALID,
                validation_issues=[
                    ValidationIssue(
                        code="free_gate.malformed_llm_output", message=err or "?", path=""
                    )
                ],
                compilation_outcome=CompilationOutcome.NOT_ATTEMPTED,
                training_outcome=TrainingOutcome.NOT_ATTEMPTED,
                failure_category=FailureCategory.INVALID_PROPOSAL,
            )
            store.append_proposal_event(
                arm, ledger.num_proposed, ProposalRecord(outcome=ProposalOutcome.INVALID)
            )
            ledger.record_proposal(ProposalRecord(outcome=ProposalOutcome.INVALID))
            operations_dump = None
        else:
            result = evaluate_free_gate_candidate(
                proposal, namespaced_task, run_seed, expected_n_qubits, readout_qubit,
                train_val, training_config, proposal_id, ledger, cache=store,
                proposal_event_store=store, run_id=arm,
            )
            operations_dump = [op.model_dump() for op in proposal.operations]

        outcome.candidates.append(
            _record_from_result(
                arm, proposal_index, proposal_id, operations_dump, result, store, namespaced_task,
                llm_response=response,
            )
        )
        _select_best(outcome)
        prior_result = result
        prior_operations = operations_dump
        proposal_index += 1

    outcome.ledger_summary = ledger.summary()
    _select_best(outcome)
    return outcome


def run_llm_open_loop_arm(
    run_seed: int, base_task_name: str, expected_n_qubits: int, readout_qubit: int,
    train_val: TrainValData, training_config: FreeAmplitudeTrainingConfig, budget_limit: int,
    store: ResultStore, provider, deadline: float, max_gates: int = 5,
) -> ArmRunOutcome:
    return _llm_arm_loop(
        ARM_OPEN_LOOP, run_seed, base_task_name, expected_n_qubits, readout_qubit, train_val,
        training_config, budget_limit, store, provider, deadline, max_gates, open_loop=True,
    )


def run_llm_closed_loop_arm(
    run_seed: int, base_task_name: str, expected_n_qubits: int, readout_qubit: int,
    train_val: TrainValData, training_config: FreeAmplitudeTrainingConfig, budget_limit: int,
    store: ResultStore, provider, deadline: float, max_gates: int = 5,
) -> ArmRunOutcome:
    return _llm_arm_loop(
        ARM_CLOSED_LOOP, run_seed, base_task_name, expected_n_qubits, readout_qubit, train_val,
        training_config, budget_limit, store, provider, deadline, max_gates, open_loop=False,
    )


def evaluate_protected_test(
    outcome: ArmRunOutcome, store: ResultStore, base_task_name: str, readout_qubit: int,
    test_split: DataSplit, run_seed: int,
    config_version: str = FREE_AMPLITUDE_TRAINING_CONFIG_VERSION,
) -> None:
    """Score the arm's selected circuit once on the protected test
    partition, reading from the arm's OWN cache namespace. `train_seed` is
    recomputed deterministically (pure function of `run_seed`,
    `selected_structural_hash`, `config_version`) rather than looked up,
    since that is exactly how `llm_vqc.free_amplitude.harness` derived it
    in the first place -- no store scan needed.
    """
    if outcome.selected_structural_hash is None:
        return
    namespaced_task = outcome.task_namespace or arm_task_name(base_task_name, outcome.arm)
    train_seed = free_amplitude_train_seed(
        run_seed, outcome.selected_structural_hash, config_version
    )
    circuit_hash = outcome.selected_structural_hash
    cached = store.get_cached(namespaced_task, circuit_hash, train_seed)
    weights = store.get_trained_weights(namespaced_task, circuit_hash, train_seed)
    if cached is None or weights is None or not cached.circuit_canonical_json:
        return
    ir = CircuitIR.model_validate_json(cached.circuit_canonical_json)
    result = evaluate_on_test(
        ir, readout_qubit, weights["learned_angles"], test_split, namespaced_task, train_seed
    )
    outcome.protected_test_rmse = result.test_rmse
    outcome.protected_test_mae = result.test_mae
    outcome.selected_train_seed = train_seed
