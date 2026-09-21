"""Main-mode 3-arm joint structure-and-theta search (no optimizer).

Random / Open-loop / Closed-loop, each budget-limited, each proposing
COMPLETE candidates (structure + angles) evaluated at exactly the proposed
theta. Per-(seed, arm) isolation is structural: each gets its own cache
namespace (`base::arm::seed`), its own `BudgetLedger` (run_id
`arm::seed`), and its own protected-test evaluation, and the CLI builds a
fresh data split per seed -- so no seed's data, cache, ledger, or test can
leak into another's.

Closed-loop feedback carries this arm's FULL compact history only (never
another arm's results, never a protected-Test metric).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from llm_vqc.evaluation.seeds import derive_child_seed
from llm_vqc.evaluation.store import ResultStore
from llm_vqc.free_amplitude.candidate_schema import CompleteCandidateProposal
from llm_vqc.free_amplitude.main_eval import (
    FixedThetaEvalResult,
    MainModeCache,
    evaluate_complete_candidate,
    evaluate_on_test_fixed_theta,
)
from llm_vqc.free_amplitude.prompts_main import (
    build_main_closed_loop_feedback_user_prompt,
    build_main_closed_loop_first_user_prompt,
    build_main_open_loop_user_prompt,
    build_main_system_prompt,
)
from llm_vqc.free_amplitude.sampler import sample_complete_candidate
from llm_vqc.ir.budget import BudgetLedger
from llm_vqc.tasks.base import DataSplit, TrainValData

SAFETY_MAX_PROPOSALS_PER_ARM = 25
TEMPERATURE = 0.2

ARM_RANDOM = "random"
ARM_OPEN_LOOP = "llm_open_loop"
ARM_CLOSED_LOOP = "llm_closed_loop"


def arm_seed_namespace(base_task_name: str, arm: str, seed: int) -> str:
    """Per-(arm, seed) cache namespace -- see module docstring."""
    return f"{base_task_name}::{arm}::seed{seed}"


def arm_seed_run_id(arm: str, seed: int) -> str:
    return f"{arm}::seed{seed}"


@dataclass
class MainCandidateRecord:
    seed: int
    arm: str
    proposal_index: int
    proposal_id: str
    operations: list[dict] | None
    theta: list[float] | None
    architecture_hash: str | None
    candidate_hash: str | None
    valid: bool
    is_duplicate: bool
    outcome: str
    train_mse: float | None
    val_rmse: float | None
    val_mae: float | None
    prediction_mean: float | None
    prediction_std: float | None
    prediction_min: float | None
    prediction_max: float | None
    constant_output: bool | None
    circuit_depth: int | None
    searched_body_depth: object | None
    searched_body_gate_count: int | None
    controlled_gate_count: int | None
    quantum_parameter_count: int | None
    parameter_count_in_causal_cone: int | None
    fraction_in_causal_cone: float | None
    best_so_far_val_rmse: float | None
    api_latency_seconds: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass
class MainArmRunOutcome:
    seed: int
    arm: str
    task_namespace: str
    n_qubits: int = 0
    candidates: list[MainCandidateRecord] = field(default_factory=list)
    ledger_summary: dict[str, int] = field(default_factory=dict)
    stop_reason: str | None = None
    selected_candidate_hash: str | None = None
    selected_architecture_hash: str | None = None
    selected_operations: list[dict] | None = None
    selected_theta: list[float] | None = None
    selected_val_rmse: float | None = None
    protected_test_rmse: float | None = None
    protected_test_mae: float | None = None
    api_successful_calls: int = 0
    api_failed_calls: int = 0
    api_outbound_attempts: int = 0


def _record_from_result(
    seed: int, arm: str, proposal_index: int, result: FixedThetaEvalResult,
    best_so_far: float | None, llm_response=None,
) -> MainCandidateRecord:
    ps = result.val_prediction_stats or {}
    return MainCandidateRecord(
        seed=seed, arm=arm, proposal_index=proposal_index, proposal_id=result.proposal_id,
        operations=result.operations, theta=result.theta,
        architecture_hash=result.architecture_hash, candidate_hash=result.candidate_hash,
        valid=result.valid, is_duplicate=result.is_duplicate, outcome=result.outcome,
        train_mse=result.train_mse, val_rmse=result.val_rmse, val_mae=result.val_mae,
        prediction_mean=ps.get("mean"), prediction_std=ps.get("std"),
        prediction_min=ps.get("min"), prediction_max=ps.get("max"),
        constant_output=ps.get("constant_output"),
        circuit_depth=result.circuit_depth, searched_body_depth=result.searched_body_depth,
        searched_body_gate_count=result.searched_body_gate_count,
        controlled_gate_count=result.controlled_gate_count,
        quantum_parameter_count=result.quantum_parameter_count,
        parameter_count_in_causal_cone=result.parameter_count_in_causal_cone,
        fraction_in_causal_cone=result.fraction_in_causal_cone,
        best_so_far_val_rmse=best_so_far,
        api_latency_seconds=(llm_response.latency_seconds if llm_response is not None else None),
        input_tokens=(llm_response.input_tokens if llm_response is not None else None),
        output_tokens=(llm_response.output_tokens if llm_response is not None else None),
    )


def _history_entry(result: FixedThetaEvalResult) -> dict:
    """One arm-local closed-loop feedback entry: this proposal's full
    compact record (never any protected-Test field)."""
    ps = result.val_prediction_stats or {}
    return {
        "operations": result.operations,
        "theta": result.theta,
        "valid": result.valid,
        "is_duplicate": result.is_duplicate,
        "outcome": result.outcome,
        "train_mse": _r(result.train_mse),
        "validation_rmse": _r(result.val_rmse),
        "validation_mae": _r(result.val_mae),
        "prediction_std": _r(ps.get("std")),
        "constant_output": ps.get("constant_output"),
        "searched_body_gate_count": result.searched_body_gate_count,
        "controlled_gate_count": result.controlled_gate_count,
        "searched_body_depth": result.searched_body_depth,
        "parameter_count_in_q0_causal_cone": result.parameter_count_in_causal_cone,
        "fraction_in_q0_causal_cone": _r(result.fraction_in_causal_cone),
    }


def _r(x, ndigits: int = 6):
    return round(x, ndigits) if isinstance(x, float) else x


def _update_selection(outcome: MainArmRunOutcome, result: FixedThetaEvalResult) -> None:
    if result.outcome != "evaluated" or result.val_rmse is None:
        return
    if outcome.selected_val_rmse is None or result.val_rmse < outcome.selected_val_rmse:
        outcome.selected_val_rmse = result.val_rmse
        outcome.selected_candidate_hash = result.candidate_hash
        outcome.selected_architecture_hash = result.architecture_hash
        outcome.selected_operations = result.operations
        outcome.selected_theta = result.theta


def run_random_arm_main(
    seed: int, base_task_name: str, n_qubits: int, readout_qubit: int, train_val: TrainValData,
    budget_limit: int, store: ResultStore, deadline: float, max_gates: int = 5,
) -> MainArmRunOutcome:
    arm = ARM_RANDOM
    namespace = arm_seed_namespace(base_task_name, arm, seed)
    run_id = arm_seed_run_id(arm, seed)
    ledger = BudgetLedger.from_store(store, run_id)
    cache = MainModeCache(store, namespace, seed)
    rng = np.random.default_rng(derive_child_seed(seed, "search", arm))
    outcome = MainArmRunOutcome(seed=seed, arm=arm, task_namespace=namespace, n_qubits=n_qubits)
    proposal_index = ledger.num_proposed

    while not ledger.is_exhausted(budget_limit):
        if proposal_index >= SAFETY_MAX_PROPOSALS_PER_ARM:
            outcome.stop_reason = "safety_cap_reached"
            break
        if time.monotonic() >= deadline:
            outcome.stop_reason = "wall_clock_deadline"
            break
        proposal = sample_complete_candidate(rng, n_qubits, max_gates)
        result = evaluate_complete_candidate(
            proposal, n_qubits, readout_qubit, train_val, f"{arm}:{seed}:{proposal_index}",
            ledger, seed, cache=cache, proposal_event_store=store, run_id=run_id,
        )
        _update_selection(outcome, result)
        outcome.candidates.append(
            _record_from_result(seed, arm, proposal_index, result, outcome.selected_val_rmse)
        )
        proposal_index += 1

    outcome.ledger_summary = ledger.summary()
    return outcome


def _run_llm_arm_main(
    arm: str, open_loop: bool, seed: int, base_task_name: str, n_qubits: int, readout_qubit: int,
    train_val: TrainValData, budget_limit: int, store: ResultStore, provider, deadline: float,
    feature_count: int, max_gates: int,
) -> MainArmRunOutcome:
    namespace = arm_seed_namespace(base_task_name, arm, seed)
    run_id = arm_seed_run_id(arm, seed)
    ledger = BudgetLedger.from_store(store, run_id)
    cache = MainModeCache(store, namespace, seed)
    system_prompt = build_main_system_prompt(n_qubits, feature_count, max_gates)
    outcome = MainArmRunOutcome(seed=seed, arm=arm, task_namespace=namespace, n_qubits=n_qubits)
    proposal_index = ledger.num_proposed
    history: list[dict] = []

    while not ledger.is_exhausted(budget_limit):
        if proposal_index >= SAFETY_MAX_PROPOSALS_PER_ARM:
            outcome.stop_reason = "safety_cap_reached"
            break
        if time.monotonic() >= deadline:
            outcome.stop_reason = "wall_clock_deadline"
            break

        if open_loop:
            user_prompt = build_main_open_loop_user_prompt()
        elif not history:
            user_prompt = build_main_closed_loop_first_user_prompt()
        else:
            user_prompt = build_main_closed_loop_feedback_user_prompt(
                history=history, best_operations=outcome.selected_operations,
                best_val_rmse=outcome.selected_val_rmse,
                remaining_budget=ledger.remaining_budget(budget_limit),
            )

        try:
            response = provider.complete(system_prompt, user_prompt, TEMPERATURE)
        except Exception as exc:
            outcome.api_outbound_attempts += 1
            outcome.api_failed_calls += 1
            outcome.stop_reason = f"api_error: {type(exc).__name__}: {exc}"
            break
        outcome.api_outbound_attempts += 1
        outcome.api_successful_calls += 1

        from llm_vqc.free_amplitude.candidate_schema import parse_complete_candidate_json

        proposal, err = parse_complete_candidate_json(response.raw_text)
        proposal_arg = proposal if proposal is not None else {"_malformed": True}
        result = evaluate_complete_candidate(
            proposal_arg, n_qubits, readout_qubit, train_val, f"{arm}:{seed}:{proposal_index}",
            ledger, seed, cache=cache, proposal_event_store=store, run_id=run_id,
        )
        _update_selection(outcome, result)
        if not open_loop:
            history.append(_history_entry(result))
        outcome.candidates.append(
            _record_from_result(
                seed, arm, proposal_index, result, outcome.selected_val_rmse, llm_response=response
            )
        )
        proposal_index += 1

    outcome.ledger_summary = ledger.summary()
    return outcome


def run_open_loop_arm_main(
    seed: int, base_task_name: str, n_qubits: int, readout_qubit: int, train_val: TrainValData,
    budget_limit: int, store: ResultStore, provider, deadline: float, feature_count: int,
    max_gates: int = 5,
) -> MainArmRunOutcome:
    return _run_llm_arm_main(
        ARM_OPEN_LOOP, True, seed, base_task_name, n_qubits, readout_qubit, train_val,
        budget_limit, store, provider, deadline, feature_count, max_gates,
    )


def run_closed_loop_arm_main(
    seed: int, base_task_name: str, n_qubits: int, readout_qubit: int, train_val: TrainValData,
    budget_limit: int, store: ResultStore, provider, deadline: float, feature_count: int,
    max_gates: int = 5,
) -> MainArmRunOutcome:
    return _run_llm_arm_main(
        ARM_CLOSED_LOOP, False, seed, base_task_name, n_qubits, readout_qubit, train_val,
        budget_limit, store, provider, deadline, feature_count, max_gates,
    )


def evaluate_protected_test_main(
    outcome: MainArmRunOutcome, readout_qubit: int, test_split: DataSplit,
) -> None:
    """Evaluate the arm's selected candidate once on the protected test
    partition, at its own proposed theta. Reconstructs the candidate from
    the stored operations (which include theta) and the outcome's recorded
    `n_qubits` -- no retraining, no optimizer."""
    if outcome.selected_operations is None or outcome.selected_val_rmse is None:
        return
    proposal = CompleteCandidateProposal.model_validate(
        {"n_qubits": outcome.n_qubits, "operations": outcome.selected_operations}
    )
    test = evaluate_on_test_fixed_theta(proposal, readout_qubit, test_split)
    outcome.protected_test_rmse = test["test_rmse"]
    outcome.protected_test_mae = test["test_mae"]
