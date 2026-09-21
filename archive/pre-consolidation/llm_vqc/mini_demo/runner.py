"""The 3-arm search loop for the mini LLM-API VQC demo: Random, LLM
Open-loop, and LLM Closed-loop, each budget-limited to B proposals.

Does NOT use `llm_vqc.search.SearchRunner` or the general search arms
(`RandomArm`/`LLMIterArm`) -- those propose over the full `CircuitIR`
grammar, not this demo's compact 4-field grammar -- so a small dedicated
loop is used instead. Every proposal still goes through the shared,
already-validated `llm_vqc.evaluation.harness.evaluate_candidate` and
`llm_vqc.ir.budget.BudgetLedger`.

**Correction-pass invariants baked in here:**
  - *Arm-local duplicate detection* (section 3): each arm caches under its
    own namespaced task name (`base::arm`), so a circuit is a duplicate
    only relative to circuits that SAME arm proposed earlier -- execution
    order across arms cannot change any arm's duplicate status or prompts,
    and Closed-loop never sees information derived from another arm.
  - *Explicit initialization + float64/CPU/backprop* (section 5): every
    candidate trains through `mini_demo_init_policy` with
    `diff_method="backprop"`.
  - *Dollar-budget + retry policy* (section 8): `preflight_checks` requires
    a positive `LLM_API_BUDGET_USD` and an explicit `OPENAI_MODEL` before
    any provider is constructed; each real call is checked against the
    budget before it is issued.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from llm_vqc.evaluation.final_test import evaluate_on_test
from llm_vqc.evaluation.harness import evaluate_candidate
from llm_vqc.evaluation.results import (
    CompilationOutcome,
    EvaluationResult,
    FailureCategory,
    TrainingOutcome,
    ValidationOutcome,
)
from llm_vqc.evaluation.seeds import derive_child_seed, train_seed_for_circuit
from llm_vqc.evaluation.store import ResultStore
from llm_vqc.evaluation.training import TrainingConfig
from llm_vqc.ir.budget import BudgetLedger, ProposalOutcome, ProposalRecord
from llm_vqc.ir.schema import CircuitIR
from llm_vqc.ir.validators import ValidationIssue
from llm_vqc.llm.budget import LLMApiBudget, LLMBudgetExceededError
from llm_vqc.llm.openai_provider import CallBudgetExceededError
from llm_vqc.llm.provider import LLMProvider
from llm_vqc.llm.records import LLMCallRecord
from llm_vqc.mini_demo.capacity import verify_fixed_capacity
from llm_vqc.mini_demo.compact_schema import (
    CompactArchitecture,
    canonicalize_compact,
    compact_to_circuit_ir,
    parse_compact_json,
)
from llm_vqc.mini_demo.init_policy import mini_demo_init_policy
from llm_vqc.mini_demo.prompts import (
    build_closed_loop_feedback_user_prompt,
    build_closed_loop_first_user_prompt,
    build_open_loop_user_prompt,
    build_system_prompt,
)
from llm_vqc.mini_demo.records import ArmRunOutcome, CandidateRecord
from llm_vqc.mini_demo.sampler import sample_compact_architecture

TEMPERATURE = 0.2
#: Explicit differentiation method (correction pass section 5): pinned to
#: backprop rather than PennyLane's "best" auto-selection.
DIFF_METHOD = "backprop"
#: Conservative per-call charge against the dollar cap. OpenAI's chat
#: completions API never returns a real dollar cost, so this estimate is
#: what is charged; far above the true cost of a ~500-token call, so the
#: cap can only ever bind early, never silently overrun.
COST_ESTIMATE_PER_CALL_USD = 0.05
#: Angles are "changed" if any moved by more than this after training.
ANGLE_CHANGE_TOLERANCE = 1e-6
#: Guards against a pathological all-INVALID loop consuming no budget.
SAFETY_MAX_PROPOSALS_PER_ARM = 25

ARM_RANDOM = "random"
ARM_OPEN_LOOP = "llm_open_loop"
ARM_CLOSED_LOOP = "llm_closed_loop"


class PreflightError(Exception):
    """Raised when a required environment precondition for a real run is
    missing or invalid -- blocks execution before any provider request."""


@dataclass
class PreflightConfig:
    api_key: str
    model: str
    budget: LLMApiBudget


def preflight_checks(env: Mapping[str, str]) -> PreflightConfig:
    """Validate the environment for a real run and return the resolved
    config, or raise `PreflightError`. Requires `OPENAI_API_KEY`, an
    explicit `OPENAI_MODEL` (no silent default), and a positive
    `LLM_API_BUDGET_USD` (via the sanctioned `LLMApiBudget.from_env`).
    Constructs no provider and issues no request."""
    api_key = (env.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise PreflightError("OPENAI_API_KEY is not set")
    model = (env.get("OPENAI_MODEL") or "").strip()
    if not model:
        raise PreflightError(
            "OPENAI_MODEL is not set (this demo refuses to silently default to a model id)"
        )
    budget = LLMApiBudget.from_env(dict(env))
    if budget is None:
        raise PreflightError(
            "a positive LLM_API_BUDGET_USD must be set in the environment "
            "(external paid API calls are blocked without an explicit hard cap)"
        )
    return PreflightConfig(api_key=api_key, model=model, budget=budget)


def arm_task_name(base_task_name: str, arm: str) -> str:
    """Arm-scoped cache namespace: keeps each arm's duplicate history and
    trained-weight cache independent of every other arm's."""
    return f"{base_task_name}::{arm}"


def _quantum_angles_changed(initial: list | None, learned: list | None) -> bool | None:
    if not initial or not learned or len(initial) != len(learned):
        return None
    return any(
        abs(a - b) > ANGLE_CHANGE_TOLERANCE for a, b in zip(initial, learned, strict=False)
    )


def _dtype_device_ok(provenance: list | None) -> bool | None:
    if not provenance:
        return None
    trainable = [p for p in provenance if p.get("requires_grad")]
    if not trainable:
        return None
    return all(p.get("dtype") == "torch.float64" and p.get("device") == "cpu" for p in trainable)


def _build_candidate_record(
    arm: str, proposal_index: int, proposal_id: str, compact: CompactArchitecture | None,
    result: EvaluationResult, store: ResultStore, namespaced_task: str,
    run_seed: int, llm_response=None,
) -> CandidateRecord:
    cost = result.circuit_cost
    trained_ok = result.training_outcome == TrainingOutcome.SUCCESS
    total_params = 51 if trained_ok else None

    changed = None
    dtype_ok = None
    unique_training_id = None
    if result.structural_hash is not None:
        train_seed = train_seed_for_circuit(run_seed, result.structural_hash)
        unique_training_id = f"{arm}:{result.structural_hash[:12]}:{train_seed}"
        weights = store.get_trained_weights(namespaced_task, result.structural_hash, train_seed)
        if weights is not None:
            changed = _quantum_angles_changed(
                weights.get("initial_circuit_weights"), weights.get("circuit_weights")
            )
            dtype_ok = _dtype_device_ok(weights.get("param_provenance"))

    # "Actually trained now" vs "reused from cache" (a within-arm duplicate).
    actually_trained = trained_ok and not result.is_duplicate and not result.cache_hit

    return CandidateRecord(
        arm=arm,
        proposal_index=proposal_index,
        proposal_id=proposal_id,
        compact=compact.model_dump() if compact is not None else None,
        valid=result.validation_outcome == ValidationOutcome.VALID,
        is_duplicate=result.is_duplicate,
        structural_hash=result.structural_hash,
        training_outcome=result.training_outcome.value,
        final_train_mse=(result.train_loss_history[-1] if result.train_loss_history else None),
        final_val_rmse=result.val_metric_value,
        circuit_depth=cost.depth if cost is not None else None,
        two_qubit_gate_count=cost.two_qubit_gate_count if cost is not None else None,
        quantum_parameter_count=cost.parameter_count if cost is not None else None,
        api_latency_seconds=(llm_response.latency_seconds if llm_response is not None else None),
        input_tokens=(llm_response.input_tokens if llm_response is not None else None),
        output_tokens=(llm_response.output_tokens if llm_response is not None else None),
        total_trainable_parameters=total_params,
        train_loss_history=list(result.train_loss_history),
        val_metric_history=list(result.val_metric_history),
        total_gate_count=cost.gate_count if cost is not None else None,
        training_runtime_seconds=result.wall_clock_seconds,
        cache_hit=result.cache_hit,
        actually_trained=actually_trained,
        unique_training_id=unique_training_id,
        dtype_device_verified=dtype_ok,
        quantum_angles_changed=changed,
    )


def _invalid_llm_output_result(
    task_name: str, run_seed: int, proposal_id: str, error_message: str
) -> EvaluationResult:
    return EvaluationResult(
        proposal_id=proposal_id,
        task_name=task_name,
        run_seed=run_seed,
        structural_hash=None,
        circuit_canonical_json=None,
        validation_outcome=ValidationOutcome.INVALID,
        validation_issues=[
            ValidationIssue(code="mini_demo.malformed_llm_output", message=error_message, path="")
        ],
        compilation_outcome=CompilationOutcome.NOT_ATTEMPTED,
        training_outcome=TrainingOutcome.NOT_ATTEMPTED,
        failure_category=FailureCategory.INVALID_PROPOSAL,
    )


def _evaluate_compact_proposal(
    proposal_id: str, compact: CompactArchitecture, namespaced_task: str, run_seed: int,
    train_val, training_config: TrainingConfig, ledger: BudgetLedger, store: ResultStore,
    run_id: str,
) -> tuple[CompactArchitecture, EvaluationResult]:
    """Canonicalize, convert to IR, verify fixed capacity, then evaluate
    through the shared harness with the explicit mini-demo init policy and
    backprop differentiation. `namespaced_task` is the arm-scoped cache
    namespace (section 3)."""
    canonical = canonicalize_compact(compact)
    ir = compact_to_circuit_ir(canonical)
    verify_fixed_capacity(ir, train_val.spec)
    result = evaluate_candidate(
        ir, namespaced_task, run_seed, train_val, training_config, proposal_id, ledger,
        cache=store, proposal_event_store=store, run_id=run_id,
        init_policy=mini_demo_init_policy, diff_method=DIFF_METHOD,
    )
    return canonical, result


def _parse_and_persist_llm_call(
    store: ResultStore, arm: str, proposal_id: str, system_prompt: str, user_prompt: str,
    response,
) -> tuple[CompactArchitecture | None, str | None]:
    """Parse one LLM response against the compact schema and durably record
    its full provenance."""
    compact, err = parse_compact_json(response.raw_text)
    store.append_llm_call(
        arm,
        LLMCallRecord(
            proposal_id=proposal_id, call_index=0, system_prompt=system_prompt,
            user_prompt=user_prompt, model=response.model, temperature=TEMPERATURE,
            raw_response=response.raw_text,
            parsed_proposal=compact.model_dump() if compact else None,
            validation_errors=[err] if err else [], input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            estimated_cost_usd=response.estimated_cost_usd,
            latency_seconds=response.latency_seconds,
        ),
    )
    return compact, err


def _select_best(outcome: ArmRunOutcome, run_seed: int) -> None:
    trained = [
        c for c in outcome.candidates
        if c.training_outcome == "success" and c.final_val_rmse is not None
    ]
    if not trained:
        return
    best = min(trained, key=lambda c: c.final_val_rmse)
    outcome.selected_structural_hash = best.structural_hash
    outcome.selected_val_rmse = best.final_val_rmse
    outcome.selected_train_seed = train_seed_for_circuit(run_seed, best.structural_hash)


def run_random_arm(
    run_seed: int, base_task_name: str, train_val, training_config: TrainingConfig,
    budget_limit: int, store: ResultStore, deadline: float,
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
        compact = sample_compact_architecture(rng)
        canonical, result = _evaluate_compact_proposal(
            proposal_id, compact, namespaced_task, run_seed, train_val, training_config,
            ledger, store, arm,
        )
        outcome.candidates.append(
            _build_candidate_record(
                arm, proposal_index, proposal_id, canonical, result, store, namespaced_task,
                run_seed,
            )
        )
        proposal_index += 1

    outcome.ledger_summary = ledger.summary()
    _select_best(outcome, run_seed)
    return outcome


def _charge_budget_or_stop(
    outcome: ArmRunOutcome, budget: LLMApiBudget | None, cost: float
) -> bool:
    """Check the dollar budget BEFORE a request. Returns True if the call
    may proceed; sets stop_reason and returns False if it must not."""
    if budget is None:
        return True
    try:
        budget.check_can_afford(cost)
    except LLMBudgetExceededError as exc:
        outcome.stop_reason = f"dollar_budget_exceeded: {exc}"
        return False
    return True


def run_llm_open_loop_arm(
    run_seed: int, base_task_name: str, train_val, training_config: TrainingConfig,
    budget_limit: int, store: ResultStore, provider: LLMProvider, deadline: float,
    task_description: str, budget: LLMApiBudget | None = None,
    cost_estimate_per_call_usd: float = COST_ESTIMATE_PER_CALL_USD,
) -> ArmRunOutcome:
    arm = ARM_OPEN_LOOP
    namespaced_task = arm_task_name(base_task_name, arm)
    ledger = BudgetLedger.from_store(store, arm)
    system_prompt = build_system_prompt(task_description)
    outcome = ArmRunOutcome(arm=arm, task_namespace=namespaced_task)
    proposal_index = ledger.num_proposed

    while not ledger.is_exhausted(budget_limit):
        if proposal_index >= SAFETY_MAX_PROPOSALS_PER_ARM:
            outcome.stop_reason = "safety_cap_reached"
            break
        if time.monotonic() >= deadline:
            outcome.stop_reason = "wall_clock_deadline"
            break
        if not _charge_budget_or_stop(outcome, budget, cost_estimate_per_call_usd):
            break

        proposal_id = f"{arm}:{proposal_index}"
        user_prompt = build_open_loop_user_prompt()
        try:
            response = provider.complete(system_prompt, user_prompt, TEMPERATURE)
        except CallBudgetExceededError as exc:
            outcome.stop_reason = f"call_budget_exceeded: {exc}"
            break
        except Exception as exc:  # a real outbound request was attempted and failed
            outcome.api_outbound_attempts += 1
            outcome.api_failed_calls += 1
            outcome.stop_reason = f"api_error: {type(exc).__name__}: {exc}"
            break
        outcome.api_outbound_attempts += 1
        outcome.api_successful_calls += 1
        if budget is not None:
            budget.record_spend(cost_estimate_per_call_usd)

        compact, err = _parse_and_persist_llm_call(
            store, arm, proposal_id, system_prompt, user_prompt, response
        )

        canonical = compact
        if compact is None:
            result = _invalid_llm_output_result(namespaced_task, run_seed, proposal_id, err or "?")
            store.append_proposal_event(
                arm, ledger.num_proposed, ProposalRecord(outcome=ProposalOutcome.INVALID)
            )
            ledger.record_proposal(ProposalRecord(outcome=ProposalOutcome.INVALID))
        else:
            canonical, result = _evaluate_compact_proposal(
                proposal_id, compact, namespaced_task, run_seed, train_val, training_config,
                ledger, store, arm,
            )
        outcome.candidates.append(
            _build_candidate_record(
                arm, proposal_index, proposal_id, canonical, result, store, namespaced_task,
                run_seed, llm_response=response,
            )
        )
        proposal_index += 1

    outcome.ledger_summary = ledger.summary()
    _select_best(outcome, run_seed)
    return outcome


def run_llm_closed_loop_arm(
    run_seed: int, base_task_name: str, train_val, training_config: TrainingConfig,
    budget_limit: int, store: ResultStore, provider: LLMProvider, deadline: float,
    task_description: str, budget: LLMApiBudget | None = None,
    cost_estimate_per_call_usd: float = COST_ESTIMATE_PER_CALL_USD,
) -> ArmRunOutcome:
    arm = ARM_CLOSED_LOOP
    namespaced_task = arm_task_name(base_task_name, arm)
    ledger = BudgetLedger.from_store(store, arm)
    system_prompt = build_system_prompt(task_description)
    outcome = ArmRunOutcome(arm=arm, task_namespace=namespaced_task)
    proposal_index = ledger.num_proposed
    prior: dict | None = None  # feedback about THIS arm's previous proposal only

    while not ledger.is_exhausted(budget_limit):
        if proposal_index >= SAFETY_MAX_PROPOSALS_PER_ARM:
            outcome.stop_reason = "safety_cap_reached"
            break
        if time.monotonic() >= deadline:
            outcome.stop_reason = "wall_clock_deadline"
            break
        if not _charge_budget_or_stop(outcome, budget, cost_estimate_per_call_usd):
            break

        proposal_id = f"{arm}:{proposal_index}"
        if prior is None:
            user_prompt = build_closed_loop_first_user_prompt()
        else:
            user_prompt = build_closed_loop_feedback_user_prompt(**prior)
        try:
            response = provider.complete(system_prompt, user_prompt, TEMPERATURE)
        except CallBudgetExceededError as exc:
            outcome.stop_reason = f"call_budget_exceeded: {exc}"
            break
        except Exception as exc:
            outcome.api_outbound_attempts += 1
            outcome.api_failed_calls += 1
            outcome.stop_reason = f"api_error: {type(exc).__name__}: {exc}"
            break
        outcome.api_outbound_attempts += 1
        outcome.api_successful_calls += 1
        if budget is not None:
            budget.record_spend(cost_estimate_per_call_usd)

        compact, err = _parse_and_persist_llm_call(
            store, arm, proposal_id, system_prompt, user_prompt, response
        )

        canonical = compact
        if compact is None:
            result = _invalid_llm_output_result(namespaced_task, run_seed, proposal_id, err or "?")
            store.append_proposal_event(
                arm, ledger.num_proposed, ProposalRecord(outcome=ProposalOutcome.INVALID)
            )
            ledger.record_proposal(ProposalRecord(outcome=ProposalOutcome.INVALID))
        else:
            canonical, result = _evaluate_compact_proposal(
                proposal_id, compact, namespaced_task, run_seed, train_val, training_config,
                ledger, store, arm,
            )

        # Build the NEXT prompt's feedback from THIS proposal only -- always
        # all six fields (validation RMSE/depth/2q are populated even for a
        # within-arm duplicate, since the cached result carries them).
        prior = _closed_loop_feedback_fields(canonical, result)

        outcome.candidates.append(
            _build_candidate_record(
                arm, proposal_index, proposal_id, canonical, result, store, namespaced_task,
                run_seed, llm_response=response,
            )
        )
        proposal_index += 1

    outcome.ledger_summary = ledger.summary()
    _select_best(outcome, run_seed)
    return outcome


def _closed_loop_feedback_fields(
    canonical: CompactArchitecture | None, result: EvaluationResult
) -> dict:
    """The exact kwargs for `build_closed_loop_feedback_user_prompt`, derived
    from one proposal's own result. For a malformed (None) proposal the
    architecture fields fall back to placeholders and metrics are null."""
    cost = result.circuit_cost
    arch = canonical.model_dump() if canonical is not None else {
        "layer_1_gate": "n/a", "entangler": "n/a",
        "entangler_direction": "n/a", "layer_2_gate": "n/a",
    }
    return {
        "prior_layer_1_gate": arch["layer_1_gate"],
        "prior_entangler": arch["entangler"],
        "prior_entangler_direction": arch["entangler_direction"],
        "prior_layer_2_gate": arch["layer_2_gate"],
        "valid": result.validation_outcome == ValidationOutcome.VALID,
        "is_duplicate": result.is_duplicate,
        "val_rmse": result.val_metric_value,
        "circuit_depth": cost.depth if cost is not None else None,
        "two_qubit_gate_count": cost.two_qubit_gate_count if cost is not None else None,
    }


def evaluate_protected_test(
    outcome: ArmRunOutcome, store: ResultStore, base_task_name: str, test_split, task_spec,
) -> None:
    """Score the arm's selected circuit once on the protected test partition.
    Reads from the arm's OWN cache namespace (section 3)."""
    if outcome.selected_structural_hash is None or outcome.selected_train_seed is None:
        return
    namespaced_task = outcome.task_namespace or arm_task_name(base_task_name, outcome.arm)
    weights = store.get_trained_weights(
        namespaced_task, outcome.selected_structural_hash, outcome.selected_train_seed
    )
    cached_eval = store.get_cached(
        namespaced_task, outcome.selected_structural_hash, outcome.selected_train_seed
    )
    if weights is None or cached_eval is None or not cached_eval.circuit_canonical_json:
        return
    ir = CircuitIR.model_validate_json(cached_eval.circuit_canonical_json)
    test_result = evaluate_on_test(
        ir, weights["classical_state"], test_split, task_spec, outcome.selected_train_seed
    )
    outcome.protected_test_rmse = test_result.test_metric_value
