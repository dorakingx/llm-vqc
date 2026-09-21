"""The search-facing evaluation harness for free-gate, fixed-readout
candidates -- the `llm_vqc.free_amplitude` analog of `llm_vqc.evaluation.
harness.evaluate_candidate`.

Cannot reuse that function directly: it always constructs a
`HybridQNNModel` (classical embed + head) via `llm_vqc.evaluation.
training.train_model`, which structurally cannot represent this
experiment's zero-classical-parameter model. This module reimplements the
same validate -> dedupe -> compile -> train -> record contract using
`llm_vqc.free_amplitude.training.train_fixed_readout_model` instead, while
reusing `llm_vqc.ir.budget.BudgetLedger`, `llm_vqc.evaluation.results.
EvaluationResult`, and the same test-quarantine discipline (this module
never imports `llm_vqc.free_amplitude.final_test`).
"""

from __future__ import annotations

from typing import Protocol

from llm_vqc.evaluation.results import (
    CompilationOutcome,
    EvaluationResult,
    FailureCategory,
    TrainingOutcome,
    ValidationOutcome,
)
from llm_vqc.free_amplitude.diagnostics import causal_cone_summary, gradient_diagnostics
from llm_vqc.free_amplitude.init_policy import free_amplitude_train_seed
from llm_vqc.free_amplitude.model import FixedReadoutModelError, FixedReadoutQuantumModel
from llm_vqc.free_amplitude.schema import (
    FreeGateProposal,
    free_gate_proposal_to_circuit_ir,
    validate_free_gate_proposal,
)
from llm_vqc.free_amplitude.structural_cost import structural_cost_summary
from llm_vqc.free_amplitude.training import FreeAmplitudeTrainingConfig, train_fixed_readout_model
from llm_vqc.ir.budget import BudgetLedger, ProposalEventStore, ProposalOutcome, ProposalRecord
from llm_vqc.ir.canonicalize import canonical_json
from llm_vqc.ir.canonicalize import structural_hash as compute_structural_hash
from llm_vqc.ir.compiler_pennylane import CompilerError
from llm_vqc.ir.metrics import circuit_cost_summary
from llm_vqc.ir.validators import validate_proposal
from llm_vqc.tasks.base import TrainValData


class CandidateCache(Protocol):
    """Same narrow protocol as `llm_vqc.evaluation.harness.CandidateCache`
    -- `llm_vqc.evaluation.store.ResultStore` satisfies it structurally."""

    def get_cached(
        self, task_name: str, structural_hash: str, train_seed: int
    ) -> EvaluationResult | None: ...

    def put_cached(
        self, task_name: str, structural_hash: str, train_seed: int,
        result: EvaluationResult, trained_weights: dict | None,
    ) -> None: ...


def evaluate_free_gate_candidate(
    raw_proposal: dict | FreeGateProposal,
    task_name: str,
    run_seed: int,
    expected_n_qubits: int,
    readout_qubit: int,
    train_val: TrainValData,
    training_config: FreeAmplitudeTrainingConfig,
    proposal_id: str,
    ledger: BudgetLedger,
    cache: CandidateCache | None = None,
    proposal_event_store: ProposalEventStore | None = None,
    run_id: str | None = None,
) -> EvaluationResult:
    """Validate, (maybe) compile, (maybe) train, and score one free-gate
    proposal. Returns an `EvaluationResult` reporting only validation-set
    metrics -- never a protected-test metric (there is no code path here
    that could reach one; see `llm_vqc.free_amplitude.final_test`, which
    this module never imports).
    """

    def _record(record: ProposalRecord) -> None:
        if proposal_event_store is not None and run_id is not None:
            ledger.record_and_persist(proposal_event_store, run_id, record)
        else:
            ledger.record_proposal(record)

    validation = validate_free_gate_proposal(raw_proposal, expected_n_qubits)
    if not validation.valid:
        result = EvaluationResult(
            proposal_id=proposal_id, task_name=task_name, run_seed=run_seed,
            structural_hash=None, circuit_canonical_json=None,
            validation_outcome=ValidationOutcome.INVALID,
            validation_issues=validation.issues,
            compilation_outcome=CompilationOutcome.NOT_ATTEMPTED,
            training_outcome=TrainingOutcome.NOT_ATTEMPTED,
            failure_category=FailureCategory.INVALID_PROPOSAL,
        )
        _record(
            ProposalRecord(
                outcome=ProposalOutcome.INVALID, structural_hash=None, issues=validation.issues
            )
        )
        return result

    proposal = validation.proposal
    ir = free_gate_proposal_to_circuit_ir(proposal, readout_qubit)

    # Defensive double-check: the converter should always produce a valid
    # IR by construction, but this turns a silent converter regression into
    # a visible INVALID result rather than a confusing downstream crash.
    ir_check = validate_proposal(ir)
    if not ir_check.valid:
        result = EvaluationResult(
            proposal_id=proposal_id, task_name=task_name, run_seed=run_seed,
            structural_hash=None, circuit_canonical_json=None,
            validation_outcome=ValidationOutcome.INVALID,
            validation_issues=ir_check.issues,
            compilation_outcome=CompilationOutcome.NOT_ATTEMPTED,
            training_outcome=TrainingOutcome.NOT_ATTEMPTED,
            failure_category=FailureCategory.INVALID_PROPOSAL,
        )
        _record(
            ProposalRecord(
                outcome=ProposalOutcome.INVALID, structural_hash=None, issues=ir_check.issues
            )
        )
        return result

    ir_hash = compute_structural_hash(ir)
    ir_json = canonical_json(ir)
    train_seed = free_amplitude_train_seed(run_seed, ir_hash, training_config.config_version)

    if cache is not None:
        cached = cache.get_cached(task_name, ir_hash, train_seed)
        if cached is not None:
            duplicate_result = cached.model_copy(
                update={"proposal_id": proposal_id, "is_duplicate": True, "cache_hit": True}
            )
            _record(ProposalRecord(outcome=ProposalOutcome.DUPLICATE, structural_hash=ir_hash))
            return duplicate_result

    try:
        cost = circuit_cost_summary(ir)
    except CompilerError as exc:
        result = EvaluationResult(
            proposal_id=proposal_id, task_name=task_name, run_seed=run_seed,
            structural_hash=ir_hash, circuit_canonical_json=ir_json,
            validation_outcome=ValidationOutcome.VALID,
            compilation_outcome=CompilationOutcome.FAILED, compilation_error=str(exc),
            training_outcome=TrainingOutcome.NOT_ATTEMPTED,
            failure_category=FailureCategory.COMPILATION_ERROR, train_seed=train_seed,
        )
        _record(ProposalRecord(outcome=ProposalOutcome.FAILED, structural_hash=ir_hash, cost=None))
        return result

    try:
        FixedReadoutQuantumModel(ir, readout_qubit=readout_qubit)
    except (FixedReadoutModelError, CompilerError, RuntimeError, ValueError) as exc:
        result = EvaluationResult(
            proposal_id=proposal_id, task_name=task_name, run_seed=run_seed,
            structural_hash=ir_hash, circuit_canonical_json=ir_json,
            validation_outcome=ValidationOutcome.VALID,
            compilation_outcome=CompilationOutcome.FAILED, compilation_error=str(exc),
            training_outcome=TrainingOutcome.NOT_ATTEMPTED, circuit_cost=cost,
            failure_category=FailureCategory.COMPILATION_ERROR, train_seed=train_seed,
        )
        _record(ProposalRecord(outcome=ProposalOutcome.FAILED, structural_hash=ir_hash, cost=cost))
        return result

    training_output = train_fixed_readout_model(
        ir, readout_qubit, train_val, training_config, train_seed
    )

    if not training_output.success:
        result = EvaluationResult(
            proposal_id=proposal_id, task_name=task_name, run_seed=run_seed,
            structural_hash=ir_hash, circuit_canonical_json=ir_json,
            validation_outcome=ValidationOutcome.VALID,
            compilation_outcome=CompilationOutcome.SUCCESS,
            training_outcome=TrainingOutcome.FAILED, training_error=training_output.error_message,
            train_loss_history=training_output.train_loss_history,
            val_metric_history=training_output.val_metric_history,
            epochs_completed=training_output.epochs_completed,
            wall_clock_seconds=training_output.wall_clock_seconds, circuit_cost=cost,
            failure_category=(
                FailureCategory.TRAINING_DIVERGED
                if "Diverged" in (training_output.error_message or "")
                else FailureCategory.TRAINING_ERROR
            ),
            train_seed=train_seed,
        )
        _record(ProposalRecord(outcome=ProposalOutcome.FAILED, structural_hash=ir_hash, cost=cost))
        return result

    causal = causal_cone_summary(proposal, readout_qubit)
    grad_diag = gradient_diagnostics(training_output.initial_gradient_vector, causal)
    structural = structural_cost_summary(ir)

    result = EvaluationResult(
        proposal_id=proposal_id, task_name=task_name, run_seed=run_seed,
        structural_hash=ir_hash, circuit_canonical_json=ir_json,
        validation_outcome=ValidationOutcome.VALID,
        compilation_outcome=CompilationOutcome.SUCCESS,
        training_outcome=TrainingOutcome.SUCCESS,
        val_metric_name="rmse", val_metric_value=training_output.final_val_metric,
        val_metric_history=training_output.val_metric_history,
        train_loss_history=training_output.train_loss_history,
        epochs_completed=training_output.epochs_completed,
        wall_clock_seconds=training_output.wall_clock_seconds, circuit_cost=cost,
        failure_category=FailureCategory.NONE, train_seed=train_seed,
        trained_weights_ref=ir_hash if cache is not None else None,
    )
    _record(ProposalRecord(outcome=ProposalOutcome.VALID, structural_hash=ir_hash, cost=cost))

    if cache is not None:
        cache.put_cached(
            task_name, ir_hash, train_seed, result,
            trained_weights={
                "readout_qubit": readout_qubit,
                "compact_proposal": proposal.model_dump(),
                "initial_angles": training_output.initial_angles,
                "learned_angles": training_output.learned_angles,
                "angle_deltas": training_output.angle_deltas,
                "initial_gradient_norm": training_output.initial_gradient_norm,
                "final_gradient_norm": training_output.final_gradient_norm,
                "initial_gradient_vector": training_output.initial_gradient_vector,
                "final_gradient_vector": training_output.final_gradient_vector,
                "causal_operation_indices": sorted(causal.causal_operation_indices),
                "parameter_count_in_causal_cone": causal.parameter_count_in_causal_cone,
                "fraction_in_causal_cone": causal.fraction_in_causal_cone,
                "zero_gradient_parameter_count": grad_diag.zero_gradient_parameter_count,
                "near_zero_gradient_parameter_count": grad_diag.near_zero_gradient_parameter_count,
                "no_trainable_parameter_affects_readout": (
                    grad_diag.no_trainable_parameter_affects_readout
                ),
                "all_gradients_numerically_zero": grad_diag.all_gradients_numerically_zero,
                "searched_body_gate_count": structural.searched_body_gate_count,
                "controlled_gate_count": structural.controlled_gate_count,
                "searched_body_depth": structural.searched_body_depth,
                "total_compiled_depth": structural.total_compiled_depth,
                "decomposed_state_preparation_gate_count": (
                    structural.decomposed_state_preparation_gate_count
                ),
                "decomposed_state_preparation_depth": structural.decomposed_state_preparation_depth,
                "total_decomposed_gate_count": structural.total_decomposed_gate_count,
                "total_decomposed_depth": structural.total_decomposed_depth,
                "classical_parameter_count": 0,
            },
        )

    return result
