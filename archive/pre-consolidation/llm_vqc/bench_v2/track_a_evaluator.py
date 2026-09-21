"""Track A: structure-only evaluation with ONE shared inner trainer.

Every Track-A arm — random, evolutionary, greedy, LLM, and the fixed
references — routes its proposals through `evaluate_structure_candidate`.
The arm supplies structure only; theta initialization and AdamW training
happen here, identically for every arm, with the theta seed a pure
function of `(task_version [via task_name], data_seed, search_seed,
structural_hash)` (protocol §5). The shared cross-arm cache guarantees
that an identical architecture receives an identical inner-training
result no matter which arm proposed it.

**Test quarantine:** this module never imports
`llm_vqc.bench_v2.test_gate`, `llm_vqc.free_amplitude.final_test`, or any
`build_test` path — locked by `tests/test_bench_v2_evaluators.py` via AST
inspection, mirroring the preserved main-mode invariant.

Proposal dispatch by space profile:

- `compact_free_v1` proposals are `{"operations": [gate dicts]}` handled
  by the preserved `validate_free_gate_proposal` converter;
- `scalable_layered_v1` proposals are `{"operations": [layer dicts]}` in
  the shared layered IR grammar (`llm_vqc.bench_v2.space`).
"""

from __future__ import annotations

from llm_vqc.bench_v2.init_policy import make_bench_v2_init_policy
from llm_vqc.bench_v2.space import (
    COMPACT_FREE_V1,
    READOUT_QUBIT,
    SpaceProfile,
    validate_layered_structure,
)
from llm_vqc.evaluation.results import (
    CompilationOutcome,
    EvaluationResult,
    FailureCategory,
    TrainingOutcome,
    ValidationOutcome,
)
from llm_vqc.evaluation.seeds import derive_child_seed
from llm_vqc.free_amplitude.diagnostics import (
    causal_cone_summary_from_ops,
    gradient_diagnostics,
)
from llm_vqc.free_amplitude.harness import CandidateCache
from llm_vqc.free_amplitude.init_policy import free_amplitude_train_seed
from llm_vqc.free_amplitude.model import FixedReadoutModelError, FixedReadoutQuantumModel
from llm_vqc.free_amplitude.schema import (
    free_gate_proposal_to_circuit_ir,
    validate_free_gate_proposal,
)
from llm_vqc.free_amplitude.structural_cost import structural_cost_summary
from llm_vqc.free_amplitude.training import (
    FreeAmplitudeTrainingConfig,
    train_fixed_readout_model,
)
from llm_vqc.ir.budget import BudgetLedger, ProposalEventStore, ProposalOutcome, ProposalRecord
from llm_vqc.ir.canonicalize import canonical_json
from llm_vqc.ir.canonicalize import structural_hash as compute_structural_hash
from llm_vqc.ir.compiler_pennylane import CompilerError
from llm_vqc.ir.expand import EntangleInstruction, build_program
from llm_vqc.ir.metrics import circuit_cost_summary
from llm_vqc.tasks.base import TrainValData


class _ProgramOpView:
    """Adapter giving expanded program instructions the (.gate, .wires)
    interface the shared causal-cone routine expects."""

    __slots__ = ("gate", "wires")

    def __init__(self, gate: str, wires: tuple[int, ...]) -> None:
        self.gate = gate
        self.wires = wires


def _program_op_views(ir) -> list[_ProgramOpView]:
    program = build_program(ir)
    views = []
    for inst in program.body:
        if isinstance(inst, EntangleInstruction):
            views.append(_ProgramOpView(inst.gate, (inst.control, inst.target)))
        else:
            views.append(_ProgramOpView(inst.gate, (inst.wire,)))
    return views


def bench_v2_run_seed(task_name: str, data_seed: int, search_seed: int) -> int:
    """The Track-A run seed: pure function of (task_name [carries the task
    version], data_seed, search_seed). Feeding this into the preserved
    `free_amplitude_train_seed(run_seed, structural_hash, config_version)`
    makes the final theta seed exactly the protocol §5 function
    f(task_version, data_seed, search_seed, structural_hash)."""
    return derive_child_seed(data_seed, "bench_v2_run", task_name, str(search_seed))


def evaluate_structure_candidate(
    raw_proposal: dict,
    space: SpaceProfile,
    task_name: str,
    val_metric_name: str,
    run_seed: int,
    train_val: TrainValData,
    training_config: FreeAmplitudeTrainingConfig,
    proposal_id: str,
    ledger: BudgetLedger,
    cache: CandidateCache | None = None,
    proposal_event_store: ProposalEventStore | None = None,
    run_id: str | None = None,
) -> EvaluationResult:
    """Validate -> dedupe -> compile -> train (shared protocol) -> record.

    Returns only validation-set metrics; no code path here reaches test
    data. `raw_proposal` must be `{"operations": [...]}` in the space
    profile's grammar.
    """

    def _record(record: ProposalRecord) -> None:
        if proposal_event_store is not None and run_id is not None:
            ledger.record_and_persist(proposal_event_store, run_id, record)
        else:
            ledger.record_proposal(record)

    def _invalid(issues) -> EvaluationResult:
        result = EvaluationResult(
            proposal_id=proposal_id, task_name=task_name, run_seed=run_seed,
            structural_hash=None, circuit_canonical_json=None,
            validation_outcome=ValidationOutcome.INVALID, validation_issues=list(issues),
            compilation_outcome=CompilationOutcome.NOT_ATTEMPTED,
            training_outcome=TrainingOutcome.NOT_ATTEMPTED,
            failure_category=FailureCategory.INVALID_PROPOSAL,
        )
        _record(
            ProposalRecord(
                outcome=ProposalOutcome.INVALID, structural_hash=None, issues=list(issues)
            )
        )
        return result

    # ---- validation / IR assembly, per space profile --------------------
    if space.name == COMPACT_FREE_V1:
        validation = validate_free_gate_proposal(raw_proposal, space.n_qubits)
        if not validation.valid:
            return _invalid(validation.issues)
        ir = free_gate_proposal_to_circuit_ir(validation.proposal, READOUT_QUBIT)
    else:
        operations = (
            raw_proposal.get("operations") if isinstance(raw_proposal, dict) else None
        )
        ir, issues = validate_layered_structure(operations, space)
        if ir is None:
            return _invalid(issues)

    ir_hash = compute_structural_hash(ir)
    ir_json = canonical_json(ir)
    train_seed = free_amplitude_train_seed(run_seed, ir_hash, training_config.config_version)

    # ---- global cross-arm dedupe/cache ----------------------------------
    if cache is not None:
        cached = cache.get_cached(task_name, ir_hash, train_seed)
        if cached is not None:
            duplicate_result = cached.model_copy(
                update={"proposal_id": proposal_id, "is_duplicate": True, "cache_hit": True}
            )
            _record(ProposalRecord(outcome=ProposalOutcome.DUPLICATE, structural_hash=ir_hash))
            return duplicate_result

    # ---- compile ---------------------------------------------------------
    try:
        cost = circuit_cost_summary(ir)
        FixedReadoutQuantumModel(ir, readout_qubit=READOUT_QUBIT)
    except (FixedReadoutModelError, CompilerError, RuntimeError, ValueError) as exc:
        result = EvaluationResult(
            proposal_id=proposal_id, task_name=task_name, run_seed=run_seed,
            structural_hash=ir_hash, circuit_canonical_json=ir_json,
            validation_outcome=ValidationOutcome.VALID,
            compilation_outcome=CompilationOutcome.FAILED, compilation_error=str(exc),
            training_outcome=TrainingOutcome.NOT_ATTEMPTED,
            failure_category=FailureCategory.COMPILATION_ERROR, train_seed=train_seed,
        )
        _record(ProposalRecord(outcome=ProposalOutcome.FAILED, structural_hash=ir_hash))
        return result

    # ---- shared inner training ------------------------------------------
    init_policy = make_bench_v2_init_policy(space.max_quantum_parameters)
    training_output = train_fixed_readout_model(
        ir, READOUT_QUBIT, train_val, training_config, train_seed, init_policy=init_policy
    )

    if not training_output.success:
        result = EvaluationResult(
            proposal_id=proposal_id, task_name=task_name, run_seed=run_seed,
            structural_hash=ir_hash, circuit_canonical_json=ir_json,
            validation_outcome=ValidationOutcome.VALID,
            compilation_outcome=CompilationOutcome.SUCCESS,
            training_outcome=TrainingOutcome.FAILED,
            training_error=training_output.error_message,
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

    # ---- diagnostics + record -------------------------------------------
    op_views = _program_op_views(ir)
    causal = causal_cone_summary_from_ops(op_views, READOUT_QUBIT)
    grad_diag = gradient_diagnostics(training_output.initial_gradient_vector, causal)
    structural = structural_cost_summary(ir)
    n_batches = -(-len(train_val.train) // training_config.batch_size)
    optimizer_objective_evaluations = training_config.epochs * n_batches
    # The recorded validation metric is the trainer's val RMSE. For the T4
    # classification task this equals sqrt(Brier) on the probability
    # output — a strictly monotone transform of the protocol's Brier
    # selection metric, so candidate ranking (and therefore selection) is
    # identical; reporting converts back to Brier where needed.
    result = EvaluationResult(
        proposal_id=proposal_id, task_name=task_name, run_seed=run_seed,
        structural_hash=ir_hash, circuit_canonical_json=ir_json,
        validation_outcome=ValidationOutcome.VALID,
        compilation_outcome=CompilationOutcome.SUCCESS,
        training_outcome=TrainingOutcome.SUCCESS,
        val_metric_name=val_metric_name,
        val_metric_value=training_output.final_val_metric,
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
                "space_profile": space.name,
                "n_qubits": space.n_qubits,
                "readout_qubit": READOUT_QUBIT,
                "operations": (
                    raw_proposal.get("operations") if isinstance(raw_proposal, dict) else None
                ),
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
                "near_zero_gradient_parameter_count": (
                    grad_diag.near_zero_gradient_parameter_count
                ),
                "searched_body_gate_count": structural.searched_body_gate_count,
                "controlled_gate_count": structural.controlled_gate_count,
                "searched_body_depth": structural.searched_body_depth,
                "total_compiled_depth": structural.total_compiled_depth,
                "optimizer_objective_evaluations": optimizer_objective_evaluations,
                "optimizer_gradient_evaluations": optimizer_objective_evaluations,
                "classical_parameter_count": 0,
                "init_policy_version": "bench_v2_init_v1",
            },
        )

    return result
