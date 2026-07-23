"""Main-mode candidate evaluation: **the LLM/sampler supplies the angles,
so there is NO optimizer.**

`evaluate_complete_candidate` compiles amplitude encoding + the proposed
circuit, loads the proposed `theta` verbatim into the model's quantum
weights, and performs a single no-grad forward pass on train and
validation data -- it never constructs a `torch.optim` optimizer and never
updates a weight. (The AdamW structure-search path lives in
`llm_vqc.free_amplitude.harness`, kept only as a separately-labelled
ablation.)

This module deliberately does not import `torch.optim` at all, which
`tests/test_free_amplitude_joint_search.py::test_no_optimizer_in_main_mode`
asserts by AST inspection -- "no optimizer exists in the main mode" is a
structural fact, not a convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from llm_vqc.free_amplitude.candidate_schema import (
    CompleteCandidateProposal,
    architecture_hash,
    candidate_hash,
    canonical_theta_values,
    complete_candidate_to_ir_and_theta,
    validate_complete_candidate,
)
from llm_vqc.free_amplitude.diagnostics import causal_cone_summary_from_ops
from llm_vqc.free_amplitude.metrics import (
    PredictionStats,
    prediction_stats,
    samplewise_mae,
    samplewise_mse,
    samplewise_rmse,
)
from llm_vqc.free_amplitude.model import FixedReadoutModelError, FixedReadoutQuantumModel
from llm_vqc.free_amplitude.structural_cost import structural_cost_summary
from llm_vqc.ir.budget import BudgetLedger, ProposalEventStore, ProposalOutcome, ProposalRecord
from llm_vqc.ir.validators import ValidationIssue
from llm_vqc.tasks.base import DataSplit, TrainValData


@dataclass
class FixedThetaEvalResult:
    """Everything one main-mode candidate evaluation produced. `train_seed`
    is absent by design -- there is no training in main mode."""

    proposal_id: str
    valid: bool
    is_duplicate: bool
    outcome: str  # "evaluated" | "invalid" | "failed"
    validation_issues: list[ValidationIssue] = field(default_factory=list)
    architecture_hash: str | None = None
    candidate_hash: str | None = None
    operations: list[dict] | None = None
    theta: list[float] | None = None
    train_mse: float | None = None
    val_rmse: float | None = None
    val_mae: float | None = None
    val_prediction_stats: dict | None = None
    circuit_depth: int | None = None
    searched_body_depth: object | None = None
    searched_body_gate_count: int | None = None
    controlled_gate_count: int | None = None
    quantum_parameter_count: int | None = None
    parameter_count_in_causal_cone: int | None = None
    fraction_in_causal_cone: float | None = None
    error_message: str | None = None


class MainModeOptimizerError(RuntimeError):
    """Raised if any optimizer construction is ever attempted in main
    mode. Referenced by the main-mode invariant tests."""


def _load_theta(model: FixedReadoutQuantumModel, theta: list[float]) -> None:
    if model.q_layer.weights.numel() != len(theta):
        raise FixedReadoutModelError(
            f"model needs {model.q_layer.weights.numel()} angles, got {len(theta)}"
        )
    with torch.no_grad():
        model.q_layer.weights.copy_(torch.tensor(theta, dtype=torch.float64))


def _forward(model: FixedReadoutQuantumModel, features) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float64)
        return model(x).cpu().numpy().reshape(-1)


class MainModeCache:
    """Narrow protocol satisfied structurally by `ResultStore` -- but main
    mode caches a plain dict blob (metrics + theta) rather than an
    `EvaluationResult`, so it uses the store's trained-weights slot only.
    Kept as a tiny wrapper so the runner can pass a `ResultStore` directly.
    """

    def __init__(self, store, task_name: str, seed: int) -> None:
        self._store = store
        self._task_name = task_name
        self._seed = seed

    def get(self, candidate_hash_value: str) -> dict | None:
        return self._store.get_trained_weights(self._task_name, candidate_hash_value, self._seed)

    def put(self, candidate_hash_value: str, blob: dict) -> None:
        # A minimal EvaluationResult stand-in is required by the store's
        # put_cached signature; main mode only ever reads back the blob.
        from llm_vqc.evaluation.results import (
            CompilationOutcome,
            EvaluationResult,
            TrainingOutcome,
            ValidationOutcome,
        )

        placeholder = EvaluationResult(
            proposal_id=candidate_hash_value, task_name=self._task_name, run_seed=self._seed,
            structural_hash=candidate_hash_value, circuit_canonical_json=None,
            validation_outcome=ValidationOutcome.VALID,
            compilation_outcome=CompilationOutcome.SUCCESS,
            training_outcome=TrainingOutcome.SUCCESS,
            val_metric_name="rmse", val_metric_value=blob.get("val_rmse"),
        )
        self._store.put_cached(self._task_name, candidate_hash_value, self._seed, placeholder, blob)


def evaluate_complete_candidate(
    raw_proposal: dict | CompleteCandidateProposal,
    expected_n_qubits: int,
    readout_qubit: int,
    train_val: TrainValData,
    proposal_id: str,
    ledger: BudgetLedger,
    seed: int,
    cache: MainModeCache | None = None,
    proposal_event_store: ProposalEventStore | None = None,
    run_id: str | None = None,
) -> FixedThetaEvalResult:
    """Validate, (maybe) evaluate-at-fixed-theta, and score one complete
    candidate. NO optimizer, NO gradient step. Duplicate detection is by
    `candidate_hash` (structure + angles), so re-proposing identical angles
    on identical structure is a duplicate, but the same structure with
    different angles is a fresh candidate."""

    def _record(record: ProposalRecord) -> None:
        if proposal_event_store is not None and run_id is not None:
            ledger.record_and_persist(proposal_event_store, run_id, record)
        else:
            ledger.record_proposal(record)

    validation = validate_complete_candidate(raw_proposal, expected_n_qubits)
    if not validation.valid:
        _record(ProposalRecord(outcome=ProposalOutcome.INVALID))
        return FixedThetaEvalResult(
            proposal_id=proposal_id, valid=False, is_duplicate=False, outcome="invalid",
            validation_issues=validation.issues,
        )

    proposal = validation.proposal
    arch_hash = architecture_hash(proposal, readout_qubit)
    cand_hash = candidate_hash(proposal, readout_qubit)
    operations = [op.model_dump() for op in proposal.operations]
    theta = canonical_theta_values(proposal)

    if cache is not None:
        cached = cache.get(cand_hash)
        if cached is not None:
            _record(ProposalRecord(outcome=ProposalOutcome.DUPLICATE, structural_hash=cand_hash))
            return _result_from_blob(proposal_id, cached, is_duplicate=True)

    ir, theta_vector = complete_candidate_to_ir_and_theta(proposal, readout_qubit)
    try:
        cost = structural_cost_summary(ir)
        model = FixedReadoutQuantumModel(ir, readout_qubit=readout_qubit)
        model.double()
        _load_theta(model, theta_vector)

        train_pred = _forward(model, train_val.train.features)
        val_pred = _forward(model, train_val.val.features)
        train_mse = samplewise_mse(train_pred, train_val.train.targets)
        val_rmse = samplewise_rmse(val_pred, train_val.val.targets)
        val_mae = samplewise_mae(val_pred, train_val.val.targets)
        pstats: PredictionStats = prediction_stats(val_pred)
    except Exception as exc:  # compilation/forward failure -> FAILED (consumes budget)
        _record(ProposalRecord(outcome=ProposalOutcome.FAILED, structural_hash=cand_hash))
        return FixedThetaEvalResult(
            proposal_id=proposal_id, valid=True, is_duplicate=False, outcome="failed",
            architecture_hash=arch_hash, candidate_hash=cand_hash, operations=operations,
            theta=theta, error_message=f"{type(exc).__name__}: {exc}",
        )

    cone = causal_cone_summary_from_ops(proposal.operations, readout_qubit)
    blob = {
        "architecture_hash": arch_hash, "candidate_hash": cand_hash,
        "operations": operations, "theta": theta,
        "train_mse": train_mse, "val_rmse": val_rmse, "val_mae": val_mae,
        "val_prediction_stats": {
            "mean": pstats.mean, "std": pstats.std, "min": pstats.min, "max": pstats.max,
            "constant_output": pstats.constant_output,
        },
        "circuit_depth": cost.total_compiled_depth,
        "searched_body_depth": cost.searched_body_depth,
        "searched_body_gate_count": cost.searched_body_gate_count,
        "controlled_gate_count": cost.controlled_gate_count,
        "quantum_parameter_count": cost.quantum_parameter_count,
        "parameter_count_in_causal_cone": cone.parameter_count_in_causal_cone,
        "fraction_in_causal_cone": cone.fraction_in_causal_cone,
    }
    _record(ProposalRecord(outcome=ProposalOutcome.VALID, structural_hash=cand_hash))
    if cache is not None:
        cache.put(cand_hash, blob)
    return _result_from_blob(proposal_id, blob, is_duplicate=False)


def _result_from_blob(proposal_id: str, blob: dict, is_duplicate: bool) -> FixedThetaEvalResult:
    return FixedThetaEvalResult(
        proposal_id=proposal_id, valid=True, is_duplicate=is_duplicate, outcome="evaluated",
        architecture_hash=blob.get("architecture_hash"), candidate_hash=blob.get("candidate_hash"),
        operations=blob.get("operations"), theta=blob.get("theta"),
        train_mse=blob.get("train_mse"), val_rmse=blob.get("val_rmse"), val_mae=blob.get("val_mae"),
        val_prediction_stats=blob.get("val_prediction_stats"),
        circuit_depth=blob.get("circuit_depth"),
        searched_body_depth=blob.get("searched_body_depth"),
        searched_body_gate_count=blob.get("searched_body_gate_count"),
        controlled_gate_count=blob.get("controlled_gate_count"),
        quantum_parameter_count=blob.get("quantum_parameter_count"),
        parameter_count_in_causal_cone=blob.get("parameter_count_in_causal_cone"),
        fraction_in_causal_cone=blob.get("fraction_in_causal_cone"),
    )


def evaluate_on_test_fixed_theta(
    proposal: CompleteCandidateProposal, readout_qubit: int, test_split: DataSplit,
) -> dict:
    """Score a selected complete candidate once on the protected test
    partition, at its own proposed theta. No optimizer; never touches
    validation data."""
    ir, theta_vector = complete_candidate_to_ir_and_theta(proposal, readout_qubit)
    model = FixedReadoutQuantumModel(ir, readout_qubit=readout_qubit)
    model.double()
    _load_theta(model, theta_vector)
    test_pred = _forward(model, test_split.features)
    return {
        "test_rmse": samplewise_rmse(test_pred, test_split.targets),
        "test_mae": samplewise_mae(test_pred, test_split.targets),
        "n_test_samples": len(test_split),
    }
