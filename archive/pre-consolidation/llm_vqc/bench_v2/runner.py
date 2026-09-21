"""bench_v2 search runners — the one execution loop per track.

Mirrors the preserved `llm_vqc.search.runner.SearchRunner` contract
(unconditional per-proposal checkpointing, exact budget termination from
a store-replayed ledger, no arm access to evaluator/store/test data) with
three bench_v2 differences:

1. **Budget axis is UNIQUE candidate evaluations** (protocol §8): the
   loop terminates when `ledger.num_unique >= budget_limit` — duplicates
   and invalids are recorded (both budget views are reported) but do not
   count toward the unique axis.
2. **Arm exhaustion**: `arm.propose` may return `None` (fixed references
   do); the runner finalizes early with `stop_reason="arm_exhausted"`.
3. **Two tracks**: `StructureSearchRunner` drives Track-A arms through
   `evaluate_structure_candidate`; `JointSearchRunner` drives Track-B
   arms through the preserved no-optimizer `evaluate_complete_candidate`.

Neither runner imports any test-gate/final-test/build_test path (AST-
audited).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from llm_vqc.bench_v2.space import READOUT_QUBIT, SpaceProfile
from llm_vqc.bench_v2.track_a_evaluator import evaluate_structure_candidate
from llm_vqc.bench_v2.track_b_evaluator import MainModeCache, evaluate_complete_candidate
from llm_vqc.evaluation.results import CompilationOutcome, TrainingOutcome
from llm_vqc.evaluation.store import ResultStore
from llm_vqc.free_amplitude.training import FreeAmplitudeTrainingConfig
from llm_vqc.ir.budget import BudgetLedger, ProposalOutcome
from llm_vqc.search.arm import SearchArm
from llm_vqc.search.feedback import SearchFeedback
from llm_vqc.tasks.base import TrainValData

#: Same pathological-arm guard as the preserved runner.
_MAX_PROPOSALS_PER_BUDGET_UNIT = 25


class BenchV2RunnerError(Exception):
    pass


@dataclass
class BenchV2RunResult:
    run_id: str
    arm_name: str
    task_name: str
    budget_limit: int
    stop_reason: str  # "budget_exhausted" | "arm_exhausted"
    ledger_summary: dict
    selected_hash: str | None
    selected_val_metric_value: float | None
    selection_timestamp: str


class _BaseRunner:
    """Shared loop; subclasses provide `_evaluate(raw_proposal, proposal_id,
    ledger) -> SearchFeedback-compatible EvaluationResult-ish`."""

    def __init__(
        self,
        arm: SearchArm,
        task_name: str,
        budget_limit: int,
        run_seed: int,
        result_store: ResultStore,
        run_id: str,
    ) -> None:
        self.arm = arm
        self.task_name = task_name
        self.budget_limit = budget_limit
        self.run_seed = run_seed
        self.result_store = result_store
        self.run_id = run_id

    def _load_or_initialize_state(self):
        from llm_vqc.evaluation.seeds import derive_child_seed

        raw = self.result_store.load_run_state(self.run_id)
        if raw is not None:
            return self.arm.deserialize_state(raw)
        arm_seed = derive_child_seed(self.run_seed, "search", self.arm.name)
        state = self.arm.initialize(arm_seed)
        self.result_store.save_run_state(self.run_id, self.arm.serialize_state(state))
        return state

    def _evaluate(self, raw_proposal: dict, proposal_id: str, ledger: BudgetLedger):
        raise NotImplementedError

    def run(self) -> BenchV2RunResult:
        ledger = BudgetLedger.from_store(self.result_store, self.run_id)
        state = self._load_or_initialize_state()
        max_proposals = self.budget_limit * _MAX_PROPOSALS_PER_BUDGET_UNIT
        proposal_index = ledger.num_proposed
        stop_reason = "budget_exhausted"

        while ledger.num_unique < self.budget_limit:
            if proposal_index >= max_proposals:
                raise BenchV2RunnerError(
                    f"arm {self.arm.name!r} made {proposal_index} proposals without "
                    f"reaching {self.budget_limit} unique evaluations — aborting "
                    "rather than looping forever."
                )
            raw_proposal = self.arm.propose(state)
            if raw_proposal is None:
                stop_reason = "arm_exhausted"
                break
            proposal_id = f"{self.run_id}:{proposal_index}"
            feedback = self._evaluate(raw_proposal, proposal_id, ledger)
            state = self.arm.update_state(state, raw_proposal, feedback)
            self.result_store.save_run_state(self.run_id, self.arm.serialize_state(state))
            proposal_index += 1

        selected_hash = self.arm.select_final(state)
        selected_metric = getattr(state, "best_metric", None)
        return BenchV2RunResult(
            run_id=self.run_id,
            arm_name=self.arm.name,
            task_name=self.task_name,
            budget_limit=self.budget_limit,
            stop_reason=stop_reason,
            ledger_summary=ledger.summary(),
            selected_hash=selected_hash,
            selected_val_metric_value=selected_metric,
            selection_timestamp=datetime.now(UTC).isoformat(),
        )


class StructureSearchRunner(_BaseRunner):
    """Track A: structure-only arms + the shared inner trainer."""

    def __init__(
        self,
        arm: SearchArm,
        space: SpaceProfile,
        task_name: str,
        val_metric_name: str,
        train_val: TrainValData,
        training_config: FreeAmplitudeTrainingConfig,
        budget_limit: int,
        run_seed: int,
        result_store: ResultStore,
        run_id: str,
    ) -> None:
        super().__init__(arm, task_name, budget_limit, run_seed, result_store, run_id)
        self.space = space
        self.val_metric_name = val_metric_name
        self.train_val = train_val
        self.training_config = training_config

    def _evaluate(self, raw_proposal, proposal_id, ledger):
        result = evaluate_structure_candidate(
            raw_proposal, self.space, self.task_name, self.val_metric_name,
            self.run_seed, self.train_val, self.training_config, proposal_id,
            ledger, cache=self.result_store,
            proposal_event_store=self.result_store, run_id=self.run_id,
        )
        return SearchFeedback.from_evaluation_result(result)


class JointSearchRunner(_BaseRunner):
    """Track B: complete candidates, verbatim theta, no optimizer."""

    def __init__(
        self,
        arm: SearchArm,
        n_qubits: int,
        task_name: str,
        train_val: TrainValData,
        budget_limit: int,
        run_seed: int,
        result_store: ResultStore,
        run_id: str,
    ) -> None:
        super().__init__(arm, task_name, budget_limit, run_seed, result_store, run_id)
        self.n_qubits = n_qubits
        self.train_val = train_val
        self.cache = MainModeCache(result_store, task_name, run_seed)

    def _evaluate(self, raw_proposal, proposal_id, ledger):
        result = evaluate_complete_candidate(
            raw_proposal, self.n_qubits, READOUT_QUBIT, self.train_val, proposal_id,
            ledger, self.run_seed, cache=self.cache,
            proposal_event_store=self.result_store, run_id=self.run_id,
        )
        # Adapt FixedThetaEvalResult -> SearchFeedback (validation-only).
        outcome = {
            "evaluated": (
                ProposalOutcome.DUPLICATE if result.is_duplicate else ProposalOutcome.VALID
            ),
            "invalid": ProposalOutcome.INVALID,
            "failed": ProposalOutcome.FAILED,
        }[result.outcome]
        return SearchFeedback(
            proposal_id=proposal_id,
            outcome=outcome,
            structural_hash=result.candidate_hash,
            is_duplicate=result.is_duplicate,
            validation_issues=result.validation_issues,
            compilation_outcome=(
                CompilationOutcome.SUCCESS
                if result.outcome == "evaluated"
                else CompilationOutcome.FAILED
                if result.outcome == "failed"
                else CompilationOutcome.NOT_ATTEMPTED
            ),
            training_outcome=TrainingOutcome.NOT_ATTEMPTED,  # no optimizer in Track B
            val_metric_name="rmse" if result.outcome == "evaluated" else None,
            val_metric_value=result.val_rmse,
            circuit_cost=None,
        )
