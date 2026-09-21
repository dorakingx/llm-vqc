"""bench_v2 LLM search arms (protocol §7): schema-constrained BATCH
proposals, open-loop and bounded-archive closed-loop, for both tracks.

Architecture per the frozen protocol: (1) proposal layer = one LLM call
returning `{"candidates": [...]}` (batch of 3); (2) the deterministic
validator/canonicalizer downstream of `propose()` (the shared evaluators)
records every issue — the arm performs NO repair loop, so the invalid
rate is measured, not hidden; (3) archive manager = bounded top-k=8 by
validation metric with hash-dedup novelty; (4) deterministic feedback
summarizer (`llm_prompts`); (5) the protected test gate is a separate
module this one never imports.

Buffering: one call yields up to `batch_size` candidates; `propose()`
serves them one at a time, so the LLM-call budget and the candidate-
evaluation budget stay separate ledgers. The freshly fetched buffer is
merged into persisted state at the next `update_state`; a crash between
call and persist repeats at most one call on resume (window identical to
the preserved runner's evaluate-vs-checkpoint window), and every call is
logged to the store the moment it returns via the injected append-only
`call_sink`.

Exhaustion honesty: on `MAX_CONSECUTIVE_PARSE_FAILURES` unparseable
responses or a provider cap/exception, the arm raises — a mandatory LLM
cell must fail loudly (and stay incomplete for the checker), never
silently degrade into a random arm.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from llm_vqc.bench_v2.llm_prompts import (
    BENCH_V2_PROMPT_VERSION,
    closed_loop_user_prompt,
    joint_system_prompt,
    open_loop_user_prompt,
    structure_system_prompt,
)
from llm_vqc.bench_v2.space import SpaceProfile
from llm_vqc.llm.records import LLMCallRecord
from llm_vqc.search.arm import SearchArm
from llm_vqc.search.comparison import is_strictly_better
from llm_vqc.search.feedback import SearchFeedback

BATCH_SIZE = 3          # frozen (protocol_v2.yaml: batch_candidates_per_call)
ARCHIVE_TOP_K = 8       # frozen (protocol_v2.yaml: archive_top_k)
DEFAULT_TEMPERATURE = 1.0
MAX_CONSECUTIVE_PARSE_FAILURES = 3


class LLMArmError(Exception):
    """A mandatory LLM arm could not proceed (parse-failure streak or
    provider failure) — the cell fails loudly and stays incomplete."""


def parse_candidate_batch(raw_text: str) -> list[dict] | None:
    """Parse a batch response. Liberal in shape (a bare single candidate
    object is accepted as a batch of one), strict in type. Returns None if
    nothing parseable emerged (the call is still logged)."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict) and isinstance(payload.get("candidates"), list):
        return [c for c in payload["candidates"] if isinstance(c, dict)] or None
    if isinstance(payload, dict) and "operations" in payload:
        return [payload]
    return None


class _ArchiveEntry(BaseModel):
    structural_hash: str
    val_metric: float
    operations: list[dict] | None = None
    n_ops: int | None = None
    n_params: int | None = None


class LLMArmState(BaseModel):
    seed: int
    n_proposed: int = 0
    llm_calls_made: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    llm_estimated_cost_usd: float = 0.0
    consecutive_parse_failures: int = 0
    buffer: list[dict] = Field(default_factory=list)
    archive: list[_ArchiveEntry] = Field(default_factory=list)
    failure_counts: dict[str, int] = Field(default_factory=dict)
    n_unique_evaluated: int = 0
    pending_operations: list[dict] | None = None
    best_hash: str | None = None
    best_metric: float | None = None
    best_candidate: dict | None = None


class _BaseLLMArm(SearchArm[LLMArmState]):
    """Shared machinery; subclasses set track ('structure'|'joint') and
    closed_loop, and provide the system prompt."""

    track: str
    closed_loop: bool

    def __init__(
        self,
        provider,
        budget_limit: int,
        call_sink=None,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        self.provider = provider
        self.budget_limit = budget_limit
        self.call_sink = call_sink
        self.temperature = temperature
        self._freshly_fetched: list[dict] = []
        self._fetch_stats: dict | None = None

    # -- subclass hooks ----------------------------------------------------

    def system_prompt(self) -> str:
        raise NotImplementedError

    # ----------------------------------------------------------------------

    def initialize(self, seed: int) -> LLMArmState:
        return LLMArmState(seed=seed)

    def _user_prompt(self, state: LLMArmState) -> str:
        remaining = max(0, self.budget_limit - state.n_unique_evaluated)
        proposal_round = state.llm_calls_made + 1
        if not self.closed_loop:
            return open_loop_user_prompt(proposal_round, remaining)
        return closed_loop_user_prompt(
            proposal_round, remaining,
            [entry.model_dump() for entry in state.archive],
            dict(sorted(state.failure_counts.items())),
            state.n_unique_evaluated, ARCHIVE_TOP_K,
        )

    def _fetch_batch(self, state: LLMArmState) -> list[dict]:
        failures = state.consecutive_parse_failures
        while failures < MAX_CONSECUTIVE_PARSE_FAILURES:
            system = self.system_prompt()
            user = self._user_prompt(state)
            response = self.provider.complete(system, user, self.temperature)
            candidates = parse_candidate_batch(response.raw_text)
            record = LLMCallRecord(
                proposal_id=f"llm_call:{state.llm_calls_made + 1}",
                call_index=0,
                system_prompt=system,
                user_prompt=user,
                model=response.model,
                temperature=self.temperature,
                raw_response=response.raw_text,
                parsed_proposal={"candidates": candidates} if candidates else None,
                validation_errors=[] if candidates else ["unparseable batch response"],
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                estimated_cost_usd=response.estimated_cost_usd,
                latency_seconds=response.latency_seconds,
            )
            if self.call_sink is not None:
                self.call_sink(record)
            self._fetch_stats = {
                "calls": 1,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost": response.estimated_cost_usd or 0.0,
                "parse_failed": candidates is None,
            }
            if candidates:
                return candidates
            failures += 1
            state.consecutive_parse_failures = failures
            state.llm_calls_made += 1  # persisted at next update_state
        raise LLMArmError(
            f"{self.name}: {MAX_CONSECUTIVE_PARSE_FAILURES} consecutive unparseable "
            "LLM responses — failing the cell loudly rather than degrading."
        )

    def propose(self, state: LLMArmState) -> dict:
        if state.buffer:
            candidate = state.buffer[0]
        else:
            self._freshly_fetched = self._fetch_batch(state)
            candidate = self._freshly_fetched[0]
        state.pending_operations = candidate.get("operations")
        return candidate

    def _resource_summary(self, feedback: SearchFeedback) -> tuple[int | None, int | None]:
        cost = feedback.circuit_cost
        if cost is None:
            return None, None
        return cost.gate_count, cost.parameter_count

    def update_state(self, state: LLMArmState, proposal: dict, feedback: SearchFeedback):
        # Two disjoint cases: the proposal was served from the persisted
        # buffer head, or from a fresh fetch made while the buffer was
        # empty (in which case the remainder of the batch becomes the new
        # buffer and the call stats are merged into persisted state).
        if self._freshly_fetched:
            fetched = self._freshly_fetched
            self._freshly_fetched = []
            stats = self._fetch_stats or {}
            self._fetch_stats = None
            buffer = [dict(c) for c in fetched[1:]]
            state = state.model_copy(update={
                "llm_calls_made": state.llm_calls_made + stats.get("calls", 1),
                "llm_input_tokens": state.llm_input_tokens + stats.get("input_tokens", 0),
                "llm_output_tokens": state.llm_output_tokens + stats.get("output_tokens", 0),
                "llm_estimated_cost_usd": (
                    state.llm_estimated_cost_usd + (stats.get("cost") or 0.0)
                ),
                "consecutive_parse_failures": 0,
            })
        else:
            buffer = [dict(c) for c in state.buffer[1:]]

        failure_counts = dict(state.failure_counts)
        outcome_key = feedback.outcome.value
        failure_counts[outcome_key] = failure_counts.get(outcome_key, 0) + 1

        archive = list(state.archive)
        n_unique = state.n_unique_evaluated
        if (
            feedback.val_metric_value is not None
            and feedback.structural_hash is not None
            and not feedback.is_duplicate
        ):
            n_unique += 1
            if all(e.structural_hash != feedback.structural_hash for e in archive):
                n_ops, n_params = self._resource_summary(feedback)
                archive.append(_ArchiveEntry(
                    structural_hash=feedback.structural_hash,
                    val_metric=feedback.val_metric_value,
                    operations=state.pending_operations,
                    n_ops=n_ops if n_ops is not None else (
                        len(state.pending_operations) if state.pending_operations else None
                    ),
                    n_params=n_params,
                ))
                archive = sorted(archive, key=lambda e: e.val_metric)[:ARCHIVE_TOP_K]

        best_hash, best_metric = state.best_hash, state.best_metric
        best_candidate = state.best_candidate
        if feedback.val_metric_value is not None and is_strictly_better(
            feedback.val_metric_value, best_metric, lower_is_better=True
        ):
            best_hash, best_metric = feedback.structural_hash, feedback.val_metric_value
            best_candidate = proposal

        return state.model_copy(update={
            "n_proposed": state.n_proposed + 1,
            "buffer": buffer,
            "archive": archive,
            "failure_counts": failure_counts,
            "n_unique_evaluated": n_unique,
            "pending_operations": None,
            "best_hash": best_hash,
            "best_metric": best_metric,
            "best_candidate": best_candidate,
        })

    def select_final(self, state: LLMArmState) -> str | None:
        return state.best_hash

    def deserialize_state(self, raw_json: str) -> LLMArmState:
        return LLMArmState.model_validate_json(raw_json)

    def llm_usage_summary(self, state: LLMArmState) -> dict:
        from llm_vqc.bench_v2.llm_providers import is_mock_provider

        return {
            "provider": type(self.provider).__name__,
            "model_snapshot": getattr(self.provider, "model_name", None),
            "prompt_version": BENCH_V2_PROMPT_VERSION,
            "temperature": self.temperature,
            "batch_size": BATCH_SIZE,
            "successful_calls": state.llm_calls_made,
            "input_tokens": state.llm_input_tokens,
            "output_tokens": state.llm_output_tokens,
            "estimated_cost_usd": state.llm_estimated_cost_usd,
            "mock": is_mock_provider(self.provider),
        }


class LLMOpenStructureArm(_BaseLLMArm):
    name = "llm_open_structure"
    track = "structure"
    closed_loop = False

    def __init__(self, provider, profile: SpaceProfile, budget_limit: int, **kwargs) -> None:
        super().__init__(provider, budget_limit, **kwargs)
        self.profile = profile

    def system_prompt(self) -> str:
        return structure_system_prompt(self.profile, BATCH_SIZE)


class LLMArchiveClosedStructureArm(LLMOpenStructureArm):
    name = "llm_archive_closed_structure"
    closed_loop = True


class LLMOpenJointArm(_BaseLLMArm):
    name = "llm_open_joint"
    track = "joint"
    closed_loop = False

    def __init__(self, provider, n_qubits: int, budget_limit: int, **kwargs) -> None:
        super().__init__(provider, budget_limit, **kwargs)
        self.n_qubits = n_qubits

    def system_prompt(self) -> str:
        return joint_system_prompt(self.n_qubits, BATCH_SIZE)


class LLMClosedJointArm(LLMOpenJointArm):
    name = "llm_closed_joint"
    closed_loop = True


def register_llm_arms() -> None:
    from llm_vqc.bench_v2.arm_registry import ARM_FACTORIES, register_arm

    for name, cls in (
        ("llm_open_structure", LLMOpenStructureArm),
        ("llm_archive_closed_structure", LLMArchiveClosedStructureArm),
        ("llm_open_joint", LLMOpenJointArm),
        ("llm_closed_joint", LLMClosedJointArm),
    ):
        if name not in ARM_FACTORIES:
            register_arm(name, cls)


register_llm_arms()
