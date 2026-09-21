"""Plain result records shared between the mini-demo runner and its
reporting/output-writing module (kept dependency-free of the script itself
so both can import from one place without a script-to-script import).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CandidateRecord:
    """One proposal attempt within one arm, flattened for CSV/report output."""

    arm: str
    proposal_index: int
    proposal_id: str
    compact: dict | None
    valid: bool
    is_duplicate: bool
    structural_hash: str | None
    training_outcome: str
    final_train_mse: float | None
    final_val_rmse: float | None
    circuit_depth: int | None
    two_qubit_gate_count: int | None
    quantum_parameter_count: int | None
    api_latency_seconds: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_trainable_parameters: int | None = None
    train_loss_history: list[float] = field(default_factory=list)
    val_metric_history: list[float] = field(default_factory=list)
    # Process metrics (correction pass, section 7).
    total_gate_count: int | None = None
    training_runtime_seconds: float | None = None
    cache_hit: bool = False
    actually_trained: bool = False
    unique_training_id: str | None = None
    dtype_device_verified: bool | None = None
    # Parameter-change auditing (correction pass, section 6).
    quantum_angles_changed: bool | None = None


@dataclass
class ArmRunOutcome:
    """Everything one arm's B-proposal search produced."""

    arm: str
    candidates: list[CandidateRecord] = field(default_factory=list)
    ledger_summary: dict[str, int] = field(default_factory=dict)
    stop_reason: str | None = None
    selected_structural_hash: str | None = None
    selected_train_seed: int | None = None
    selected_val_rmse: float | None = None
    protected_test_rmse: float | None = None
    # Arm-scoped cache namespace actually used for this arm's candidates
    # (so protected-test lookup and reporting read from the same namespace).
    task_namespace: str | None = None
    # API-level counters (correction pass, section 7). Random arm leaves
    # these at zero (it makes no API calls).
    api_successful_calls: int = 0
    api_failed_calls: int = 0
    api_outbound_attempts: int = 0
