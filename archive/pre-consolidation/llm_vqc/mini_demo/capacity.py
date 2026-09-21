"""Hard, fail-loud capacity check (Codex instruction section 2):

"Fail loudly if any candidate does not have exactly: 2 qubits; 4 quantum
parameters; 51 total trainable parameters."

By construction, every IR `llm_vqc.mini_demo.compact_schema.
compact_to_circuit_ir` can produce already has exactly this shape, so
this check should never fire in normal operation -- it exists to turn a
silent regression in that converter (or in the fixed task/model wiring)
into an immediate, loud crash instead of a quietly-wrong experiment.
This is therefore a hard `raise`, not a `ValidationIssue` fed back into
the ordinary invalid/duplicate/failed proposal bookkeeping.
"""

from __future__ import annotations

from llm_vqc.evaluation.model import HybridQNNModel
from llm_vqc.ir.metrics import circuit_cost_summary
from llm_vqc.ir.schema import CircuitIR
from llm_vqc.tasks.base import TaskSpec

EXPECTED_N_QUBITS = 2
EXPECTED_QUANTUM_PARAMETERS = 4
EXPECTED_TOTAL_TRAINABLE_PARAMETERS = 51


class FixedCapacityViolation(RuntimeError):
    """Raised when a candidate circuit does not match the demo's fixed
    2-qubit / 4-quantum-parameter / 51-total-parameter contract."""


def verify_fixed_capacity(ir: CircuitIR, task_spec: TaskSpec) -> None:
    """Build a throwaway model for `ir` and hard-assert its parameter
    counts. Raises `FixedCapacityViolation` (never returns issues) on any
    mismatch -- see module docstring for why this is a hard failure.
    """
    if ir.n_qubits != EXPECTED_N_QUBITS:
        raise FixedCapacityViolation(
            f"expected n_qubits={EXPECTED_N_QUBITS}, got {ir.n_qubits}"
        )

    cost = circuit_cost_summary(ir)
    if cost.parameter_count != EXPECTED_QUANTUM_PARAMETERS:
        raise FixedCapacityViolation(
            f"expected {EXPECTED_QUANTUM_PARAMETERS} quantum parameters, "
            f"got {cost.parameter_count}"
        )

    model = HybridQNNModel(
        ir, raw_feature_dim=task_spec.raw_feature_dim, head_out_dim=task_spec.classical_head_out_dim
    )
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if total_trainable != EXPECTED_TOTAL_TRAINABLE_PARAMETERS:
        raise FixedCapacityViolation(
            f"expected {EXPECTED_TOTAL_TRAINABLE_PARAMETERS} total trainable parameters, "
            f"got {total_trainable}"
        )
