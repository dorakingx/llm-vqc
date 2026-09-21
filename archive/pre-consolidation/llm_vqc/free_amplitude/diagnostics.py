"""Fixed-readout trainability diagnostics (Codex instruction section 12):
because only q0 is measured, some proposed gates may not affect the
prediction at all. This module distinguishes two different things that
are easy to conflate:

- **Structural causal-cone membership** (`compute_q0_causal_cone_indices`):
  a backward-lightcone computation over the ordered operation sequence --
  which operations *could possibly* influence the q0 expectation value,
  for ANY parameter values. A parameter outside the cone has an *exactly*
  zero gradient for every input, by construction (nothing connects it to
  the measured qubit).
- **Empirical near-zero gradient** (`gradient_diagnostics`): a parameter
  *inside* the causal cone can still have a numerically tiny gradient for
  particular parameter values or input data (e.g. a controlled rotation
  whose control qubit happens to be in a Z-eigenstate branch) -- this is a
  real, observed phenomenon, not a bug, and is reported separately from
  cone membership rather than assumed to be the same thing.

Also implements the diagnostics-only all-qubit measurement path (never
entering the training loss or prediction -- see
`llm_vqc.free_amplitude.model`, whose prediction path measures only q0).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from llm_vqc.free_amplitude.schema import PARAMETERIZED_GATES, FreeGateProposal
from llm_vqc.ir.compiler_pennylane import to_qnode
from llm_vqc.ir.expand import build_program
from llm_vqc.ir.schema import CircuitIR, MeasurementSpec

#: Gradient magnitude thresholds for the empirical diagnostics below. These
#: are engineering defaults (not derived from the task), chosen to be well
#: below typical nonzero gradients observed on this Gaussian-regression
#: task while still comfortably above float64 rounding noise.
ZERO_GRADIENT_ATOL = 1e-10
NEAR_ZERO_GRADIENT_ATOL = 1e-4
CONSTANT_PREDICTION_STD_ATOL = 1e-9


def compute_q0_causal_cone_indices_from_ops(operations, readout_qubit: int) -> set[int]:
    """Backward-lightcone operation indices over a generic operations list
    (each element having `.gate` and `.wires`) -- shared by both the
    structure-only `FreeGateProposal` and the main-mode
    `CompleteCandidateProposal`, whose operations differ only by an extra
    `theta` field that is irrelevant to the causal structure.

    Standard reverse-time lightcone construction: start with the single
    relevant wire (the readout), scan operations from last to first, and
    whenever an operation touches an already-relevant wire, mark it as
    causal and add all of its wires to the relevant set (an entangling
    gate can carry information between its wires in either direction).
    """
    relevant_wires = {readout_qubit}
    causal_indices: set[int] = set()
    for i in reversed(range(len(operations))):
        op = operations[i]
        if set(op.wires) & relevant_wires:
            causal_indices.add(i)
            relevant_wires |= set(op.wires)
    return causal_indices


def compute_q0_causal_cone_indices(
    proposal: FreeGateProposal, readout_qubit: int
) -> set[int]:
    """Structure-only convenience wrapper -- see `_from_ops`."""
    return compute_q0_causal_cone_indices_from_ops(proposal.operations, readout_qubit)


@dataclass
class CausalConeSummary:
    causal_operation_indices: set[int]
    parameter_count_in_causal_cone: int
    total_quantum_parameter_count: int

    @property
    def fraction_in_causal_cone(self) -> float:
        if self.total_quantum_parameter_count == 0:
            return 0.0
        return self.parameter_count_in_causal_cone / self.total_quantum_parameter_count


def causal_cone_summary_from_ops(operations, readout_qubit: int) -> CausalConeSummary:
    causal_indices = compute_q0_causal_cone_indices_from_ops(operations, readout_qubit)
    parameterized_positions = [
        i for i, op in enumerate(operations) if op.gate in PARAMETERIZED_GATES
    ]
    in_cone = sum(1 for i in parameterized_positions if i in causal_indices)
    return CausalConeSummary(
        causal_operation_indices=causal_indices,
        parameter_count_in_causal_cone=in_cone,
        total_quantum_parameter_count=len(parameterized_positions),
    )


def causal_cone_summary(proposal: FreeGateProposal, readout_qubit: int) -> CausalConeSummary:
    return causal_cone_summary_from_ops(proposal.operations, readout_qubit)


@dataclass
class GradientDiagnostics:
    zero_gradient_parameter_count: int | None
    near_zero_gradient_parameter_count: int | None
    total_parameter_count: int | None
    no_trainable_parameter_affects_readout: bool | None
    all_gradients_numerically_zero: bool | None


def gradient_diagnostics(
    gradient_vector: list[float] | None, causal: CausalConeSummary
) -> GradientDiagnostics:
    """Empirical zero/near-zero gradient counts from an observed gradient
    vector (e.g. `FreeAmplitudeTrainingOutput.initial_gradient_vector`).
    `None` in, `None` out -- honest "not computed" rather than a fabricated
    zero.
    """
    if gradient_vector is None:
        return GradientDiagnostics(None, None, None, None, None)

    zero_count = sum(1 for g in gradient_vector if abs(g) < ZERO_GRADIENT_ATOL)
    near_zero_count = sum(1 for g in gradient_vector if abs(g) < NEAR_ZERO_GRADIENT_ATOL)
    total = len(gradient_vector)
    return GradientDiagnostics(
        zero_gradient_parameter_count=zero_count,
        near_zero_gradient_parameter_count=near_zero_count,
        total_parameter_count=total,
        no_trainable_parameter_affects_readout=(causal.parameter_count_in_causal_cone == 0),
        all_gradients_numerically_zero=(total > 0 and zero_count == total),
    )


def is_constant_prediction(predictions: np.ndarray) -> bool:
    """True if predictions are (numerically) constant across the batch --
    a candidate whose prediction does not depend on the input at all."""
    return bool(np.std(np.asarray(predictions, dtype=np.float64)) < CONSTANT_PREDICTION_STD_ATOL)


def diagnostic_all_qubit_expectations(
    ir: CircuitIR, inputs, weights
) -> dict[int, float] | str:
    """Compute Z expectation values on EVERY qubit for one sample, using a
    SEPARATE QNode built from a copy of `ir` with `measurements.wires =
    "all"`. This is a diagnostics-only path: its output never enters the
    training loss or the model's `forward()` (see `llm_vqc.free_amplitude.
    model.FixedReadoutQuantumModel`, whose QNode is built directly from the
    original `ir`, unmodified, with a single-wire measurement).

    Returns a `{wire: expval}` dict, or the string `"unavailable"` if the
    diagnostic circuit cannot be compiled (never fabricates a value).
    """
    try:
        diagnostic_ir = ir.model_copy(
            update={"measurements": MeasurementSpec(observable="Z", wires="all")}
        )
        program = build_program(diagnostic_ir)
        qnode = to_qnode(diagnostic_ir, diff_method="backprop")
        raw = qnode(inputs, weights)
        return {wire: float(raw[i]) for i, wire in enumerate(range(program.n_qubits))}
    except Exception:
        return "unavailable"
