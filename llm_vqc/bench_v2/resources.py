"""Resource metrics and robustness evaluation for selected circuits
(protocol §9, E4; contract C11/C12).

- `logical_resource_summary`: backend-independent counts from the shared
  IR (gate/depth/1q/2q/controlled-rotation/parameter counts, causal-cone
  fraction) — computed for EVERY evaluated candidate.
- `transpiled_resource_summary`: Qiskit transpilation of the SEARCHED
  BODY under fixed coupling profiles (line / ring / all_to_all), fixed
  basis [rz, sx, x, cx], optimization_level 1, fixed transpile seed.
  Amplitude state-preparation is deliberately excluded (its decomposition
  would swamp the body signal; it is identical for every candidate at a
  given n and already tracked separately by `structural_cost_summary`).
  SWAPs inserted by routing are decomposed into the cx count by the
  basis, so `transpiled_two_qubit_count` is the native-2q cost including
  routing overhead.
- `finite_shot_eval`: the selected circuit's metric under 1,024 / 4,096
  shots (seeded sampler device).
- `noisy_eval`: the frozen `depol_ro_v1` profile — single-qubit
  depolarizing p1 after every 1q body gate, depolarizing p2 applied to
  BOTH wires of every 2q body gate (a stated 2q-noise approximation, not
  a two-qubit channel), and a readout bit-flip before measurement.

E4 runs ONLY on already-selected circuits; nothing here feeds back into
search or selection.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
import torch
from qiskit import transpile
from qiskit.transpiler import CouplingMap

from llm_vqc.free_amplitude.model import FixedReadoutQuantumModel
from llm_vqc.ir.compiler_pennylane import to_qnode
from llm_vqc.ir.expand import EntangleInstruction, RotationInstruction, build_program
from llm_vqc.ir.metrics import circuit_cost_summary
from llm_vqc.ir.schema import CircuitIR

TRANSPILE_BASIS = ("rz", "sx", "x", "cx")
TRANSPILE_SEED = 7            # frozen (protocol_v2.yaml E4)
TRANSPILE_OPT_LEVEL = 1
COUPLING_PROFILES = ("line", "ring", "all_to_all")
NOISE_PROFILE_V1 = {"name": "depol_ro_v1", "p1": 0.001, "p2": 0.01, "readout": 0.02}
CONTROLLED_ROTATION_GATES = frozenset({"CRX", "CRY", "CRZ"})


def logical_resource_summary(ir: CircuitIR) -> dict:
    cost = circuit_cost_summary(ir)
    program = build_program(ir)
    one_q = sum(1 for i in program.body if isinstance(i, RotationInstruction))
    two_q = sum(1 for i in program.body if isinstance(i, EntangleInstruction))
    controlled_rot = sum(
        1 for i in program.body
        if isinstance(i, EntangleInstruction) and i.gate in CONTROLLED_ROTATION_GATES
    )
    return {
        "logical_gate_count": cost.gate_count,
        "logical_depth": cost.depth,
        "one_qubit_count": one_q,
        "two_qubit_count": two_q,
        "controlled_rotation_count": controlled_rot,
        "parameter_count": cost.parameter_count,
    }


def _coupling_map(profile: str, n_qubits: int) -> CouplingMap | None:
    if profile == "all_to_all":
        return None
    edges = [[i, i + 1] for i in range(n_qubits - 1)]
    if profile == "ring" and n_qubits > 2:
        edges.append([n_qubits - 1, 0])
    both_ways = edges + [[b, a] for a, b in edges]
    return CouplingMap(couplinglist=both_ways)


def _body_only_circuit(ir: CircuitIR):
    """The searched body as a standalone Qiskit circuit (no state prep)."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter

    program = build_program(ir)
    circuit = QuantumCircuit(ir.n_qubits)
    params = [Parameter(f"w{i}") for i in range(program.num_parameters)]
    gate_map_1q = {"RX": circuit.rx, "RY": circuit.ry, "RZ": circuit.rz}
    gate_map_2q = {"CRX": circuit.crx, "CRY": circuit.cry, "CRZ": circuit.crz}
    for instr in program.body:
        if isinstance(instr, RotationInstruction):
            if instr.gate == "H":
                circuit.h(instr.wire)
            else:
                gate_map_1q[instr.gate](params[instr.param_index], instr.wire)
        else:
            if instr.gate == "CNOT":
                circuit.cx(instr.control, instr.target)
            elif instr.gate == "CZ":
                circuit.cz(instr.control, instr.target)
            else:
                gate_map_2q[instr.gate](
                    params[instr.param_index], instr.control, instr.target
                )
    return circuit


def transpiled_resource_summary(
    ir: CircuitIR, coupling_profile: str, transpile_seed: int = TRANSPILE_SEED
) -> dict:
    body = _body_only_circuit(ir)
    transpiled = transpile(
        body,
        coupling_map=_coupling_map(coupling_profile, ir.n_qubits),
        basis_gates=list(TRANSPILE_BASIS),
        optimization_level=TRANSPILE_OPT_LEVEL,
        seed_transpiler=transpile_seed,
    )
    ops = transpiled.count_ops()
    return {
        "coupling_profile": coupling_profile,
        "transpile_seed": transpile_seed,
        "basis_gates": list(TRANSPILE_BASIS),
        "transpiled_depth": transpiled.depth(),
        "transpiled_gate_count": sum(ops.values()),
        "transpiled_two_qubit_count": int(ops.get("cx", 0)),
        "transpiled_swap_count": int(ops.get("swap", 0)),  # 0 when decomposed into cx
    }


def _predictions_to_metrics(pred: np.ndarray, targets: np.ndarray) -> dict:
    residual = pred - targets
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
    }


def finite_shot_eval(
    ir: CircuitIR,
    theta: list[float],
    features: np.ndarray,
    targets: np.ndarray,
    shots: int,
    shot_seed: int,
    readout_qubit: int = 0,
) -> dict:
    """Selected-circuit metric under a finite-shot sampler device."""
    device = qml.device("default.qubit", wires=ir.n_qubits, shots=shots, seed=shot_seed)
    qnode = to_qnode(ir, diff_method=None, device=device)
    weights = np.asarray(theta, dtype=np.float64)
    pred = np.array([
        (1.0 - float(qnode(torch.tensor(x, dtype=torch.float64), weights)[0])) / 2.0
        for x in features
    ])
    return {
        "shots": shots, "shot_seed": shot_seed,
        **_predictions_to_metrics(pred, targets),
    }


def noisy_eval(
    ir: CircuitIR,
    theta: list[float],
    features: np.ndarray,
    targets: np.ndarray,
    profile: dict = NOISE_PROFILE_V1,
    readout_qubit: int = 0,
) -> dict:
    """Selected-circuit metric under the frozen depol_ro_v1 profile on a
    density-matrix simulator (exact expectation of the noisy channel; no
    shot noise on top)."""
    program = build_program(ir)
    p1, p2, p_ro = profile["p1"], profile["p2"], profile["readout"]
    device = qml.device("default.mixed", wires=ir.n_qubits)
    weights = np.asarray(theta, dtype=np.float64)

    @qml.qnode(device)
    def qnode(x):
        qml.AmplitudeEmbedding(x, wires=range(ir.n_qubits), normalize=True, pad_with=0.0)
        for instr in program.body:
            if isinstance(instr, RotationInstruction):
                if instr.gate == "H":
                    qml.Hadamard(wires=instr.wire)
                else:
                    getattr(qml, instr.gate)(weights[instr.param_index], wires=instr.wire)
                qml.DepolarizingChannel(p1, wires=instr.wire)
            else:
                if instr.gate == "CNOT":
                    qml.CNOT(wires=[instr.control, instr.target])
                elif instr.gate == "CZ":
                    qml.CZ(wires=[instr.control, instr.target])
                else:
                    getattr(qml, instr.gate)(
                        weights[instr.param_index], wires=[instr.control, instr.target]
                    )
                qml.DepolarizingChannel(p2, wires=instr.control)
                qml.DepolarizingChannel(p2, wires=instr.target)
        qml.BitFlip(p_ro, wires=readout_qubit)
        return qml.expval(qml.PauliZ(readout_qubit))

    pred = np.array([(1.0 - float(qnode(x))) / 2.0 for x in features])
    return {
        "noise_profile": dict(profile),
        **_predictions_to_metrics(pred, targets),
    }


def exact_eval(
    ir: CircuitIR, theta: list[float], features: np.ndarray, targets: np.ndarray,
    readout_qubit: int = 0,
) -> dict:
    """Noise-free reference on the same selected circuit (E4 baseline row)."""
    model = FixedReadoutQuantumModel(ir, readout_qubit=readout_qubit)
    model.double()
    with torch.no_grad():
        model.q_layer.weights.copy_(torch.tensor(theta, dtype=torch.float64))
        pred = model(torch.tensor(features, dtype=torch.float64)).cpu().numpy().reshape(-1)
    return _predictions_to_metrics(pred, targets)
