# QAE prior-matched baselines — protocol (frozen before execution)

Status: committed before any result of this study was computed.
Disclosure: the arm definitions below were chosen *after* reading the
structure of the LLM proposals already committed under
`outputs/qae_robustness/` (nearest-neighbour CNOTs in 77–100% of LLM
open-pool proposals, RY in 42–100% of rotations). No candidate of the arms
below had been trained when this file was written. No LLM API call is made.

## Question

Is the LLM-Open advantage over uniform Random reproduced by a *hand-coded*
proposal distribution that encodes the same stated facts (real amplitudes →
RY; nearest-neighbour Ising chain → nearest-neighbour CNOTs), with no
language model? Which of the stated facts carries the effect, and is the
gate *ordering* the LLM chooses important beyond its gate multiset?

## Fixed elements (identical to `QAE_ROBUSTNESS_PROTOCOL.md`)

Capacity rule (3n rotations, n CNOTs, 4n ordered gates), latent = first n/2
qubits, trash = last n/2, state streams `state_sets(seed, family, n)`, 12
paired seeds 0–11, trainer (Adam lr 0.05, 60 epochs, U(−0.05, 0.05) init,
training seed `train_seed(seed, architecture)`, best-validation checkpoint),
selection by lowest validation loss, held-out test `F_trash` read after
selection. The trainer is the tensor-contraction re-implementation in
`llm_vqc/experiments/qae_baselines/fast_eval.py`, verified against the frozen
dense trainer to ≤1e-12.

Conditions: `reference` (n=4 TFIM), `qubits_n6`, `qubits_n8`,
`hamiltonian_xxz`. Budget conditions B ∈ {4, 16} reuse the n=4 TFIM pools.

## Arms (all non-LLM unless marked *diagnostic*)

| arm | rotation axis | rotation qubit | CNOT pair | order |
|---|---|---|---|---|
| `Uniform` | uniform {X,Y,Z} | uniform | uniform directed pair | uniform permutation |
| `RY` | Y only | uniform | uniform directed pair | uniform permutation |
| `NN` | uniform {X,Y,Z} | uniform | uniform nearest-neighbour directed pair | uniform permutation |
| `RY+NN` | Y only | uniform | uniform nearest-neighbour directed pair | uniform permutation |
| `Hand` | two fixed templates (below), B = 2 | | | |
| `LLM-shuffled` (*diagnostic*) | the condition's committed LLM-Open pool, each member's gate list uniformly permuted | | | |

`Hand` templates (n qubits, h = n/2):
- `ladder-in`: RY on every qubit; CNOT q_{i+1}→q_i for i = n−2 … 0; RY on every
  qubit; CNOT q_h→q_{h−1}; RY on every qubit.
- `ladder-out`: the same with every CNOT reversed.

Random streams: `numpy.random.default_rng([20261011, arm_index, n, family_index, seed])`.

Pool sizes per seed: `Uniform` 64, each prior arm 32, `LLM-shuffled` 4 shuffles
of each of the 8 pool members (32).

## Primary pre-declared analysis

The B-budget arm of a pool is its **first B draws** (B = 8 primary; 4 and 16
at n=4 TFIM). Selected = lowest validation loss. Paired per-seed contrasts with
the robustness study's conventions (bootstrap 95% CI, exact two-sided
Wilcoxon, Cohen's dz, wins), reported for:
`RY+NN − Uniform`, `LLM-Open − RY+NN`, `LLM-Closed − RY+NN`.
Holm correction is applied across the 4 conditions × 3 contrasts.

Secondary (descriptive): best-of-B curves (expected selected test `F_trash`
over 2,000 random B-subsets of each pool, B = 1 … pool size), `RY − Uniform`,
`NN − Uniform`, `LLM-shuffled` vs LLM-Open, `Hand`.
