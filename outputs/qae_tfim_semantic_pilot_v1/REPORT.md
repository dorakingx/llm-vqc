# QAE-TFIM semantic-prior pilot — report

## Headline
On a frozen 4-qubit quantum-autoencoder verification task, the task-aware GPT-5.6 Sol proposal pool achieved mean selected protected-test trash fidelity **0.9676**, versus **0.9181** for unconstrained random search and **0.9474** for a stronger RY-only random topology control.

Across 12 independent train/validation seeds, the LLM-selected circuit beat ordinary random in **10/12** seeds and the RY-only control in **11/12** seeds.

## Why this task
Earlier controlled VQC results showed that when all methods search essentially the same small gate grammar, Random / Evolutionary / Greedy can become practically equivalent. A language model has little useful semantic information to exploit. QAE changes the question: the proposer sees the Hamiltonian, the latent/trash split, the fact that the ground states are real, and the local interaction graph. These facts can be translated into circuit design choices.

## Result
| Method | Mean test fidelity | Median | SD |
|---|---:|---:|---:|
| Random | 0.9181 | 0.9166 | 0.0357 |
| RY-random | 0.9474 | 0.9468 | 0.0166 |
| LLM-semantic | **0.9676** | **0.9679** | 0.0029 |

Paired fidelity gains:
- LLM − Random: mean **+0.0495**, bootstrap 95% CI **[0.0297, 0.0693]**, exact Wilcoxon **p=0.00244**.
- LLM − RY-random: mean **+0.0203**, bootstrap 95% CI **[0.0114, 0.0294]**, exact Wilcoxon **p=0.00244**.

The LLM pool selected `L2_pair_then_funnel` in 11/12 seeds and `L4_latent_targets` once.

## What the RY control tells us
The gap Random → RY-random is large, so a substantial part of the benefit comes from the simple, explainable prior that the target state family is real and RY rotations are a natural parameterization. The remaining LLM → RY-random gain tests topology reasoning after that prior is held fixed; it remains positive in 11/12 verification seeds.

## What can be claimed now
Supported, for this pilot only:
1. A task-aware semantic proposal pool can outperform an equal-budget random circuit sampler on a QAE task where the prompt exposes physically meaningful structure.
2. The advantage is not explained only by choosing RY instead of X/Y/Z: the LLM pool also outperforms an RY-only random topology control in this verification run.
3. The winning motif is interpretable: pair local degrees of freedom, then route information from trash toward latent qubits.

Not supported:
- no general LLM-superiority claim;
- no quantum advantage;
- no hardware/noise claim;
- no claim that GPT-5.6 Sol is uniquely responsible (the pool has not yet been replayed across models);
- no claim beyond four qubits or this Ising ground-state family.

## Next confirmatory steps
1. Replay the exact prompt through version-pinned API models and store raw call provenance.
2. Add Evolutionary and a hand-designed tree-tensor-network/QAE baseline.
3. Increase to 6 and 8 qubits, vary latent size, and add noise/hardware topology.
4. Use independent state families (Heisenberg, molecular ground states).
5. Run a blinded, pre-frozen multi-model/multi-seed benchmark before a publication claim.
