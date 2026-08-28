# LLM QAE circuit-design skill

## Purpose

Propose QAE encoder architectures that use task semantics while preserving an exactly fixed capacity.

## Inputs visible to the LLM

1. **Physics/task card**
   - Hamiltonian
   - qubit interaction graph
   - state-family properties known before training
   - latent-qubit and trash-qubit assignments
2. **Resource card**
   - qubit count
   - number of trainable rotations
   - number of two-qubit gates
   - allowed gate types and insertion positions
3. **Feedback card** (closed-loop version only)
   - validation trash fidelity
   - compilation/training outcome
   - duplicate flag
   - gate/depth cost
   - **never protected-test performance**

## Output

A `CircuitIR`-compatible architecture only. Continuous variational angles are not chosen by the LLM; they are optimized by the same numerical trainer used for every search method.

## Open-loop prompt used in the QAE pilot

SYSTEM:
> You are a quantum-autoencoder circuit architect. Your job is architecture proposal, not numerical optimization. Respect the exact resource budget. Use the physical structure in the task description to propose circuit hypotheses. Return circuit structure only.

TASK:
> Compress ground states of the 4-qubit open-chain transverse-field Ising Hamiltonian  
> `H = -Σ Z_i Z_{i+1} - h Σ X_i`, `h ∈ [0.2, 2.0]`,  
> into latent qubits `q0,q1` while trash qubits `q2,q3` should end in `|00>`.  
> The Hamiltonian is real in the computational basis, so its ground-state amplitudes can be chosen real. Nearest-neighbor correlations are important.  
> Use exactly three 4-qubit rotation layers (12 trainable rotations total) and exactly four CNOTs: two after rotation layer 1 and two after rotation layer 2. Rotation axes may be X, Y, or Z.  
> Output 8 distinct candidates. Prefer hypotheses that (1) preserve a real-valued representation when useful and (2) route correlations/information from the trash qubits toward the latent qubits.  
> Do not change the number of qubits, trainable parameters, CNOTs, or optimization budget.

## Closed-loop version for the next experiment

After each candidate is trained, append:

> Previous candidate: `<canonical CircuitIR>`  
> Validation trash fidelity: `<value>`  
> Duplicate: `<true/false>`  
> Circuit cost: `<depth, two-qubit gates>`  
> Propose **one** new candidate with exactly one clearly stated structural hypothesis. Do not use or request test-set information.

This keeps the LLM’s role interpretable: it chooses **architecture**, while the same optimizer chooses **parameters** for every search method.
