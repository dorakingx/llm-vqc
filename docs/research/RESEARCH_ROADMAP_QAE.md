# Research roadmap: from controlled VQC search to semantic QAE search

## Central research question

**When does language-model reasoning actually help variational quantum-circuit architecture search?**

The project originally asked whether an LLM can propose better VQC gate structures than random or evolutionary search. The controlled experiments showed that this question is too broad: when the search task contains little semantic structure, different search strategies can be practically equivalent.

The new roadmap makes the role of semantics explicit.

## Step 1 — Establish a fair search benchmark (completed)

Capacity-controlled T1/T2 fixed the main confounders:
- identical parameter budgets across candidate circuits,
- identical training and validation pipelines,
- protected test data not exposed during search,
- equal candidate-evaluation budgets,
- explicit classical and analytical baselines.

On T1-v2, Random / Evolutionary / Greedy were practically equivalent within the frozen margin, even though selecting the best of a budget of candidates helped. This says that *architecture selection can matter while the particular search algorithm does not*.

## Step 2 — Diagnose task degeneracy before comparing searchers (completed)

The first HIGGS study showed that a difficult-looking real dataset can still be a bad architecture-search benchmark if the learning pipeline is under-qualified. The follow-up data-scale study traced the near-baseline behavior to:
1. too few training examples,
2. an inherited learning rate that was too large,
3. information loss from PCA(8).

After increasing the data scale, using raw features, and auditing the optimizer, trainable quantum angles and entanglement became mildly useful again. The lesson is methodological: **qualify the task before judging the search strategy**.

The HIGGS branches are retained as archives because their Git ancestry is disconnected from `main`. Their scientific conclusions are summarized here instead of force-merging their history.

## Step 3 — Give the LLM a decision that can use meaning (new QAE study)

A Quantum Autoencoder (QAE) provides an architecture-search task with explicit semantic structure:
- which qubits are latent and which are trash,
- which correlations should be routed toward the latent subsystem,
- which gate axes match the state family,
- which entanglement topology matches the Hamiltonian interaction graph.

The first QAE task compresses 4-qubit transverse-field Ising ground states to 2 latent qubits. All methods receive exactly 12 trainable rotations and 4 CNOTs. The LLM is given the Hamiltonian, the latent/trash assignment, and the fact that the ground states can be chosen real.

This isolates a new hypothesis:

> **Semantic-prior hypothesis.** An LLM gains sample efficiency when the prompt contains task information that can be converted into useful circuit-architecture priors.

## Step 4 — Controls that distinguish LLM reasoning from a trivial heuristic

The QAE pilot includes:
- **Random**: axes and CNOT topology both random.
- **RY-only random**: all rotations fixed to RY, but CNOT topology remains random.
- **LLM semantic pool**: uses the same exact capacity but may use the Hamiltonian and latent/trash roles.

RY-only random is important. If the LLM only beats ordinary random, the result could be explained by one simple heuristic: "the target states are real, so use RY." Beating the RY-only control tests whether topology reasoning adds value beyond that heuristic.

## Step 5 — Confirm before generalizing

The current QAE result is an exploratory/frozen verification pilot, not a final publication claim. Before claiming general LLM superiority:
1. ~~replay the exact prompt through version-pinned APIs and save raw provenance~~ — **done** (v2: `gpt-5.4-mini-2026-03-17`, open- and closed-loop, full call provenance in `outputs/qae_tfim_api_v2/`),
2. ~~add Evolutionary and a hand-designed QAE baseline~~ — **done** (v2; both underperform even unconstrained Random at B=8),
3. repeat on 6–8 qubits and other Hamiltonians,
4. add noise and hardware connectivity,
5. freeze a confirmatory multi-model protocol (several pinned models, pre-registered) before inspecting protected test results.

The v2 verification confirmed the pilot: the API open-loop pool (mean test trash
fidelity 0.9700) beats Random (+0.0518, 11/12 seeds), the RY-only control
(+0.0276, 12/12), Evolutionary (+0.0695, 11/12), and the hand-designed
reference (+0.0811, 12/12). Closed-loop feedback added no measurable value over
open-loop priors at this budget (23/96 proposals were duplicates).

## Step 6 — Neutral-space 2×2 (v3, primary experiment, completed)

v3 (`docs/research/QAE_PROTOCOL_V3.md`, pre-registered) removed the rigid
layer layout (free ordering of 12 rotations + 4 CNOTs) and reduced the
comparison to a clean 2×2 over semantics × validation feedback:
Random / Greedy / LLM-Open / LLM-Closed, B=8, paired seeds, pinned
`gpt-5.4-mini-2026-03-17`. Result (`outputs/qae_tfim_neutral_v3/REPORT.md`):
**semantics is the decisive axis** — LLM-Open 0.9653 vs Random 0.8721
(+0.0932, 12/12 seeds, p=0.00049, dz=1.73); LLM-Closed 0.9553 beats both
non-semantic arms 12/12; Greedy 0.7667 lands *significantly below* Random —
single-start local refinement sacrificed exploration breadth at B=8 — and
no additional closed-loop benefit over open-loop was detected (p=0.68,
26/96 duplicate/invalid proposals). v3 is preserved unchanged as the
single-start diagnostic.

## Step 7 — Multi-start 4+4 follow-up (v4, primary experiment, completed)

Motivated by the v3 diagnosis and pre-registered before running
(`docs/research/QAE_PROTOCOL_V4.md`, commit 69f8335): Greedy and
LLM-Closed get 4 diverse warm starts before 4 refinement evaluations;
Random and LLM-Open unchanged in design (fresh draws/pool). Result
(`outputs/qae_tfim_neutral_v4/REPORT.md`): the diagnosis holds —
**Greedy recovers to Random-parity** (0.8793 vs 0.8677, p=0.42) with a
genuine refinement gain (+0.0635, 10/12 seeds, p=0.009); the **semantic
advantage replicates with an independent pool** (LLM-Open 0.9643, 12/12
vs Random, dz=3.5); LLM-Closed's duplicates drop 26/96 → 8/96 and its
refinement gain is small but positive (+0.0185), yet **no detectable
closed-loop benefit over the open-loop batch** (−0.0085, p=0.13).

## Paper story

The paper is no longer "LLM beats random at VQC design."

The stronger story is:

1. Controlled benchmarks reveal that search strategies are interchangeable when the task exposes little semantic information.
2. Task qualification explains why misleading negative or positive results appear.
3. A QAE benchmark exposes meaningful physical structure to the proposer.
4. Under exact capacity control, a task-aware language-model proposal pool becomes more sample-efficient than random controls.
5. The boundary of the claim is explicit: the benefit is **conditional on useful semantic priors**.
