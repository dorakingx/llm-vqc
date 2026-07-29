# Literature Review — Quantum Architecture Search and LLM-Guided Circuit Design

Date: 2026-07-29. Written before protocol freeze (goal §5), as the basis for
`BENCHMARK_V2_PROTOCOL.md` claim discipline.

Provenance tags used throughout:

- **[VERIFIED-WEB]** — checked against the primary arXiv page / publisher /
  official repo during this session (2026-07-29).
- **[REPO-AUDIT]** — taken from the in-repo `LLM-VQC_MASTER_PLAN.md` §2,
  whose author read the paper in full; treated as reliable secondary record.
- **[TRAINING-KNOWLEDGE]** — from model background (pre-2026 papers with
  stable, widely cited content); to be spot-re-verified before any external
  publication that leans on the specific number cited.

## 1. Per-paper structured review

Fields required by the goal: search space / task / qubits / architecture-vs-
parameter treatment / baselines / budget definition / noise setting /
metrics / replication / remaining gap.

### 1.1 Sim, Johnson, Aspuru-Guzik 2019 — "Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms" (arXiv:1905.10876, Adv. Quantum Technol. 2(12)) [TRAINING-KNOWLEDGE]

- **Search space:** no search — a fixed catalog of 19 four-qubit PQC
  templates from the literature.
- **Task:** none (descriptor study, not task optimization).
- **Qubits:** 4 (descriptor definitions are general).
- **Architecture/parameter treatment:** architectures fixed; parameters
  sampled uniformly to estimate state-distribution descriptors.
- **Baselines:** Haar-random states (the reference distribution).
- **Budget:** number of parameter samples for the estimators; not a search
  budget.
- **Noise:** none (statevector).
- **Metrics:** expressibility = KL divergence between the sampled
  state-fidelity distribution and the Haar fidelity distribution;
  entangling capability = mean Meyer–Wallach Q; circuit cost = parameter
  count, gate count, depth.
- **Replication:** estimator sampling only; no seeds/statistics questions
  of the search kind.
- **Gap this project addresses:** Sim et al. define the descriptors but do
  not test whether they predict *task* performance. Our RQ5 treats
  expressibility/Q as **descriptive covariates whose association with task
  metrics must be measured, not assumed** — exactly the discipline the
  descriptor literature itself recommends.

### 1.2 Knipfer, Roman, Matchev, Matcheva, Gleyzer 2026 — "AI Agents for Variational Quantum Circuit Design" (arXiv:2602.19387) [VERIFIED-WEB, details REPO-AUDIT]

- **Search space:** free-form PennyLane code written by a single LLM agent
  (Claude 3.7 Sonnet / Llama 3.3 70B in the Orchestral framework); three
  QNN wrappers (Simple QNN, QuanvNN, Full-Quantum QNN) plus a Lie-EQGNN
  jet-tagging internal-ansatz variant.
- **Task:** synthetic 1-D Gaussian-peak position regression (21-point
  signals); jet tagging (Lie-EQGNN).
- **Qubits:** small (≤ ~6 effective).
- **Architecture/parameter treatment:** agent proposes architecture; angles
  trained by a classical optimizer inside the evaluation pipeline —
  architecture and parameter effects not separated.
- **Baselines:** none of random/evolutionary/greedy/fixed-ansatz kind;
  comparisons are across the agent's own iterations and between LLMs.
- **Budget definition:** none — runs are open-ended agent sessions with
  mid-run human steering.
- **Noise:** simulator.
- **Metrics:** test RMSE + gate/parameter counts, **with feedback computed
  on the test set** (leakage; acknowledged nowhere in the paper).
- **Replication:** single runs; no seeds, no variance, no tests.
- **Gap:** the paper's own limitations list (exploration collapse,
  context-window forgetting, no principled exploration mechanism) plus
  everything quantitative: budget-matched baselines, statistics, leakage
  hygiene. **This is the direct predecessor our benchmark evaluates.**

### 1.3 Sakka, Mizukami, Mitarai 2026 — "An LLM System for Autonomous Variational Quantum Circuit Design" (arXiv:2606.13380, U. Osaka QIQB) [VERIFIED-WEB]

- **Search space:** LLM-generated feature maps and VQE ansätze under
  explicit design constraints; seven-component closed loop (Exploration,
  Generation, Discussion, Validation, Storage, Evaluation, Review) with
  web knowledge acquisition and literature-grounded critique.
- **Task:** image-classification kernels (quantum feature maps) and
  molecular ground-state estimation (7 molecules).
- **Qubits:** small-to-moderate; scaling experiments for the kernel task.
- **Architecture/parameter treatment:** joint pipeline; parameters trained
  downstream; no isolation of architecture-only value.
- **Baselines:** representative fixed feature maps, RBF kernel,
  chemically-inspired and hardware-efficient ansätze — but **no
  budget-matched non-LLM search baselines** (no random/evolutionary QAS at
  equal evaluation count).
- **Budget definition:** not candidate-evaluation-matched.
- **Noise:** simulator.
- **Metrics:** classification accuracy; VQE energy accuracy.
- **Replication:** system-demonstration style; not seed-replicated
  statistics.
- **Gap:** occupies the "system demonstration" slot; the controlled
  evaluation layer (our contribution) remains open.

### 1.4 Zhang, Hsieh, Zhang, Yao — "Differentiable Quantum Architecture Search" (DQAS, arXiv:2010.08561; Quantum Sci. Technol. 7 045023) [TRAINING-KNOWLEDGE]

- **Search space:** a layered operation pool with a continuous relaxation —
  a probabilistic/supernet parameterization over discrete op choices.
- **Task:** QAOA/state preparation and related VQA objectives.
- **Qubits:** small (≤ ~6 in the main studies).
- **Architecture/parameter treatment:** *joint* gradient-based optimization
  of architecture distribution parameters and shared gate angles
  (weight sharing across sampled architectures).
- **Baselines:** random search and fixed ansätze in some experiments;
  budgets defined in optimization steps, not unique candidate evaluations.
- **Noise:** some noisy-simulation demonstrations.
- **Metrics:** task objective (energy/fidelity).
- **Replication:** limited seed reporting.
- **Gap:** weight-sharing confounds architecture quality with shared-angle
  transfer; our Track A explicitly avoids supernet sharing (each unique
  architecture gets its own deterministic inner training), and DQAS-style
  arms are declared **optional** extensions only.

### 1.5 Wang, Ding, Cheng, et al. — "QuantumNAS: Noise-Adaptive Search for Robust Quantum Circuits" (arXiv:2107.10845, HPCA 2022) [TRAINING-KNOWLEDGE]

- **Search space:** SuperCircuit with weight-sharing; subcircuit sampling;
  co-search of ansatz and qubit mapping.
- **Task:** QML classification (vowel/MNIST-family) and VQE.
- **Qubits:** up to ~10 (hardware runs on IBM devices).
- **Architecture/parameter treatment:** weight-sharing supernet then
  evolutionary subcircuit search with noise-aware evaluation; final
  from-scratch retraining of selected circuits.
- **Baselines:** fixed ansätze, random baselines, human designs.
- **Budget:** measured in supernet training + search evaluations; not
  unique-candidate-matched across method families.
- **Noise:** central — real-device noise feedback and noise models.
- **Metrics:** task accuracy on hardware, gate counts after pruning.
- **Replication:** standard DL-style; not paired-seed statistical design.
- **Gap:** engineering-first co-search; does not isolate search-strategy
  value at matched budgets. Informs our E4 (noise/transpilation
  robustness of *selected* circuits, no search on test).

### 1.6 Evolutionary QAS: EVQE (Rattew, Shaydulin, Sun, Pistoia, Wood, arXiv:1910.09694), MoG-VQE (Chivilikhin, Samarin, Ulyantsev, Iorsh, Oganov, Kyriienko, arXiv:2007.04424), EQAS-PQC (Ding & Spector, GECCO 2022) [TRAINING-KNOWLEDGE]

- **Search space:** variable-length gate-sequence genomes (EVQE);
  two-objective topology+rotation genomes under NSGA-II (MoG-VQE);
  PQC building-block sequences (EQAS-PQC).
- **Task:** VQE ground-state chemistry/spin models; PQC benchmarks.
- **Qubits:** 2–8 typical.
- **Architecture/parameter treatment:** evolutionary structure search with
  inner parameter optimization per candidate (EVQE adds parameter
  inheritance asexually; MoG-VQE optimizes angles per individual).
- **Baselines:** fixed ansätze (UCCSD/HEA); occasionally random.
- **Budget:** generations × population; objective-evaluation counting is
  inconsistent across papers.
- **Noise:** EVQE argues noise resilience via shallow circuits; mostly
  simulator.
- **Metrics:** energy error, CNOT count (MoG-VQE Pareto: energy vs #CNOT).
- **Replication:** limited; no paired-seed cross-method design.
- **Gap:** establishes evolutionary search as the *mandatory classical
  competitor class* — any LLM-vs-random-only comparison (Knipfer, ours
  pre-v2) understates the classical bar. Motivates our
  `evolutionary_structure` and the **mandatory `evolutionary_joint`**
  (mixed discrete/continuous chromosome) arms.

### 1.7 He et al. — "Topology-Driven Quantum Architecture Search Framework" (TD-QAS, arXiv:2502.14265; Sci. China Inf. Sci.) [VERIFIED-WEB]

- **Search space:** decoupled: first search circuit *topology*, then
  fine-tune *gate types*, inheriting parameters between phases.
- **Task:** VQA benchmarks (VQE and related).
- **Qubits:** small-to-moderate.
- **Architecture/parameter treatment:** two-phase decoupling; parameter
  inheritance across phases (a weight-transfer confound they accept for
  efficiency).
- **Key empirical claim:** topology matters more than gate type — 76% of
  circuits showed no significant performance change after gate-type
  alteration under noise.
- **Noise:** noisy scenarios included.
- **Gap:** supports our diagnostic interest in entanglement
  pattern/topology descriptors (RQ5) and motivates the optional
  "noise-aware topology-first" arm; not a budget-matched
  strategy-comparison study.

### 1.8 Lipardi, Dibenedetto, Stamoulis, Winands — "Quantum Circuit Design using a Progressive Widening Enhanced Monte Carlo Tree Search" (PWMCTS, arXiv:2502.03962; Adv. Quantum Technol. 2025) [VERIFIED-WEB]

- **Search space:** MCTS over circuit-building actions with a sampling-based
  action-space formulation and progressive widening; **gradient-free** —
  designs both topology and parameters during search.
- **Task:** random-circuit approximation across stabilizer-Rényi-entropy
  (magic) regimes; quantum chemistry; linear-systems applications.
- **Qubits:** small-to-moderate.
- **Architecture/parameter treatment:** joint (topology + parameters inside
  the tree search) — the closest classical analog to our Track B "direct
  joint proposal" setting besides evolutionary joint search.
- **Baselines:** prior MCTS variants.
- **Budget:** tree-search simulations; not cross-family matched.
- **Noise:** simulator.
- **Replication:** robustness across domains reported; not paired-seed.
- **Gap:** motivates our optional MCTS arm; mandatory matrix retains
  evolutionary joint as the classical joint baseline (simpler, more
  standard).

### 1.9 Kundu et al. — "BenchRL-QAS: Benchmarking reinforcement learning algorithms for quantum architecture search" (arXiv:2507.12189; AAAI-SS 2025; code: github.com/azhar-ikhtiarudin/bench-rlqas) [VERIFIED-WEB]

- **Search space:** RL gate-placement environments.
- **Task:** VQE, quantum state diagonalization, variational classification,
  state preparation — a multi-task suite.
- **Qubits:** 2–8 (matches our E3 scaling range).
- **Architecture/parameter treatment:** RL proposes structure; parameters
  trained in the environment loop.
- **Baselines:** 9 RL agents (value-based and policy-gradient) compared to
  each other under noiseless and noisy execution.
- **Budget:** RL training episodes; agents compared within-framework, and
  the paper proposes a weighted ranking metric (accuracy, depth, gate
  count, training time).
- **Replication:** benchmark-style, multiple agents/tasks; RL-internal.
- **Gap:** benchmarks RL-vs-RL, not LLM-vs-classical-search at matched
  unique-candidate budgets; no protected-test discipline of our kind. It
  is the closest existing *benchmark* infrastructure in spirit; our study
  differs by (a) LLM arms, (b) paired-seed budget-matched design, (c)
  architecture-vs-joint-proposal separation (Track A/B).

### 1.10 Tool-constrained scientific agents and schema-validated orchestration [TRAINING-KNOWLEDGE + REPO-AUDIT]

- **FunSearch** (Romera-Paredes et al., Nature 625:468, 2024) and
  **AlphaEvolve** (DeepMind 2025): LLM-as-mutation-operator inside an
  evolutionary loop with programmatic evaluation — the strongest evidence
  that LLM proposal quality should be measured *inside* a controlled
  search scaffold with automatic scoring, exactly our design.
- **QiboAgent** (arXiv:2603.15538) and the Orchestral/Conductor framework
  (Nielsen et al.): agentic quantum-code assistants; free-form code
  action spaces with validation layers.
- **Structured-output / JSON-schema constrained LLM interfaces** (provider
  structured-output modes; pydantic-validated tool I/O): reliability
  literature and practice consistently show schema-constrained action
  spaces cut invalid-action rates and make failures classifiable. Our
  proposal layer is therefore schema-constrained JSON + a deterministic
  validator/canonicalizer that records every issue (goal §10), not
  free-form code generation (the Knipfer/Sakka action space).

### 1.11 Adjacent benchmarks noted for claim discipline

- **QAS-Bench** (Lu et al., ICML 2023): unitary-reproduction and
  circuit-optimization QAS benchmark — task family disjoint from ours.
- **GQE/GPT-QE** (Nakaji et al. 2024, arXiv:2401.09253): trains a
  transformer *as* the generator (domain-trained, not a frozen
  general-purpose LLM); different question from ours (frozen-LLM prior
  value at matched budget).
- **HamQASBench** (arXiv:2607.04845): Hamiltonian-informed QAS diagnostics
  [VERIFIED-WEB, surfaced in search]; descriptor-oriented, not
  LLM-vs-classical.

## 2. Cross-cutting synthesis

1. **No prior work compares a frozen general-purpose LLM proposer against
   random, evolutionary, greedy, AND fixed-reference arms at matched
   unique-candidate-evaluation budgets with paired seeds and protected
   test discipline.** Knipfer and Sakka lack non-LLM search baselines;
   DQAS/QuantumNAS/TD-QAS lack LLM arms and use weight sharing;
   BenchRL-QAS compares RL agents only.
2. **No prior work separates architecture-proposal quality from joint
   structure+parameter proposal quality** (our Track A vs Track B). DQAS,
   QuantumNAS, TD-QAS all entangle the two via weight sharing/inheritance;
   PWMCTS is joint-only; Knipfer/Sakka train parameters downstream of
   architecture proposals without an isolation ablation.
3. **Descriptor discipline:** Sim-style expressibility/entanglement are
   descriptors, not objectives. TD-QAS's topology-dominance finding makes
   entanglement-pattern covariates worth recording, but association must
   be measured per-task with circuit-size controls (RQ5).
4. **Budget accounting is the field's weakest point:** budgets appear as
   RL episodes, generations, optimizer steps, or agent iterations —
   rarely unique candidate evaluations, and duplicates are never
   accounted. Our ledger design (proposals vs valid vs unique vs
   duplicates vs failures, plus LLM tokens/cost and optimizer calls) is
   itself a contribution.

## 3. Claim-to-evidence table (frozen wording for the final report)

| # | Claim we intend to be able to make | Evidence that will support it | What we must NOT claim |
|---|---|---|---|
| C1 | "At matched unique-candidate budgets on this suite, LLM-guided structure search {does/does not} outperform random/evolutionary/greedy/fixed references (per-task effect sizes + CIs)." | E2 paired replicates, Holm-corrected paired tests, effect sizes | Any generalization beyond the task suite, qubit range, model snapshot, and prompt version tested |
| C2 | "Archive-based closed-loop feedback {changes/does not change} LLM search performance and sample efficiency vs open-loop." | E2 (+E1 for joint track), anytime curves + AUC of best-so-far | Mechanistic claims about *why* the LLM behaves differently without ablation evidence |
| C3 | "When theta is proposed verbatim (no optimizer), LLM joint proposals {beat/match/lose to} random joint sampling and a classical mixed-chromosome evolutionary search." | E1 + E5 exact objective-call-matched comparisons | That verbatim-theta results transfer to trained-theta settings (that is Track A's question) |
| C4 | "Method rankings {are/are not} stable from 3 to 8 qubits on T1/T2." | E3, ≥5 paired replicates per cell, per-n rankings with uncertainty | Asymptotic scaling laws from 4 points |
| C5 | "Expressibility/MW-Q/gradient/causal-cone descriptors {are/are not} associated with test performance after circuit-size control." | RQ5 regression/correlation analyses with size covariates + uncertainty | 'Lower KL / higher Q is better' as a design rule |
| C6 | "Method X populates the best validation/test-vs-resource Pareto frontier under logical and transpiled cost metrics." | E2/E3 Pareto fronts (test metric vs transpiled 2q count/depth) | Hardware-performance claims (we simulate; transpiled costs are estimates under fixed coupling profiles) |
| C7 | Task-difficulty context: "classical sanity baselines achieve Y." | Phase 1 baseline table per task | **Any quantum-advantage claim (prohibited outright)** |
| C8 | Novelty: "To our knowledge, the first budget-matched, seed-paired comparison of frozen-LLM structure and joint proposal against random/evolutionary/greedy/fixed-reference QAS with architecture-vs-joint separation on a controlled synthetic suite." | §1–2 documented search (this file), qualified "to our knowledge", scope-limited | Unqualified "first LLM-QAS benchmark" (BenchRL-QAS, QAS-Bench exist; Knipfer/Sakka are LLM-QAS systems) |

## 4. Sources

Primary pages verified this session: arXiv:2602.19387, arXiv:2606.13380,
arXiv:2507.12189 (+ github.com/azhar-ikhtiarudin/bench-rlqas),
arXiv:2502.03962 (Adv. Quantum Technol. 2025), arXiv:2502.14265
(Sci. China Inf. Sci.), arXiv:2607.04845. Pre-2026 canonical papers cited
from training knowledge: arXiv:1905.10876 (Sim), arXiv:2010.08561 (DQAS),
arXiv:2107.10845 (QuantumNAS), arXiv:1910.09694 (EVQE), arXiv:2007.04424
(MoG-VQE), Ding & Spector GECCO 2022, Lu et al. ICML 2023 (QAS-Bench),
arXiv:2401.09253 (GQE), Romera-Paredes et al. Nature 2024 (FunSearch).
