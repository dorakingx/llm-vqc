# BENCHMARK v2 PROTOCOL — Budget-Matched LLM-vs-Classical VQC Architecture Search

Status: **FROZEN at commit time** (see git history for the freeze SHA).
Changes after the Phase 6 gate require a new protocol version
(`bench_v2.1`, new namespaces) and must never follow inspection of
protected-test results. Machine-readable mirror:
`configs/bench_v2/protocol_v2.yaml`.

## 0. North-star question

At equal candidate-evaluation and computational budgets, when and why does
LLM guidance improve variational quantum circuit architecture search
compared with random, evolutionary, greedy, and fixed-reference methods —
and how does the conclusion change with task family, qubit count, resource
constraints, and whether the LLM proposes only architecture (Track A) or
jointly proposes architecture and numerical gate angles (Track B)?

## 1. Research questions and preregistered hypotheses

Hypotheses are directional *expectations*, not conclusions; null and
contradictory outcomes are valid results.

- **RQ1 (architecture search).** At equal unique-candidate budget, does
  LLM-guided structure search outperform random, evolutionary, greedy, and
  fixed-reference ansätze on validation-selected protected-test
  performance?
  *H1: LLM-open beats random at small budgets (prior transfer) on T1/T2;
  evolutionary catches up by B=24; fixed references are hard to beat at
  n=5.*
- **RQ2 (feedback).** Does archive-based closed-loop feedback beat
  open-loop LLM sampling; does it improve sample efficiency or only
  variance?
  *H2: closed-loop improves best-so-far AUC (sample efficiency) modestly
  and increases across-seed variance.*
- **RQ3 (joint theta proposal).** With theta evaluated verbatim, does the
  LLM add value beyond random complete-candidate sampling and classical
  mixed discrete/continuous evolutionary search?
  *H3: LLM-joint beats random-joint at B≤8; evolutionary-joint wins at
  B=16.*
- **RQ4 (scaling).** How do rankings change from 3 to 8 qubits; do
  resource costs grow differently by method?
  *H4: rankings compress toward the fixed reference as n grows at fixed
  B; LLM circuit resource costs grow faster than evolutionary ones.*
- **RQ5 (diagnostics).** Are expressibility (KL), Meyer–Wallach Q,
  gradient statistics, or causal-cone coverage associated with task
  performance after controlling for circuit size? Empirical association
  only; never assumed proxies.
  *H5: after size control, causal-cone fraction shows the strongest
  association; expressibility shows none consistent.*
- **RQ6 (resource trade-offs).** Which methods populate the best
  validation/test-vs-resource Pareto frontier under logical and transpiled
  cost metrics?
  *H6: greedy/evolutionary populate the cheap end; LLM arms the
  expressive end; the frontier is method-mixed.*

## 2. Model contract (both tracks, all tasks)

The quantum model is the preserved `FixedReadoutQuantumModel`:

- `2**n_qubits` real features → single L2 normalization inside
  `AmplitudeEmbedding(normalize=True)` → searched body → `<Z>` on fixed
  readout qubit 0 → `mu_hat = (1 - <Z0>)/2 ∈ [0,1]`.
- Zero classical parameters (structural invariant, AST/parameter-count
  tested). Regression targets are normalized to `[0,1]`; for T4 the output
  is the class-1 probability.
- Exact statevector simulation (PennyLane `default.qubit`,
  `diff_method="backprop"` where gradients are needed), float64, CPU.

## 3. Task suite (`llm_vqc/tasks/signal_suite/`, version `signal_suite_v1`)

All tasks generate length-`2**n` signals on a fixed grid `x_i = i/N`,
`i = 0..N-1`, `N = 2**n`. Frozen train/val/test generation with disjoint
sample IDs; generator parameters and split hashes stored in every dataset
manifest; task version string in every cache key. No MNIST/digits anywhere.

- **T1 `gauss_peak`** — Gaussian peak location regression. Target `mu`.
  Nuisances: amplitude, width, baseline, noise. Documented limitation:
  amplitude encoding removes global scale; the task never asks for
  absolute amplitude. A **legacy profile** reproduces the existing
  `amplitude_n3_smoke_v1` behaviour (n=3, sigma∈[0.08,0.2]) for E0/E1
  continuity; the **main profile** (n=5) widens nuisance ranges.
- **T2 `sin_freq`** — noisy sinusoid frequency regression.
  `y = A sin(2π f x + φ) + c + ε`. Frequency range is a profile function
  of N: `f ∈ [1, N/4]` cycles per unit interval (max = half of Nyquist
  N/2, guaranteeing identifiability margin); target = `(f-1)/(N/4-1)`
  normalized to [0,1].
- **T3 `change_point`** — piecewise-level signals with random pre/post
  levels (separation margin enforced), optional small slopes, noise;
  randomized sign/amplitude conventions against leakage. Target =
  normalized change-point location ∈ [0.2, 0.8].
- **T4 `peak_count`** — balanced binary classification: single-peak vs
  double-peak signals (second peak separation ≥ 4 grid points), randomized
  locations/widths/amplitudes/baselines/noise. Output = P(class 1).

Exact generator constants are frozen in the versioned task package at
Phase 1 (before any v2 search run) and hashed into
`configs/bench_v2/protocol_v2.yaml`; they may not change afterwards
without a task-version bump.

**Split sizes.** Main profiles target 256 train / 256 val / 2048 test
(provisional; confirmed or reduced only by the prewritten compute-sizing
rule in §10 *before* any main-matrix run). Legacy T1 replication keeps its
original committed sizes for E0 only. Required data checks: deterministic
regeneration hashes; disjointness; target-distribution plots; classical
sanity baselines; no zero-norm amplitude vectors; aliasing guard for T2;
class balance for T4.

## 4. Search-space profiles

- **`compact_free_v1`** (preserved): 1–5 body gates, each
  `H|RX|RY|RZ|CRX|CRY|CRZ` with explicit wires (and one angle for
  parameterized gates in Track B); fixed amplitude encoding; fixed Z(q0)
  readout. Used for E0/E1 replication continuity at n∈{3,5}.
- **`scalable_layered_v1`** (new, frozen here): for n∈{3..8}, a proposal
  is an ordered list of at most `max_ops = min(2n, 16)` layer operations,
  each either
  `rot(gate ∈ {RX,RY,RZ,H}, wires ⊆ {0..n-1})` or
  `entangle(gate ∈ {CNOT,CZ,CRX,CRY,CRZ}, pattern ∈ {line,ring,star,pairs,all_to_all}, valid wires)`;
  encoding fixed amplitude-on-all-wires; measurement fixed Z(q0); grammar
  version `scalable_layered_v1`; identical grammar, mutation operators,
  size priors, and validity rules for every arm (validity = shared IR
  validators; no arm-specific actions).

## 5. Tracks

### Track A — primary: structure-only search, one shared inner trainer

Arms emit architecture IR only. One shared evaluator performs theta
initialization + training identically for every arm:

- optimizer AdamW (`FreeAmplitudeTrainingConfig`): lr 0.05, weight decay
  1e-5, betas (0.9, 0.999), eps 1e-8, full-batch-shuffled minibatches,
  fixed epochs and batch size chosen at the Phase 6 compute-sizing gate
  (provisional: epochs 40, batch 32) and then frozen for every arm;
- deterministic theta seed = f(task_version, data_seed, search_seed,
  structural_hash); identical architecture ⇒ identical result across arms
  via the global evaluation cache;
- validation metric returned to the searcher (regression: val RMSE; T4:
  val Brier score, both lower-better); test data unreachable from search
  modules (structural quarantine);
- no arm-specific learning rates, epochs, restarts, preprocessing, or
  measurement heads.

Post-selection robustness: each selected architecture is re-trained under
5 predeclared additional theta-init seeds; reported as robustness spread,
never used for selection.

### Track B — ablation: direct joint structure+theta, verbatim evaluation

Preserved `free_amplitude` main mode: no classical layer, no optimizer,
theta evaluated verbatim, candidate identity includes theta
(`candidate_hash`), fixed readout, test quarantine. Plus the mandatory
classical `evolutionary_joint` baseline (chromosome = gate choices, wires,
ordering, continuous theta; mutation + crossover; identical budget
accounting) and the E5 paired theta-quality analysis (LLM theta vs random
theta vs fixed-budget classical theta optimization on identical frozen
architectures, exact objective-call matching).

## 6. Arms

Track A mandatory: `random_structure`, `evolutionary_structure`
((mu+lambda), mu=4 lambda=8, documented mutation set), `greedy_growth`,
`llm_open_structure`, `llm_archive_closed_structure` (bounded top-k=8
archive + diversity/resource summary), fixed references
`ref_realamp_d1`, `ref_realamp_d2` (RY layer + CNOT line, depth 1/2) and
`ref_strongent_d1`, `ref_strongent_d2` (RZ-RY-RZ layers + CNOT ring,
depth 1/2).

Track B mandatory: `random_joint`, `evolutionary_joint`, `llm_open_joint`,
`llm_closed_joint`.

Optional (only after every mandatory cell is complete): DQAS/supernet,
progressive-widening MCTS, noise-aware topology-first, multi-agent critic.

## 7. LLM architecture

Six-stage tool-constrained pipeline (no free-form code generation):
schema-constrained JSON proposal layer (batched, 3 candidates/call);
deterministic validator/canonicalizer recording every issue; deterministic
evaluator/trainer; top-k archive manager (score, novelty, resource
summary); deterministic feedback summarizer (compact numeric history,
failure categories, diversity, remaining budgets); protected test gate
called only after selection freeze. Full logging per call: prompt version,
model snapshot, sampling config, response, tokens, latency, repair
attempts, provider errors. No test-derived information of any kind may
enter any prompt. Paid calls require `OPENAI_API_KEY` **and** positive
`LLM_API_BUDGET_USD`; the preflight is enforced in code before any call.
The pre-existing full-history closed loop may be kept as a cheap ablation;
the primary closed-loop condition is the bounded archive.

## 8. Budgets

Separate ledger fields: total proposals; valid; unique candidate
evaluations; duplicates/cache hits; failed evaluations; LLM outbound
attempts; successful calls; input/output tokens and estimated/actual USD;
inner-optimizer objective and gradient evaluations; wall-clock and CPU
time. Primary x-axis: **unique candidate evaluations**; secondary views:
all proposals, LLM calls, wall-clock, estimated cost. Duplicates are
reported in both proposal-budget and unique-budget views, never silently
free.

## 9. Metrics and statistics

- Regression primary: **validation-selected protected-test RMSE** (defined
  in §12); secondary MAE, R², NRMSE, residual calibration over target
  bins.
- T4 primary: **AUROC** (fixed here, before any run); secondary balanced
  accuracy, log loss, Brier, calibration.
- Cross-task: per-task first; standardized regret vs best predeclared
  reference; average rank secondary only.
- Search metrics: per-seed points (never bar-only), best-so-far validation
  curves + AUC, evaluations-to-threshold, invalid/duplicate/failed/repair
  /stop-reason distributions, pairwise edit-distance diversity, archive
  turnover, motif frequency.
- Resource metrics: logical gate count/depth, 1q/2q counts,
  controlled-rotation count, parameter count, causal-cone count/fraction,
  transpiled depth and native-2q/SWAP counts under fixed line, ring, and
  all-to-all coupling (fixed transpile seeds), optimizer convergence and
  gradient norm/variance (Track A), finite-shot sensitivity at 1,024 and
  4,096 shots for selected circuits. Expressibility KL and MW-Q are
  descriptive only (RQ5 association analysis with size controls).
- Statistics: paired replicates (same data-seed and search-seed block for
  every arm); data/search/theta-init/LLM identities logged separately.
  Report median, IQR, mean, bootstrap 95% CI (10,000 resamples), and
  individual points; paired Wilcoxon/permutation tests; Holm correction
  within each RQ family; effect sizes (Hodges–Lehmann shift, paired
  Cliff's delta). `p > 0.05` is never equivalence; with small n emphasize
  intervals and effect sizes.

## 10. Experiment matrix and sizing rule

| ID | Content | Cells |
|---|---|---|
| **E0** | Provenance replication of the committed 2-seed real-LLM joint run from stored artifacts (`outputs/free_amplitude_fixed_readout_v1/`), T1 legacy n=3, arms random/llm_open/llm_closed, B=4, **zero new API calls** | verification, not evidence |
| **E1** | Direct joint extension: T1-legacy@n=3 (pinned by its committed definition), T2@n=3, T2@n=5; `random_joint`, `evolutionary_joint`, `llm_open_joint`, `llm_closed_joint`, B=16 unique, checkpoints 4/8/16, 10 paired replicates | 3 task-n pairs × 4 arms × 10 |
| **E2** | Main Track-A matrix: T1–T4, n=5, 5 search arms + 4 fixed references, B=24, checkpoints 4/8/16/24, 10 paired replicates | 4 × 9 × 10 |
| **E3** | Scaling: T1, T2; n∈{3,4,6,8}; `random_structure`, `evolutionary_structure`, `llm_open_structure`, `llm_archive_closed_structure`, strongest fixed reference (chosen on *validation*); B=16, checkpoints 4/8/16, ≥5 paired replicates | 2 × 4 × 5 × 5 |
| **E4** | Selected top circuits from E2/E3: exact vs 1,024/4,096 shots vs one documented noisy profile (depolarizing p1=0.001, p2=0.01 + readout 0.02 — frozen here); fixed transpile seeds/coupling maps; no tuning on test | selected circuits only |
| **E5** | Theta isolation: representative LLM-proposed architectures frozen; LLM theta vs random theta vs fixed-budget classical theta optimizer; exact objective-call matching | paired analysis |

**Prewritten compute-sizing rule** (may be applied only *before* the main
matrix, from pilot timing, never after seeing method rankings): if
projected E2 wall-clock exceeds 48 h: first reduce test split 2048→1024;
then inner epochs 40→25 (all arms identically); then B 24→16 (checkpoints
4/8/16). Replicate counts of mandatory cells are reduced last and only
under a documented external blocker; optional cells are dropped before any
mandatory reduction.

## 11. Classical sanity baselines (task-difficulty context only)

Per task, same splits: ridge regression (T1–T3) / logistic regression
(T4); fixed-capacity MLP (one hidden layer, 32 units); an interpretable
signal-processing estimator (T1: Gaussian fit; T2: periodogram argmax;
T3: CUSUM-style split; T4: peak-count heuristic). These contextualize
difficulty. **Quantum-advantage claims are prohibited.**

## 12. Definition: validation-selected protected-test RMSE

> RMSE computed on a held-out test partition that is inaccessible to every
> search arm and inner tuning decision, evaluated only after the
> candidate/architecture has been selected using validation data, and
> never fed back into prompts, early stopping, hyperparameter choices, or
> protocol changes.

The classification analog substitutes the T4 primary metric (AUROC). This
wording is used in docs, CLI help, reports, and figure captions.

## 13. Seeds and pairing

Replicate r ∈ {0..9}: `data_seed = 1000 + r`, `search_seed = 2000 + r`,
identical for every arm in a comparison (paired blocks). Theta-init seed
derived per §5. LLM run identity = (model snapshot, prompt version,
call index) logged per call. E3 uses r ∈ {0..4} minimum.

## 14. Store and artifact layout (checker-facing)

- Durable stores: `runs/bench_v2/<EID>/<cell_id>/results.sqlite` where
  `cell_id = {task}_{n}q_{arm}_r{replicate}`; plus
  `runs/bench_v2/<EID>/cells/<cell_id>.json` completion manifests
  (config hash, git SHA, seeds, budget ledger summary, selected candidate,
  test-gate timestamp, artifact hashes).
- Sanitized derived artifacts + figures: `outputs/bench_v2/<EID>/`
  (committed), each figure with source CSV/JSON and config/store hashes.
- Completion is verified only by `scripts/check_goal_completion.py`
  against `GOAL_CONTRACT.yaml`.
