# v2 pre-registered protocol — `capacity_controlled_t1_v2`

**Status: predeclared before any v2 result was inspected.** Machine-readable
mirror: `protocol.json`. Nothing below (margins, grammars, seeds, budgets,
decision rules) may be revised after v2 outcomes are seen.

## 0. Fixed inheritance from v1

The primary VQC condition is byte-for-byte the v1 controlled space (see
`SPACE_SPEC.md`, copied unchanged): 4 qubits, angle-RY on all wires, Z on all
wires, no amplitude encoding, embed `Linear(21→4)`, head `Linear(4→1)`, exactly
**12 trainable quantum params → 105 total**, init `explicit_kaiming_uniform_qc_v1`
(quantum angles U(−π,π)), AdamW lr=0.05/20 epochs/batch 16/wd 1e-5/LR×0.5 @
{7,13,17}, CPU float64, no early stopping, validation-only selection, protected
test once. Structural limits: MAX_BLOCKS=6, MAX_DEPTH=30, MAX_TOTAL_GATES=32,
MAX_TWO_QUBIT_GATES=24.

## 1. Design matrix (primary)

- **Data-split seeds:** 0, 1, 2 (3 independent T1 partitions).
- **Repetition (search/train) seeds:** 0–9 (10 per split).
- **Search arms:** Controlled Random, Controlled Evolutionary, Controlled Greedy.
- **Budget:** B = 25 candidate-budget units per run. **Fixed; never changed
  after seeing outcomes.**
- 3 splits × 10 seeds × 3 arms = **90 search runs**; ≤ 90 × 25 = 2,250 budget units.
- Within a split, all arms and seeds use the **identical** train/val/test
  partitions. Each split has its own durable store (no cross-split cache
  collision). Runs are resumable without extra budget or stream changes.

## 2. Primary hypothesis and equivalence margin

**H1 (primary, confirmatory):** Random, Evolutionary, and Greedy are *practically
equivalent* on protected-test RMSE within margin **δ = 0.002 RMSE**, under the
capacity-controlled T1 protocol.

**Margin rationale (fixed a priori):**
- Target μ ∈ [0, 1]; RMSE is in μ-units, so δ = 0.002 = peak localization within
  0.2% of the full target range — practically negligible for this task.
- v1 seed-to-seed SD of protected-test RMSE was ≈ 0.0025–0.003 (per arm); δ is
  *below* that noise floor, i.e. a difference smaller than δ is smaller than one
  seed's worth of variability.
- v1 observed arm median gaps were ≈ 0.0008 (0.01772 vs 0.01690); δ ≈ 2.5× that,
  i.e. "even a gap a couple times larger than what v1 hinted is not meaningful."

**Confirmatory test:** paired **TOST** on protected-test RMSE for each arm pair,
pairing by (split, seed) cell (arms share the partition within a cell). Two
one-sided tests at α = 0.05; conclude equivalence for a pair iff the 90% CI of
the mean paired difference lies entirely within (−δ, +δ). A non-significant
difference test alone is **not** treated as equivalence.

**Sensitivity:** re-report the equivalence conclusion at δ ∈ {0.001, 0.002,
0.003}.

## 3. Secondary hypotheses (exploratory unless stated)

- **H2 anytime efficiency:** arms may differ in best-val-RMSE vs (proposals |
  unique trainings | wall-clock) even if final RMSE is equivalent.
- **H3 efficiency:** arms may differ in duplicate rate / unique-training count.
- **H4 quantum source:** trainable entangled VQCs outperform frozen-quantum (QF)
  and product-state (QP) ablations.
- **H5 fair classical:** searched VQCs vs a *fairly searched* classical MLP (C3,
  same B=25 selection budget) and vs strong analytical estimators (A1–A4).

## 4. Baselines and ablations (all per split × seed unless noted)

**Analytical estimators (no training; deterministic; never see test labels):**
- **A1 argmax:** μ̂ = grid position of the max normalized input.
- **A2 quadratic peak:** parabola through the argmax and its two valid neighbors,
  sub-grid vertex; edge samples fall back to A1 deterministically.
- **A3 weighted centroid:** μ̂ = Σ wᵢxᵢ / Σ wᵢ with predeclared weights
  wᵢ = (normalized height)² (nonnegative; not tuned).
- **A4 Gaussian NLLS:** per-sample least-squares fit of A·exp(−(x−μ)²/2σ²)+c with
  μ∈[0,1], A≥0, σ∈[0.005,0.5], c free; deterministic init (A1 for μ, grid-std
  for σ); fit failures recorded and fall back to A1. Never uses true A/σ/μ.

**Classical (learned):**
- **C1 bottleneck** (v1, 93 params) and **C2 param-matched MLP** (v1, 107 params)
  — preserved unchanged for continuity.
- **C3 fair classical MLP search:** random search over a predeclared grammar,
  **B=25** validation-selected candidates, protected test once. Grammar:
  hidden ∈ {(4,), (5,), (4,3), (4,2), (3,3), (5,2), (6,2)}; activation ∈
  {relu, tanh, sigmoid}; lr ∈ {0.01, 0.05, 0.1}; weight_decay ∈ {0, 1e-5, 1e-4};
  optimizer AdamW; 20 epochs; batch 16; float64. **Allowed param mismatch:**
  total trainable params ∈ [90, 120] (|Δ| ≤ 15 vs 105); exact count recorded per
  candidate. Selection on validation only; test never used for selection/tuning.

**Quantum ablations (capacity-controlled; per split × seed):**
- **Q0 no-quantum:** controlled shell with the VQC replaced by identity
  (embed→sigmoid·π→head→sigmoid); trainable 93, no quantum params.
- **QF frozen random quantum features:** fixed Q2 architecture, quantum angles
  initialized then **frozen**; train only embed+head (93 trainable, 12 frozen,
  105 total).
- **QP trainable product-state VQC:** predeclared 3-single-qubit-rotation-layer
  circuit (RX,RY,RZ on all wires), **no two-qubit gates**, 12 trainable quantum
  params, 105 total.
- **QE fixed entangled VQC:** v1 Q1 hardware-efficient (RY×3 + CNOT-ring×2), fixed
  structure, 12 trainable quantum params, 105 total.
- **QR fixed random VQC:** v1 Q2 fixed random architecture, 12 trainable quantum
  params, 105 total.
- **QS searched entangled VQC:** the primary Random/Evolutionary/Greedy condition.

## 5. Statistics

- **Primary:** paired TOST (§2) + blocked bootstrap 95% CI of the median paired
  difference (resample seeds within each split, 2000 resamples, seeded).
- **Secondary (continuity):** Kruskal–Wallis across arms (pooled and per split),
  Mann–Whitney U + Holm, Cliff's δ, anytime AUC, duplicate/invalid/failed rates,
  runtime, unique-training efficiency. Marked exploratory.
- **Baselines:** searched-VQC vs {A1–A4, C1–C3, Q0/QF/QP/QE/QR} on protected-test
  RMSE. No "quantum advantage" wording — only "outperformed / did not outperform /
  equivalent within margin / inconclusive under this protocol."

## 6. Power analysis (predeclared, run alongside)

Simulation-based, using v1 within-arm SD (≈ 0.0025) as the assumed paired-diff SD
proxy. Estimate P(conclude equivalence via TOST at δ) for true |Δ| ∈ {0.001,
0.002, 0.003} at n = 30 paired cells (3 splits × 10 seeds). Report assumed
variance, splits, seeds, expected power, and the limitation that v1 variance is a
small-sample estimate. The design is **not** revised because a scenario looks
favorable.

## 7. Decision gates (do not force a favorable conclusion)

- **Search:** S1 equivalent within δ / S2 at least one arm differs meaningfully /
  S3 inconclusive.
- **Quantum:** Q1 trainable > frozen quantum / Q2 entangled > product-state /
  Q3 search > fixed VQC / Q4 inconclusive.
- **Classical:** C1 searched-VQC outperforms fair classical search / C2 classical
  matches-or-beats VQC / C3 depends on baseline family or inconclusive.

## 8. Integrity (binding)

No slides. No LLM API. No amplitude encoding in the primary condition. Test
metrics never enter search or selection (VQC or classical). δ, B=25, seeds, and
splits are frozen. No seed/split exclusion or cherry-picking. Raw stores/weights
never committed. v1 never overwritten.
