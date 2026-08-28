# v1 audit — starting point for v2

**Audited branch/commit:** `experiment/capacity-controlled-t1-v1` @
`03ad011dea0001758fbf2fafe52faf294c210217` (v2 branches from this exact commit).
The v1 experiment, code, and outputs are **not modified** by v2.

## What v1 established (reproduced and confirmed)

- **Capacity was the legacy confound.** Legacy unrestricted candidates ranged
  **24 → 22,609** trainable params (942×); larger amplitude-encoding models
  systematically scored better (Spearman ρ(val, params) = −0.34, p<1e-6; amplitude
  selected in 5/9 legacy runs).
- **Under fixed 105-param capacity, the Random advantage disappears.** All 331
  v1 candidates verified at exactly 105 trainable params.
- **Arms indistinguishable at B=25, n=5** (protected-test medians):
  Random 0.01772, Evolutionary 0.01690, Greedy 0.01736; Kruskal–Wallis p≈0.40;
  all pairwise Holm-corrected p ≥ 0.93.
- **Zero invalid and zero failed** controlled proposals across all arms.
- **Searched VQCs beat the v1 fixed classical baselines** (C1 bottleneck 0.051,
  C2 param-matched MLP 0.029 vs arms ≈ 0.017).
- **399 tests pass** on the v1 commit (re-verified at the start of v2).

## What v1 did NOT establish

1. **Practical equivalence** of the arms — a non-significant difference test is
   not evidence of equivalence (no equivalence/TOST analysis was run).
2. **Generalization across dataset splits** — v1 used a single frozen split
   (seed 0). Split-to-split variability is unmeasured.
3. **A strong "VQC adds value" claim** — the v1 classical baselines were mostly
   *fixed* while the VQC side received *architecture selection*: an unfair
   comparison. No fairly-searched classical model, and no strong analytical
   estimator (argmax / parabola / centroid / Gaussian fit), was evaluated.
4. **Where any VQC benefit comes from** — quantum layer vs. trainable quantum
   params vs. entanglement vs. architecture search were not separated.
5. **Efficiency differences** — v1 reported duplicate rates (Evolutionary 28%)
   but not anytime efficiency by unique trainings or wall-clock time.

## Why v2 is necessary

To move from "no detectable difference (underpowered)" to a **pre-registered
equivalence test** with **more power** (3 splits × 10 seeds) and **fair, strong
baselines + quantum ablations**, so the four scientific questions (search
equivalence; cross-split generalization; VQC-vs-fair-classical; source of any
quantum benefit) can be answered rather than left open.

## Design choices held fixed from v1 (unchanged in v2 primary condition)

Task T1; 4 qubits; angle-RY encoding on all 4 wires; **no amplitude encoding**;
Z measurement on all 4 wires; embed `Linear(21→4)`; head `Linear(4→1)`; **exactly
12 trainable quantum params → 105 total**; init policy
`explicit_kaiming_uniform_qc_v1` (quantum angles U(−π,π)); AdamW lr=0.05, 20
epochs, batch 16, wd=1e-5, LR×0.5 @ {7,13,17}, CPU float64, no early stopping;
validation-only selection; protected test once after selection; identical
depth/gate/layer/two-qubit limits (MAX_BLOCKS=6, MAX_DEPTH=30, MAX_TOTAL_GATES=32,
MAX_TWO_QUBIT_GATES=24). No v1 implementation bug was found that requires
correcting these.

## New hypotheses tested in v2

See `PROTOCOL.md` / `protocol.json` (predeclared before any v2 run). In brief:
primary = **practical equivalence of Random/Evolutionary/Greedy** within a
predeclared RMSE margin across 3 splits × 10 seeds; secondary = anytime
efficiency, duplicate/unique-training efficiency, quantum ablations (Q0/QF/QP/QE/
QR/QS), and searched-VQC vs fairly-searched-classical + analytical estimators.
