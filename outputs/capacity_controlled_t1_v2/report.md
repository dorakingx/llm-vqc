# Capacity-controlled LAQS-Bench v2 — research report

**Experiment:** `capacity_controlled_t1_v2` · **Branches from:** v1 @
`03ad011` · **Design:** 3 data-split seeds × 10 repetition seeds × 3 search arms,
B=25 (90 search runs) + strong analytical/classical baselines + quantum ablations.
**Status:** completed. Lower RMSE is better. Everything pre-registered in
`PROTOCOL.md` before results were seen.

> **Headline.** With power raised to n=30 paired cells across 3 independent
> splits, Random / Evolutionary / Greedy are **statistically equivalent** on
> protected-test RMSE within the pre-registered margin δ=0.002 (paired TOST, all
> three pairs). The searched VQC beats a *fairly searched* classical MLP, but
> **simple analytical estimators (Gaussian fit, quadratic peak) dramatically
> outperform every learned model** — so "VQC adds value" holds only against
> learned baselines, not against task-appropriate analytical ones. Ablations show
> the benefit comes from **entanglement** and from **selecting the best of B=25**,
> not from training the quantum angles of a fixed circuit.

## 1. What v1 left open and how v2 closes it

v1 (single split, n=5) found no detectable arm difference but ran no equivalence
test, one split, and only weak fixed classical baselines. v2 adds: a
pre-registered TOST equivalence test with a justified margin, 3 splits × 10 seeds
for power, strong analytical estimators, a fair classical *architecture search*,
and a full quantum-ablation ladder. See `V1_AUDIT.md`, `PROTOCOL.md`.

## 2. Capacity verification

All **1,997** evaluated controlled VQC candidates had exactly **105** trainable
parameters (`fig14`; analysis raises if any deviates). Fair comparison confirmed.

## 3. Power analysis (pre-registered, `power_analysis.json`)

Using the v1 within-arm SD (≈0.0025) as the paired-diff SD proxy, the n=30 design
concludes equivalence via TOST (δ=0.002) with probability **0.99 when arms are
truly equal**, **0.69 at a true 0.001 gap**, and **~0 at a 0.002+ gap**; it
detects a true 0.002 gap with power **0.99**. So the design is adequately powered
to *establish* equivalence at δ=0.002 and to *detect* a gap ≥ δ. The observed v2
paired-diff SD (0.0022–0.0027) matched the assumption.

## 4. Primary result — search equivalence (confirmatory)

Protected-test RMSE, pooled over 30 cells (`fig01`, `statistical_analysis.json`):

| Arm | median | mean | SD | boot 95% CI (median) |
|---|---:|---:|---:|---|
| Random | 0.01775 | 0.01776 | 0.00214 | [0.01686, 0.01818] |
| Evolutionary | 0.01715 | 0.01734 | 0.00151 | [0.01667, 0.01784] |
| Greedy | 0.01689 | 0.01684 | 0.00174 | [0.01614, 0.01773] |

Kruskal–Wallis (secondary): H=2.89, **p=0.24** (no significant difference).

**Paired TOST (primary, `equivalence_analysis.json`), δ=0.002:**

| Pair | mean paired Δ | 90% CI | equivalent @0.002 |
|---|---:|---|:--:|
| Random − Evolutionary | +0.00042 | [−0.00041, +0.00124] | **yes** |
| Random − Greedy | +0.00093 | [+0.00017, +0.00168] | **yes** |
| Evolutionary − Greedy | +0.00051 | [−0.00017, +0.00119] | **yes** |

**All three pairs are equivalent within δ=0.002** (`fig03`). Sensitivity: at the
stricter δ=0.001 no pair is equivalent (CIs exceed 0.001); at δ=0.002 and 0.003
all pairs are equivalent. Note Random−Greedy's CI is entirely positive: Greedy is
*reliably* a hair better than Random, but by a practically negligible amount.
**Decision → S1: the arms are practically equivalent within the pre-registered
margin.**

## 5. Split-level variability

Arm medians are stable across the 3 splits (`fig15`, `split_summary.csv`):
Random [0.01821, 0.01749, 0.01688], Evolutionary [0.01756, 0.01676, 0.01763],
Greedy [0.01753, 0.01655, 0.01636]. No split reverses the (near-tied) ordering;
the equivalence conclusion is not split-specific.

## 6. Anytime efficiency and resource accounting (secondary)

Per arm over 30 runs (`resource_accounting.csv`, `fig04`–`fig08`), all with
**0 invalid and 0 failed** proposals:

| Arm | proposals | duplicate rate | unique trainings (median) | median cumulative walltime |
|---|---:|---:|---:|---:|
| Random | 750 | **1.6%** | 25 | 22.5 s |
| Evolutionary | 750 | **25.1%** | 19 | 18.6 s |
| Greedy | 750 | 7.1% | 24 | 21.5 s |

Evolutionary reaches equivalent final RMSE while training **fewer unique
circuits** (19 vs 25) due to its high duplicate rate — modestly more
*compute-efficient per unique training* but exploring less. Anytime curves vs
proposals, unique trainings, and wall-clock (`fig04`–`fig06`) overlap heavily; no
arm dominates the efficiency frontier. Total VQC compute: 1,997 unique trainings,
median 0.93 s, ~1,871 s.

## 7. Quantum ablations (`fig11`, `quantum_ablation_summary.csv`)

Protected-test RMSE medians (all 105 total params unless noted):

| Condition | test RMSE | reads as |
|---|---:|---|
| Q0 no-quantum (93 trainable) | 0.05345 | worst |
| QP product-state (no entanglement) | 0.03495 | |
| QF frozen random quantum (93 trainable + 12 frozen) | 0.02275 | |
| QR fixed random entangled | 0.02219 | |
| QE fixed entangled HEA | 0.02207 | |
| QS searched (best arm) | **0.01689** | best |

- **Is the quantum layer useful?** Yes — Q0 (0.053) is far worse than any VQC.
- **Is entanglement useful? (Q2)** Yes — product-state QP (0.035) ≫ entangled
  QR/QE (0.022).
- **Is architecture search useful beyond a fixed VQC? (Q3)** Yes — QS searched
  (0.017) beats fixed QE/QR (0.022). The gain is from **selecting the best of
  B=25 candidates on validation**, not from the search *strategy* (which §4 shows
  is interchangeable).
- **Is training the quantum angles useful? (Q1)** **Barely.** On the *same* fixed
  QR architecture, training the 12 quantum angles moved 0.02275 (QF frozen) →
  0.02219 (QR trained) — a negligible 0.0006. → **Q1 inconclusive/weak; Q2 yes;
  Q3 yes.**

## 8. Classical comparison (`fig09`, `fig10`)

Protected-test RMSE medians:

| Model | params | test RMSE |
|---|---:|---:|
| **A4 Gaussian NLLS fit** | 0 (analytical) | **0.00116** |
| A2 quadratic peak | 0 | 0.00536 |
| A3 weighted centroid | 0 | 0.00619 |
| A1 argmax | 0 | 0.01451 |
| **QS searched VQC (best arm)** | 105 | 0.01689 |
| C3 fair classical MLP search | ~107 | 0.02369 |
| C2 param-matched MLP | 107 | 0.02944 |
| C1 bottleneck | 93 | 0.05169 |

- **VQC vs learned classical:** the searched VQC (0.017) **outperforms the fairly
  searched classical MLP** (C3, 0.024) and both fixed classical models.
- **VQC vs analytical:** the Gaussian-fit estimator A4 (0.00116) is **~15× better**
  than the VQC; A2/A3 are ~3× better; even A1 argmax (0.0145, zero parameters) is
  competitive with the 105-param VQC. A4 had **0 fit failures** over 9,000 fits.
- **Decision → C3 (depends on baseline family):** VQC > *learned* classical, but
  *analytical* classical ≫ VQC. Any "VQC adds value" statement must be qualified
  as "over learned baselines of equal capacity," not over task-appropriate
  analytical methods.

## 9. Prediction and residual analysis

The best run's predicted-vs-true and residual plots on protected test (`fig12`,
`fig13`) show the VQC tracks μ with test RMSE ≈ 0.0135 and residuals centered near
0 with mild heteroscedasticity — competent but far from the analytical fit's
near-exact recovery.

## 10. Decision gates

- **Search — S1:** arms practically equivalent within δ=0.002 (all pairs; robust
  across 3 splits; well-powered). Not equivalent at the stricter δ=0.001.
- **Quantum — Q2 + Q3 supported, Q1 inconclusive:** entanglement helps; selecting
  the best of B=25 helps; training the quantum angles of a fixed circuit does not
  meaningfully help.
- **Classical — C3:** result depends on the baseline family — searched VQC beats
  fair *learned* classical search but is far behind simple analytical estimators.

## 11. Claims supported

Pre-registered equivalence of the three search arms under capacity control, on
T1, at δ=0.002, replicated across 3 splits; entanglement and B=25 selection are
the load-bearing sources of VQC performance (not quantum-angle training);
searched VQC > fair learned classical search; **analytical estimators ≫ all
learned models on T1**; uniform 105-param capacity verified over 1,997 candidates;
0 invalid / 0 failed across 90 runs.

## 12. Claims NOT supported

No arm is meaningfully better than another (S2 rejected). No equivalence at
δ=0.001. **No quantum advantage** — a classical Gaussian fit crushes the VQC. No
HEP-domain claim (T1 is a synthetic 1-D probe). No generalization beyond 4 qubits,
12 quantum params, T1, B=25, this init policy, or these baseline families. No LLM
claim (LLM arms excluded by design).

## 13. Remaining confounders / limitations

- **Task-specificity:** T1's generative form exactly matches A4's fit model, so
  A4's dominance is expected and does not generalize to tasks without a known
  parametric form. A learned model's value would show on such tasks.
- **Single quantum-param budget (12) and single encoding (angle-RY):** the
  ablations fix these; a different budget/encoding could shift the ladder.
- **QF uses only the QR architecture** for the frozen contrast; a frozen-vs-trained
  comparison averaged over many architectures would strengthen the Q1 conclusion.
- **B=25 selection budget:** larger budgets might separate the search strategies
  that are equivalent here.
- **δ=0.002 is a value judgment;** equivalence fails at 0.001.

## 14. Recommendation for the next simulation

1. **Second task without a known parametric form** (e.g. T2 digits, or a Gaussian
   with structured/correlated noise) under the identical protocol — the decisive
   test of whether learned models (VQC or MLP) ever beat analytical baselines.
2. **Averaged frozen-vs-trained quantum** across many sampled architectures to
   settle Q1 with power.
3. **Budget sweep** (B ∈ {10, 25, 60}) to see whether search strategies separate
   at larger budgets.
4. Only then, add LLM arms under this same capacity-controlled, equivalence-tested
   protocol.

---

### Reproduction

```bash
python scripts/capacity_controlled/run_experiment_v2.py --splits 0,1,2 --seeds 0,1,2,3,4,5,6,7,8,9 --budget 25 --tag pilot
python scripts/capacity_controlled/analyze_v2.py --tag pilot
python scripts/capacity_controlled/power_analysis.py
```

Machine-readable: `protocol.json`, `experiment_config.json`,
`candidate_results.csv`, `run_summary.{csv,json}`, `split_summary.csv`,
`classical_search_summary.csv`, `analytical_baseline_summary.csv`,
`quantum_ablation_summary.csv`, `resource_accounting.csv`,
`statistical_analysis.json`, `equivalence_analysis.json`, `power_analysis.json`,
`experiment_manifest.json`, `artifact_manifest.json`. Figures `fig01`–`fig16`
(PNG+SVG). Raw stores/weights are never published.
