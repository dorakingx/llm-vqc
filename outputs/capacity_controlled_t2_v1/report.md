# Capacity-controlled T2-v1 — research report

**Experiment:** `capacity_controlled_t2_v1` · **Branches from:** T1-v2 @
`f24ea14` · **Design:** 5 data-split seeds × 6 repetition seeds × 3 search arms,
B=25 (90 runs) + classical baselines C0–C6 + paired quantum ablations
(freeze/train ×20 archs, product/entangled ×20 pairs) + selection-gain. **Status:**
completed. **Methodology-validation task, NOT HEP.** All choices pre-registered
(`PROTOCOL.md`) before the matrix ran (`b978c52`).

> **Headline.** On the near-separable T2 (digits 3-vs-8) task, the three search
> arms are **statistically equivalent** on protected-test balanced-accuracy error
> — at both the pre-registered δ=0.03 and the stricter δ=0.02. And with proper
> paired designs, **none of the "quantum knobs" help on T2**: training the quantum
> angles gives no gain, entanglement gives no gain, and best-of-B architecture
> selection gives no gain. Strong classical models (RBF-SVM ≈ 0, logistic
> regression / trees ≈ 0.014) **beat** the 61-param VQC (≈ 0.028). The methodology
> is validated; the task is simply too easy to separate the quantum conditions.

## 1. Metric decision (audit-driven; see `T2_TASK_AUDIT.md`)

T2 is near-separable — logistic-regression test AUC = 0.996, and **validation AUC
saturates at 1.0** for every controlled VQC (15/15 probed), so AUC cannot rank
candidates. Pre-registered fix: **selection on validation log-loss** (discriminative,
spread 0.046–0.062), **primary protected-test metric = balanced-accuracy error**,
**equivalence margin δ=0.03** (justified from the 71-sample test resolution).
Documented, not silent.

## 2. Capacity verification

All **2,004** evaluated controlled VQC candidates had exactly **61** trainable
params (44 embed + 12 quantum + 5 head; `fig12`). 0 invalid, 0 failed across 90 runs.

## 3. Power (post-run, observed SD, `power_analysis.json`)

Observed paired-diff SD = 0.0138 (matching the pre-registered assumption). With
n=30 cells, TOST at δ=0.03 concludes equivalence with P=1.00 when arms are truly
equal, and detects a 0.03 gap with P=1.00. The equivalence conclusion is
well-powered.

## 4. Primary result — search equivalence (confirmatory, `fig01`, `fig03`)

Protected-test balanced-accuracy error, pooled over 30 cells:

| Arm | median | mean | SD | boot 95% CI (median) |
|---|---:|---:|---:|---|
| Random | 0.0282 | 0.0377 | 0.0266 | [0.0212, 0.0425] |
| Evolutionary | 0.0286 | 0.0401 | 0.0286 | [0.0210, 0.0425] |
| Greedy | 0.0282 | 0.0339 | 0.0245 | [0.0143, 0.0286] |

Kruskal–Wallis p=0.66; all pairwise Holm p=1.00. **Paired TOST:**

| Pair | mean paired Δ | 90% CI | equiv @0.03 | @0.02 | split-mean cluster |
|---|---:|---|:--:|:--:|:--:|
| Random − Evolutionary | −0.0024 | [−0.0062, +0.0015] | ✅ | ✅ | ✅ |
| Random − Greedy | +0.0038 | [−0.0005, +0.0081] | ✅ | ✅ | ✅ |
| Evolutionary − Greedy | +0.0062 | [+0.0019, +0.0105] | ✅ | ✅ | ✅ |

**All pairs equivalent at δ=0.03 and δ=0.02**; the conclusion survives the
split-mean cluster re-analysis and split-blocked bootstrap. **→ S1: practically
equivalent** (more strongly than T1, which failed at δ=0.001).

## 5. Split-level variability (`fig18`)

Arm medians are flat and interleaved across all 5 splits; no split separates the
arms. The equivalence is not split-specific.

## 6. Efficiency / resource accounting (`fig04`–`fig08`, `fig13`, `fig14`)

0 invalid, 0 failed across all 90 runs. Duplicate rates replicate the T1 pattern
(Evolutionary highest). Anytime curves (proposals / unique trainings / wall-clock)
overlap; no arm dominates. See `resource_accounting.csv`.

## 7. Paired quantum ablations (the key improvement over T1-v2)

Unlike T1-v2 (single fixed structures), these use 20 architectures / 20 paired
circuits with identical initialization.

- **Trainable vs frozen quantum angles** (`fig09`; n=300 paired): frozen − trainable
  mean = **−0.0076**, median **0.0000**, Cohen's dz = −0.13, Wilcoxon p=0.12,
  fraction favouring training = 0.23. Training the 12 quantum angles gives **no
  meaningful gain** — if anything a weak (non-significant) lean toward *frozen*
  being better. **→ QT2: no meaningful improvement.**
- **Entangled vs product** (`fig10`; n=300 paired): product − entangled mean =
  **+0.0000**, median **0.0000**, dz = 0.000, fraction favouring entanglement =
  0.18 (Wilcoxon p=0.028 flags a distributional asymmetry of *negligible
  magnitude*). Entanglement gives **no practical benefit** on T2. **→ E2: no
  meaningful improvement.** (Contrast T1-v2, where entanglement clearly helped.)
- **Selection benefit** (`fig11`): expected fixed-architecture test error 0.0282;
  best-of-B median 0.0282 (B=1), 0.0212 (B=5), 0.0286 (B=10), 0.0282 (B=25).
  Best-of-B ≈ fixed — **validation selection provides essentially no test gain**
  on T2 (because every controlled circuit already classifies this easy task well).
  **→ A2: little or no selection gain.** (Contrast T1, where best-of-25 clearly
  beat a fixed circuit.)

## 8. Classical comparison (`fig07`, `fig08`)

Protected-test balanced-accuracy error medians:

| Model | test error |
|---|---:|
| **C3 RBF-SVM** | **0.0000** |
| C4 trees | 0.0139 |
| C1 logistic regression | 0.0143 |
| C2 linear SVM | 0.0278 |
| **QS searched VQC (best arm)** | 0.0282 |
| C5 fixed MLP | 0.0282 |
| C6 fair classical NAS | 0.0353 |
| C0 trivial | 0.5000 |

Strong classical models (RBF-SVM, logistic regression, trees) **outperform** the
VQC; the neural baselines (MLP, fair NAS) tie or trail it. **→ C3: family-dependent
— the VQC does not outperform strong classical models on T2.**

## 9. Classification diagnostics

Aggregate confusion matrices per arm (`fig15`), the best run's ROC (`fig16`,
AUC ≈ 0.98) and calibration (`fig17`) show competent, reasonably-calibrated
classifiers; secondary metrics (accuracy ≈ 0.94–0.97, AUC ≈ 0.97–0.99) are in
`run_summary.csv`.

## 10. Runtime / compute

2,004 unique VQC candidate trainings (median ≈ 0.9 s) + 1,200 paired-ablation
trainings + 30 classical-baseline cells (incl. C6 NAS 25 fits each). CPU, float64.

## 11. Decision gates

- **Search — S1:** arms practically equivalent (δ=0.02 and 0.03; well-powered;
  robust to splits and clustering).
- **Quantum-training — QT2:** trainable quantum angles give no meaningful gain over
  frozen quantum features (weak lean toward frozen; n.s.).
- **Entanglement — E2:** entanglement gives no practical gain over parameter-matched
  product circuits (effect ≈ 0).
- **Selection — A2:** best-of-B validation selection gives little/no test gain.
- **Classical — C3:** family-dependent; strong classical models beat the VQC.

## 12. Claims supported

Pre-registered equivalence of the three search arms on T2 (δ≤0.03, 5 splits ×
6 seeds, well-powered); with clean paired designs, **no measurable contribution
from quantum-angle training, entanglement, or architecture selection on T2**;
strong classical models outperform the 61-param VQC; uniform 61-param capacity
verified over 2,004 candidates; 0 invalid / 0 failed; no preprocessing leakage.

## 13. Claims NOT supported

No arm is better than another (S2 rejected). **No quantum advantage** — strong
classical models win. No claim that quantum-angle training or entanglement helps
*in general* (they don't help *here*; T1-v2 showed entanglement can help on a
harder task). No HEP claim (T2 is sklearn digits). No generalization beyond this
task, 4 qubits, 12 quantum params, B=25, δ=0.03, or these baseline families.

## 14. Remaining confounders / limitations

- **Task too easy:** near-separability (val AUC = 1.0) means the quantum knobs have
  little room to matter; T2 validates the *methodology* but cannot discriminate
  quantum mechanisms. A harder classification task is needed for that.
- **Entanglement mapping** replaces CRZ→RY and free-entangle→H; gate counts match to
  within ≤2 (recorded), a minor structural confound.
- **Selection metric switch** (AUC→log-loss) was necessary but is task-specific.
- **12-param budget / angle-RY encoding fixed;** other budgets/encodings untested.

## 15. Cross-task synthesis (T1-v2 vs T2-v1)

| Question | T1 (Gaussian regression) | T2 (digits classification) |
|---|---|---|
| Arms equivalent? | yes @δ=0.002 (not 0.001) | **yes @δ=0.02 & 0.03** |
| Quantum-angle training helps? | barely (single arch) | **no** (20 archs, paired) |
| Entanglement helps? | **yes** | **no** |
| Best-of-B selection helps? | yes | **no** |
| VQC vs strong classical? | analytical ≫ VQC | strong classical > VQC |

The consistent thread: **search-strategy choice never matters under capacity
control**, and the value of the quantum components is entirely task-dependent —
present on T1, absent on the easy T2 — while strong classical baselines match or
beat the VQC on both.

## 16. Recommendation for the next simulation

1. **A genuinely hard classification task** (harder digit pairs, more classes with
   overlap, or label noise) where validation does not saturate — the decisive test
   of whether entanglement/quantum-training ever help for classification.
2. **Encoding/param-budget sweep** (angle vs re-uploading; 8/12/16 quantum params)
   to see whether the quantum knobs matter at higher capacity.
3. Only then, add LLM arms under this same capacity-controlled, equivalence-tested,
   paired-ablation protocol.

---

### Reproduction

```bash
python scripts/capacity_controlled/run_experiment_t2.py --splits 0,1,2,3,4 --seeds 0,1,2,3,4,5 --abl-seeds 0,1,2 --budget 25 --tag pilot
python scripts/capacity_controlled/analyze_t2.py --tag pilot
```

Machine-readable: `protocol.json`, `experiment_config.json`, `candidate_results.csv`,
`run_summary.{csv,json}`, `split_summary.csv`, `classical_baseline_summary.csv`,
`classical_search_summary.csv`, `quantum_ablation_summary.csv` (as
`paired_freeze_train_results.csv` + `paired_entanglement_results.csv`),
`selection_gain_summary.csv`, `resource_accounting.csv`, `statistical_analysis.json`,
`equivalence_analysis.json`, `power_analysis.json`, manifests. Figures fig01–fig18
(PNG+SVG). Raw stores/weights never published.
