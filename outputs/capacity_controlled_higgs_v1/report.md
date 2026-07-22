# Capacity-controlled HIGGS-v1 — research report

**Experiment:** `capacity_controlled_higgs_v1` · **Branches from:** T2-v1 @
`6f95aec` · **Task:** official UCI **HIGGS** (id 280) binary classification, 21
low-level features → PCA(8), methodology-validation (NOT a HEP-domain claim).
**Status:** completed. Lower log-loss is better. Pre-registered (`PROTOCOL.md`,
committed `680b693`) before any VQC run; external holdout accessed once after
freeze.

> **Headline.** On a genuinely hard official HEP task, analyzed with correct
> **block-level** inference, the earlier "arms are equivalent" story does **not**
> reproduce cleanly: search-arm equivalence at δ=0.01 is **inconclusive**
> (underpowered — block-to-block SD 0.016 exceeds the margin). More importantly,
> the paired ablations **reverse** the T1 intuition: on 500-sample blocks,
> **training the quantum angles and adding entanglement both slightly HURT
> generalization** (clustered CIs exclude 0 in the "simpler-is-better"
> direction). Validation **selection genuinely helps** (best-of-25 beats a fixed
> circuit). And the 53-param VQC is **worse than a trivial majority classifier** —
> no quantum advantage.

## 1. Task qualification & why HIGGS

Unlike T2 (AUROC ≈ 1.0, degenerate), HIGGS 21 low-level **passed all 6
pre-registered criteria** (D1): best dev-val AUROC **0.673** ∈ [0.65, 0.92],
log-loss 0.646, balance 0.526, non-saturated (`TASK_QUALIFICATION.md`). SUSY
fallback not needed. This is the non-saturated regime the design required.

## 2. Integrity verified

11,000,000 rows verified (SHA-256 `ea302c18…`); last 500k held out and untouched
until the final audit. **10 mutually disjoint blocks** (500/250/2000, stratified,
immutable row-index IDs); preprocessing (StandardScaler→PCA8) fit on block-train
only (leakage-tested). All **2,005** evaluated candidates have exactly **53**
trainable params (`fig14`); **0 invalid, 0 failed** across 90 runs.

## 3. Primary — search-arm equivalence (block-level, confirmatory)

Protected-test log-loss, pooled 30 cells: Random 0.7287, Evolutionary 0.7139,
Greedy 0.7267. **Block-level paired TOST (n=10 blocks), δ=0.01** (`fig04`,
`equivalence_analysis.json`):

| Pair | block-level mean Δ | 90% CI | eq @0.01 | @0.02 |
|---|---:|---|:--:|:--:|
| Random − Evolutionary | +0.0030 | [−0.0076, +0.0137] | ❌ | ✅ |
| Random − Greedy | −0.0028 | [−0.0085, +0.0030] | ✅ | ✅ |
| Evolutionary − Greedy | −0.0058 | [−0.0150, +0.0034] | ❌ | ✅ |

**Not all pairs equivalent at δ=0.01**; all equivalent only at the looser δ=0.02.
Kruskal–Wallis on block medians p=0.048 (marginal). **Power is the key caveat:**
with n=10 blocks and observed block-contrast SD **0.0159** (> the 0.01 margin),
the probability of concluding equivalence via TOST at δ=0.01 when the true
difference is 0 is only **0.22** (`power_analysis.json`). The design is
**underpowered to establish δ=0.01 equivalence**. No CI cleanly excludes the
equivalence region with a consistent sign either. **→ S3 (inconclusive):** the
arms are within ~0.006 log-loss but block variance is too large, at 10 blocks, to
declare either equivalence (δ=0.01) or a meaningful difference. (At δ=0.02 they
are equivalent.)

This is a **more conservative and honest** result than T1/T2, and it directly
reflects the corrected inference: seeds are within-block repeats, so the true n is
10 blocks, not 30 or 90.

## 4. Paired quantum ablations (block + architecture clustered) — the key finding

Clusters: 10 blocks × 20 architectures × 2 seeds = **400 pairs each**; reported at
the cluster level with a hierarchical bootstrap (`statistical_analysis.json`,
`fig11`, `fig12`):

| Contrast | overall mean | hierarchical-bootstrap 95% CI | direction |
|---|---:|---|---|
| frozen − trainable quantum angles | **−0.0062** | [−0.0097, −0.0027] (excludes 0) | **frozen better** |
| product − entangled | **−0.0087** | [−0.0114, −0.0061] (excludes 0) | **product better** |

Both CIs exclude 0 **in the "less-quantum-is-better" direction**:

- **Training the 12 quantum angles does not help — frozen random quantum features
  are ~0.006 log-loss BETTER** (only 38% of pairs favor training). **→ QT2 (no
  improvement; effect reversed):** on 500-sample hard-task blocks, optimizing the
  quantum angles overfits relative to freezing them and training only the
  classical shell.
- **Entanglement does not help — parameter-matched product circuits are ~0.009
  log-loss BETTER** (27% of pairs favor entanglement). **→ E2 (no improvement;
  effect reversed):** the exact CRZ→RZ product counterparts generalize better than
  their entangled originals. This is the **opposite** of T1 (where entanglement
  helped) and confirms that entanglement's value is task- and data-regime-
  dependent, not universal.

## 5. Selection benefit (block-aggregated)

Expected fixed-architecture test log-loss 0.7522; best-of-B: 0.7709 (B=1), 0.7372
(B=5), **0.7258 (B=10)**, 0.7287 (B=25). **Best-of-B clearly improves over a fixed
architecture** (~0.023 by B=10). **→ A1: validation selection improves
protected-test performance** — framed as the benefit of best-of-B validation
selection, not proof a sophisticated algorithm is needed. (Unlike T2, where
selection did not help; on this harder task there is real spread to select from.)

## 6. Classical comparison

Protected-test log-loss medians: majority (C0) **0.6917**, RBF-SVM 0.6910, linear
SVM 0.6917, RandomForest 0.6937, fair NAS 0.6959, logreg 0.6963, HistGB 0.7159,
fixed MLP 0.7266 — vs **QS best VQC arm 0.7139**. **Every non-trivial classical
model, and even the trivial majority classifier, matches or beats the VQC.**
**→ C2: classical models match/outperform the searched VQC.** Note all models sit
near the majority log-loss (~0.69): with only **500 training samples** on a task
needing thousands, little is learned, and the extra-capacity VQC is the *worst* of
the group. **No quantum advantage.**

## 7. Hardware-aware secondary (`fig19`–`fig21`, `pareto_summary.json`)

Every candidate transpiled to the linear topology 0–1–2–3 (fixed config, seed
20260719): logical→transpiled two-qubit inflates (e.g. 4→~14 cx) with ~2 inserted
SWAPs. Per-arm performance–cost Pareto **hypervolumes are near-identical** (Random
12.10, Evolutionary 12.13, Greedy 12.05). **→ H2: Pareto fronts practically
equivalent** across arms. The hardware-aware selection rule (retain within 0.01
val-log-loss → fewest transpiled two-qubit → tie depth → tie hash) yields circuits
with lower two-qubit cost at similar test log-loss.

## 8. External-holdout audit (once, after freeze)

Consensus architecture (lowest mean selected-validation log-loss across the 10
blocks), retrained on the combined 5,000-sample training set, evaluated **once** on
the fixed 20,000-row subset of the official final 500k holdout: **AUROC 0.572,
log-loss 0.687, accuracy 0.556** (`external_holdout_result.json`, `fig23`). The
53-param VQC barely exceeds chance even with 10× more training data — consistent
with §6. No design decision was changed from this result.

## 9. Decision gates

- **Task — D1** (HIGGS low-level qualified).
- **Search — S3 (inconclusive):** not equivalent at δ=0.01, not robustly
  different; **underpowered** (P(equiv|true=0)=0.22 at n=10). Equivalent only at
  δ=0.02.
- **Quantum-training — QT2 (reversed):** trainable does not beat frozen; frozen is
  ~0.006 better (clustered CI excludes 0).
- **Entanglement — E2 (reversed):** entangled does not beat product; product is
  ~0.009 better (clustered CI excludes 0).
- **Selection — A1:** best-of-B improves over fixed (~0.023).
- **Hardware — H2:** Pareto fronts equivalent across arms.
- **Classical — C2:** classical (and trivial majority) match/beat the VQC.

## 10. Claims supported

On official HIGGS with correct block-level inference: (a) search-arm equivalence
at δ=0.01 is **inconclusive/underpowered**, equivalent only at δ=0.02; (b) with
proper clustering, **training the quantum angles and entanglement both slightly
HURT** generalization on small hard-task blocks; (c) validation selection helps;
(d) the 53-param VQC does **not** outperform classical models — it underperforms
even majority; (e) arms have equivalent hardware-cost Pareto fronts; (f) uniform
53-param capacity verified over 2,005 candidates; leakage-free; holdout quarantined.

## 11. Claims NOT supported

No practically meaningful arm ranking is *established* (S3, not S2). **No quantum
advantage.** No claim that quantum-angle training or entanglement helps — here
they hurt. No HEP-domain physics claim. No generalization beyond 4 qubits, 12
quantum params, PCA(8), 500-sample blocks, B=25, δ=0.01, or these baseline
families.

## 12. Remaining confounders / limitations

- **Small training blocks (500 samples)** on a task needing thousands: nothing
  learns much (all near majority log-loss), so the ablation reversals hold in a
  **low-data / high-variance regime**; larger blocks could change the sign. This
  is the dominant confound.
- **Underpowered search test:** 10 blocks with block SD 0.016 > δ 0.01 → cannot
  establish δ=0.01 equivalence; more blocks are needed.
- **Classical baselines run on the same PCA(8)** representation as the VQC (fair
  same-input comparison); classical models on the raw 21 features would likely do
  better still.
- Entanglement mapping keeps gate/layer count but not backend depth (recorded).
- `statsmodels` unavailable → block bootstrap + cluster-averaged CIs used instead
  of a random-effects model (the intended equivalent).

## 13. Corrected cross-task synthesis

| Question | T1 (RMSE) | T2 (easy, bal-acc) | **HIGGS (hard, log-loss, block-level)** |
|---|---|---|---|
| Arms equivalent? | yes @0.002 | yes @0.02–0.03 | **inconclusive @0.01 (underpowered); eq @0.02** |
| Quantum-angle training helps? | not established | no | **no — frozen slightly better** |
| Entanglement helps? | provisional yes | no | **no — product slightly better** |
| Selection helps? | yes | no | **yes** |
| VQC vs strong classical? | analytical ≫ VQC | classical > VQC | **classical (& majority) ≥ VQC** |

The only cross-task-robust statement remains the **conditional** one: under the
tested conditions no *practically meaningful* arm difference is established;
and the quantum components never provide a robust benefit — on the hardest,
correctly-clustered task they slightly hurt.

## 14. Reproduction

```bash
python scripts/capacity_controlled/qualify_higgs.py
python scripts/capacity_controlled/run_experiment_higgs.py --blocks 0-9 --seeds 0,1,2 --abl-seeds 0,1 --budget 25 --tag pilot
python scripts/capacity_controlled/analyze_higgs.py --tag pilot
python scripts/capacity_controlled/external_holdout_higgs.py
```

## 15. Is the project ready to add LLM search arms?

**Not yet on this task.** Before spending LLM-API budget: (1) the classical
baselines (including majority) match/beat the VQC and the search test is
underpowered — an LLM arm would be compared against a VQC family that does not
beat trivial baselines and against an inconclusive arm comparison. First **raise
training-block size** (≥2–5k) so models clear the majority baseline, and **add
blocks** (≥20–30) so δ=0.01 equivalence is powered. Only when the controlled VQC
meaningfully beats majority, and the arm comparison is adequately powered, will an
LLM arm's effect be measurable against a fair, non-degenerate baseline. The
infrastructure (blocks, quarantine, capacity control, hierarchical inference,
paired ablations, hardware analysis, external holdout) is now in place and
reusable for that step.
