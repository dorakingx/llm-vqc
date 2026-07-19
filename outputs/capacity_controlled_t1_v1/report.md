# Capacity-controlled LAQS-Bench simulation on T1 — research report

**Experiment:** `T1_capacity_controlled_v1` · **Date:** 2026-07-19 ·
**Status:** completed pilot (descriptive, n=5 seeds). Lower RMSE is better.

> **One-line result.** After fixing model capacity so that *every* candidate has
> exactly **105 trainable parameters**, the legacy Random-Search advantage
> **disappears**: the three search arms are statistically indistinguishable
> (Kruskal–Wallis p≈0.40 on protected test). The capacity-controlled VQC still
> beats a parameter-matched classical MLP, but *which search strategy* is used no
> longer matters at this scale.

This is a simulation study only. No slides. No LLM API calls. No amplitude
encoding. Test metrics were computed once per run after selection and never
entered search feedback.

---

## 1. The problem this experiment fixes

In the legacy unrestricted pilot the classical embedding is
`Linear(21 → circuit.num_inputs)`, and `num_inputs` is the number of angle-
encoding wires or `2**n_wires` for amplitude encoding. So candidate models did
not differ only in VQC architecture — they differed in **total trainable
capacity**. Audited over the 222 legacy circuits
(`legacy_capacity_audit.{csv,json,png}`):

- total trainable parameters ranged **24 → 22,609** — a **942×** swing;
- angle-encoding circuits: 24–284 params (median 79);
- amplitude-encoding circuits: 47–**22,609** params (median 2,831).

"Which arm won" was therefore confounded by classical capacity (see §7).

## 2. What was controlled

Every candidate, every arm: 4 qubits, angle-RY encoding on all 4 wires, Z
measurement on all 4 wires, `Linear(21→4)` embed, `Linear(4→1)` head, **exactly
12 trainable quantum angles → 105 total trainable parameters**, one shared
training protocol (AdamW, lr 0.05, 20 epochs, batch 16, LR×0.5 @ {7,13,17}), and
one shared explicit init policy `explicit_kaiming_uniform_qc_v1` (U(−π,π) quantum
angles). Only the internal VQC architecture varied. Full spec: `SPACE_SPEC.md`.

**Capacity was verified, not assumed:** all **331** evaluated candidates had
exactly 105 trainable params (`fig10_capacity_verification.png`); the analysis
raises `RuntimeError` if any candidate deviates.

## 3. Design

- **Arms:** Random, Evolutionary (μ+λ = 5+5), Greedy (best-of-5 neighbors), all
  proposing from the same controlled grammar via genome operators that preserve
  the 12-param invariant by construction.
- **Budget:** B = 25 candidates per run. **Seeds:** 0–4 (n = 5). **Task:** T1,
  split seed 0. **Protected test:** 2000 samples, scored once per run after
  selection, never in feedback.
- **Baselines** (same split, same schedule, separate from budget curves):
  - **C1** classical bottleneck `21→4→1` (93 params, no VQC);
  - **C2** parameter-matched MLP `21→4→3→1` (107 params; +2 vs 105, recorded);
  - **Q1** fixed hardware-efficient VQC (RY×3 + CNOT-ring×2, 105 params);
  - **Q2** fixed random architecture (predeclared seed 20260719, 105 params).

## 4. Results — search arms (protected test, n=5)

| Arm | Test RMSE median | IQR | bootstrap 95% CI (median) | seeds |
|---|---:|---:|---|---|
| Random | 0.01772 | 0.00357 | [0.01633, 0.02428] | 0.017 0.016 0.024 0.018 0.021 |
| Evolutionary | **0.01690** | 0.00165 | [0.01521, 0.02121] | 0.017 0.021 0.016 0.015 0.017 |
| Greedy | 0.01736 | 0.00060 | [0.01587, 0.01809] | 0.017 0.017 0.018 0.018 0.016 |

Validation (selection) medians: Random 0.01755, Evolutionary 0.01515, Greedy
0.01509.

**No arm is distinguishable from another.** Kruskal–Wallis: test H=1.82,
**p=0.40**; validation H=1.26, **p=0.53**. All pairwise Mann–Whitney U with Holm
correction: **p_holm ≥ 0.93** (test), = 1.00 (validation). Cliff's deltas are
small-to-moderate and inconsistent in sign across val/test. See
`statistical_analysis.json`, `fig01`, `fig02`, `fig03`.

Notably, Random — the clear leader in the legacy pilot — has the **worst**
validation median here (0.01755) and the widest test IQR. Its legacy dominance
did not survive capacity control.

## 5. Results — baselines (protected test median, n=5)

| Baseline | Params | Test RMSE median |
|---|---:|---:|
| **Search arms (any)** | 105 | **≈ 0.017** |
| Q2 fixed random VQC | 105 | 0.02095 |
| Q1 fixed hand-designed VQC | 105 | 0.02645 |
| C2 parameter-matched MLP | 107 | 0.02937 |
| C1 classical bottleneck | 93 | 0.05073 |

The capacity-controlled VQC search arms (~0.017) **beat the parameter-matched
classical MLP** (0.029) and both fixed VQCs, and are far better than the
classical bottleneck (0.051). See `fig11_arms_vs_baselines.png`. So a 4-qubit VQC
does add value over a same-size classical model on T1 — but *searching* its
architecture (vs. taking a random or fixed valid one) adds little at B=25/n=5.

## 6. Process metrics

- **Proposal quality:** invalid = **0** for all arms (generators are valid by
  construction; the controlled validator independently confirmed every proposal).
  Failed = 0. Duplicate rate: Random 3.2%, Greedy 4.0%, **Evolutionary 28.0%**
  (population convergence), consistent with the legacy pattern (`fig12`).
- **Selected-circuit structure** varied across depth 4–21 and two-qubit gates
  0–19 at fixed capacity; validation RMSE shows no strong trend with depth or
  two-qubit count (`fig07`, `fig08`) — architecture within this controlled space
  is a weak predictor of performance.
- **Compute:** 331 candidate trainings, median **0.95 s** each, ~319 s total
  training wall-clock (`fig13`); plus 20 baseline trainings. CPU, float64.

## 7. Legacy post-hoc under the capacity lens (EXPLORATORY, not confirmatory)

Re-reading the legacy pilot's published numbers against capacity
(`legacy_posthoc.{json,png}`; the legacy experiment is **not** redefined):

- across all 222 legacy circuits, **lower validation RMSE correlates with larger
  total capacity**: Spearman ρ = **−0.34**, p = 1.9×10⁻⁷;
- legacy validation RMSE by encoding: angle median 0.0441 vs **amplitude 0.0126**
  (Mann–Whitney p = 1.3×10⁻¹²) — amplitude circuits (largest embeddings) were far
  "better";
- among the 9 legacy selected circuits, **amplitude encoding was selected in 5/9
  runs** (including all 3 Random winners), and protected test tracks capacity at
  Spearman ρ = **−0.80** (p = 0.01).

**Interpretation:** the legacy "Random advantage" largely reflected Random's
tendency to land on amplitude-encoding circuits with 2,800–22,600-parameter
classical embeddings — a *capacity* effect, not a *search-strategy* effect. Fixed
at 105 params, no arm reaches the legacy ~0.008 RMSE; all land near 0.017.

## 8. Decision gate

Best-supported statement: **B — the advantage substantially shrinks after
controlling capacity** (in fact it vanishes), with **D — results are too
underpowered (n=5) to distinguish the search arms** as the operative caveat.

- **A (Random advantage persists):** ❌ not supported — Random is no better than,
  and on validation slightly worse than, Greedy/Evolutionary.
- **C (classical matches/outperforms VQC):** ❌ not supported — the 105-param VQC
  beats the 107-param MLP and 93-param bottleneck on protected test.
- **E (another confound revealed):** partially — architecture-within-capacity is
  a weak performance predictor here, hinting the real lever on T1 is capacity and
  the classical shell, not fine VQC structure. Not forced.

## 9. Claims supported / not supported

**Supported:** (a) a fully capacity-controlled LAQS-Bench experiment is
implementable and was run with verified uniform 105-param capacity; (b) with
capacity fixed, Random/Evolutionary/Greedy are statistically indistinguishable on
T1 at B=25/n=5; (c) the legacy Random-Search advantage was substantially a
capacity confound (amplitude-encoding embeddings); (d) a same-capacity 4-qubit
VQC outperforms a parameter-matched classical MLP on T1.

**Not supported:** any ranking among the three search strategies; any LLM claim
(LLM arms deliberately excluded); quantum advantage; any HEP-domain claim (T1 is
a synthetic 1-D probe); generalization beyond 4 qubits, 12 quantum params, T1,
B=25, or n=5.

## 10. Remaining confounders / limitations

- **n=5 is underpowered** — arm differences at ~0.001 RMSE are within seed noise.
- **Encoding fixed to angle-RY**: the controlled space cannot test whether an
  amplitude-style larger encoding *would* help — it deliberately removes that
  capacity lever. A separate, capacity-matched amplitude study would be needed to
  disentangle "encoding type" from "capacity".
- **C2 mismatch +2 params** (107 vs 105) — small, recorded, not equalizable
  exactly with an integer-width MLP.
- **Single task** (T1) and **single quantum-param budget** (12).

## 11. Recommendation for the next simulation

1. **Increase power:** rerun at n=10–20 seeds before making any arm-ranking claim
   (the infrastructure and budget are unchanged; only seeds grow).
2. **Add a capacity-matched encoding sweep:** compare 4-qubit angle vs a
   capacity-matched amplitude variant (equalize *total* params by shrinking the
   embed) to separate encoding-type from capacity.
3. **Second task (T2 digits)** under the same controlled protocol to test whether
   "search strategy doesn't matter at fixed capacity" generalizes.
4. Only after the above, introduce the LLM arms under identical capacity control —
   so any LLM effect is measured against a real, capacity-fair baseline.

---

### Reproduction

```bash
# controlled pilot (writes gitignored runs/, sanitized outputs/)
python scripts/capacity_controlled/run_experiment.py --budget 25 --seeds 0,1,2,3,4 --tag pilot
python scripts/capacity_controlled/analyze.py --tag pilot
python scripts/capacity_controlled/legacy_posthoc.py
# legacy capacity audit (needs the gitignored legacy store)
python scripts/capacity_controlled/audit_legacy_capacity.py --store runs/pilot_t1/results.sqlite
```

Machine-readable results: `experiment_config.json`, `candidate_results.csv`,
`run_summary.{csv,json}`, `baseline_summary.csv`, `statistical_analysis.json`,
`legacy_posthoc.json`, `experiment_manifest.json`. Figures: `fig01`–`fig13`
(+ `legacy_capacity_audit`, `legacy_posthoc`), PNG + SVG.
