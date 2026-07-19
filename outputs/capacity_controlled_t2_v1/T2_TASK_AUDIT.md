# T2 task audit (`T2_digits_3_vs_8`)

Audited from `llm_vqc/tasks/t2_digits.py`, `tests/`, and `DECISIONS.md`.
**This is a methodology-validation task, NOT a HEP-domain result.**

## Task facts (verified)

| Property | Value |
|---|---|
| Source dataset | `sklearn.datasets.load_digits()` (public-domain UCI optical digits) |
| Classes | digit **3** (label 0) vs digit **8** (label 1) |
| Total samples (this pair) | **357** (183 digit-3, 174 digit-8) |
| Class balance | 0.487 positive — well balanced |
| Raw feature dim | 64 pixels (8×8, 0–16) |
| Post-preprocessing dim | **10** (PCA components) |
| Train / val / test | **214 / 72 / 71** (stratified 60/20/20) |
| Stratification | per-class shuffle then 60/20/20 — balance preserved (train 0.486, val 0.486, test 0.493) |
| Preprocessing | PCA(10) → per-component min-max → ×π, **fit on train only** |
| Target encoding | binary {0,1} |
| Task loss | BCE |
| Task metric (existing) | AUC (`lower_is_better=False`) |
| Test quarantine | `build_test` refits PCA on the **train** indices only (same seed) — test never touches preprocessing fit |

## Leakage / integrity checks (all pass)

- **Preprocessing fit on train only:** ✅ PCA + min-max fit on `train_idx` pixels in
  both `build` and `build_test`; applied unchanged to val/test.
- **Splits disjoint:** ✅ verified 0 overlap (train∩val, train∩test, val∩test) for
  splits 0,1,2 via sample ids.
- **Stratified & balanced:** ✅ each split ≈ 0.486–0.493 positive.
- **Deterministic split:** ✅ `rng = default_rng(data_split_seed(seed, name))`.
- **No duplicated samples:** ✅ sample ids carry the original dataset index; disjoint.
- **Feature range:** ✅ train features in [0, π].

**No leakage, no unstable splitting, no duplication found. No task-code change is
required for correctness.**

## The one issue that DOES require a documented decision: metric saturation

**Audited difficulty:** the 3-vs-8 pair after PCA(10) is *nearly separable*.
- Logistic regression protected-test AUC = **0.996**.
- Every controlled 61-param VQC reaches **validation AUC = 1.0000** (15/15 probed
  circuits; n_unique = 1).

**Consequence:** validation AUC is **non-discriminative** — it cannot rank
candidates, so architecture *selection by validation AUC* would be degenerate
(all candidates tie; the first-seen perfect-AUC circuit "wins"). Using AUC as the
selection metric would make the whole search-arm comparison meaningless for an
uninteresting reason. This is an "ambiguous task semantics" issue for a
selection-based experiment, which the audit is required to resolve.

**Resolution (documented, justified, chosen before the full matrix — see
`PROTOCOL.md`):**

- **Selection metric (search-visible): validation log-loss (BCE)**,
  `lower_is_better=True`. Log-loss stays discriminative even at AUC = 1.0 (probed
  spread 0.0463–0.0618 across circuits) and is exactly the training loss. This is
  an **explicit, documented** switch driven by the audit — not a silent change of
  the project metric. The underlying task loss (BCE) is unchanged.
- **Primary protected-test metric (equivalence): balanced-accuracy error**
  (1 − balanced accuracy at threshold 0.5), `lower_is_better=True` — appropriate
  for the balanced classes, interpretable, and non-saturating on the 71-sample
  test (probed spread ≈ 0.056–0.114).
- **Secondary reported test metrics:** AUROC, accuracy, log-loss, Brier score,
  confusion matrix, ROC/PR curves, calibration.

The controlled T2 VQC's own accuracy (≈0.89–0.94) has genuine spread, so the
protected-test comparison remains meaningful even though AUC saturates.

## Controlled T2 capacity (computed, fixed before execution)

With raw feature dim 10 → embed `Linear(10→4)`:

| Component | Params |
|---|---:|
| Classical embed `Linear(10→4)` | 10·4 + 4 = **44** |
| Quantum angles | **12** |
| Classical head `Linear(4→1)` | 4·1 + 1 = **5** |
| **Total trainable** | **61** |

Every primary T2 controlled candidate must have exactly **61** trainable params
(vs T1's 105 — the difference is only the embed input dim, 10 vs 21). The circuit
IR grammar, validator, and structural limits are identical to T1-v2.
