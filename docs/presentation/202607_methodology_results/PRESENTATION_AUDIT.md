# Presentation audit — `20260716_GSoC.pdf` vs. the repository

**Audited:** 2026-07-19 · **Deck:** `20260716_GSoC.pdf` (10 slides) ·
**Repository:** `github.com/dorakingx/llm-vqc` · **Method:** every technical
claim and number on every slide was checked against the authoritative source
(`llm_vqc/**`, `scripts/**`, and the durable stores `runs/pilot_t1`,
`runs/groq_pilot_t1`, `runs/mini_llm_experiment`).

This audit does **not** preserve a statement merely because it is already in the
PDF. Findings are graded:

- 🔴 **Incorrect** — the slide states something the code/data contradict. Must change.
- 🟠 **Misleading / imprecise** — technically defensible wording that a domain
  reviewer will read as a stronger claim than the evidence supports.
- 🟡 **Missing** — a gap the restructured package fills with a new figure or doc.
- 🟢 **Verified** — checked and correct; keep as-is.

The companion documents (`01`–`06`) and `figures/` are the corrected package.

---

## Headline finding (🔴 must fix before presenting)

**Slide 8 labels `0.00845 / 0.01897 / 0.02438` as "Mean Test RMSE T1". They are
neither the mean, nor the test metric.** They are the **median _validation_
RMSE at budget = 25** across the 3 seeds of each non-LLM arm.

Reproduced directly from the stored pilot (`runs/pilot_t1/pilot_analysis.json`,
checkpoint 25; and re-derived in `data/pilot_nonllm.json`):

| Arm | Slide 8 value | What it actually is | True **mean test** RMSE | True **median test** RMSE |
|---|---:|---|---:|---:|
| Random | 0.00845 | median **validation** RMSE @ B=25 | **0.00910** | 0.00910 |
| Evolutionary | 0.01897 | median **validation** RMSE @ B=25 | **0.01817** | 0.02081 |
| Greedy | 0.02438 | median **validation** RMSE @ B=25 | **0.02756** | 0.02759 |

Two independent errors compound here: (a) **validation** is presented as
**test**, collapsing the project's central quarantine distinction on the one
slide that shows numbers; and (b) **median** is presented as **mean**. The
values are internally traceable — they match the search-selection metric
exactly — but the label misrepresents what a reader is looking at.

**Correction:** relabel as "Median validation RMSE at B=25 (selection metric)"
and add the protected-test column beside it. See
[`figures/pilot_val_vs_test.png`](figures/pilot_val_vs_test.png), which plots
both quantities side by side, and [`06_RESULTS.md`](06_RESULTS.md).

---

## Slide-by-slide

### Slide 1 — Title
🟢 Title, presenter, repo URL, and the four-phase arc (prototype → audit →
benchmark infra → real-API pilots) are accurate.

### Slide 2 — "Previous Milestone: Autonomous Clifford Exploration"
🟢 Accurate description of the earlier BFS/equivalence-class prototype
(`llm_vqc/circuit_explorer.py`). The self-critical quote ("did not isolate or
quantify the scientific contribution of the LLM") is a fair framing of the pivot.

### Slide 3 — "A Correctness Bug Invalidated the Original State Counts"
🟢 The signed-zero (`-0.0` vs `+0.0`) hashing bug and its fix (canonicalize zero
after rounding) match `llm_vqc/circuit_explorer.py` and `DECISIONS.md`. Corrected
counts (N₂ saturation 82→60; N₃ depth≤5 893→666; N₃ saturation 1780→1080) are
consistent with the archived outputs. Keep.
🟡 *Optional:* none of these counts are re-derived in the shipped presentation
package because they belong to the superseded prototype, not LAQS-Bench.

### Slide 4 — "Research Pivot / Central Research Question"
🟢 The central question (same IR, budget, training, feedback, task → does LLM
guidance beat simple search?) is exactly what the codebase operationalizes.
🟠 **Five vs. six ambiguity.** The slide lists **six** comparison conditions
(Random, Evolutionary, Greedy, LLM Open-loop, LLM Closed-loop, LLM Evolutionary)
but Slide 10 says "**five** distinct search arms." Both are true of different
things and the deck never says which:

- **Five implementation classes** (one file each):
  `random_arm.py`, `evolutionary_arm.py`, `greedy_arm.py`, `llm_iter_arm.py`,
  `llm_evo_arm.py`.
- **Six experimental conditions:** open-loop and closed-loop are the **same
  class** `LLMIterArm`, distinguished only by its `open_loop` constructor flag
  (verified in `llm_vqc/search/arms/llm_iter_arm.py`). So 5 classes → 6 conditions.

**Correction:** state it as "five arm implementations → six experimental
conditions (open/closed-loop share one class)." See
[`04_ARCHITECTURE_SEARCH_AND_SELECTION.md`](04_ARCHITECTURE_SEARCH_AND_SELECTION.md).

### Slide 5 — "One Shared Circuit Language for Every Search Arm"
🔴 **Code shown does not match the schema.** The slide's `CircuitIR` snippet
reads `num_qubits: int` and `layers: list[NonRepeatLayer]`. The real schema
(`llm_vqc/ir/schema.py`) uses **`n_qubits`** and **`layers: list[Layer]`** where
`Layer` includes `RepeatBlock` (repeat blocks *are* allowed at top level; only
_nested_ repeats are disallowed). Also the real `CircuitIR` carries `encoding`
and `measurements` fields the snippet omits.
🟡 The slide introduces the IR abstractly but **never shows a concrete candidate
VQC or its initialized trainable parameters** — the single most-requested
missing artifact. Filled by [`figures/circuit_greedy_s1.png`](figures/circuit_greedy_s1.png)
and [`figures/circuit_random_s0.png`](figures/circuit_random_s0.png) (actual
selected circuits, drawn from their stored IR) and
[`02_MODEL_AND_PARAMETERS.md`](02_MODEL_AND_PARAMETERS.md).
🟢 "Strict Pydantic validation, no arbitrary code execution" and "structural
hashing via canonical JSON for duplicate detection" are accurate.

### Slide 6 — "Search Feedback and Final Test Are Structurally Separated"
🟢 The quarantine claim is real and, unusually, *structural*: `harness.py` takes
`TrainValData` (no test attribute) and returns `EvaluationResult` (no test
field); `final_test.py` is never imported by the search path. This is the
project's strongest methodological point and is under-sold as a bullet.
🟠 "Fixed Training (AdamW, 20 Epochs)" is correct but **opaque**: the slide never
explains the forward → loss → backward → update cycle, the learning rate (0.05),
batch size (16), LR-decay schedule (×0.5 at epochs 7/13/17), or that MSE is the
loss while RMSE is the reported metric. Filled by
[`03_TRAINING.md`](03_TRAINING.md) and
[`figures/training_curves_random_s0.png`](figures/training_curves_random_s0.png).
🟡 No visual of the boundary. Filled by
[`figures/validation_vs_test_quarantine.png`](figures/validation_vs_test_quarantine.png).

### Slide 7 — "Measuring Both Circuit Properties and Search Behavior"
🟢 Diagnostics (expressibility via KL-to-Haar, Meyer–Wallach entanglement,
gradient variance) exist under `llm_vqc/diagnostics/`, and the "high
expressibility ≠ task performance" caution is scientifically sound.
🟠 The arm grid on this slide again shows **six** conditions — fine, but reconcile
with Slide 10's "five" per the Slide 4 note.

### Slide 8 — "The Infrastructure Reached Real VQC Training and Real API Execution"
🔴 **"Mean Test RMSE" mislabel** — see the headline finding above. This is the
priority fix.
🟠 **"225 real VQC training evaluations."** 225 = 9 runs × B=25 *candidate
evaluations*, but duplicates are cache hits that are **not** re-trained. The
ledgers record 183 unique within-arm trainings (random 75, evolutionary 39,
greedy 69), fewer still with global cross-run caching (the store holds 222 total
evaluation rows including the separate LLM runs). Say "225 candidate evaluations
(B=25 × 9 runs); duplicates served from cache, not retrained."
🟠 **OpenAI mini claim.** "Validated Open-loop and Closed-loop using
`gpt-5.4-mini-2026-03-17`… B=2 mini-experiment confirmed end-to-end
communication." Verified against `runs/mini_llm_experiment` and
`docs/presentation/mini_llm_experiment/`: 4 real calls, n=1 seed, B=2, 1 epoch,
9.8 s wall-clock. Correct — but the slide should state n=1/B=2/1-epoch so nobody
reads it as a performance result. (Its RMSEs are ~0.13–0.29, i.e. 1-epoch smoke
numbers, an order of magnitude worse than the 20-epoch pilot — *not* comparable.)
🟢 Kruskal–Wallis p = 0.079 and "n=3, descriptive, underpowered" match
`pilot_analysis.json` (checkpoint 25 p = 0.07939) and are appropriately hedged.
🟡 The slide is a table with no graphical evidence. Filled by
[`figures/pilot_val_vs_test.png`](figures/pilot_val_vs_test.png) and
[`figures/pilot_anytime_curves.png`](figures/pilot_anytime_curves.png).

### Slide 9 — "Groq Pilot Revealed Practical Model and Rate-Limit Failures"
🔴 **"triggering system timeouts."** The stored evidence
(`docs/presentation/groq_pilot/`, `runs/groq_pilot_t1`) shows **severe free-tier
rate-limit backoff** (117.34–768.20 s per stored call on seed 1) and a
**deliberate experiment interruption** — not a system timeout. Reword to "severe
free-tier rate-limit backoff and experiment interruption."
🟢 "83.9% invalid rate" is exact (26 of 31 proposals across the two started
`llm_open` cells), and the root cause (wire fields emitted as strings, not
integer arrays) is confirmed in the stored validation errors.
🟢 "Mean latency 9 s → over 600 s" matches the two cells (seed 0 mean 9.38 s;
seed 1 mean 657.69 s).
🟠 The slide **emphasizes failure status but hides the real evidence** already in
the repo: validation/test RMSE for the 9 completed non-LLM cells, anytime curves,
proposal-quality bars, and token/latency plots
(`docs/presentation/groq_pilot/*.png`). Surface at least the anytime and
LLM-usage plots so the slide shows what *did* run, not only what stopped.
🟡 Run matrix: the deck's grid is directionally right but the sanitized report is
precise — **9 complete non-LLM cells, 2 interrupted `llm_open` cells, 7
not-started cells.** Use those counts.

### Slide 10 — "Current Status and Next Steps"
🟢 Contributions and next steps (B=60 full runs, ablations, QMLHEP paper) are
consistent with `LLM-VQC_MASTER_PLAN.md`.
🟠 "Implemented and validated five distinct search arms" — keep, but add "(six
experimental conditions)" per the Slide 4 reconciliation.

---

## Cross-cutting gaps the restructured package closes

| Gap on the deck | New artifact |
|---|---|
| No concrete candidate VQC / initialized params shown | `figures/circuit_greedy_s1.*`, `figures/circuit_random_s0.*`, `02_MODEL_AND_PARAMETERS.md` |
| Architecture vs. quantum-angle vs. classical-embed vs. classical-head vs. input-encoding params never distinguished | `figures/model_parameter_taxonomy.*` (5-role diagram) |
| Forward/loss/backward/update never explained | `03_TRAINING.md` |
| Regression target never stated plainly | `01_TASK_AND_DATA.md` (target = peak position μ) |
| Input data / target generation never visualized | `figures/t1_data_generation.*`, `figures/t1_example_curves.*` |
| Parameter-initialization policy never stated | `03_TRAINING.md` (§ init) |
| No training-loss or validation-RMSE curves | `figures/training_curves_random_s0.*` |
| No predicted-vs-true plot | `figures/pred_vs_true_random_s0.*` |
| How an architecture is chosen from validation not shown | `figures/pilot_anytime_curves.*`, `04_ARCHITECTURE_SEARCH_AND_SELECTION.md` |
| Search-time validation vs protected test under-visualized | `figures/validation_vs_test_quarantine.*`, `05_VALIDATION_AND_FINAL_TEST.md` |
| Tables without graphical evidence | `figures/pilot_val_vs_test.*` and the Groq/mini plot packages |

All figures are regenerated deterministically and offline by
`scripts/build_methodology_figures.py` from committed data
(`data/pilot_nonllm.json`) plus the deterministic task/model/training code; the
re-training determinism check (`random_s0` final val RMSE = 0.008114592) is
asserted at build time.
