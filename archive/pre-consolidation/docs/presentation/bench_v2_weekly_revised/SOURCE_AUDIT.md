# Source-of-truth audit for the revised GSoC benchmark deck

Every scientific statement in `20260731_GSoC_revised.pptx` is traced here
to code or a durable store. Where the previous deck
(`docs/presentation/bench_v2_weekly/`, preserved unchanged) asserted
something this audit could not confirm, the correction is recorded in
§9.

## 1. Repository state

| Item | Value |
|---|---|
| Audit date | 2026-07-31 |
| Base commit (branch point) | `f159d3597daaaf3584296c857f360d0a6fa5b7d4` |
| Base branch | `research/rigorous-qas-benchmark-v2` |
| Working branch | `presentation/rebuild-gsoc-benchmark-v2` |
| Working tree at branch point | clean (no uncommitted or untracked work) |
| Running experiment processes | none |
| Remote | https://github.com/dorakingx/llm-vqc.git |

## 2. Protocol reference (immutable content, not external registration)

- Protocol text: `docs/research/BENCHMARK_V2_PROTOCOL.md`
- Machine-readable mirror: `configs/bench_v2/protocol_v2.yaml`
- Content hashes: `outputs/bench_v2/protocol_hashes.json`
  (`BENCHMARK_V2_PROTOCOL.md` = `9b6dd316590fa8aa…`,
  `protocol_v2.yaml` = `4b6430eb07520237…`), verified by goal-contract
  criterion `C02`.
- Commits that touched the protocol: `9362fea` (Phase 0c, initial freeze,
  before any v2 experiment ran) and `e2afa37` (Phase 6a, encoding
  correction only: E1 re-expressed as explicit `task_n_pairs` because the
  legacy T1 profile is pinned at n=3; arms, budgets, checkpoints and
  replicate counts unchanged).

**There was no external preregistration.** The deck therefore says
"protocol-frozen" / "pre-specified", never "preregistered".

## 3. Experiment status (from `runs/bench_v2/*/cells/*.json`)

Regenerated for this deck; identical to
`docs/presentation/bench_v2_weekly/data/matrix_status.csv`.

| Experiment | Cells | Expected | Complete | Failed | Pending |
|---|---|---|---|---|---|
| E0 | provenance replay | 1 | 1 | 0 | 0 |
| E1 | classical (Track B) | 60 | 60 | 0 | 0 |
| E1 | real-LLM | 60 | 0 | 0 | 60 |
| E2 | classical + references (Track A) | 280 | 280 | 0 | 0 |
| E2 | real-LLM | 80 | 0 | 0 | 80 |
| E3 | classical + reference (Track A) | 120 | 120 | 0 | 0 |
| E3 | real-LLM | 80 | 0 | 0 | 80 |
| E4 | shot/noise robustness | 1 | 0 | 0 | 1 |
| E5 | θ-isolation | 1 | 0 | 0 | 1 |

**Classical cells: 460/460 complete, 0 failed. Real-LLM cells: 0/220.
E4 and E5 pending (they depend on the LLM cells).** The benchmark as a
whole is therefore **not** complete, and the deck never says it is.

`scripts/check_goal_completion.py --skip-commands` reports **9/17
criteria pass**. The failing criteria (C06 n=8 LLM cells, C07/C09/C17
matrix completeness, C08 real-LLM cells, C12 E4, C16 final report) fail
*correctly* because the LLM matrix has not run. The checker was not
modified.

## 4. Definitions used by the deck

| Concept | Implementation | Notes for the deck |
|---|---|---|
| Track A | `llm_vqc/bench_v2/track_a_evaluator.py` | Arms propose **structure only**; one shared AdamW loop trains θ. |
| Track B | `llm_vqc/bench_v2/track_b_evaluator.py` → preserved `free_amplitude/main_eval.py` | Complete candidates (gates+wires+θ); θ evaluated **verbatim**, module imports no optimizer (AST-tested). |
| Budget axis | `llm_vqc/bench_v2/runner.py` (`while ledger.num_unique < budget_limit`) | **Unique candidate evaluations.** Duplicates/invalids are recorded but never satisfy the budget. |
| Shared trainer | `configs/bench_v2/sizing_freeze.json` | AdamW, lr 0.05, **40 epochs, batch 32**, identical for every arm; splits 256 train / 256 val / 2048 test. |
| θ seed | `free_amplitude_train_seed(run_seed, structural_hash, config_version)` with `run_seed = f(task, data_seed, search_seed)` | Same architecture ⇒ same trained result in any arm (global cache). |
| Paired seeds | `cell_runner.py`: `data_seed = 1000+r`, `search_seed = 2000+r` | Every arm in a comparison sees the identical block. |
| Protected test | `llm_vqc/bench_v2/test_gate.py`, called only by `cell_runner` after `runner.run()` returns | Evaluated **once**, after selection is frozen; search-side modules cannot import it (AST-tested). |
| Fixed references | `FixedReferenceArm` in `arms/structure_arms.py` | Proposes its template **once**, then returns `None` → `stop_reason="arm_exhausted"`. **Not budget-matched**; they are anchors evaluated through the identical training/test pipeline. |

## 5. Statistics (implementation: `scripts/bench_v2/analyze_experiment.py`)

- Paired tests within each (task, n) family: exact **Wilcoxon signed-rank**
  and a **paired sign-flip permutation test** on the mean difference
  (10,000 Monte-Carlo resamples, seed 20260729).
- **Multiplicity:** Holm step-down applied to the permutation p-values
  *within* each (task, n) family.
- **Effect sizes:** **Hodges–Lehmann shift** (median of Walsh averages of
  the paired differences, in RMSE units) and **paired Cliff's delta**
  ((#positive − #negative)/n).
- Bootstrap 95% CIs of the median (10,000 resamples) in
  `stats_arm_summaries.csv`.

**Verified minimum attainable p-value** (measured, not assumed, by
re-running the shipped test functions):

| Replicates | Exact Wilcoxon two-sided min | Monte-Carlo permutation min | Holm × 3 comparisons |
|---|---|---|---|
| 5 (E3) | 0.0625 = 2/2⁵ | ≈0.061 (MC noise around 2/32) | **≈0.19** |
| 10 (E2) | 0.00195 | ≈0.0016 | ≈0.005 |

So **no E3 comparison can reach α = 0.05 after Holm correction, by
construction** — a design limit of 5 replicates, not evidence of absence.
The deck states this explicitly.

## 6. Simulation regime

All reported results, **including n = 8**, are **noiseless analytic
statevector simulations**: PennyLane `default.qubit` with
`diff_method="backprop"`, float64, expectation values computed exactly
(`llm_vqc/ir/compiler_pennylane.py:139`). No shot sampling, no noise
model, **no quantum hardware**.

Finite-shot (1,024 / 4,096) and the frozen `depol_ro_v1` noisy profile
are implemented in `llm_vqc/bench_v2/resources.py` but belong to **E4,
which has not run**.

## 7. Resource metrics

`llm_vqc/bench_v2/resources.py`:

- Logical counts from the shared IR (1q, 2q, controlled-rotation,
  parameters, depth).
- Transpiled counts via **Qiskit**, basis `[rz, sx, x, cx]`,
  `optimization_level=1`, `seed_transpiler=7`, coupling maps
  **line / ring / all-to-all**. SWAPs introduced by routing are
  decomposed into `cx`, so `transpiled_two_qubit_count` includes routing
  overhead.
- **Amplitude state preparation is deliberately excluded** from the
  transpiled counts (`_body_only_circuit`, "no state prep"). It is
  identical for every arm at a given n, so excluding it isolates the
  searched body — but it means the numbers are a **relative proxy**, not
  a total hardware cost. The deck says exactly this.

## 8. Diagnostic circuits (28)

`docs/presentation/bench_v2_weekly/build_deck_figures.py::fig_e2_diagnostics`
selects, for **each (task, arm) pair in E2**, the replicate whose
selected candidate has the best (lowest) **validation** metric, then
computes expressibility KL and Meyer–Wallach Q on that one circuit.
4 tasks × 7 arms = **28 circuits**. Selection uses validation only.

Diagnostics come from `llm_vqc/free_amplitude/expressibility.py`
(200 sampled states, 2,000 fidelity pairs, θ ~ U[−π, π], seeded by
architecture hash). They are **descriptive circuit properties**; the
protocol (RQ5) treats their association with task performance as an
open empirical question with circuit-size controls, and that analysis
has not been run.

## 9. Corrections applied relative to the previous deck

| # | Previous deck | Verified position in this deck |
|---|---|---|
| 1 | "preregistered benchmark" | "protocol-frozen"; no external registration exists (§2). |
| 2 | "statistically indistinguishable" / "statistically close" | "No Holm-adjusted difference was detected at the current sample size", plus effect sizes. Not an equivalence claim. |
| 3 | Prior work had "no seed-replicated statistics" | Narrowed to what this project verified: prior work does not isolate the proposal strategy against **budget-matched** classical search with paired replicates and a protected test. |
| 4 | "Accuracy is bought with hardware cost" | "Searched circuits reach lower RMSE with larger compiled bodies" — a Pareto trade-off, with the 2q count named a **proxy** that excludes state preparation. |
| 5 | Headline cited only `random vs StrongEnt-d2, p_Holm = 0.032` | That is the **weakest** of the reference contrasts (HL = −0.016). The deck now reports the systematic result: on T1 and T2 **every** searched arm beats **every** shallow reference, Cliff's δ = −1.00, p_Holm ≈ 0.021–0.046. |
| 6 | E3 framed as "reference degrades with n while search improves" | Verified as a **crossover**: at n=3 on T2 the reference is *better* (HL = +0.033, δ = +1.0); search wins from n=4 on. |
| 7 | Slide 4 said "(E3, running)" while slide 8 said complete | One status everywhere, generated from the stores. |
| 8 | "n=8 actually simulated" | "noiseless statevector simulation, no shots, no noise, no hardware". |
| 9 | "expected actual cost ≈ $5–20" | Replaced by a reproducible derivation (§10) with the model dependency stated honestly. |
| 10 | Title implied a completed main matrix | Title states classical baselines complete, LLM evaluation pending. |

## 10. API budget derivation (reproducible)

Constants, from code:

- `BATCH_SIZE = 3` candidates per call (`llm_vqc/bench_v2/llm_arms.py:47`)
- `PER_CELL_CALL_CAP = 40` (`llm_vqc/bench_v2/llm_providers.py:26`) —
  a **failure guard**, not expected usage
- `COST_ESTIMATE_PER_CALL_USD = 0.05`
  (`llm_vqc/free_amplitude/openai_provider.py:49`) — the amount reserved
  against the cap **before** each call, deliberately conservative
- Budgets: E1 B=16, E2 B=24, E3 B=16 unique evaluations per cell

Expected successful calls per cell = ceil(B / 3) plus a small allowance
for duplicate/invalid proposals: **6 calls** (E1/E3, B=16) and **8 calls**
(E2, B=24).

| Experiment | Cells | Calls/cell | Calls |
|---|---|---|---|
| E1 | 60 | 6 | 360 |
| E2 | 80 | 8 | 640 |
| E3 | 80 | 6 | 480 |
| **Total** | **220** | — | **≈1,480** |

- **Reserved against the cap** (what the guard actually charges):
  1,480 × $0.05 = **$74**; worst case at the 40-call guard for every
  cell would be 8,800 × $0.05 = $440, which the global cap must bound.
- **Actual spend** depends on the model, which is read from
  `OPENAI_MODEL` at run time and is **not currently pinned in the
  protocol**. Prompts are ~600–900 input tokens and ~300–600 output
  tokens per call. At a mini-class rate the true cost is well under the
  reserved figure; at a frontier-model rate it can exceed it. Because
  the model is not yet fixed, this deck presents a **model-specific
  estimate table in the appendix** instead of a single dollar figure,
  and asks for approval of the **hard cap**, which is the quantity that
  is actually enforced.
- Enforcement: `LLMApiBudget.from_env` refuses to run without a positive
  `LLM_API_BUDGET_USD`; `BudgetEnforcedProvider` checks affordability
  before every request; `CallCountLimitedProvider` caps calls per cell;
  retries sit outside both wrappers so each retry is itself checked.
  `LLM_API_BUDGET_USD` is currently **unset**, so all LLM cells are
  blocked by policy.

## 11. Files consulted

`docs/research/BENCHMARK_V2_PROTOCOL.md`,
`docs/research/LITERATURE_REVIEW_QAS.md`,
`docs/research/BREAK_TESTS.md`, `docs/research/BLOCKED.md`,
`GOAL_CONTRACT.yaml`, `scripts/check_goal_completion.py`,
`scripts/bench_v2/analyze_experiment.py`, `configs/bench_v2/*.{yaml,json}`,
`llm_vqc/bench_v2/*.py`, `llm_vqc/free_amplitude/{model,main_eval,expressibility}.py`,
`runs/bench_v2/*/cells/*.json`,
`outputs/bench_v2/{E0,E1,E2,E3}/*.{csv,json}`,
`outputs/free_amplitude_fixed_readout_v1/EXPERIMENT_CONFIG.json`.
