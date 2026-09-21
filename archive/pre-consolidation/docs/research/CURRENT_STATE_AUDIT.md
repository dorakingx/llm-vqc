# Current-State Audit — Rigorous QAS Benchmark v2 preflight

Date: 2026-07-29 (UTC)
Auditor: Claude (principal implementation/experiment engineer session)
Working branch created by this audit: `research/rigorous-qas-benchmark-v2`

## 1. Git preflight (recorded before any edit)

| Item | Value |
|---|---|
| Worktree | `.claude/worktrees/github-author-config-faee6e` (dedicated session worktree of `dorakingx/llm-vqc`) |
| Branch found at session start | `claude/code-llm-vqc-research-ace654` @ `734122e8b004392a20bcad82d4a370a473edbbd4` (== `origin/main`, == local `main`) |
| `git status --short --branch` | clean (no staged, unstaged, or untracked tracked-path changes) |
| `git diff` / `git diff --stat` | empty |
| Audit reference branch | `experiment/free-amplitude-fixed-readout-v1` exists **locally and on origin**, both exactly at `2b9084af716c9eebee01529981c1d25f670b05b0` — identical to the external-audit reference. The local checkout is **not** ahead of it and has no uncommitted work on it. |
| Git author identity | `Doraking <120563040+dorakingx@users.noreply.github.com>` (correct; must be preserved for all commits) |

### 1.1 History topology (important)

`main` and `experiment/free-amplitude-fixed-readout-v1` have **no merge base**
(`git merge-base` exits 1; disjoint histories). Two lineages exist:

1. **`main` lineage** (19 commits): original Clifford circuit-explorer work →
   LAQS-Bench framework commit `0c3141a` → Groq/OpenAI pilot publishing →
   presentation packages (HEAD `734122e`).
2. **Experiment lineage** (42 commits, rewritten history that re-implements and
   supersedes the `main` science code): capacity-controlled chain
   `experiment/capacity-controlled-t1-v1` → `-t1-v2` → `-t2-v1` →
   `-higgs-v1` → `higgs-data-scale-qualification-v1` (`24d6f1e`) →
   `experiment/free-amplitude-fixed-readout-v1` (`2b9084a`, this goal's start
   point). None of these experiment branches are merged into `main`.

The experiment lineage contains the authoritative science framework: typed
`CircuitIR` (`llm_vqc/ir/`), shared `SearchRunner`/`SearchArm`
(`llm_vqc/search/`), evaluation harness + SQLite `ResultStore` + budget
ledger (`llm_vqc/evaluation/`, `llm_vqc/ir/budget.py`), diagnostics
(`llm_vqc/diagnostics/`), the complete `llm_vqc/free_amplitude/` package,
`LLM-VQC_MASTER_PLAN.md`, `DECISIONS.md`, 55 test files, and 363 committed
sanitized artifact files under `outputs/`.

**Decision:** `research/rigorous-qas-benchmark-v2` was created from
`2b9084af716c9eebee01529981c1d25f670b05b0` (the goal's mandated start point,
verified bit-identical to the remote audit reference). Nothing was deleted,
rebased, force-pushed, or merged.

## 2. Process / job inspection

- `ps aux` filtered for `llm-vqc`, python experiment scripts, tmux, screen:
  **no experiment processes running** (only desktop apps: Claude, Notion,
  Docker Desktop idle, Spotify).
- `tmux`: not installed. `screen -ls`: no sockets.
- **Conclusion: no active or paid run exists; nothing to attach to or
  preserve in-flight.**

## 3. Durable result stores

| Store | Location | Contents | Status |
|---|---|---|---|
| `runs/free_amplitude_fixed_readout_v1/free_amplitude_fixed_readout_v1.sqlite` (this worktree, gitignored) | 64K | run_meta: budget=2, seeds=1, arms `[random, scripted_open_loop, scripted_closed_loop]`, epochs=5, git SHA `24d6f1e`, 2026-07-23T19:40Z; 5 evaluations | **Early smoke only** (pre-dates the final no-optimizer mode). Not resumable toward any goal cell; preserved untouched. |
| `runs/mini_llm_api_vqc_demo_v1/` (this worktree, gitignored) | 256K | mini demo results.sqlite + provisional | Demo artifact; preserved untouched. |
| Sibling worktree `llm-vqc-architecture-plan-8417fd/runs/` | 12M | `pilot_t1`, `groq_pilot_t1`, `groq_smoke`, `mini_llm_experiment`, `search_smoke`, `dummy_phase0_smoke` | Historical pilots (main-lineage framework); preserved untouched. |
| Sibling worktree `llm-vqc-presentation-restructure-605dda/runs/` | 43M | `capacity_controlled_{t1_v1,t1_v2,t2_v1,higgs_v1}`, `higgs_data_scale_qualification_v1` | Capacity-controlled experiment stores (experiment lineage); preserved untouched. |
| Committed sanitized artifacts | `outputs/free_amplitude_fixed_readout_v1/` on `2b9084a` | EXPERIMENT_CONFIG.json, candidate_trace.csv, cross_seed_summary.json, per-arm figures + CSVs, selected circuits | **This is the E0 replication source**: REAL-LLM joint-search run, OpenAI, n=3, B=4/arm, 2 seeds, arms `[random, llm_open_loop, llm_closed_loop]`, executed at clean SHA `712dbe4`, no optimizer, 0 classical params. |

**No partial resumable experiment toward the v2 goal exists.** All prior
stores are complete historical runs; none will be overwritten. New protocol
work uses new run namespaces (`runs/bench_v2_*`).

## 4. What already exists and must be preserved (audit-not-reimplement)

Verified directly in the source at `2b9084a`:

- **`llm_vqc/free_amplitude/model.py`** — `FixedReadoutQuantumModel`:
  amplitude encoding (`2**n` features, single normalization inside
  `AmplitudeEmbedding(normalize=True)`), free 1–5-gate body, fixed `Z(q0)`
  readout, `mu_hat = (1 - <Z0>)/2`, structurally zero classical parameters.
- **`llm_vqc/free_amplitude/main_eval.py`** — direct joint mode: complete
  candidates (gates + wires + theta), theta loaded verbatim, **no
  `torch.optim` import** (AST-asserted by
  `tests/test_free_amplitude_joint_search.py`), duplicate identity =
  `candidate_hash` (structure + theta), separate `architecture_hash`;
  protected-test scoring in a separate function/module path.
- **`llm_vqc/free_amplitude/harness.py` + `training.py`** — the
  AdamW-on-quantum-angles-only training loop (explicit
  epochs/batch/lr/betas/eps config; optimizer-params invariant), i.e. the
  natural Track-A inner optimizer for the zero-classical-parameter model.
- **`llm_vqc/search/`** — one `SearchRunner` loop for every arm; arms
  implement `SearchArm` (initialize/propose/update_state/select_final +
  serialized state); unconditional per-proposal checkpointing; exact budget
  termination via `BudgetLedger.from_store` (resume-safe); arms
  structurally cannot reach test data. Existing arms:
  `random_arm`, `evolutionary_arm`, `greedy_arm`, `llm_iter_arm`,
  `llm_evo_arm`, plus `mutation.py` operators.
- **`llm_vqc/evaluation/`** — deterministic seed tree
  (`SeedSequence`-based; `train_seed_for_circuit(run_seed,
  structural_hash)` gives identical inner-training results across arms),
  SQLite result store with global cross-arm cache, metrics, final-test
  quarantine (`TrainValData` has no test field by construction).
- **`llm_vqc/ir/`** — typed layered CircuitIR (pydantic, extra=forbid),
  validators that collect all issues, canonicalization + structural hash,
  PennyLane and Qiskit compilers, budget ledger, IR-level metrics.
- **`llm_vqc/diagnostics/`** — expressibility (KL), Meyer–Wallach
  entanglement, gradient statistics, sampling harness, store.
- **`llm_vqc/free_amplitude/` extras** — candidate schema + canonical theta,
  causal-cone summary, structural cost, per-arm reporting (1061-line
  main_reporting), OpenAI provider with rate-limit-aware retry/backoff and
  model-compat guard, prompt versioning, provenance capture.
- **Tests** — 55 files covering IR, evaluation, search, free_amplitude,
  diagnostics, budget persistence, capacity-controlled experiments.

## 5. Environment

- Shared venv: `/Users/hatanakatomoya/Developer/Sim/llm-vqc/.venv` —
  Python 3.14.4, torch 2.13.0, pennylane 0.45.1, numpy 2.4.6,
  qiskit 2.4.1, openai 2.41.1 (versions cross-checked against committed
  EXPERIMENT_CONFIG.json library_versions).
- Disk: 154 GiB free on the volume; stores total < 60 MB. No constraint.
- `.env` present in this worktree (gitignored). Variable names (values
  never read): `OPENAI_API_KEY`, `OPENAI_MODEL`. **No `LLM_API_BUDGET_USD`
  (or equivalent explicit positive cap) is currently set — therefore paid
  API calls are prohibited at this moment.** Real-LLM cells will require
  the user to export an explicit cap; all earlier phases (0–3, 5, and all
  classical arms) are unblocked regardless. This is recorded now and will
  be re-checked at Phase 4/6/7 gates; it is not yet a blocker because
  substantial unblocked work precedes any needed API call.

## 6. Baseline verification

- Full `pytest tests/ -q` run on `research/rigorous-qas-benchmark-v2`
  (= `2b9084a`, clean): **567 passed, 12 skipped, 0 failed** in 112 s
  (skips are provider/optional-dependency guards). The inherited baseline
  is green before any modification.

## 7. Preservation commitments for this project

1. No existing run store is deleted, rewritten, or reused as a write
   target; v2 experiments write to new namespaces `runs/bench_v2_*`.
2. No branch deletion, history rewrite, force-push, or merge to `main`
   without explicit authorization.
3. Commits are small and phase-scoped, authored as
   `Doraking <120563040+dorakingx@users.noreply.github.com>`.
4. Secrets: `.env` is never read, printed, or committed; provider headers
   and raw keys never enter logs, prompts, or artifacts.
5. Paid API calls only when both a credential AND an explicit positive
   `LLM_API_BUDGET_USD` cap are present in the environment; the preflight
   check is enforced in code before any provider call.
6. Existing committed sanitized artifacts under `outputs/` are treated as
   immutable evidence (E0 provenance source).

## 8. Gap analysis: goal requirements vs existing code

What the v2 goal requires that does **not** yet exist (the implementation
surface for Phases 1–8), versus what is reused:

| Goal requirement | Status at `2b9084a` |
|---|---|
| T1 signal-suite main profile (n=5) + T2 sinusoid-frequency + T3 change-point + T4 waveform classification, all `2**n`-point signals | **Missing.** Existing tasks: T1 Gaussian (angle-encoding hybrid profile in `llm_vqc/tasks/t1_gaussian.py`; amplitude n=3 profile in `llm_vqc/free_amplitude/tasks.py`), T2 = sklearn digits (**prohibited** for v2 — no MNIST/digits), HIGGS. New `llm_vqc/tasks/signal_suite/` package required. |
| Track A structure-only arms on the fixed-readout model with one shared inner optimizer | **Partially exists.** `free_amplitude/harness.py` + `training.py` implement AdamW-on-angles; a structure-only search loop over the free-gate space with global caching keyed by `(task_version, data_seed, search_seed, structural_hash)` must be assembled and generalized to n∈{3..8}. |
| Track B direct joint mode | **Exists** (`main_eval.py`, `main_runner.py`); needs `evolutionary_joint` classical baseline and paired theta-quality analysis (E5). |
| `scalable_layered_v1` search-space profile (3–8 qubits, `max_ops = min(2n,16)`, line/ring/star/pairs/all-to-all) | **Missing** as a frozen profile; IR grammar already supports the primitives. |
| Fixed reference ansatze (RealAmplitudes/StronglyEntangling-style at predeclared depths) | **Missing** as arms. |
| Bounded top-k archive closed-loop LLM arm + batch (2–4/call) proposals | **Partially exists** (`llm_iter_arm` full-history closed loop; free_amplitude open/closed prompts). Archive manager + batch parsing required. |
| Budget ledgers incl. LLM outbound/success/token/cost + inner-optimizer objective/grad evals | **Partially exists** (proposal/eval ledger, llm_calls table); optimizer-call accounting and cost ledger consolidation required. |
| Transpiled resource metrics under line/ring/all-to-all coupling; finite-shot 1,024/4,096; noisy-simulator profile | **Missing** (qiskit compiler exists as substrate). |
| Statistical machinery: paired replicates, bootstrap CIs, permutation tests, Holm correction, effect sizes | **Partially exists** in capacity-controlled analysis scripts; must be generalized into reusable v2 analysis modules. |
| `GOAL_CONTRACT.yaml` + `scripts/check_goal_completion.py` | **Missing.** |
| `docs/research/` protocol/lit-review/audit docs | **Missing** (this file is the first). |

## 9. Baseline test-suite result

`./.venv/bin/python -m pytest tests/ -q` at `2b9084a` on
`research/rigorous-qas-benchmark-v2`:

```
567 passed, 12 skipped in 112.23s
```

Zero failures. This is the preserved baseline; every subsequent phase gates
on keeping it green.
