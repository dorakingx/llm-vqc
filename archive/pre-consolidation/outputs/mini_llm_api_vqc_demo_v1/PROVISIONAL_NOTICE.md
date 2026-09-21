# PROVISIONAL NOTICE — mini_llm_api_vqc_demo_v1

**Status: RESOLVED.** A corrected real execution has completed from the
clean, committed implementation at git SHA `b57a84c` (see
`EXPERIMENT_CONFIG.json`'s `execution_code_sha` in the current
`report.md`/`selected_circuits.json`/etc. for this directory's present
contents). All defects listed below are fixed in that corrected run. The
original provisional run's raw result store is preserved at
`runs/mini_llm_api_vqc_demo_v1/results.sqlite.provisional_run1` for
auditability; this directory's files now reflect the corrected run.

This notice is kept (not deleted) as the historical record of what was
wrong with the first run and why. It no longer describes the current
contents of this directory.

---

**Original notice (first, uncorrected run) below:**

The artifacts currently in this directory were produced by the **first**
real OpenAI run (committed in `5640347`). A subsequent review found several
blocking issues. The artifacts are **kept for auditability** and are **not
deleted or silently rewritten**, but they must **not** be presented as the
final corrected experiment. A corrected real run — executed from a clean,
committed implementation SHA — will overwrite these files.

## Why these measurements are provisional

1. **Cross-arm duplicate contamination.** The first run used a single
   shared candidate cache keyed only by `(task, structural_hash,
   train_seed)`. Because the LLM Open-loop arm proposed architecture
   `RY / CNOT(0→1) / RY` and the LLM Closed-loop arm's proposal 1 proposed
   the *same* architecture, closed-loop proposal 1 was recorded as a
   **duplicate of another arm's** candidate (not trained fresh). Its
   duplicate status — and therefore the feedback the closed-loop LLM saw
   before proposal 2 — depended on arm execution order.

2. **Closed-loop feedback omitted metrics on the duplicate branch.** The
   feedback prompt used mutually-exclusive prose branches: for a duplicate
   (or invalid) prior proposal it printed only "duplicated an earlier one"
   and **omitted validation RMSE, circuit depth, and two-qubit-gate
   count**. So it is **incorrect** to claim closed-loop proposal 2 received
   validation RMSE in this run — on the duplicate branch it did not.

3. **Initialization was not explicit.** Training relied on PyTorch/PennyLane
   dependency-default initialization rather than a declared, versioned
   policy.

4. **All-parameter float64 was not guaranteed.** PennyLane's `TorchLayer`
   initializes its weights as `float32` by default; the quantum rotation
   angles in the first run were therefore float32, not float64.

5. **Initial quantum angles were not recorded.** Only the learned angles
   were stored, so "did the angles actually change, and from what" was not
   auditable.

6. **Execution provenance pointed to the base commit.** The run recorded
   git SHA `24d6f1e` (the base commit) as though it were the implementation
   SHA, even though the worktree contained uncommitted implementation
   changes.

## What the corrected implementation changes

- **Arm-local duplicate detection** via an arm-namespaced cache
  (`task::arm`): each arm has an independent duplicate history; arm
  execution order can no longer change any arm's duplicate status or
  prompts; the closed-loop arm never receives information derived from
  another arm.
- **Complete, deterministic closed-loop feedback**: a single JSON-like
  block that always includes architecture, validation RMSE (or explicit
  null), validity, duplicate status, circuit depth (or null), and
  two-qubit-gate count (or null) — never any protected-test field.
- **Explicit versioned initialization** (`mini_demo_init_v1`: embedding &
  head Xavier-uniform, biases zero, quantum angles Uniform[-π, π]),
  `diff_method="backprop"`, every trainable tensor float64 on CPU, with
  fail-loud checks (exactly 4 quantum / 51 total parameters).
- **Initial and learned quantum angles recorded** for every selected arm.
- **Corrected provenance**: the real HEAD SHA and a dirty flag, plus
  Python and library versions.
- **Random search samples uniformly over the 36 canonical architectures**
  (not the 54 raw field combinations).
- **Provider built with `max_retries=0`**; a positive `LLM_API_BUDGET_USD`
  and an explicit `OPENAI_MODEL` are required before any request.
