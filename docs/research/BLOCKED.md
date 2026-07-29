# BLOCKED — real-LLM cells await an explicit spending cap

Status at 2026-07-29 (UTC). Everything runnable without a paid API call
is running or complete; this file documents the ONE genuine external
blocker (goal §2, blocker class 1) and the exact resume path.

## What is blocked, exactly

The scientific real-provider LLM cells (mocks count only for tests):

| Experiment | Cells | Arms |
|---|---|---|
| E1 | 60 | `llm_open_joint`, `llm_closed_joint` (3 task-n pairs × 10 replicates) |
| E2 | 80 | `llm_open_structure`, `llm_archive_closed_structure` (4 tasks × 10) |
| E3 | 80 | same two structure arms (8 task-n combos × 5) |
| E5 | dependent | needs E1 LLM cells' selected architectures |
| E4 (LLM rows) | dependent | needs E2/E3 LLM cells' selected circuits |

## Evidence of the blocker

- `.env` (worktree + main repo root) defines `OPENAI_API_KEY` and
  `OPENAI_MODEL` only (variable names inspected; values never read).
- `LLM_API_BUDGET_USD` is absent from the environment and from `.env`.
- Controlling policy (frozen; `llm_vqc/llm/budget.py`, goal §3.7): paid
  API calls are prohibited unless BOTH a credential AND a positive
  explicit cap are present — a key alone is not authorization, and the
  session does not self-authorize spending. `preflight_real_run` refuses
  before any client is constructed (covered by
  `tests/test_bench_v2_llm_arms.py::test_real_provider_requires_explicit_cap`).

## What is NOT blocked (running / complete)

- E0 provenance replication: COMPLETE, exact match, zero API calls.
- E1 classical joint cells (60/60): COMPLETE.
- E2/E3 classical + reference cells: running via the detached driver
  (`runs/bench_v2/logs/driver_classical.log`).
- All infrastructure, tests, break-tests, analyses of completed cells.

## Cost expectation for the cap

Every call reserves a conservative $0.05 against the cap
(`COST_ESTIMATE_PER_CALL_USD`; true mini-model cost is far lower), and
each cell has a hard 40-call cap. The full LLM matrix is ~220 cells and
an estimated ~1,500–2,200 calls:

- **`LLM_API_BUDGET_USD=120`** lets the whole matrix run uninterrupted
  under the conservative reservation; **actual** spend with a mini-class
  model (e.g. the committed E0 run's gpt-5.4-mini) is expected around
  $5–20.
- A smaller cap (e.g. 30) also works: the run stops cleanly when the cap
  binds and RESUMES from its stores when re-invoked with a fresh cap —
  no work is lost, but the matrix completes in several installments.

## Exact resume command

```bash
cd /Users/hatanakatomoya/Developer/Sim/llm-vqc/.claude/worktrees/github-author-config-faee6e \
  && export LLM_API_BUDGET_USD=120 \
  && nohup bash scripts/bench_v2/run_llm_matrix.sh > runs/bench_v2/logs/driver_llm.log 2>&1 &
```

(The script sources `.env` itself for the key/model, enforces the cap
before any request, retries transient failures, resumes cells from their
durable stores, and finishes with E5, E4, and the per-experiment
analyses.)
