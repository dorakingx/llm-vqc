# Deliberate break-test transcripts (goal §19)

Protocol: temporarily break a critical safeguard, prove the intended test
FAILS, restore, prove it passes again. Performed 2026-07-29 on
`research/rigorous-qas-benchmark-v2`; every break was reverted via
`git checkout --` immediately after the failing run (verified by the
final passing run of each block). No break was ever committed.

## Break 1 — Track B "no optimizer" invariant

Break: insert `import torch.optim` into `llm_vqc/free_amplitude/main_eval.py`.

- **First attempt: the preserved `test_no_optimizer_in_main_mode` did NOT
  fail** — it checked `ast.ImportFrom` (`from torch.optim import ...`)
  and `AdamW/SGD/Adam` attribute references, but a plain `import
  torch.optim` is an `ast.Import` node. Genuine safeguard gap found by
  the break-test.
- Fix: the test now also rejects `ast.Import` aliases containing
  `optim` (comment in the test points here).
- Retry transcript:
  - broken: `FAILED tests/test_free_amplitude_joint_search.py::test_no_optimizer_in_main_mode` (1 failed, 12 deselected)
  - restored: `1 passed, 12 deselected`

## Break 2 — protected-test quarantine (search side cannot import the gate)

Break: insert `from llm_vqc.bench_v2.test_gate import evaluate_selected_on_test`
into `llm_vqc/bench_v2/track_a_evaluator.py`.

- broken: `FAILED tests/test_bench_v2_evaluators.py::test_search_side_modules_cannot_reach_test_data[llm_vqc/bench_v2/track_a_evaluator.py]` (1 failed, 4 passed)
- restored: `5 passed`

## Break 3 — unique-evaluation budget axis

Break: make `BudgetLedger.num_unique` return `num_proposed` (duplicates
and invalids would silently satisfy the budget).

- **First attempt did NOT fail**: the existing budget test asserted
  `ledger_summary["num_unique"] == B` — measured through the same broken
  property (self-consistent break). Gap closed by adding an independent
  behavioural test, `test_duplicates_never_satisfy_unique_budget`
  (an always-duplicate arm must hit the pathological-arm guard, never
  falsely complete).
- Retry transcript:
  - broken: `FAILED tests/test_bench_v2_arms.py::test_duplicates_never_satisfy_unique_budget` (1 failed, 15 deselected)
  - restored: `1 passed, 15 deselected`

## Outcome

Two of three first-attempt breaks exposed real blind spots (a safeguard
that looks tested but is not is worse than an untested one); both are now
closed with strictly stronger tests. All three safeguards demonstrably
fail when broken and pass when restored.
