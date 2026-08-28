# Historical artifacts for the 2026-08-04 GSoC deck (mini5 ctx run)

Recovered on 2026-08-29 from the previously gitignored data directory of the
`github-author-config-faee6e` worktree (`docs/presentation/mini5/data/`),
which the 20260804_GSoC deck cited in its footers ("data/ctx_analysis.json").
Committed here so that slide/paper claims about the historical experiment are
generated from machine-readable artifacts rather than from slide images.

- `ctx_analysis.json` — THE source for the 2026-08-04 deck: 16-gate, 5-qubit,
  B=8, 30-seed run on the four synthetic signal tasks
  (gauss_peak = T1 peak position, sin_freq = T2 frequency,
  change_point = T3 change point, peak_count = T4 one-vs-two bumps).
  Arms: random, evolutionary, llm_open_ctx, llm_closed_ctx, llm_hybrid_ctx,
  reference (fixed ansatz). **There was no Greedy arm in this run.**
  `median_test_rmse` holds per-task medians (lower is better);
  `primary` holds the Holm-corrected paired tests: LLM-open vs random is
  "no difference detected" on all four tasks.
- `condition_comparison.json` — supporting condition analysis from the same
  campaign.
- `per_cell_mini_5gate_v1.csv` — an EARLIER, smaller pilot
  (`mini_5gate_v1`: exactly 5 gates, 3–5 qubits, 10 seeds) that did include a
  greedy arm. It is NOT the run shown in the 2026-08-04 deck; kept only for
  completeness.
