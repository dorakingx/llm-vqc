# QAE budget-target audit

Date: 2026-09-08 (JST).
Status: historical validation reanalysis completed; new LLM boundary experiments NOT executed.

## Basis and execution

The requested basis is the final slide of `20260904_GSoC`:
https://docs.google.com/presentation/d/14aqubJaVVc4sxlXlY4GxwVti5ULAlQnhl82znxxQ_K8/edit

That slide proposes fixing validation trash-fidelity targets 0.95 and 0.99, requiring at least 10 of the same 12 paired seeds, reusing logs before making new calls, and evaluating only unresolved budget boundaries. This audit uses both stated targets; it is retrospective, not a newly preregistered confirmatory experiment.

Source data: `outputs/qae_robustness/` at commit `e846c27fc21d06aa20e026590fcc06429f8bcfb4`.
Analysis branch: `analysis/qae-budget-target-20260908`.
Executed audit revision: `1b3f5d8cafa4fdb0653a2242694623fa72a12667`.
Successful workflow: https://github.com/dorakingx/llm-vqc/actions/runs/34151502872

The audit processed 2,880 candidate records and 336 selected-result records, including reused control records, across 7 conditions and 4 methods. These are record counts, not counts of independent LLM generations. All 56 condition/method/target endpoint cells were evaluated. No new model calls, training runs, or test evaluations were made.

For every method/condition, the audit checked seeds 0 through 11, contiguous candidate orders, finite validation fidelity, and agreement between each trajectory's validation maximum and its stored selected result. Built-in audit self-tests passed. A separate local run reproduced six analysis output files byte-for-byte. This is not a claim that the repository's entire test suite was executed.

`OPENAI_API_KEY` was not configured in the checked GitHub Actions context or the local execution environment. Only availability was checked; no credential value was exported. This blocks additional paid LLM experiments here, not necessarily in other user environments. No model substitution or fabricated completion was used.

## Metric and decision rule

The metric is **validation trash fidelity**, not reconstruction fidelity. At candidate index k within a configured-budget-B run, compute each seed's maximum validation fidelity among candidates 1 through k. Count seeds whose maximum is greater than or equal to the target. A cell passes when this count is at least 10 out of 12. A mean fidelity above the target is not sufficient.

Test columns are discarded by an explicit input allowlist. The old `anytime_mean.csv` files contain `best_so_far_test_fid`; they are not used to choose budgets. Random fallbacks still consume candidate budget and remain in the primary analysis.

## Endpoint attainment: validation fidelity >= 0.95

Entries are successful seeds out of 12. At least 10 is required.

| Condition | Configured B | Random | Greedy | LLM-Open | LLM-Closed |
|---|---:|---:|---:|---:|---:|
| 4-qubit TFIM, reference model | 4 | 0 | 0 | 0 | 8 |
| 4-qubit TFIM, reference model | 8 | 1 | 1 | 10 | 12 |
| 4-qubit TFIM, reference model | 16 | 2 | 3 | 12 | 11 |
| 6-qubit TFIM, reference model | 8 | 0 | 0 | 2 | 0 |
| 8-qubit TFIM, reference model | 8 | 0 | 0 | 0 | 0 |
| 4-qubit XXZ, reference model | 8 | 2 | 3 | 0 | 9 |
| 4-qubit TFIM, alternative model | 8 | 1 | 1 | 7 | 5 |

Reference model: `gpt-5.4-mini-2026-03-17`. Alternative model: `gpt-4.1-mini-2025-04-14`. These are the historical recorded snapshots, not a current model recommendation. Alternative-model Random and Greedy reuse the baseline controls and must not be counted as independent evidence.

For target 0.99, no method/condition satisfies 10/12. The only nonzero endpoint counts are 1/12 for B=4 LLM-Closed and 1/12 for XXZ Greedy; every other endpoint count is zero.

For the baseline task and the actually tested grid {4, 8, 16}, the smallest passing budget is 8 for both LLM-Open and LLM-Closed at target 0.95. Random and Greedy do not pass on this grid. No method passes at target 0.99.

**This does not certify a global B_min of 8.** Untested budgets remain unresolved. Because the search policy depends on configured B, failure at B=4 and success at B=8 do not by themselves establish a mathematically certified monotone interval (4, 8]. Likewise, failure at B=16 does not prove B_min > 16 without further assumptions.

## Within-run first-hit analysis

The first candidate index at which at least 10/12 seeds reach 0.95 is:

| Configured run | LLM-Open first-hit k | LLM-Closed first-hit k |
|---|---:|---:|
| B=4 | Not reached | Not reached |
| B=8 | 8 | 3 |
| B=16 | 3 | 9 |

All other condition/method traces fail the 10/12 target within their recorded length. No recorded trace satisfies 10/12 at target 0.99.

The B=8 Closed success-count trajectory is [4, 8, 10, 10, 11, 12, 12, 12]. Its first four candidates are semantic warm starts; feedback-based refinement begins at candidate 5. Therefore its early 10/12 attainment at candidate 3 cannot be attributed to feedback. This does not rule out feedback effects on later scores.

The B=16 Open run also reaches 10/12 after three evaluations, but its full 16-candidate pool was generated in advance. Stopping evaluation after three candidates would not refund the already incurred pool-generation cost. Neither finding is an independently executed B=3 result.

Budget-dependent warm-start/refinement splits (2+2, 4+4, 8+8) and requested batch sizes mean the B=4, B=8 and B=16 runs are not interchangeable prefixes of one search. Report configured budget, evaluated candidates, generated proposals, API calls, repair calls and tokens separately.

The Open pool is shared across data/training seeds. Its 12-seed attainment is conditional on that pool, not an estimate over 12 independently generated LLM pools.

## Generation validity

Fallback counts remain in the operational score. The alternative Open cell has 60 fallback evaluations out of 96, corresponding to five of eight shared pool entries evaluated across 12 seeds. Recorded invalid-error histories and final fallback flags are distinct: a repaired proposal can have recorded errors without being a fallback.

The bundle includes a separate diagnostic that excludes fallback candidates when calculating best scores, but it is not a causal validity-controlled experiment. Adaptive search histories are not regenerated by filtering existing logs. The original fixed-budget operational comparison remains primary.

## Proposed next experiments: NOT executed

These are targeted follow-up choices derived from this audit, not additional results from the source slides.

1. Baseline TFIM, B=6, Open and Closed, seeds 0 through 11, with Closed split 3+3. This probes an intermediate even budget between the observed B=4 failure and B=8 success. Nominal model calls without repairs: 1 shared Open batch plus 12 times (1 warm batch + 3 redesigns) = 49. A passing B=6 would still not establish a global minimum without checking all admissible smaller budgets or proving a valid monotonicity property.
2. XXZ anchor, B=10, Closed only, the same 12 seeds, split 5+5. The current B=8 count is 9/12. Nominal calls without repairs: 12 times (1 warm batch + 5 redesigns) = 72. This must be declared as a budget study anchored at XXZ, not silently admitted as a one-factor change from the old TFIM/B=8 reference.
3. Leave 6-qubit, 8-qubit and alternative-model boundaries unresolved rather than launch a blind paid sweep. Diagnose validity and optimization/capacity separately. Do not lower targets after observing these results.

Before any paid run, require available credentials and an explicit hard spend cap, freeze the admissible budget grid, preserve each anchor's trainer/gates/data/model/repair policy, and select on validation only. Current CLI support for arbitrary B=6/B=10 was not implemented or verified by this audit; the follow-up specification is not an executed runner.

Historical logs already contain historical test outputs. This retrospective analysis cannot certify those outputs as a new untouched confirmation set. Any subsequent confirmation must clearly distinguish historical descriptive results from a newly frozen evaluation protocol.

## Reproduction and deliverables

From the repository analysis branch:

```bash
python scripts/qae/analyze_budget_targets.py \
  --root outputs/qae_robustness \
  --out /tmp/qae-budget-audit-reproduced
```

From the downloaded bundle, using only the validation-projected input files:

```bash
python analyze_budget_targets.py --root data --out ../qae-budget-audit-reproduced
python build_budget_figures.py --root .
```

The analysis uses the Python standard library. Figure rendering was executed with Python 3.13.5 and Matplotlib 3.10.8. The bundle contains all 7 conditions' mean best-validation and 0.95-attainment charts (14 charts, PNG and SVG), CSVs, projected validation data, source manifests/hashes, local verification records, and the explicitly unexecuted follow-up specification.

Workflow artifact ID: 10029507492. Original artifact SHA-256: `6236fe4a50fa962e7791a48f5d9c6951c2d15b0ec6ecc7d216e4515490e68138`. It contains the validation audit; figures and local reproduction checks were added to the separate deliverable bundle. The original meeting deck and main branch were not modified.
