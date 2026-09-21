# Archive — superseded research lines, preserved verbatim

This directory exists so that `main` is **self-contained**. Everything under
`archive/` was originally developed on branches whose Git history is
*disconnected* from `main` (an author-identity rewrite early in the project
split the history in two). Those branches have been deleted from the remote;
their scientifically relevant content was copied here **byte-for-byte**,
at its original repository-relative paths, before deletion.

Nothing in this directory is part of the final result. It is kept because
these studies produced real measurements, real negative results and the
design rules that the final experiment is built on. See
[`../GSoC2026_FINAL_REPORT.md`](../GSoC2026_FINAL_REPORT.md) for the work
that *is* the final result.

## Status of this archive

- **Read-only.** Files here are not imported by `llm_vqc`, not collected by
  pytest (`testpaths = ["tests"]` in `pyproject.toml`), and not linted as
  part of `./scripts/check.sh`.
- **Not re-runnable as-is.** The archived code was written against the
  package layout of the pre-consolidation line. Reproducing these studies
  means checking out the original commit listed below, not running the
  copies here.
- **Paths are original.** `archive/pre-consolidation/outputs/bench_v2/…`
  was `outputs/bench_v2/…` on its source branch, so internal relative links
  inside archived reports still resolve within the archive subtree.

## Provenance

`archive/pre-consolidation/` is the union of the unique content of two
disconnected branch tips. The two tips agreed byte-for-byte on all 137
paths they shared, so the union is unambiguous.

| Source branch (deleted) | Tip SHA | Files taken | Notes |
|---|---|---:|---|
| `experiment/bench-v2-real-llm-cost-minimal` | `6afa91d979c49bc856cf650c1cee9fc740e66454` | 592 | Tip of the disconnected line; itself contains `experiment/capacity-controlled-higgs-v1`, `experiment/higgs-data-scale-qualification-v1`, `experiment/free-amplitude-fixed-readout-v1`, `research/rigorous-qas-benchmark-v2` and `presentation/rebuild-gsoc-benchmark-v2` as ancestors. |
| `experiment/mini-llm-api-vqc-demo-v1` | `112cf2d39d5c3d05c0cac325e0d59fe2ec35846b` | 38 (the paths not already present) | Separate tip of the same disconnected line. |

Only files **absent from `main`** were copied. No file already present on
`main` was overwritten, and no archived file conflicts with a tracked file
outside `archive/`.

## What each archived line contains

### `outputs/capacity_controlled_higgs_v1/` — HIGGS-v1 (negative / task-qualification)

Capacity-controlled VQC study on the official HIGGS task (500 training
examples per block, PCA(8)). Search-arm differences were small and the
strict equivalence test was underpowered; frozen quantum parameters and
product circuits slightly **outperformed** trainable and entangled
counterparts, and the VQC stayed near or below trivial/classical baselines.
The study concluded it was **not yet a valid setting for an LLM search
comparison**. The frozen protocol, statistical analysis plan, data-split
spec and cross-task claim audit are all included.

### `outputs/higgs_data_scale_qualification_v1/` — HIGGS data-scale follow-up

Isolated the causes of the HIGGS-v1 null: data scarcity at n=500,
mis-tuned inherited optimizer settings (lr 0.05), and PCA(8) information
loss. With raw 21 features, n=10,000 and an audited protocol the entangled
VQC reached approximately AUROC 0.635 / log-loss 0.664, and the signs of
the earlier frozen-vs-trainable and product-vs-entangled comparisons
**flipped** at larger data scale.

A condensed synthesis of both HIGGS studies, and the design rule they
produced, is on `main` at
[`docs/research/HIGGS_ARCHIVE_SYNTHESIS.md`](../docs/research/HIGGS_ARCHIVE_SYNTHESIS.md).

### `outputs/bench_v2/` + `docs/research/BENCHMARK_V2_PROTOCOL.md` — rigorous QAS benchmark v2 (**incomplete: blocked**)

A benchmark protocol frozen before execution, with a literature review,
search-space parity proof, break tests, a cumulative spend ledger and a
completion checker. The **classical** cells ran (E0 exact-replication check
passed with 0 API calls; E1–E3 analysis snapshots are present). The
**LLM cells were never executed**: the API credential returned
`insufficient_quota` at account level. The blocker is documented, with the
exact request that failed and a recorded cumulative spend of **$0.000000**,
in [`pre-consolidation/docs/research/BLOCKED.md`](pre-consolidation/docs/research/BLOCKED.md).

This line is archived as a **blocked, incomplete study**. Its protocol,
parity proof and budget-ledger machinery informed the later QAE work; its
LLM-versus-classical comparison was never measured and is not reported
anywhere as a result.

### `outputs/free_amplitude_fixed_readout_v1/` — free-amplitude joint search (integration demo)

Joint structure-and-angle search in which the LLM proposes both the circuit
and the numerical angles, with no optimizer and no classical layer. Real
OpenAI execution, 2 seeds. The report labels itself explicitly as
**"an integration/smoke demonstration, not a scientific performance
claim"**, and it is archived on those terms — 2 seeds cannot support a
method comparison.

### `outputs/mini_llm_api_vqc_demo_v1/` — mini real-API demo

Small end-to-end demonstration of the real-API search loop (Random /
LLM-Open / LLM-Closed) under a hard-verified fixed capacity of 51 trainable
parameters. A first run had defects; a corrected run from clean committed
SHA `b57a84c` replaced the artifacts, and `PROVISIONAL_NOTICE.md` is kept
as the record of what was wrong and why. A demonstration, not a benchmark.

### `docs/presentation/` — earlier presentation packages

Methodology-and-results slide material from the pre-consolidation line.
Superseded by the decks under `outputs/qae_robustness/deck/` and
`outputs/qae_budget_targets_v2_20260908/deck/`.

## Why these were superseded

Each line failed a gate that the next one was designed to pass:

1. HIGGS-v1 could not qualify its own task → data-scale study.
2. The data-scale study qualified the task but left the LLM comparison
   confounded by capacity and training protocol → capacity-controlled work.
3. Benchmark v2 fixed the protocol but could not pay for the LLM cells →
   blocked.
4. The QAE line replaced a classification task with a task carrying a
   **direct quantum objective** (state-compression fidelity) and an exactly
   controlled circuit budget, which is what made a clean, budget-matched
   LLM-versus-classical comparison possible at a cost the project could
   actually afford.
