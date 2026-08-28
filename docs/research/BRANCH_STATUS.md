# Research branch status

This document makes `research/unified-qae-v1` the single active integration line for the project.

## Active line

| Branch | Status | Action |
|---|---|---|
| `main` | stable baseline | merge only through the unified PR |
| `research/unified-qae-v1` | **ACTIVE** | canonical research integration branch; based on `experiment/capacity-controlled-t2-v1` |

## Incorporated / superseded lines

| Branch | Status | Reason |
|---|---|---|
| `experiment/capacity-controlled-t1-v1` | superseded | T1-v1 is contained in the later capacity-controlled line |
| `experiment/capacity-controlled-t1-v2` | historical checkpoint | T1-v2 is contained in the T2-v1 ancestry and its conclusions are preserved |
| `experiment/capacity-controlled-t2-v1` | incorporated base | chosen as the common-ancestor base for the unified branch |

## Preserve as read-only scientific archives; do not merge directly

| Branch | Status | Reason |
|---|---|---|
| `experiment/capacity-controlled-higgs-v1` | archive | useful HIGGS diagnosis but Git history is disconnected from `main`; conclusions are summarized in the unified docs rather than force-merging unrelated history |
| `experiment/higgs-data-scale-qualification-v1` | archive | follow-up HIGGS data/training qualification; same disconnected-history issue |
| `experiment/free-amplitude-fixed-readout-v1` | legacy archive | ancestor of the old rigorous-QAS line, but disconnected from current `main` history |
| `research/rigorous-qas-benchmark-v2` | legacy archive | preserves earlier benchmark infrastructure/results; not a merge target |
| `presentation/rebuild-gsoc-benchmark-v2` | superseded presentation branch | old draft presentation work |
| `experiment/bench-v2-real-llm-cost-minimal` | superseded experiment branch | old draft real-LLM cost work |
| `experiment/mini-llm-api-vqc-demo-v1` | archive | small real-API demonstration, not the new scientific benchmark |
| `codex/publish-output-graphs` | archive | presentation/output helper branch |
| `claude/llm-vqc-architecture-plan-8417fd` | archive | planning branch |

## Pull requests

The two open draft PRs on the disconnected `research/rigorous-qas-benchmark-v2` line are superseded by the unified branch:
- PR #2 `presentation/rebuild-gsoc-benchmark-v2` -> `research/rigorous-qas-benchmark-v2`
- PR #3 `experiment/bench-v2-real-llm-cost-minimal` -> `research/rigorous-qas-benchmark-v2`

They should remain in GitHub history for provenance, but are closed rather than merged.

## Why we do not force-merge disconnected histories

A force merge would combine unrelated ancestry and make it difficult to audit which result came from which experimental protocol. The unified branch instead:
1. keeps the capacity-controlled T1/T2 history intact,
2. summarizes the HIGGS lessons with explicit branch pointers,
3. adds the QAE benchmark as the next scientific step,
4. creates one clean PR to `main`.
