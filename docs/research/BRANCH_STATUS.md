# Research branch status

The consolidation PR (#4) and the API-verification PR (#5) have been squash-merged. `main` is the single active research line.

## Active line

| Branch | Status | Action |
|---|---|---|
| `main` | **ACTIVE** | canonical research branch containing capacity-controlled T1/T2, the semantic QAE pilot, reproducibility docs, and the manuscript |
| `research/unified-qae-v1` | merged checkpoint | PR #4 was squash-merged into `main`; keep only as temporary provenance if desired |

## Incorporated / superseded lines

| Branch | Status | Reason |
|---|---|---|
| `experiment/capacity-controlled-t1-v1` | superseded | T1-v1 is contained in the later capacity-controlled line now incorporated into `main` |
| `experiment/capacity-controlled-t1-v2` | historical checkpoint | T1-v2 conclusions and artifacts are incorporated into `main` |
| `experiment/capacity-controlled-t2-v1` | incorporated | T2-v1 formed the base of the consolidation and is now in `main` |

## Preserve as read-only scientific archives; do not merge directly

| Branch | Status | Reason |
|---|---|---|
| `experiment/capacity-controlled-higgs-v1` | archive | useful HIGGS diagnosis but Git history is disconnected from `main`; conclusions are summarized in `docs/research/HIGGS_ARCHIVE_SYNTHESIS.md` rather than force-merging unrelated history |
| `experiment/higgs-data-scale-qualification-v1` | archive | follow-up HIGGS data/training qualification; same disconnected-history issue |
| `experiment/free-amplitude-fixed-readout-v1` | legacy archive | ancestor of the old rigorous-QAS line, disconnected from current `main` history |
| `research/rigorous-qas-benchmark-v2` | legacy archive | preserves earlier benchmark infrastructure/results; not a merge target |
| `presentation/rebuild-gsoc-benchmark-v2` | superseded presentation branch | old draft presentation work |
| `experiment/bench-v2-real-llm-cost-minimal` | superseded experiment branch | old draft real-LLM cost work |
| `experiment/mini-llm-api-vqc-demo-v1` | archive | small real-API demonstration, not the new scientific benchmark |
| `codex/publish-output-graphs` | archive | presentation/output helper branch |
| `claude/llm-vqc-architecture-plan-8417fd` | archive | planning branch |

## Pull requests

- PR #2 was closed without merge and retained only as provenance.
- PR #3 was closed without merge and retained only as provenance.
- PR #4 `research/unified-qae-v1 -> main` was squash-merged successfully.
- PR #5 `claude/llm-vqc-consolidate-qae-673a04 -> main` (QAE v2 version-pinned API verification + Evolutionary/hand-designed baselines + paper update) was squash-merged on 2026-08-28.
- There are no open pull requests.

## External deliverables

- Meeting deck (2026-08-29): `20260829_GSoC.pptx` + `20260829_GSoC.pdf` in the shared Google Drive `slides` folder; the earlier draft is retained as `20260829_GSoC_archive`.
- Manuscript: `paper/main.tex` (canonical, in this repo) is synced to the Overleaf project (git remote `git.overleaf.com/6a1b64ed666041bd4b9014fe`, commit "Update manuscript with QAE v2 version-pinned API verification results").
- Literature: the six queued references in `REFERENCES_TO_ADD_TO_PAPERPILE.md` were imported into the Paperpile `llm-vqc` folder on 2026-08-28.

## Why disconnected histories remain archives

Force-merging unrelated ancestry would make it difficult to audit which result came from which experimental protocol. Instead, `main` now contains the clean scientific narrative and reproducible active benchmark, while disconnected histories remain read-only evidence for earlier experiments.

If branch deletion is desired later, delete only branches marked `superseded`, `archive`, or `merged checkpoint` after confirming no external work still points to them.
