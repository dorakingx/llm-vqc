# bench_v2 weekly deck — LLM-Guided VQC Design: From Pilot to Rigorous Benchmark

10-slide ML4SCI / GSoC 2026 weekly update (2026-07-29), generated
entirely by script from durable experiment stores and committed analysis
artifacts on branch `research/rigorous-qas-benchmark-v2`.

## Files

- `LLM_VQC_weekly_benchmark_v2.pptx` / `.pdf` — the deck (PDF exported
  via LibreOffice from the same PPTX).
- `figures/` — every deck figure as PNG **and** SVG.
- `data/` — the source CSV for every figure (same stem), plus the
  status/stat tables cited on slides:
  - `fig_*.csv` — one per figure;
  - `matrix_status.csv` — live complete/running/blocked/pending counts
    from `runs/bench_v2/*/cells/` at build time;
  - `e0_replication_summary.csv` — E0 offline-replay match results;
  - `E1_*`, `E2_*` — copies of the committed per-seed and statistics
    tables from `outputs/bench_v2/`.
- `build_deck_figures.py` — regenerates `figures/` + `data/` from stores.
- `generate_deck.js` — regenerates the PPTX (pptxgenjs).

## Regeneration

```bash
.venv/bin/python docs/presentation/bench_v2_weekly/build_deck_figures.py
node docs/presentation/bench_v2_weekly/generate_deck.js   # needs pptxgenjs
python docs/presentation/bench_v2_weekly/recompress_pptx.py   # deflate the zip
soffice --headless --convert-to pdf LLM_VQC_weekly_benchmark_v2.pptx
```

## Google Drive / Google Slides

The deck is published to the `GSoC2026` folder
(https://drive.google.com/drive/folders/1YV3KxZe4OA4ysh9YSqPGJbZxpkpN_fqf),
copied there through the Google Drive Desktop mount so the transfer is
byte-exact and full-resolution:

- `LLM_VQC_weekly_benchmark_v2.pptx` — 277,201 bytes, file id
  `1lK8lsWA4jod1XXRo7o4z9sV_mu5vHot1`
- Opens directly in Google Slides:
  https://docs.google.com/presentation/d/1lK8lsWA4jod1XXRo7o4z9sV_mu5vHot1/edit

**Converting it to a native Google Slides file needs one manual step**
(`File → Save as Google Slides` in the link above). It could not be
automated from this session, and doing so would also have *lowered*
quality:

- the Drive connector converts only on upload, and only accepts file
  content inline as base64 — 370 k characters for this deck, beyond what
  one message can carry intact, so a corrupt zip was the likely outcome;
- shrinking the deck enough to fit (≈130 KB, figures at 62 dpi) would
  have produced a *softer* Slides file than the full-quality PPTX that
  is already in the folder — strictly worse for the reader;
- the Chrome extension (which could drive the Drive UI) is not
  connected, and the `gcloud` token has no Drive scope.

With the Chrome extension connected, the upload-and-convert step can be
fully automated in a later session.

## Provenance and claim discipline

- Status at build time: E0 complete (exact offline replay, zero API
  calls); E1 classical 60/60 and E2 classical+references 280/280
  complete (10 paired replicates); **E3 running**; all real-LLM cells
  (220), E4, and E5 **blocked pending `LLM_API_BUDGET_USD`** (see
  `docs/research/BLOCKED.md`).
- The July pilot appears only as E0 provenance (2 seeds, B=4,
  integration demo) — it is not presented as evidence.
- Every numerical claim on the slides traces to a CSV in `data/` derived
  from `runs/bench_v2/` stores or committed `outputs/bench_v2/`
  artifacts; statistical claims use the Holm-corrected paired tests in
  `E1_stats_paired_tests.csv` / `E2_stats_paired_tests.csv`.
- Expressibility KL and Meyer–Wallach Q are shown as descriptive
  characterizations only (Sim et al. framing); RQ5 tests association
  empirically later.
- No raw prompts, raw LLM responses, SQLite stores, or secrets are
  included in these artifacts.
