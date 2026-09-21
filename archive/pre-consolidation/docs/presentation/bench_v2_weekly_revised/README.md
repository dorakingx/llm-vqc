# Revised GSoC benchmark deck — 20260731_GSoC_revised

An 8–10 minute talk for ML4SCI / GSoC mentors on the LLM-guided VQC
architecture-search benchmark. Rebuilt from
`docs/presentation/bench_v2_weekly/` (**preserved unchanged** as the
historical artifact) for claim discipline and readability.

## Contents

| Path | What it is |
|---|---|
| `20260731_GSoC_revised.pptx` / `.pdf` | 10 main slides + 8 appendix slides, 16:9 |
| `renders/slide-NN.png` | Every slide at 1734 × 975, used for the visual QA |
| `figures/` | Every figure as PNG and SVG |
| `data/` | The source CSV behind every figure |
| `SOURCE_AUDIT.md` | What is true at HEAD, traced to code and stores |
| `claim_source_map.yaml` | Each deck claim → file, column, derivation, kind |
| `speaker_notes.md` | 40–60 s per slide, plus Q&A preparation |
| `DECK_REVIEW.md` | Slide-by-slide visual QA, defects found and fixed |
| `artifact_manifest.json` | SHA-256, size and generation commands |
| `build_figures.py`, `generate_deck.js`, `recompress_pptx.py` | Generators |

## Regeneration

```bash
python docs/presentation/bench_v2_weekly_revised/build_figures.py
node   docs/presentation/bench_v2_weekly_revised/generate_deck.js
python docs/presentation/bench_v2_weekly_revised/recompress_pptx.py 20260731_GSoC_revised.pptx
soffice --headless --convert-to pdf 20260731_GSoC_revised.pptx
pdftoppm -png -r 130 20260731_GSoC_revised.pdf renders/slide
python scripts/verify_bench_v2_presentation.py
```

## Verified status at the time of writing

- **460 / 460 classical cells complete, 0 failures** (E1 60, E2 280, E3 120).
- **0 / 220 real-LLM cells** — blocked because `LLM_API_BUDGET_USD` is unset.
- **E4 and E5 pending**; they depend on the LLM cells.
- `scripts/check_goal_completion.py` reports **9 / 17** criteria. It fails
  correctly and was not modified.
- All results, including n = 8, are **noiseless statevector simulation** —
  no shots, no noise model, no quantum hardware.

## What changed relative to the previous deck

Ten corrections, each recorded with its evidence in `SOURCE_AUDIT.md` §9.
The substantive ones:

1. "Preregistered" → "protocol-frozen"; there was no external registration.
2. "Statistically indistinguishable" → "no Holm-adjusted difference was
   detected at the current sample size", with effect sizes reported.
3. The literature gap is narrowed to what was verified per paper.
4. Search arms and fixed reference anchors are separated, and the deck
   states what budget matching does and does not cover.
5. The headline result is the systematic pattern across contrasts, not a
   single cherry-picked p-value — the previous deck quoted the *weakest*
   reference contrast.
6. The scaling story is corrected to a **crossover**: the reference is
   better at n = 3 on T2 and degrades from n = 4 onward.
7. One status everywhere; the deck never calls the benchmark complete.
8. n = 8 is explicitly labelled simulation, not hardware.
9. Transpiled two-qubit counts are called a proxy that excludes state
   preparation; the causal "accuracy is bought with hardware cost" is gone.
10. The API request is a reproducible derivation of a hard cap, not a
    fabricated dollar estimate.

## Google Slides

The live deck was **not** modified. To publish this version, copy the
PPTX into the target Drive folder and use **File → Save as Google
Slides**; see `docs/presentation/bench_v2_weekly/README.md` for the route
that was verified to work on this machine.
