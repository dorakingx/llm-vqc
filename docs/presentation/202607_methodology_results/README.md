# Methodology & results package — 2026-07 research meeting

A restructured, evidence-checked companion to `20260716_GSoC.pdf`. It exists to
make six things unambiguous for the next research meeting:

1. **the exact initial configuration** — task, data, splits, seeds;
2. **what was trained** — the five distinct kinds of parameter;
3. **how training proceeded** — the full forward → loss → backward → update cycle;
4. **how architectures were searched and selected** — five arms, six conditions;
5. **how validation and final testing were performed** — the structural quarantine;
6. **what the results actually show** — with graphical evidence, honestly hedged.

Everything here is derived from the authoritative repository (`llm_vqc/**`,
`scripts/**`) and the durable experiment stores — nothing was copied from the
slide deck without re-verification.

## Read in order

| Doc | Contents |
|---|---|
| [`PRESENTATION_AUDIT.md`](PRESENTATION_AUDIT.md) | Slide-by-slide audit of `20260716_GSoC.pdf` with corrections. **Start here.** |
| [`01_TASK_AND_DATA.md`](01_TASK_AND_DATA.md) | T1 task, generative model, nuisances, normalization, splits, loss vs. metric |
| [`02_MODEL_AND_PARAMETERS.md`](02_MODEL_AND_PARAMETERS.md) | Forward path; the five parameter roles; two concrete selected circuits |
| [`03_TRAINING.md`](03_TRAINING.md) | Config, initialization policy, the training cycle, training curves |
| [`04_ARCHITECTURE_SEARCH_AND_SELECTION.md`](04_ARCHITECTURE_SEARCH_AND_SELECTION.md) | Five arms → six conditions; shared loop; selection on validation |
| [`05_VALIDATION_AND_FINAL_TEST.md`](05_VALIDATION_AND_FINAL_TEST.md) | Why leakage is structurally impossible |
| [`06_RESULTS.md`](06_RESULTS.md) | Numbers + figures for the non-LLM, OpenAI, and Groq pilots |

## The single most important correction

Slide 8's `0.00845 / 0.01897 / 0.02438`, labeled **"Mean Test RMSE,"** are the
**median validation RMSE at B=25** — validation, not test; median, not mean. The
true mean-test values are `0.00910 / 0.01817 / 0.02756`. See
[`figures/pilot_val_vs_test.png`](figures/pilot_val_vs_test.png).

## Figures (`figures/`, PNG for slides + SVG for editing)

| File | Shows | In doc |
|---|---|---|
| `t1_data_generation` | same μ, different nuisances | 01 |
| `t1_example_curves` | raw vs normalized, coloured by μ | 01 |
| `t1_split_summary` | 150/250/2000 frozen split | 01 |
| `model_parameter_taxonomy` | forward path + 5 parameter roles | 02 |
| `circuit_greedy_s1` | concrete angle-encoding selected circuit | 02 |
| `circuit_random_s0` | best-overall amplitude selected circuit | 02 |
| `training_curves_random_s0` | train loss + val RMSE per epoch | 03 |
| `pred_vs_true_random_s0` | predicted vs true μ (validation) | 03, 06 |
| `pilot_anytime_curves` | best-so-far validation vs proposals (selection) | 04 |
| `validation_vs_test_quarantine` | the quarantine boundary schematic | 05 |
| `pilot_val_vs_test` | median-val vs mean-test bars (the correction) | 06 |

## Reproduce every figure (offline, no API, no raw store)

```bash
.venv/bin/python scripts/build_methodology_figures.py
```

Figures are built from the deterministic task/model/training code plus the
committed sanitized extract [`data/pilot_nonllm.json`](data/pilot_nonllm.json)
(re-derivable from the durable store by `scripts/analyze_pilot.py`). The build
asserts the re-trained `random_s0` final validation RMSE equals the stored
`0.008114592`, so a training-code regression cannot silently alter a figure.

## Scope & integrity

All pilots are small and descriptive (n ≤ 3 seeds). Nothing here supports an
LLM-superiority, quantum-advantage, or HEP-domain claim. No prompts, responses,
credentials, or raw SQLite stores are published — only sanitized numeric extracts
and figures.
