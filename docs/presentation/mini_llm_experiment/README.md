# OpenAI mini LLM experiment — presentation package

> **This is a time-boxed, real-API smoke integration run (n=1 seed, budget=2 per arm), not a completed budget-matched LLM comparison.**

This package exposes the verified, repository-safe results of the first
real-API LAQS-Bench mini experiment without publishing the raw SQLite store,
run directories, prompts, responses, provider headers, or credentials. Its
purpose is to support slides and human review of what actually ran and what
can and cannot be concluded.

## Exact configuration

| Field | Value |
|---|---|
| Task | T1 1D Gaussian-peak regression; frozen data split seed 0 |
| Provider / model | `openai` / `gpt-5.4-mini` (server-reported snapshot `gpt-5.4-mini-2026-03-17`) |
| Declared candidate budget | B=2 per arm |
| Seeds | 0 (single seed — time-boxed run) |
| Arms | `random`, `evolutionary`, `greedy`, `llm_iter` open-loop, `llm_iter` closed-loop |
| Omitted | `llm_evo` (kept combined real-call count safely under the 12-call cap) |
| Training | 1 epoch (reduced for the 5-minute wall-clock cap) |
| Feedback | Validation only during search; protected test once after each arm's search completed |
| Real API calls | 4 of a 12-call cap; wall clock 9.8 s of a 5-minute cap |

Lower RMSE is better.

## Run-completion matrix (n=1 seed)

| Arm | Consumed budget | Validation RMSE | Protected test RMSE |
|---|---:|---:|---:|
| random | 2/2 | 0.292942 | 0.292311 |
| evolutionary | 2/2 | 0.131343 | 0.133476 |
| greedy | 2/2 | 0.130074 | 0.132825 |
| llm_iter (open-loop) | 2/2 | 0.174413 | 0.172725 |
| llm_iter (closed-loop) | 2/2 | 0.132116 | 0.132024 |

All five arms completed their declared budget with zero invalid, duplicate,
or failed proposals. These are single-seed descriptive results, not
confirmatory statistics.

## Real API usage

| Call | Model (server-reported) | Input tokens | Output tokens | Latency |
|---|---|---:|---:|---:|
| llm_iter (open-loop) #1 | gpt-5.4-mini-2026-03-17 | 302 | 138 | 1.99 s |
| llm_iter (open-loop) #2 | gpt-5.4-mini-2026-03-17 | 302 | 242 | 1.63 s |
| llm_iter (closed-loop) #1 | gpt-5.4-mini-2026-03-17 | 308 | 139 | 3.60 s |
| llm_iter (closed-loop) #2 | gpt-5.4-mini-2026-03-17 | 335 | 137 | 1.27 s |

Dollar cost is not reported: OpenAI's chat completions API does not return a
per-call price, and none was fabricated. Token counts and latency are the
authoritative usage record. The server-reported model string
(`gpt-5.4-mini-2026-03-17`) is independent confirmation that these were real
API responses, not a mocked provider.

## Included plots

- `mini_val_performance`: selected-circuit validation RMSE by arm.
- `mini_test_performance`: protected final-test RMSE by arm.
- `mini_anytime_curve`: best-so-far validation RMSE vs. evaluated-candidate
  count, per arm.
- `mini_proposal_rates`: invalid and duplicate proposal fractions by arm
  (all zero in this run).
- `mini_llm_usage`: real-call latency and token usage for the two LLM arms.

Each plot is included as PNG for slide tools and SVG for scalable editing.
`mini_pilot_summary.csv` holds the same per-arm numbers in tabular form.

## Supported claims

- The OpenAI provider path made 4 real calls to `gpt-5.4-mini` and completed
  candidate evaluation, protected final-test scoring, and plot generation
  entirely within a 5-minute wall-clock cap.
- All 5 arms (3 classical, 2 LLM feedback modes) completed their declared
  budget with no invalid, duplicate, or failed proposals in this single-seed
  run.
- The LLM arms' proposals were confirmed real (server-reported model
  snapshot, real token counts, real latency), parsed through the approved
  schema, and never received test-set information during search.

## Unsupported claims

- No LLM superiority or inferiority claim is supported (n=1 seed, budget=2).
- No statistically confirmatory comparison across arms is supported.
- No quantum-advantage claim is supported.
- No HEP-domain claim is supported by T1.
- Results must not be generalized to other seeds, budgets, epoch counts, or
  models.

## Reproduction and resume

With the ignored durable store available at
`runs/mini_llm_experiment/results.sqlite`, regenerate all summaries and plots
without making API calls:

```bash
.venv/bin/python scripts/plot_mini_experiment.py
```

To rerun the real-API experiment, provide the required OpenAI credential to
the process environment using a secure local mechanism, then run:

```bash
OPENAI_API_KEY="$(pbpaste | tr -d '\r\n')" .venv/bin/python scripts/mini_llm_experiment.py
```
