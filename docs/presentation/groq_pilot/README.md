# Groq GPT-OSS 20B pilot — presentation package

> **This is an interrupted pilot and descriptive integration study, not a completed budget-matched LLM comparison.**

This package exposes the verified, repository-safe results of the Groq
free-tier integration pilot without publishing the raw SQLite store, run
directories, prompts, responses, provider headers, or credentials. Its purpose
is to support slides and human review of what actually ran, what stopped, and
what can and cannot be concluded.

## Exact configuration

| Field | Value |
|---|---|
| Task | T1 1D Gaussian-peak regression; frozen data split seed 0 |
| Provider / model | `groq` / `openai/gpt-oss-20b` |
| Declared candidate budget | B=10 per arm and seed |
| Seeds | 0, 1, 2 |
| Arms | `random`, `evolutionary`, `greedy`, `llm_open`, `llm_closed`, `llm_evo` |
| Training | AdamW, 20 epochs, batch 16, learning rate 0.05, float64 CPU |
| Feedback | Validation only during search; protected test once after a completed run |
| LLM decoding | strict JSON Schema, reasoning effort `low`, temperature 0.2, at most 512 output tokens |
| Free-tier controls | sequential calls, 8 s minimum pacing, 180,000-token internal cap, 250-request cap |

Lower RMSE is better.

## Run-completion matrix

| Arm | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| random | complete 10/10 | complete 10/10 | complete 10/10 |
| evolutionary | complete 10/10 | complete 10/10 | complete 10/10 |
| greedy | complete 10/10 | complete 10/10 | complete 10/10 |
| llm_open | interrupted 4/10 | interrupted 1/10 | not started |
| llm_closed | not started | not started | not started |
| llm_evo | not started | not started | not started |

The matrix therefore contains 9 complete cells, 2 interrupted cells, and 7
not-started cells. Incomplete LLM runs must not be compared as completed cells.

## Verified completed non-LLM results

| Arm | Seed | Validation RMSE | Protected test RMSE | Duplicates |
|---|---:|---:|---:|---:|
| random | 0 | 0.009216 | 0.010305 | 0 |
| random | 1 | 0.021769 | 0.023468 | 0 |
| random | 2 | 0.020101 | 0.017630 | 0 |
| evolutionary | 0 | 0.023310 | 0.025868 | 3 |
| evolutionary | 1 | 0.020298 | 0.024421 | 4 |
| evolutionary | 2 | 0.017298 | 0.020915 | 5 |
| greedy | 0 | 0.028831 | 0.032229 | 0 |
| greedy | 1 | 0.026678 | 0.028523 | 0 |
| greedy | 2 | 0.034912 | 0.038617 | 0 |

Median validation RMSE was 0.020101 for random, 0.020298 for evolutionary,
and 0.028831 for greedy. These are descriptive n=3 results, not confirmatory
statistics.

## Partial LLM observations

| Cell | Consumed budget | Proposals | Invalid | Duplicate | Best validation RMSE | Protected test |
|---|---:|---:|---:|---:|---:|---|
| llm_open seed 0 | 4/10 | 22 | 18 | 1 | 0.024052 | not run |
| llm_open seed 1 | 1/10 | 9 | 8 | 0 | 0.042719 | not run |

Across the two partial cells, 26 of 31 proposals (83.9%) were invalid.
Stored validation errors identify the recurring problem as wire fields emitted
in a string-like form instead of semantically valid arrays of integer wire
indices. Under the tested low-reasoning configuration, GPT-OSS 20B frequently
failed to produce semantically valid CircuitIR proposals. This observation is
model-, prompt-, and configuration-specific.

## API usage, latency, and rate limits

| Cell | Calls | Reported input tokens | Reported output tokens | Mean latency | Median latency |
|---|---:|---:|---:|---:|---:|
| llm_open seed 0 | 22 | 5,056 | 964 | 9.38 s | 8.02 s |
| llm_open seed 1 | 9 | 1,264 | 245 | 657.69 s | 767.02 s |
| Total | 31 | 6,320 | 1,209 | — | — |

The store reports 7,529 tokens in total. Seed 1 experienced very long
rate-limit backoffs (117.34–768.20 s per stored call), and the experiment was
interrupted rather than waiting for the remaining matrix. The provider's
controller paced sequential requests and retried transient rate-limit/server
errors once; the durable store preserved the partial state.

## Included plots

- `groq_val_performance`: completed-cell validation RMSE medians with seed dots.
- `groq_test_performance`: protected final-test RMSE for completed cells only.
- `groq_anytime_curves`: per-run best-so-far validation RMSE; partial LLM runs
  remain visibly shorter and are not filled to B=10.
- `groq_proposal_rates`: invalid and duplicate fractions for started cells.
- `groq_llm_usage`: reported tokens and mean call latency for the two started
  LLM cells.

Each plot is included as PNG for slide tools and SVG for scalable editing.

## Supported claims

- The Groq provider path made 31 real calls to `openai/gpt-oss-20b` and stored
  7,529 reported tokens.
- All 9 non-LLM cells completed; two `llm_open` cells were interrupted and the
  other 7 LLM cells were not started.
- Strict schema conformance did not guarantee semantically valid CircuitIR
  proposals under this tested configuration.
- Protected test results exist only for the 9 completed cells and never entered
  search feedback.

## Unsupported claims

- No LLM superiority or inferiority claim is supported.
- No completed Groq-vs-classical or model-vs-model comparison is supported.
- No quantum-advantage claim is supported.
- No HEP-domain claim is supported by T1.
- Results must not be generalized to other prompts, Groq models, open-source
  models, tasks, budgets, or reasoning settings.

## Reproduction and resume

With the ignored durable store available at `runs/groq_pilot_t1/results.sqlite`,
regenerate all summaries and plots without making API calls:

```bash
.venv/bin/python scripts/analyze_groq_pilot.py
```

To resume the interrupted matrix, first provide the required Groq credential
to the process environment using a secure local mechanism, then run:

```bash
.venv/bin/python scripts/groq_pilot_experiment.py pilot
```

Resume is permitted only with the same provider/model identity. The matrix is
still incomplete until every declared cell reaches B=10.
