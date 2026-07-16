# Sanitized report: interrupted Groq GPT-OSS 20B pilot

## 1. Experiment status

The T1 Groq free-tier matrix was interrupted after all 9 non-LLM cells
completed, two `llm_open` cells partially ran, and the other 7 LLM cells had
not started. It is a descriptive integration pilot, not a completed comparison.

## 2. Provider and model

The isolated condition used provider `groq` and model
`openai/gpt-oss-20b`. Run IDs use the `groq_gpt_oss_20b_` namespace and the
result-store compatibility identity prevents a resume under a different
provider or model. No OpenAI result store was mixed into this analysis.

The provider requested strict JSON Schema output, low reasoning effort,
temperature 0.2, and at most 512 output tokens. It used sequential free-tier
pacing and a client-side quota controller.

## 3. Protocol

- Task: T1 1D Gaussian-peak regression, frozen data split seed 0.
- Repetition seeds: 0, 1, 2.
- Declared budget: 10 consumed candidate evaluations per cell.
- Arms: random, evolutionary, greedy, LLM open-loop, LLM closed-loop, and
  LLM evolutionary.
- Training: AdamW, 20 epochs, batch size 16, float64 CPU.
- Search feedback: validation metric only.
- Protected test: evaluated only after a run completed, using its selected
  circuit and already-trained weights.

Lower RMSE is better. Invalid proposals are budget-free; duplicate and failed
evaluations consume budget under the declared accounting rule.

## 4. Run completion matrix

| Arm | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| random | complete 10/10 | complete 10/10 | complete 10/10 |
| evolutionary | complete 10/10 | complete 10/10 | complete 10/10 |
| greedy | complete 10/10 | complete 10/10 | complete 10/10 |
| llm_open | interrupted 4/10 | interrupted 1/10 | not started |
| llm_closed | not started | not started | not started |
| llm_evo | not started | not started | not started |

## 5. API usage

The two started LLM cells contain 31 real call records: 22 for seed 0 and 9
for seed 1. Stored usage totals are 6,320 input tokens and 1,209 output tokens,
or 7,529 reported tokens overall. No monetary cost is fabricated because the
free-tier API did not report a dollar amount.

## 6. Completed non-LLM results

| Run | Validation RMSE | Protected test RMSE |
|---|---:|---:|
| random s0 | 0.009216 | 0.010305 |
| random s1 | 0.021769 | 0.023468 |
| random s2 | 0.020101 | 0.017630 |
| evolutionary s0 | 0.023310 | 0.025868 |
| evolutionary s1 | 0.020298 | 0.024421 |
| evolutionary s2 | 0.017298 | 0.020915 |
| greedy s0 | 0.028831 | 0.032229 |
| greedy s1 | 0.026678 | 0.028523 |
| greedy s2 | 0.034912 | 0.038617 |

The median validation RMSEs were 0.020101, 0.020298, and 0.028831 for
random, evolutionary, and greedy respectively. With only three seeds and an
incomplete overall matrix, these are descriptive estimates only.

## 7. Partial LLM observations

`llm_open` seed 0 made 22 proposals, of which 18 were invalid, 3 valid, and 1
duplicate. It consumed 4/10 budget and reached best validation RMSE 0.024052.
Seed 1 made 9 proposals, of which 8 were invalid and 1 valid. It consumed 1/10
budget and reached best validation RMSE 0.042719. Neither partial cell received
a protected final-test result.

Under the tested low-reasoning configuration, GPT-OSS 20B frequently failed
to produce semantically valid CircuitIR proposals. This provisional observation
does not extend to other models, prompts, or decoding configurations.

## 8. Invalid and duplicate proposals

Across the partial LLM cells, 26/31 proposals were invalid (83.9%) and one was
a duplicate. Stored validation categories point to string-like wire encodings
instead of arrays of integer wire indices, despite strict schema and repeated
format instructions. The completed evolutionary cells recorded 3, 4, and 5
duplicates respectively; random and greedy recorded none.

## 9. Latency and rate limits

For `llm_open` seed 0, mean call latency was 9.38 s and median latency was
8.02 s. For seed 1, mean latency rose to 657.69 s and median latency to
767.02 s, with stored calls spanning 117.34–768.20 s. This is consistent with
severe free-tier rate-limit backoff. The long delays led to interruption rather
than completion of the remaining cells.

## 10. Test-quarantine verification

Protected test RMSE is present only for the 9 complete non-LLM cells. Both
interrupted LLM rows and all not-started rows have no test value. The search
feedback type contains no test metric, and test evaluation is called only after
completed search selection.

## 11. Interpretation

The pilot verifies that the Groq provider, strict structured-output path,
durable accounting, interruption, and offline analysis all operate end to end.
It also identifies model-specific semantic-output and free-tier pacing problems
that must be addressed before a complete comparison is practical.

No LLM superiority or inferiority conclusion is supported. The partial LLM
cells are not budget-matched completed observations and must not be merged into
completed-arm medians.

## 12. Limitations

- The declared 18-cell matrix is incomplete: 9 complete, 2 interrupted, 7 not
  started.
- There are only three declared seeds per arm and only one synthetic task.
- T1 does not establish HEP-domain performance.
- Results are specific to one model, prompt/schema implementation, low
  reasoning effort, and free-tier rate limits.
- No quantum-advantage or general open-source-LLM claim is supported.

## 13. Reproduction

With the ignored SQLite store present, regenerate tables and plots without API
calls:

```bash
.venv/bin/python scripts/analyze_groq_pilot.py
```

After securely supplying the required Groq credential to the process
environment, resume the same provider/model condition with:

```bash
.venv/bin/python scripts/groq_pilot_experiment.py pilot
```
