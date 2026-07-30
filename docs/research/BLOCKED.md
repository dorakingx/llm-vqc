# BLOCKED — the API credential has no quota

Status: 2026-07-31, branch `experiment/bench-v2-real-llm-cost-minimal`.

Every preflight fix required before spending was completed, and the
authorisation itself was in place: the model was pinned, the price
manifest committed, the cumulative ledger implemented and tested, the
search-space parity proved, and `LLM_API_BUDGET_USD` was set. The run
still cannot proceed, for a reason no cap or code change can fix.

## The blocker

The first real request returned:

```
openai.RateLimitError: Error code: 429 — {'error':
  {'message': 'You exceeded your current quota, please check your plan
   and billing details.',
   'type': 'insufficient_quota', 'code': 'insufficient_quota'}}
```

`insufficient_quota` is an **account-level billing state**, not a rate
limit and not a per-key throttle: the credential is valid and accepted,
but the account has no remaining credit to draw on. Retrying, waiting, a
smaller batch, a cheaper model or a larger cap all fail identically.

This is blocker class 1 from the original goal — "a paid API credential
or explicit API spending cap is absent/**exhausted**".

## What was attempted, exactly

| Item | Value |
|---|---|
| Endpoint | `chat.completions` with strict Structured Outputs |
| Model | `gpt-5-nano-2025-08-07` (pinned; env verified before any client was built) |
| Preflight cap | `LLM_API_BUDGET_USD=0.02`, separate ledger `runs/bench_v2/preflight_ledger.sqlite` |
| Requests that reached the provider | 1 logical call → 3 outbound attempts (the retry wrapper's 2 retries) |
| Requests that succeeded | 0 |
| **Cumulative spend** | **$0.000000** |

The ledger recorded three reservations totalling $0.00252 and
**released all three** when the requests failed, leaving committed spend
at exactly zero. That is the reservation/release path working under a
real provider failure rather than in a test.

## Consequence for the matrix

| Experiment | State | Reason |
|---|---|---|
| E1 real-LLM (60 cells) | not started | no quota |
| E2 real-LLM (80 cells) | not started | no quota |
| E3 real-LLM (80 cells) | not started | no quota |
| **E5 θ-isolation** | **blocked** | the protocol freezes *LLM-proposed* architectures; with no LLM cells there are none. Running it on classical joint architectures instead would silently redefine the experiment, so it was not done. |
| E4 shot/noise robustness | **runnable and run** | it selects circuits from completed E2/E3 cells, which are classical and complete |

The classical matrix (460/460 cells) is unaffected and remains complete.

## To unblock

Add credit to the OpenAI account that owns `OPENAI_API_KEY` (Billing →
add a payment method or top up credits), then:

```bash
cd /Users/hatanakatomoya/Developer/Sim/llm-vqc/.claude/worktrees/github-author-config-faee6e
set -a; source .env; set +a
export OPENAI_MODEL=gpt-5-nano-2025-08-07
export LLM_API_BUDGET_USD=0.02
python scripts/bench_v2/preflight_real_provider.py      # must pass first
```

Then, and only then, the matrix under its own $2.00 cumulative cap:

```bash
export LLM_API_BUDGET_USD=2.00
bash scripts/bench_v2/run_llm_matrix.sh
```

Both ledgers are durable: a resumed run continues the same cumulative
total rather than restarting it, and cells already completed are skipped.

## Cost expectation once quota exists

From the committed manifest (`configs/bench_v2/model_prices.json`,
gpt-5-nano at $0.05/1M input and $0.40/1M output) and the measured
prompt sizes, one proposal call costs roughly $0.0004. The full matrix is
~1,480 calls ≈ **$0.6**, so the $2.00 cap carries the whole run with
headroom — a figure now derived from real prices and token counts rather
than the previous invented flat rate.
