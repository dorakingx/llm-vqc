# QAE-TFIM v3 — neutral-space four-method report

Protocol: `docs/research/QAE_PROTOCOL_V3.md`, pre-registered at commit
`6620cd4` and amended once pre-inspection at `92641ec` (bounded capacity
repair for the open pool). No condition was changed after protected-test
inspection.

## Headline

In the neutral space (free ordering of 12 rotations + 4 CNOTs), **task
semantics is the axis that matters, and validation feedback is not**:

| Method | Semantics | Feedback | Mean test fidelity | Median | SD |
|---|---|---|---:|---:|---:|
| **LLM-Open** | yes | no | **0.9653** | 0.9665 | 0.0042 |
| LLM-Closed | yes | yes | 0.9553 | 0.9693 | 0.0272 |
| Random | no | no | 0.8721 | 0.8802 | 0.0524 |
| Greedy | no | yes | 0.7667 | 0.8093 | 0.1143 |

Paired statistics (12 paired seeds, exact two-sided Wilcoxon, bootstrap
95% CI of mean gain, Cohen's dz):

- **Add semantics without feedback** (LLM-Open − Random):
  **+0.0932** [0.0642, 0.1228], dz = 1.73, **12/12 wins**, p = 0.00049.
- **Add semantics with feedback** (LLM-Closed − Greedy):
  **+0.1886** [0.1257, 0.2627], dz = 1.49, **12/12 wins**, p = 0.00049.
- **Add feedback without semantics** (Greedy − Random):
  **−0.1055** [−0.1702, −0.0474], dz = −0.93, 2/12 wins, p = 0.00488 —
  greedy hill-climbing is *significantly worse* than Random here.
- **Add feedback with semantics** (LLM-Closed − LLM-Open):
  −0.0100 [−0.0260, +0.0028], dz = −0.37, 6/12 wins, p = 0.68 —
  no detectable effect.
- LLM-Open − Random and LLM-Closed − Random are both 12/12 (p = 0.00049);
  LLM-Open − Greedy is +0.1986 [0.1436, 0.2661], 12/12, p = 0.00049.

## Sample efficiency (anytime curves, `anytime_mean.csv`)

Both LLM arms start near test fidelity 0.90 with their **first** proposal —
above where Random ends after all 8 (0.8721) — and saturate by the third
evaluation. Random climbs gradually. Greedy climbs the slowest: it spends
its whole budget mutating one random starting circuit.

## Why Greedy loses to Random (diagnosis, honest)

Greedy is not broken — 34/84 mutations were accepted and every seed
improved over its own starting point. The problem is search-strategy
geometry at this budget: 7 one-gate mutations of a single random start
explore a small neighbourhood of one (usually mediocre) circuit, while
Random draws 8 independent circuits and keeps the best. With B = 8 in a
large neutral space, breadth beats local depth. This is a budget-regime
statement, not a general indictment of greedy search.

## Proposal quality (`proposal_quality.json`, fig04)

- Random / Greedy: 96/96 evaluations each, 0 fallbacks.
- LLM-Open: pool of 8 valid distinct candidates from 2 calls (1 candidate
  of the first call was capacity-invalid and was repaired per the
  pre-registered amendment); 0 fallbacks in evaluation.
- LLM-Closed: 26/96 proposals replaced by flagged random fallbacks
  (25 structural duplicates of already-evaluated candidates, 1 invalid) —
  duplicates consume budget per protocol. Despite this handicap it still
  wins 12/12 against both non-semantic methods.

## API usage

Model `gpt-5.4-mini-2026-03-17` (single pinned snapshot, recorded in every
call record), temperature 0.7. 98 benchmark calls + 1 smoke call;
99,758 input + 18,342 output tokens; hard `LLM_API_BUDGET_USD=2.00` cap
charged conservatively pre-call (the API returns no dollar figures; token
counts are the authoritative usage record). All raw requests/responses/
retries in `llm_calls/`.

## Relation to v2

The v2 rigid-layout study (`outputs/qae_tfim_api_v2/`) found the same
semantic advantage against different baselines (RY-only random,
Evolutionary, hand-designed). v3 strengthens it: in the *larger, freer*
space the Random baseline drops (0.9182 → 0.8721) while the LLM arms stay
near 0.96 — the harder the space is to search blindly, the bigger the
semantic-prior gain. v2 results remain valid as historical context; v3 is
the primary experiment.

## Scope

Noiseless statevector simulation, 4 qubits, one Hamiltonian family, one
model snapshot, B = 8. The prompt states the physics; the benchmark tests
exploiting stated structure, not discovering it. No general-LLM,
quantum-advantage, or hardware claim.

## Reproduction

```bash
export LLM_API_BUDGET_USD=2.00           # required for the API arms
python scripts/qae/run_qae_tfim_neutral_v3.py            # resumable
python scripts/qae/run_qae_tfim_neutral_v3.py --no-api   # Random+Greedy only
python scripts/qae/build_qae_v3_figures.py               # figures from CSVs
```
