# QAE-TFIM v4 — multi-start four-method report

Protocol: `docs/research/QAE_PROTOCOL_V4.md`, pre-registered at commit
`69f8335` BEFORE the matrix ran. v4 was designed after inspecting v3 and
says so explicitly; nothing in v4 was tuned after v4 protected-test
inspection. v3 (`outputs/qae_tfim_neutral_v3/`) is preserved unchanged as
the single-start diagnostic experiment.

Metric: **held-out test trash fidelity**
`F_trash = ⟨00|ρ_trash|00⟩ = P(q2q3 = 00)`, higher is better. "Held-out
(protected) test" refers only to the data split (64 unseen h values); the
fidelity formula is identical on every split.

## Headline

The multi-start (4 warm starts + 4 refinements) design **confirms the v3
diagnosis**: Greedy recovers from significantly-below-Random to parity,
and its refinement phase now demonstrably adds value. The semantic
advantage replicates with a fresh, independently drawn LLM pool. Adaptive
closed-loop search still shows no detectable benefit over the open-loop
batch.

| Method | Design | Mean | Median | SD |
|---|---|---:|---:|---:|
| **LLM-Open** | semantics, open-loop batch of 8 | **0.9643** | 0.9675 | 0.0082 |
| LLM-Closed | semantics, 4 warm + 4 adaptive | 0.9558 | 0.9559 | 0.0127 |
| Greedy | no semantics, 4 warm + 4 refine | 0.8793 | 0.8824 | 0.0320 |
| Random | no semantics, 8 independent | 0.8677 | 0.8691 | 0.0264 |

Paired statistics (12 paired seeds; exact two-sided Wilcoxon; bootstrap
95% CI of mean gain; Cohen's dz):

- LLM-Open − Random: **+0.0965** [0.0824, 0.1120], dz = 3.51, 12/12,
  p = 0.00049.
- LLM-Closed − Random: **+0.0880** [0.0737, 0.1039], dz = 3.10, 12/12,
  p = 0.00049.
- LLM-Open − Greedy: +0.0850 [0.0664, 0.1030], dz = 2.52, 12/12, p = 0.00049.
- LLM-Closed − Greedy: +0.0765 [0.0584, 0.0953], dz = 2.26, 12/12, p = 0.00049.
- Greedy − Random: +0.0115 [−0.0131, +0.0347], dz = 0.26, 7/12, p = 0.42 —
  **parity**; the v3 deficit (−0.1055, p = 0.005) is gone.
- LLM-Closed − LLM-Open: −0.0085 [−0.0177, −0.0007], dz = −0.55, 4/12,
  p = 0.13 — **no significant difference**; we do not claim closed-loop is
  worse, only that no additional closed-loop benefit was detected at B=8.

## Refinement gains (pre-declared): did the 4 adaptive evaluations help?

Final selected minus best-of-first-4 warm starts (held-out test):

- **Greedy: +0.0635** (95% CI [+0.0179, +0.1276]), 10/12 seeds improved,
  Wilcoxon p = 0.0093. After a decent multi-start, validation-guided local
  refinement genuinely adds value — single-start v3 Greedy's failure was
  an exploration problem, not a refinement problem.
- **LLM-Closed: +0.0185** (95% CI [+0.0053, +0.0341]), 5/12 improved,
  7/12 unchanged (refinement never made the selected result worse, since
  selection keeps the best-validation candidate). A small but positive
  contribution — yet not enough to beat the open-loop batch, whose 8
  no-feedback proposals already cover the good region.

## Proposal quality and diversity

- Fallbacks: Random 0/96, Greedy 0/96, LLM-Open 0/96 (pool: 8/8 valid,
  distinct, in ONE call — no repair needed), LLM-Closed **8/96**
  (5 duplicates + 3 invalid), down from 26/96 in v3: the diverse warm
  start largely removed the duplicate-proposal failure mode.
- Diversity (mean pairwise structural distance, fraction of differing
  gate slots): LLM-Closed warm batch 0.832 (most diverse), refinement
  proposals 0.658; LLM-Open first four 0.708, last four 0.615.

## API usage

`gpt-5.4-mini-2026-03-17` (single pinned snapshot recorded per call),
temperature 0.7. 66 benchmark calls + 1 smoke: 48,952 input + 19,177
output tokens, under a hard `LLM_API_BUDGET_USD=2.00` cap (conservative
pre-call estimates; the API returns no dollar figures — token counts are
the authoritative record). Raw calls in `llm_calls/`.

## Interpretation (within scope)

1. **Semantic proposal quality is the dominant ingredient** — replicated
   with an independent pool (v3 pool: 0.9653; v4 pool: 0.9643).
2. **Initial exploration matters for adaptive search**: 4 warm starts
   moved Greedy from significantly-below-Random (v3) to parity, and its
   refinement phase then added +0.0635.
3. **Adaptive refinement on top of good semantic priors added little**:
   LLM-Closed's refinement gain (+0.0185) did not close the gap to the
   open-loop batch; no significant Closed-vs-Open difference.

## Scope

Noiseless statevector simulation, 4 qubits, one Hamiltonian family, one
model snapshot, B = 8. The methods differ in allocation policy as well as
information content, so no strict causal factorial claims are made.

## Reproduction

```bash
export LLM_API_BUDGET_USD=2.00
python scripts/qae/run_qae_tfim_neutral_v4.py            # resumable
python scripts/qae/run_qae_tfim_neutral_v4.py --no-api   # Random + Greedy only
python scripts/qae/build_qae_v4_figures.py               # figures from CSVs
```
