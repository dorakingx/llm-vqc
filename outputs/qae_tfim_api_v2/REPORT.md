# QAE-TFIM v2 — version-pinned API verification report

## Headline

The semantic-prior result of the chat pilot **replicates through a real,
version-pinned API**. On the frozen 4-qubit QAE verification task, one open-loop
call to `gpt-5.4-mini-2026-03-17` produced a pool of 8 capacity-checked
candidates whose validation-selected circuit reaches mean protected-test trash
fidelity **0.9700** — above every equal-budget baseline, including the
physics-informed RY-only random control (0.9424) and a newly added Evolutionary
arm (0.9005) and hand-designed reference encoder (0.8889).

| Method | Mean test fidelity | Median | SD | Search? |
|---|---:|---:|---:|---|
| **LLM-API-open** | **0.9700** | 0.9703 | 0.0016 | pool of 8, one API call |
| LLM-chat-frozen (pilot pool) | 0.9674 | 0.9681 | 0.0030 | frozen chat pool of 8 |
| LLM-API-closed | 0.9617 | 0.9665 | 0.0171 | 8 sequential API proposals |
| RY-random (physics-informed control) | 0.9424 | 0.9400 | 0.0178 | 8 random RY topologies |
| Random | 0.9182 | 0.9166 | 0.0360 | 8 random circuits |
| Evolutionary | 0.9005 | 0.9060 | 0.0504 | 4 random + 4 mutations |
| Reference-QAE (hand-designed) | 0.8889 | 0.8894 | 0.0021 | none (1 fixed circuit) |

Paired statistics across the same 12 verification seeds (exact two-sided
Wilcoxon; full table in `paired_stats.json`):

- LLM-API-open − Random: **+0.0518**, 11/12 wins, p = 0.00146
- LLM-API-open − RY-random: **+0.0276**, 12/12 wins, p = 0.00049
- LLM-API-open − Evolutionary: **+0.0695**, 11/12 wins, p = 0.00098
- LLM-API-open − Reference-QAE: **+0.0811**, 12/12 wins, p = 0.00049
- LLM-API-closed − Random: +0.0436, 11/12 wins, p = 0.00244

## Protocol

Identical to the frozen pilot (`docs/research/QAE_PROTOCOL.md`): 4-qubit
open-chain TFIM ground states, latent q0,q1 / trash q2,q3, exactly 12 trainable
rotations + 4 CNOTs per candidate, Adam lr 0.05 × 60 epochs, selection by
validation loss only, protected 64-point test grid read once after selection,
12 verification seeds, B = 8 candidate evaluations per method per seed.
New in v2: Evolutionary (4 random parents + 4 mutations of the top-2 by
validation loss), a hand-designed textbook reference encoder (RY–RZ–RY,
brickwork nearest-neighbour CNOTs, deliberately not trash/latent-aware), and
the two real-API LLM arms below.

## LLM arms: exact, reproducible pipeline

Task description → version-pinned LLM (`gpt-5.4-mini-2026-03-17`, temperature
0.7, chat completions) → structured JSON circuit proposal (schema in
`llm_vqc/experiments/qae_tfim/api_v2.py`, frozen prompt from
`docs/research/LLM_QAE_SKILL.md`) → capacity validation → shared trainer →
validation feedback (closed loop only) → final protected test.

- **Open-loop:** one call requesting 8 distinct candidates; the pool is fixed,
  then evaluated on every seed. The call returned 8 valid, distinct,
  capacity-correct candidates on the first attempt (0 repairs, 0 fallbacks).
- **Closed-loop:** per seed, 8 sequential calls; the prompt contains only
  validation trash fidelities and duplicate flags of this seed's earlier
  candidates — never protected-test values. Invalid or duplicate proposals
  consume budget and are replaced by flagged random circuits.
- **Retry policy:** up to 2 JSON-repair retries per call, every attempt stored.
- **Provenance:** every raw request/response, token count, latency, and model
  snapshot id is stored in `llm_calls/` (98 total calls including 1 smoke
  test: 65,280 input + 8,524 output tokens). Cost was charged against a hard
  `LLM_API_BUDGET_USD=2.00` cap using conservative per-call estimates
  ($0.01 pool / $0.005 closed-loop); the API does not return dollar figures,
  so token counts are the authoritative usage record.

## Findings

1. **The API replay confirms the chat pilot.** The pinned-model open-loop pool
   (0.9700) performs at least as well as the frozen chat pool (0.9674); the
   semantic-prior effect is not an artifact of the interactive session.
2. **The winning motif is stable and interpretable.** The open-loop selection
   chose `center-to-latent` in 12/12 seeds: all-RY rotations with CNOTs
   (1→0),(2→1) then (3→2),(2→1) — route correlations from the trash side
   toward the latent qubits, exactly the motif family the pilot found
   (`L2_pair_then_funnel`).
3. **Closed-loop feedback did not beat open-loop priors.** The closed-loop arm
   still beats all non-LLM baselines, but 23/96 of its proposals were
   duplicates of already-evaluated circuits (0 were invalid JSON) and were
   replaced by flagged random fallbacks. At this tiny budget, iterating on
   validation feedback added no measurable value over stating good priors
   upfront.
4. **Evolutionary search underperforms Random here** (0.9005 vs 0.9182): with
   only 8 evaluations, spending half the budget mutating two early parents
   explores less of the space than 8 independent draws — a budget effect, not
   a general statement about evolutionary search.
5. **The hand-designed generic encoder is the floor** (0.8889): a sensible
   textbook ansatz without task-aware information routing is worse than every
   search method, confirming that architecture choice genuinely matters on
   this task.

## Scope and honesty

- Noiseless statevector simulation, 4 qubits, one Hamiltonian family, B = 8.
- Deterministic protocol, but floating-point training is platform-dependent:
  re-evaluating the frozen chat pool on this machine gives 0.9674 (pilot
  archive: 0.9676) and RY-random 0.9424 (pilot archive: 0.9474). All v2
  comparisons are internally consistent (identical machine, identical seeds).
- No general LLM-superiority claim, no quantum-advantage claim, no
  hardware/noise claim, single pinned model (no cross-model replay yet).

## Reproduction

```bash
export LLM_API_BUDGET_USD=2.00  # required for the API arms
python scripts/qae/run_qae_tfim_api_v2.py            # resumes from stored pool/seeds
python scripts/qae/run_qae_tfim_api_v2.py --no-api   # deterministic arms only
python scripts/qae/build_qae_v2_figures.py
```
