# Paper source

`main.tex` and `references.bib` build the manuscript
"Semantic Priors, Not Feedback Loops: A Capacity-Controlled Evaluation of
LLM-Proposed Quantum-Autoencoder Circuits".

Build: `pdflatex main && bibtex main && pdflatex main && pdflatex main`
(natbib + `unsrtnat`; no biber needed, so the source uploads to arXiv as is).

Every number in the paper comes from committed artifacts:
- robustness study: `outputs/qae_robustness/` (`scripts/qae/run_qae_robustness.py`)
- prior-matched baselines and the matched-warm feedback ablation:
  `outputs/qae_prior_baselines/` — run `scripts/qae/run_prior_baselines.py`
  and `scripts/qae/run_matched_warm.py` (no API calls), then
  `scripts/qae/analyze_prior_baselines.py`, which also writes `figures/`.
- protocol: `docs/research/QAE_PRIOR_BASELINES_PROTOCOL.md` (committed before the runs).
