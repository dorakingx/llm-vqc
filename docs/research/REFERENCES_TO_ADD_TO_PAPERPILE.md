# References to add to the `llm-vqc` literature collection

The Google Drive `llm-vqc` folder already contains Knipfer et al. (2026), Sim et al. (2019), QiboAgent, Conductor, HEPTAPOD/Diagrammatica/ASTER, the magic-entanglement paper, and FiD-QAE.

Recommended additions:

1. **Romero, Olson, Aspuru-Guzik (2017), “Quantum autoencoders for efficient compression of quantum data.”**  
   Quantum Science and Technology 2, 045001. DOI: 10.1088/2058-9565/aa8072. arXiv:1612.02806.  
   Why: foundational QAE formulation; compresses quantum-state families including Hamiltonian ground states.

2. **Agha, Chen, Tseng, Yoo (2025), “Neural Architecture Search for Quantum Autoencoders.”**  
   arXiv:2511.19246.  
   Why: directly studies automated QAE circuit architecture search with a genetic algorithm; critical baseline/related work for LLM-guided QAE search.

3. **Kulshrestha, Liu, Ushijima-Mwesigwa, Safro (2025), “Neural Architecture Search Algorithms for Quantum Autoencoders.”**  
   arXiv:2509.15451.  
   Why: proposes two QAE-specific Quantum-NAS algorithms and reports efficient autoencoder designs on denoising, classical compression, and pure-quantum compression tasks; this is a particularly important confirmatory baseline for the next experiment.

4. **Du et al. (2022), “Quantum circuit architecture search for variational quantum algorithms.”**  
   npj Quantum Information 8, 62. DOI: 10.1038/s41534-022-00570-y.  
   Why: established automated quantum architecture search under VQA constraints.

5. **Frehner & Stockinger (2025), “Applying quantum autoencoders for time series anomaly detection.”**  
   Quantum Machine Intelligence 7, 59. DOI: 10.1007/s42484-025-00285-1. arXiv:2410.04154.  
   Why: compares QAE ansätze/depths and demonstrates QAE behavior on a practical anomaly-detection setting.

6. **Dai et al. (2026), “EAQAS: Embedding-Aware Quantum Architecture Search via cross-attention fusion and hierarchical representation learning.”**  
   EPJ Quantum Technology 13, 22. DOI: 10.1140/epjqt/s40507-026-00478-y.  
   Why: explicitly argues that data embedding and circuit structure should be searched jointly; supports the project’s move from flat gate search to semantically meaningful design choices.

## Paperpile note

The connected tools expose Google Drive files but not Paperpile’s internal collection/tag API. Therefore the repository records the exact references to add, while the Drive `llm-vqc` literature folder is used as the visible shared reference location. Do not claim Paperpile tagging succeeded unless it is confirmed in Paperpile itself.

## Status update (2026-08-28)

All six references above were imported into the Paperpile `llm-vqc` folder on
2026-08-28 via the Paperpile web app (folder count 9 → 15). The list above is
retained as the canonical record of what was added and why.
