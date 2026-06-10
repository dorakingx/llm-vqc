# LLM Agent for Quantum Circuit Equivalence Exploration

A Python agent that uses the OpenAI API with function calling to explore equivalence classes of quantum circuits. The backend performs breadth-first search (BFS) over Clifford circuits and prunes equivalent stabilizer states efficiently using Qiskit.

## Features

- **Clifford BFS explorer** (`explore_circuit_space`): explores circuits built from `{H, X, Y, Z, CX, CY, CZ}` up to a maximum depth.
- **Stabilizer pruning**: hashes stabilizer states (not full unitaries) to merge equivalent outcomes.
- **Custom agent loop**: OpenAI tool calling with local execution and follow-up analysis.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Copy the environment template and add your API key:

```bash
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`. Optionally override `OPENAI_MODEL` (default: `gpt-4o-mini`).

## Usage

Run the agent with the bundled example prompt:

```bash
python -m llm_vqc.main
```

Call the explorer directly from Python:

```python
from llm_vqc import explore_circuit_space

result = explore_circuit_space(num_qubits=3, max_depth=5)
print(result["total_unique_states"])
print(result["new_states_at_depth"])
```

## How pruning works

1. BFS starts from the identity Clifford on `N` qubits (state `|0...0>`).
2. Each gate appends a precomputed Clifford operator.
3. The resulting stabilizer state is hashed via its tableau bytes.
4. If the state was seen before, that branch is discarded.
5. The first circuit reaching each state (minimum depth) is retained.

The return payload includes:

- `total_unique_states`: all unique stabilizer states with minimum depth `<= G`
- `new_states_at_depth`: first-time discoveries per depth
- `states_at_exact_depth_G`: states whose minimum depth equals `G`
- `sample_circuits`: representative minimum-depth circuits

## Project layout

```
llm_vqc/
├── circuit_explorer.py   # BFS + stabilizer hashing
├── agent.py              # OpenAI agent loop
└── main.py               # CLI entry point
```
