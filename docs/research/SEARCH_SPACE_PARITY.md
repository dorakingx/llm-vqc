# Search-space parity between classical and LLM arms

Written before the real-LLM run, as required: an LLM arm must face the
same action space as the classical arms, or a win could just mean it was
allowed a richer move set.

Reproduce with:

```bash
python scripts/bench_v2/scan_search_space_parity.py \
    --json-out outputs/bench_v2/search_space_parity.json
```

## The rules now enforced on every search proposal

| id | Rule | Why it matters |
|---|---|---|
| R1 | a rotation operation carries **exactly one** gate | a multi-gate rotation layer packs several physical layers into one operation, buying more circuit per unit of the `max_ops = min(2n, 16)` budget |
| R2 | an entangling operation uses `wires: "all"` | the only policy the frozen classical sampler emits |
| R3 | a `pairs` entangler carries 1..⌊n/2⌋ disjoint, non-duplicate two-element pairs | prevents overlapping or repeated pairs that the sampler cannot produce |
| R4 | no operation packs several gate layers into one | the general form of R1 |

Implemented in `llm_vqc/bench_v2/space.py::_strict_grammar_issues`,
applied by `validate_layered_structure(..., strict_search_grammar=True)`,
which is the default on every search path. A violating proposal becomes
an ordinary INVALID proposal — recorded in the ledger, consuming no
unique-evaluation budget — not a crash.

## Result of scanning the completed classical run

| population | cells | candidates | operations | violations |
|---|---|---|---|---|
| search arms (random, evolutionary, greedy) | 200 | 4,160 | **25,973** | **0** |
| fixed references | 200 | 200 | 640 | 200 (R1, see below) |

**Every one of the 25,973 operations produced by a classical search arm
already satisfies R1–R4.** Enforcing the rules on new proposals therefore
cannot change the completed comparison — the classical results stand
exactly as run. That is what makes it legitimate to hold the LLM to the
same rules rather than having to re-run 460 cells.

## The one asymmetry, stated rather than hidden

The `ref_strongent_d1` and `ref_strongent_d2` templates use a three-gate
rotation layer, `{"type": "rot", "gates": ["RZ","RY","RZ"]}` — 200
operations that violate R1.

They are **fixed reference anchors, not search candidates**: they are
built through `reference_ir(..., strict_search_grammar=False)` and never
enter the search space. The protocol already classifies them as anchors
evaluated through the identical training and test pipeline without
consuming a search budget.

The precise consequence, which the deck states:

* This is an **encoding** asymmetry, not an expressivity gap. A searcher
  can build the identical circuit as three consecutive single-gate
  rotation operations — it simply costs three of its `max_ops`
  operations instead of one. At n=5 (`max_ops = 10`) the StronglyEntangling
  d1 body costs a searcher 4 operations rather than 2, which is well
  inside the budget.
* So the reference is reachable in circuit space and cheaper in operation
  space. When a searched circuit beats it, that is not an artefact of the
  reference being unreachable.

## Prompt consequence

`bench_v2_prompt_v2` states R1–R3 explicitly, so the model is told the
constraint rather than being silently penalised for a rule it cannot
see. The version was bumped from v1 because the wording changed; v1 was
never used for a scientific call (mocks only), so nothing already
measured depends on it.
