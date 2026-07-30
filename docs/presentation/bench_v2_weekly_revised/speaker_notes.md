# Speaker notes — 20260731_GSoC_revised

Target: 8–10 minutes for slides 1–10, then Q&A from the appendix.
Roughly 40–60 seconds per main slide. The same text is embedded in the
PPTX notes pane.

Acronyms on first use: **VQC** variational quantum circuit ·
**QAS** quantum architecture search · **RMSE** root-mean-square error ·
**AUROC** area under the ROC curve · **IR** intermediate representation.

---

## Slide 1 — Title and status (~30 s)

This is an interim update. The classical half of the benchmark is
finished; the LLM half has not run because it needs API spending
approval. Everything on these slides is generated from durable stores at
commit `da6ef34` — no number was typed in by hand.

## Slide 2 — Question (~55 s)

Two questions, deliberately separated. The primary one compares *search
strategies* at an equal budget of unique candidate evaluations. The
secondary one asks whether searching is worth anything at all compared
with a textbook fixed ansatz.

Say clearly why they are separate: a fixed ansatz does not run a search,
so it cannot be "budget-matched" against one. It is an anchor, evaluated
through the identical training and test pipeline.

On the literature: I am **not** claiming prior papers lack replication in
general — I did not verify that per paper. The gap I verified is
narrower: nobody isolates the proposal strategy against budget-matched
classical search with paired replicates and a protected final test.

## Slide 3 — Pipeline (~60 s)

Walk it left to right. A 32-point signal is L2-normalised once, inside
PennyLane's `AmplitudeEmbedding`, and loaded into 5 qubits — the length
is 2ⁿ precisely so it fills the amplitudes with no padding or truncation.
The searched body is the only thing that differs between methods. We
measure Pauli-Z on qubit 0 and map it to a prediction in [0, 1].

The critical detail: **zero classical trainable parameters.** There is no
dense head that could compensate for a weak circuit. A consequence of
amplitude encoding is that global scale is normalised away, which is why
every task target is a shape parameter — a position or a frequency —
never an absolute amplitude.

## Slide 4 — Tasks (~45 s)

Four synthetic tasks, so I control difficulty and leakage: peak position,
sinusoid frequency below the Nyquist limit, change-point location, and
one-peak-versus-two classification. Nuisance parameters — amplitude,
phase, baseline, width, noise — are randomised so the label cannot be
read off a fixed convention. Splits are 256 / 256 / 2048 per replicate.
No image datasets anywhere.

## Slide 5 — Fairness (~60 s)

This split is the methodological core. In Track A every arm proposes only
a structure, and one shared AdamW loop fits the angles with an identical
schedule — so a win means the *architecture* is better. In Track B the
proposer also supplies the angles and they are used verbatim, which tests
a different skill. Without the split, one lucky set of angles would look
like a good architecture.

Be explicit about what is *not* matched: wall-clock time, training FLOPs
and generated tokens all differ, and the fixed references consume no
search budget at all.

## Slide 6 — Progress (~45 s)

E0 replays the earlier pilot offline and reproduces it exactly with zero
API calls. E1 is the joint track, E2 the main 5-qubit architecture
search, E3 the qubit-scaling study. All 460 classical cells finished with
no failures. The 220 LLM cells are blocked by spending policy, and E4 and
E5 depend on them.

The completion checker still reports 9 of 17 criteria. It fails
*correctly*, and I did not weaken it to make the picture look better.

## Slide 7 — Main result (~60 s)

Left: each grey line joins one replicate's searched circuit to the fixed
reference on the same data and search seed. Right: the paired contrasts
with Holm-adjusted p-values and Cliff's delta.

Two messages. First, the three search strategies are not separated by
this experiment — the smallest adjusted p is 0.071. I say *no difference
was detected*, not that they are equivalent; with ten replicates I cannot
rule out small effects. Second, versus a shallow reference searching does
pay, but versus the strongest reference the advantage is task-dependent:
clear on T2, marginal on T1.

## Slide 8 — Scaling (~60 s)

Five paired replicates per point; faint dots are individual replicates,
the line is the median. On T2 the fixed reference is actually *better* at
3 qubits and then degrades steadily while the searched arms improve — a
crossover, not a uniform win.

Be honest about the statistics: five replicates cap the smallest
achievable two-sided p at 0.0625, so Holm correction across three
contrasts makes significance unreachable *by construction*. That is why I
label this descriptive.

And say plainly: this is exact statevector simulation. No shots, no noise
model, no quantum hardware.

## Slide 9 — Cost and diagnostics (~55 s)

Predictive error against compiled two-qubit gate count. The references
sit far left — cheap but mostly worse. Searched circuits cluster low and
to the right: better error, roughly three to twenty times more entangling
gates. This is a Pareto trade-off, not a free win.

Call the number a *proxy*: it excludes state preparation, which is common
to every arm, and it is a transpiler estimate under a fixed line coupling
— not a device measurement.

The 28 diagnostic circuits are the best-validation replicate for each of
4 tasks × 7 arms. Expressibility and entangling capability are
descriptors; I am not claiming they predict accuracy.

## Slide 10 — Interim answer and decision (~60 s)

Lead with the honest answer: the headline question is not answered yet.
What I can defend is the classical baseline and the machinery around it.

The spending question is now settled in code rather than by a promise.
A durable SQLite ledger reserves the estimated cost before each request
and settles it from the returned token counts, so the cap is one
cumulative total shared by every cell, every retry and every process
restart — the earlier per-cell object reset to zero and was never a
global cap. The model is pinned to gpt-5-nano-2025-08-07 and priced from
a dated manifest ($0.05 / 1M input, $0.40 / 1M output), which puts the
whole 220-cell matrix at roughly $0.6 against a $2.00 cumulative cap.

The measured spend is exactly $0.00. The API account returned
insufficient_quota on the very first preflight request, and the ledger
released all three reservations, so committed spend stayed at zero. The
block is billing, not authorisation. If the cap ever binds mid-run the
run stops cleanly, reports cumulative tokens, calls and remaining cells,
and is never auto-raised; cells resume from their stores, no work lost.

---

# Q&A preparation

**Why use candidate evaluations as the budget?**
It is the quantity every method spends and the one that dominates cost:
each unique candidate triggers a full inner training run. Counting
proposals instead would let an arm win by emitting duplicates; the ledger
records duplicates and invalid proposals separately and they never
consume the budget.

**Is wall-clock or training compute matched?**
No, and the deck says so. The training schedule is identical per
candidate, so compute scales with unique evaluations, but wall-clock,
total FLOPs and token counts are not equalised. An LLM arm will also
spend network latency that classical arms do not.

**What model and prompt will be used for the LLM arm?**
The model is read from `OPENAI_MODEL` at run time and is not yet pinned
in the protocol — that is why the budget slide asks for a cap rather than
quoting a cost. The prompt is versioned (`bench_v2_prompt_v1`),
schema-constrained JSON, three candidates per call, with a bounded top-8
archive in the closed-loop condition. No test-derived information can
enter a prompt; that is enforced structurally and tested.

**Is amplitude state preparation included in the resource counts?**
No. Transpilation is applied to the searched body only. State preparation
is identical for every arm at a given n and would dominate the signal.
That makes the number a *relative* proxy, not a total device cost.

**Why only five scaling replicates?**
Compute. E3 is 120 cells across four qubit counts. The consequence is
explicit on the slide: five replicates cap the attainable p-value, so I
report the scaling result as descriptive rather than confirmatory.

**Why 28 circuits for the diagnostics?**
Four tasks times seven arms in E2, taking for each pair the replicate
whose selected candidate had the best *validation* metric. Selection
never touches the test set.

**Could the searched methods win simply by producing larger circuits?**
That is exactly what slide 9 is for, and it is a live possibility: the
searched circuits are three to twenty times larger in compiled two-qubit
gates. Distinguishing "better structure" from "more parameters" needs a
size-controlled analysis, which is RQ5 and has not been run.

**Why are random, evolutionary and greedy so similar?**
Two plausible explanations, and this experiment does not separate them:
the budget of 16–24 evaluations may be too small for the smarter
strategies to pay off, or the search space may be benign enough that
random sampling finds a good region quickly. T3 in particular looks
saturated — every arm lands at the same RMSE.

**What exactly remains before the benchmark is complete?**
220 real-LLM cells, then E5 theta-isolation and E4 shot-and-noise
robustness, then the regenerated statistical report. The completion
checker must go from 9/17 to 17/17 without any criterion being weakened.
