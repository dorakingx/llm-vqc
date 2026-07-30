// Revised GSoC deck: 10 main slides + appendix.
// Every number here is mirrored in claim_source_map.yaml and checked by
// scripts/verify_bench_v2_presentation.py.
const pptxgen = require("pptxgenjs");
const path = require("path");

const HERE = __dirname;
const F = (n) => path.join(HERE, "figures", n);

const INK = "1A1A2E";
const NAVY = "1E3A5F";
const ACCENT = "B3261E";
const OK = "009E73";
const MUTED = "51566B";
const RULE = "D5DAE3";
const BG_SOFT = "EFF3F9";
const BG_WARN = "FDF0E4";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in, 16:9
const W = 13.333, H = 7.5;
const FONT = "Arial";

const DATE = "July 31, 2026";
// Read from git so the title slide can never quote a stale commit.
const SHA = require("child_process")
  .execSync("git rev-parse --short HEAD", { cwd: __dirname }).toString().trim();
const BRANCH = require("child_process")
  .execSync("git rev-parse --abbrev-ref HEAD", { cwd: __dirname }).toString().trim();

function slide(kicker, title) {
  const s = pres.addSlide();
  s.background = { color: "FFFFFF" };
  if (kicker) {
    s.addText(kicker.toUpperCase(), {
      x: 0.55, y: 0.30, w: W - 1.1, h: 0.3, fontSize: 13, bold: true,
      color: ACCENT, fontFace: FONT, charSpacing: 1.5, margin: 0,
    });
  }
  const longTitle = title.length > 60;
  s.addText(title, {
    x: 0.55, y: kicker ? 0.60 : 0.40, w: W - 1.1, h: longTitle ? 0.88 : 0.72,
    fontSize: longTitle ? 23 : 30,
    bold: true, color: NAVY, fontFace: FONT, margin: 0,
  });
  return s;
}

function footer(s, text) {
  s.addText(text, {
    x: 0.55, y: H - 0.40, w: W - 1.1, h: 0.28, fontSize: 11, color: MUTED,
    fontFace: FONT, margin: 0,
  });
}

function bullets(s, items, opts) {
  s.addText(items.map((t, i) => ({
    text: t,
    options: {
      fontSize: opts.size || 20, color: opts.color || INK, bullet: true,
      breakLine: i < items.length - 1, paraSpaceAfter: opts.gap || 10,
    },
  })), { x: opts.x, y: opts.y, w: opts.w, h: opts.h, fontFace: FONT, margin: 0, valign: "top" });
}

// ------------------------------------------------------------------ S1
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addText("LLM-Guided VQC Design", {
    x: 0.9, y: 1.55, w: 11.6, h: 0.9, fontSize: 44, bold: true,
    color: "FFFFFF", fontFace: FONT, margin: 0,
  });
  s.addText("Building a Rigorous, Budget-Matched Benchmark", {
    x: 0.9, y: 2.45, w: 11.6, h: 0.7, fontSize: 30, color: "C9D6E8",
    fontFace: FONT, margin: 0,
  });
  s.addShape(pres.ShapeType.roundRect, {
    x: 0.9, y: 3.55, w: 11.55, h: 0.85, rectRadius: 0.1,
    fill: { color: "FFFFFF", transparency: 88 },
    line: { color: "8FA6C4", width: 1 },
  });
  s.addText([
    { text: "Interim update — ", options: { color: "FFFFFF", bold: true, fontSize: 20 } },
    { text: "classical baselines complete; the real-LLM matrix is blocked on API quota, not on budget approval",
      options: { color: "FFFFFF", fontSize: 20 } },
  ], { x: 1.15, y: 3.62, w: 11.0, h: 0.7, fontFace: FONT, margin: 0, valign: "middle" });
  s.addText([
    { text: "Tomoya Hatanaka", options: { bold: true, breakLine: true, fontSize: 20 } },
    { text: "ML4SCI / Google Summer of Code 2026", options: { breakLine: true, fontSize: 20 } },
    { text: `${DATE}  ·  base commit ${SHA}  ·  ${BRANCH}`, options: { fontSize: 16 } },
  ], { x: 0.9, y: 5.05, w: 11.6, h: 1.5, color: "C9D6E8", fontFace: FONT, margin: 0 });
  s.addNotes(
    "Short framing: this is an interim update. The classical half of the benchmark is finished, " +
    "the LLM half has not run yet because it needs API spending approval. " +
    "VQC = variational quantum circuit; QAS = quantum architecture search. " +
    "Everything shown is generated from durable stores at commit " + SHA + "."
  );
}

// ------------------------------------------------------------------ S2
{
  const s = slide("Question", "Does LLM guidance actually help — compared with what?");
  s.addShape(pres.ShapeType.roundRect, {
    x: 0.6, y: 1.55, w: 6.0, h: 1.85, rectRadius: 0.1,
    fill: { color: BG_SOFT }, line: { color: NAVY, width: 1.2 },
  });
  s.addText([
    { text: "Primary\n", options: { bold: true, fontSize: 18, color: ACCENT } },
    { text: "At the same unique-candidate evaluation budget, does LLM-guided search beat random, evolutionary and greedy search?",
      options: { fontSize: 20, color: INK } },
  ], { x: 0.85, y: 1.72, w: 5.5, h: 1.5, fontFace: FONT, margin: 0, valign: "top" });
  s.addShape(pres.ShapeType.roundRect, {
    x: 6.75, y: 1.55, w: 6.0, h: 1.85, rectRadius: 0.1,
    fill: { color: "F5F6F8" }, line: { color: MUTED, width: 1.2 },
  });
  s.addText([
    { text: "Secondary\n", options: { bold: true, fontSize: 18, color: MUTED } },
    { text: "How do the selected circuits compare with fixed reference ansätze in predictive error and compiled resource cost?",
      options: { fontSize: 20, color: INK } },
  ], { x: 7.0, y: 1.72, w: 5.5, h: 1.5, fontFace: FONT, margin: 0, valign: "top" });

  bullets(s, [
    "Budget matching applies to the search arms. Fixed ansätze are anchors: they are evaluated through the identical training and test pipeline, but they do not run a search.",
    "Anytime improvement alone proves nothing — random search also improves with more evaluations.",
    "Gap addressed here: prior agentic-VQC work demonstrates feasibility, but does not isolate the value of the proposal strategy against budget-matched classical search with paired replicates and a protected final test.",
  ], { x: 0.7, y: 3.75, w: 12.0, h: 2.6, size: 20, gap: 14 });
  footer(s, "Literature basis: docs/research/LITERATURE_REVIEW_QAS.md · claim C8");
  s.addNotes(
    "The primary question compares search strategies at equal budget. The secondary question asks " +
    "whether searching is worth it at all versus a textbook fixed ansatz. " +
    "Be careful with the literature claim: I am not asserting that prior papers lack replication in general. " +
    "The verified gap is narrower — nobody isolates the proposal strategy against budget-matched classical " +
    "search with paired seeds and a protected test. That is the hole this benchmark fills."
  );
}

// ------------------------------------------------------------------ S3
{
  const s = slide("What is measured", "One pipeline; only the searched body changes");
  s.addImage({ path: F("fig_pipeline.png"), x: 0.55, y: 1.75, w: 12.2, h: 2.73 });
  bullets(s, [
    "A length-2ⁿ signal fills the amplitudes of n qubits exactly — no padding, no truncation.",
    "Zero classical parameters: no trainable dense layer anywhere, so a weak circuit cannot be rescued by a classical head.",
    "Protected-test metric: the same metric formula, re-evaluated once per selected circuit on a final held-out split, after selection is frozen on validation. That formula is RMSE for T1–T3; T4 is classification, so the pipeline-level term is \"protected-test metric\", never \"protected-test RMSE\".",
  ], { x: 0.7, y: 4.60, w: 12.0, h: 2.25, size: 17, gap: 10 });
  footer(s, "Implementation: llm_vqc/free_amplitude/model.py · llm_vqc/ir/compiler_pennylane.py");
  s.addNotes(
    "Walk left to right. A 32-point signal is L2-normalised once, inside PennyLane's AmplitudeEmbedding, " +
    "and loaded into 5 qubits. The searched body is the only thing that differs between methods. " +
    "We measure Pauli-Z on qubit 0 and map it to a prediction in [0,1]. " +
    "Crucially there are zero classical trainable parameters — no dense head can compensate for a bad circuit. " +
    "Amplitude encoding removes global scale, which is why every task target is a shape parameter, never an absolute amplitude."
  );
}

// ------------------------------------------------------------------ S4
{
  const s = slide("Tasks", "Four controlled signal tasks, sized for amplitude encoding");
  s.addImage({ path: F("fig_tasks.png"), x: 0.9, y: 1.50, w: 7.3, h: 3.20 });
  const hdr = (txt) => ({ text: txt, options: { bold: true, fill: { color: NAVY }, color: "FFFFFF" } });
  s.addTable([
    [hdr("Search-space constraint"), hdr("Value")],
    ["Qubits n", "5 for T1–T4; T1 and T2 also at n = 3, 4, 6, 8"],
    ["Allowed gates", "rotations RX / RY / RZ / H · entanglers CNOT / CZ / CRX / CRY / CRZ over line, ring, star, pairs or all-to-all"],
    ["Max layer operations", "max_ops = min(2n, 16) — 10 at n=5, 16 at n=8"],
    ["Operation \u2260 physical gate", "one operation expands to many physical gates: a rotation layer becomes n single-qubit gates, an entangling layer up to n two-qubit gates"],
  ], { x: 0.9, y: 4.92, w: 7.4, colW: [1.95, 5.45], fontSize: 11.5, fontFace: FONT,
    border: { type: "solid", color: RULE, pt: 0.75 }, rowH: 0.32, valign: "middle",
    margin: 0.05 });
  const cards = [
    ["2ⁿ points exactly", "32 values at n=5; the signal is the amplitude vector."],
    ["Nuisances randomised", "Amplitude, phase, baseline, width and noise vary, so no fixed convention leaks the label."],
    ["Scaling", "T1 and T2 also run at n = 3, 4, 6 and 8."],
    ["Splits", "256 train / 256 validation / 2048 protected test, per replicate."],
  ];
  cards.forEach(([h, b], i) => {
    const y = 1.62 + i * 1.28;
    s.addText([
      { text: h + "\n", options: { bold: true, fontSize: 19, color: NAVY } },
      { text: b, options: { fontSize: 17, color: MUTED } },
    ], { x: 9.25, y, w: 3.55, h: 1.2, fontFace: FONT, margin: 0, valign: "top" });
  });
  footer(s, "Line colour distinguishes the three example signals in a panel only — it encodes no method, arm or class · Generator: llm_vqc/tasks/signal_suite (signal_suite_v1) · data seed 1000");
  s.addNotes(
    "Four tasks, all synthetic so I control difficulty and leakage. T1 regress the Gaussian peak position, " +
    "T2 the sinusoid frequency below the Nyquist limit, T3 a change-point location, T4 one peak versus two peaks. " +
    "Every signal has exactly 2^n points. Nuisance parameters are randomised so the label cannot be read off a " +
    "fixed amplitude convention. No image datasets are used anywhere."
  );
}

// ------------------------------------------------------------------ S5
{
  const s = slide("Fairness", "Two tracks separate architecture quality from angle luck");
  const box = (x, colour, title, lines, bg) => {
    s.addShape(pres.ShapeType.roundRect, {
      x, y: 1.55, w: 6.0, h: 2.95, rectRadius: 0.1,
      fill: { color: bg }, line: { color: colour, width: 1.4 },
    });
    s.addText(title, { x: x + 0.25, y: 1.68, w: 5.5, h: 0.42, bold: true,
      fontSize: 21, color: colour, fontFace: FONT, margin: 0 });
    bullets(s, lines, { x: x + 0.32, y: 2.30, w: 5.4, h: 2.1, size: 17, gap: 8 });
  };
  box(0.6, NAVY, "Track A — structure search (primary)", [
    "Arms propose structure only",
    "One shared AdamW loop trains θ: 40 epochs, batch 32, identical for every arm",
    "θ-seed derived from the structural hash, so the same circuit gives the same result in any arm",
  ], BG_SOFT);
  box(6.75, "7A4E9E", "Track B — joint proposal (ablation)", [
    "Candidates carry gates, wires and numeric angles",
    "θ is evaluated verbatim — no optimizer exists on this path",
    "Candidate identity includes θ, so new angles make a new candidate",
  ], "F4F0F8");
  s.addShape(pres.ShapeType.roundRect, {
    x: 0.6, y: 4.75, w: 12.15, h: 1.75, rectRadius: 0.1,
    fill: { color: BG_WARN }, line: { color: "B8860B", width: 1.2 },
  });
  s.addText([
    { text: "Matched across arms: ", options: { bold: true, fontSize: 19, color: INK } },
    { text: "unique candidate evaluations, training schedule, data splits, seeds, evaluator, protected-test rule.   ",
      options: { fontSize: 19, color: INK } },
    { text: "Not matched: ", options: { bold: true, fontSize: 19, color: ACCENT } },
    { text: "wall-clock time, training FLOPs, generated tokens — and fixed references run no search at all.",
      options: { fontSize: 19, color: INK } },
  ], { x: 0.85, y: 4.9, w: 11.6, h: 1.45, fontFace: FONT, margin: 0, valign: "middle" });
  footer(s, "Protocol §5 · runner enforces the budget on unique evaluations only");
  s.addNotes(
    "This split is the methodological core. In Track A every arm proposes only a structure and one shared " +
    "trainer fits the angles, so a win means the architecture is better. In Track B the proposer also supplies " +
    "the angles and they are used verbatim, which tests a different skill. Without the split, a lucky set of " +
    "angles would look like a good architecture. " +
    "Note what is not matched: wall-clock, FLOPs and tokens differ, and fixed references do not consume a search budget."
  );
}

// ------------------------------------------------------------------ S6
{
  const s = slide("Progress", "Classical matrix complete; real-LLM matrix not started");
  s.addImage({ path: F("fig_matrix.png"), x: 0.65, y: 1.5, w: 11.9, h: 4.05 });
  s.addShape(pres.ShapeType.roundRect, {
    x: 0.65, y: 5.75, w: 11.9, h: 0.85, rectRadius: 0.08,
    fill: { color: BG_SOFT }, line: { color: RULE, width: 1 },
  });
  s.addText([
    { text: "460 / 460 classical cells complete, 0 failures", options: { bold: true, fontSize: 20, color: OK } },
    { text: "   ·   0 / 220 real-LLM cells — the API account returned insufficient_quota, so $0.00 was spent   ·   ", options: { fontSize: 17, color: INK } },
    { text: "the benchmark as a whole is not complete", options: { bold: true, fontSize: 18, color: ACCENT } },
  ], { x: 0.9, y: 5.85, w: 11.4, h: 0.65, fontFace: FONT, margin: 0, valign: "middle" });
  footer(s, "Counts regenerated from runs/bench_v2/*/cells/*.json · completion checker: 10 of 17 criteria pass, unweakened");
  s.addNotes(
    "E0 replays the February pilot offline and reproduces it exactly with zero API calls. " +
    "E1 is the joint track, E2 the main n=5 architecture search, E3 the qubit-scaling study. " +
    "All 460 classical cells finished with no failures. The 220 LLM cells are blocked by spending policy, " +
    "and E4 and E5 depend on them, so they are pending too. " +
    "The completion checker reports 10 of 17 criteria — it fails correctly and I did not weaken it."
  );
}

// ------------------------------------------------------------------ S7
{
  const s = slide("Result — 5 qubits", "Search beats the fixed reference on T1/T2; no search-method difference was detected");
  s.addImage({ path: F("fig_e2_main.png"), x: 0.45, y: 1.58, w: 7.9, h: 4.42 });
  s.addImage({ path: F("fig_e2_forest.png"), x: 8.30, y: 1.58, w: 4.48, h: 3.03 });
  bullets(s, [
    "No Holm-adjusted difference was detected among random, evolutionary and greedy on any task (smallest p = 0.071). This is not an equivalence claim.",
    "Against the strongest reference the picture is task-dependent: on T2 random and evolutionary win (p = 0.046, δ = −0.8); on T1 only random is detected (p = 0.032) and the shift is small.",
  ], { x: 8.30, y: 4.80, w: 4.48, h: 1.95, size: 15, gap: 8 });
  footer(s, "10 paired replicates · Hodges–Lehmann shift and paired Cliff's δ · outputs/bench_v2/E2/stats_*.csv");
  s.addNotes(
    "Left: each line joins one replicate's searched circuit to the fixed reference on the same data and search seed. " +
    "Right: the paired contrasts with Holm-adjusted p-values and Cliff's delta. " +
    "Two messages. First, the three search strategies are not separated by this experiment — smallest adjusted p is 0.071. " +
    "I say no difference was detected, not that they are equivalent; with ten replicates I cannot rule out small effects. " +
    "Second, versus a shallow reference searching does pay, but versus the strongest reference the advantage is task dependent."
  );
}

// ------------------------------------------------------------------ S8
{
  const s = slide("Result — qubit scaling", "A descriptive crossover: search pulls ahead at larger qubit counts");
  s.addImage({ path: F("fig_e3_scaling.png"), x: 0.6, y: 1.5, w: 11.9, h: 4.34 });
  s.addShape(pres.ShapeType.roundRect, {
    x: 0.6, y: 6.02, w: 11.9, h: 0.92, rectRadius: 0.08,
    fill: { color: BG_WARN }, line: { color: "B8860B", width: 1.1 },
  });
  s.addText([
    { text: "Descriptive, not confirmatory: ", options: { bold: true, fontSize: 18, color: INK } },
    { text: "at n=8 on T2 all 5 paired replicates favour search (δ = −1.0, median 0.058 vs 0.295), but with 5 replicates the smallest attainable p is 0.0625, so after Holm correction across 3 contrasts no result can reach 0.05.",
      options: { fontSize: 18, color: INK } },
  ], { x: 0.85, y: 6.12, w: 11.4, h: 0.75, fontFace: FONT, margin: 0, valign: "middle" });
  footer(s, "All results are noiseless statevector simulation — no shots, no noise model, no quantum hardware");
  s.addNotes(
    "Both panels: 5 paired replicates per point, faint dots are individual replicates, the line is the median. " +
    "On T2 the fixed reference is actually better at 3 qubits and then degrades steadily, while the searched arms " +
    "improve — a crossover, not a uniform win. " +
    "Be explicit about the statistics: five replicates cap the smallest achievable p-value at 0.0625, so Holm " +
    "correction makes significance unreachable by construction. I therefore call this descriptive. " +
    "And say clearly: this is exact statevector simulation, not hardware."
  );
}

// ------------------------------------------------------------------ S9
{
  const s = slide("Cost", "Lower error comes with substantially larger compiled circuits");
  s.addImage({ path: F("fig_pareto.png"), x: 0.5, y: 1.5, w: 7.4, h: 4.25 });
  s.addImage({ path: F("fig_diagnostics.png"), x: 8.15, y: 1.5, w: 4.35, h: 3.36 });
  bullets(s, [
    "Median compiled 2-qubit gates: 58–85 for searched circuits vs 4–29 for the references.",
    "A hardware-relevant proxy: state preparation is excluded (identical for every arm).",
    "KL and Meyer–Wallach Q are descriptors; their link to accuracy is untested here.",
  ], { x: 8.15, y: 4.98, w: 4.75, h: 1.85, size: 15, gap: 8 });
  footer(s, "Qiskit, basis [rz,sx,x,cx], line coupling, opt_level 1, seed 7 · 28 circuits = best-validation replicate per (task, arm)");
  s.addNotes(
    "Left: predictive error against compiled two-qubit gate count for T1. The references sit far left — cheap but mostly worse. " +
    "Searched circuits cluster low and right: better error, three to twenty times more entangling gates. " +
    "Call it a resource proxy, not hardware cost: it excludes state preparation, which is common to all arms, " +
    "and it is a transpiler estimate under a fixed line coupling, not a device measurement. " +
    "Right: the 28 diagnostic circuits are the best-validation replicate for each of 4 tasks times 7 arms. " +
    "Expressibility and entangling capability are descriptors — I am not claiming they predict accuracy."
  );
}

// ------------------------------------------------------------------ S10
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addText("Interim answer, and what is blocking the rest", {
    x: 0.7, y: 0.45, w: 12.0, h: 0.7, fontSize: 30, bold: true,
    color: "FFFFFF", fontFace: FONT, margin: 0,
  });
  s.addShape(pres.ShapeType.roundRect, {
    x: 0.7, y: 1.30, w: 12.0, h: 1.42, rectRadius: 0.1,
    fill: { color: "FFFFFF" }, line: { color: ACCENT, width: 1.8 },
  });
  s.addText([
    { text: "Not yet. ", options: { bold: true, fontSize: 26, color: ACCENT } },
    { text: "The LLM hypothesis is still untested. What exists today is a reproducible classical benchmark that defines the bar the LLM has to beat.",
      options: { fontSize: 21, color: INK } },
  ], { x: 1.0, y: 1.42, w: 11.4, h: 1.2, fontFace: FONT, margin: 0, valign: "middle" });

  s.addShape(pres.ShapeType.roundRect, {
    x: 0.7, y: 3.0, w: 5.85, h: 3.05, rectRadius: 0.1,
    fill: { color: "FFFFFF", transparency: 90 }, line: { color: "8FA6C4", width: 1 },
  });
  s.addText("Spending control: implemented, not promised", {
    x: 0.95, y: 3.12, w: 5.5, h: 0.42, bold: true, fontSize: 18, color: "FFFFFF",
    fontFace: FONT, margin: 0,
  });
  bullets(s, [
    "Durable SQLite ledger shared by every cell, retry and restart — cumulative, not per cell",
    "Cost settled from returned token counts against a dated price manifest",
    "Model pinned to gpt-5-nano-2025-08-07; cumulative cap $2.00",
    "Measured spend $0.00 — account returned insufficient_quota, reservations released",
  ], { x: 1.05, y: 3.62, w: 5.25, h: 2.3, size: 14, gap: 7, color: "E8EEF7" });

  s.addShape(pres.ShapeType.roundRect, {
    x: 6.85, y: 3.0, w: 5.85, h: 3.05, rectRadius: 0.1,
    fill: { color: "FFFFFF", transparency: 90 }, line: { color: "8FA6C4", width: 1 },
  });
  s.addText("Then, in order", {
    x: 7.1, y: 3.12, w: 5.4, h: 0.4, bold: true, fontSize: 19, color: "FFFFFF",
    fontFace: FONT, margin: 0,
  });
  bullets(s, [
    "Restore API quota, then run the 220 real-LLM cells (both tracks)",
    "E5 θ-isolation: is it the architecture or the angles?",
    "E4 shot-noise and noisy-simulator robustness",
    "Regenerate the statistical report; completion checker must pass unweakened",
  ], { x: 7.2, y: 3.6, w: 5.3, h: 2.3, size: 16, gap: 8, color: "E8EEF7" });

  s.addText("Blocked on API quota, not on approval: cap $2.00 cumulative, measured spend $0.00", {
    x: 0.7, y: 6.30, w: 12.0, h: 0.60, fontSize: 20, bold: true, color: "FFD9A0",
    fontFace: FONT, margin: 0, align: "center",
  });
  s.addNotes(
    "Lead with the honest answer: the headline question is not answered yet. " +
    "What I can defend is the classical baseline and the machinery around it. " +
    "The spending question is now settled in code rather than by a promise. A durable SQLite ledger reserves " +
    "the estimated cost before each request and settles it from the returned token counts, so the cap is one " +
    "cumulative total shared by every cell, every retry and every process restart. The model is pinned and " +
    "priced from a dated manifest. The cumulative cap is $2.00 and the measured spend is exactly zero, because " +
    "the API account returned insufficient_quota on the very first preflight request and the ledger released " +
    "all three reservations. So the block is billing, not authorisation. If the cap ever binds mid-run the run " +
    "stops cleanly and cells resume from their stores — no work is lost and the cap is never auto-raised."
  );
}

// ============================== APPENDIX ==============================
function appendix(title, subtitle) {
  const s = pres.addSlide();
  s.background = { color: "FFFFFF" };
  s.addText("APPENDIX", {
    x: 0.55, y: 0.28, w: 6, h: 0.3, fontSize: 12, bold: true, color: MUTED,
    fontFace: FONT, charSpacing: 1.5, margin: 0,
  });
  s.addText(title, {
    x: 0.55, y: 0.56, w: W - 1.1, h: 0.6, fontSize: 26, bold: true,
    color: NAVY, fontFace: FONT, margin: 0,
  });
  if (subtitle) {
    s.addText(subtitle, { x: 0.55, y: 1.14, w: W - 1.1, h: 0.4, fontSize: 17,
      color: MUTED, fontFace: FONT, margin: 0 });
  }
  return s;
}

{ // A1/A2
  const s = appendix("A1–A2  Pilot replay (E0) and joint search (E1)");
  s.addImage({ path: F("fig_ap_e1.png"), x: 0.7, y: 1.75, w: 11.0, h: 3.57 });
  bullets(s, [
    "E0: all 21 candidates of the committed 2-seed real-LLM pilot were re-evaluated offline from stored operations and angles — validation and test values matched exactly, with zero API calls.",
    "E1 (Track B, verbatim θ, B=16): no Holm-adjusted difference between random and evolutionary joint search on any of the three task/qubit cells (all p ≥ 0.28). T2 at n=5 sits near 0.29 RMSE for both.",
  ], { x: 0.7, y: 5.5, w: 12.0, h: 1.5, size: 16, gap: 8 });
  footer(s, "outputs/bench_v2/E0/replication_report.json · outputs/bench_v2/E1/stats_paired_tests.csv");
}
{ // A3
  const s = appendix("A3  Full E2 results, all four tasks");
  s.addImage({ path: F("fig_ap_e2_grid.png"), x: 0.5, y: 1.75, w: 12.3, h: 3.28 });
  bullets(s, [
    "T3 (change point) is saturated: every arm lands at RMSE ≈ 0.10 and all adjusted p-values are 1.00.",
    "T4 (peak classification) shows no arm separation either: AUROC ≈ 0.73–0.75 across all arms.",
  ], { x: 0.6, y: 5.3, w: 12.0, h: 1.3, size: 17, gap: 8 });
  footer(s, "outputs/bench_v2/E2/per_seed_E2.csv · 10 paired replicates per arm");
}
{ // A4
  const s = appendix("A4  Anytime search behaviour (E2)");
  s.addImage({ path: F("fig_ap_anytime.png"), x: 0.5, y: 1.8, w: 12.3, h: 2.95 });
  bullets(s, [
    "Median and interquartile band of the best-so-far validation metric against unique candidate evaluations.",
    "Greedy growth is the slowest starter on T2 but converges to the same region by the 24-evaluation budget.",
  ], { x: 0.6, y: 5.05, w: 12.0, h: 1.3, size: 17, gap: 8 });
  footer(s, "Budget axis is unique candidate evaluations; duplicates and invalid proposals never consume it");
}
{ // A6/A8
  const s = appendix("A6, A8  Statistics, seeds and the protected test");
  bullets(s, [
    "Paired tests within each (task, n) family: exact Wilcoxon signed-rank plus a paired sign-flip permutation test on the mean difference (10,000 resamples, seed 20260729).",
    "Multiplicity: Holm step-down applied to the permutation p-values inside each family.",
    "Effect sizes: Hodges–Lehmann shift (median of Walsh averages, RMSE units) and paired Cliff's δ.",
    "Verified p-value floor: 5 replicates ⇒ smallest two-sided p = 0.0625 ⇒ smallest Holm-adjusted value ≈ 0.19 across 3 contrasts. 10 replicates ⇒ 0.00195.",
    "Seeds: data seed 1000+r, search seed 2000+r, shared by every arm in a comparison; θ-seed derived from (task, data seed, search seed, structural hash).",
    "Protected-test metric: the same metric formula used for validation, re-evaluated once per selected circuit on a final held-out split of 2,048 examples, after selection has been frozen. RMSE = sqrt(mean((ŷ − y)²)) for the regression tasks T1–T3; AUROC for the T4 classification task, which is why the pipeline-level term is \"protected-test metric\".",
    "Search modules cannot import the test gate, and the restriction is enforced by an AST test rather than by convention.",
  ], { x: 0.7, y: 1.7, w: 12.0, h: 5.0, size: 16, gap: 11 });
  footer(s, "scripts/bench_v2/analyze_experiment.py · llm_vqc/bench_v2/test_gate.py");
}
{ // A7/A10
  const s = appendix("A7, A10  Search space, candidate identity, diagnostic selection");
  bullets(s, [
    "Grammar scalable_layered_v1: ordered list of at most min(2n, 16) operations — rotation layers (RX/RY/RZ/H) and entangling layers (CNOT/CZ/CRX/CRY/CRZ) over line, ring, star, pairs or all-to-all patterns.",
    "Encoding and readout are fixed by the profile, never proposed. Every structure must contain at least one trainable rotation.",
    "Track A identity = structural hash of the canonical IR; Track B identity additionally includes θ, so re-proposing the same shape with new angles is a new candidate.",
    "Diagnostic circuits: for each of the 4 tasks × 7 arms in E2, the replicate whose selected candidate had the best validation metric — 28 circuits, selected on validation only.",
    "Expressibility: KL between the sampled fidelity distribution and the Haar distribution, 200 sampled states and 2,000 fidelity pairs, θ ~ U[−π, π], seeded by architecture hash.",
    "Enforced parity for the LLM arms: one gate per rotation operation, entangler wires \"all\", and 1..floor(n/2) disjoint non-duplicate pairs — so no arm can pack several gate layers into one operation. A scan of all 25,973 operations produced by the completed classical search arms found zero violations, so the comparison already run is unaffected.",
  ], { x: 0.7, y: 1.7, w: 12.0, h: 5.0, size: 17, gap: 12 });
  footer(s, "llm_vqc/bench_v2/space.py · llm_vqc/free_amplitude/expressibility.py");
}
{ // A9
  const s = appendix("A9  Resource and transpilation assumptions");
  bullets(s, [
    "Compiler: Qiskit; basis gates [rz, sx, x, cx]; optimization_level 1; seed_transpiler 7; coupling maps line, ring and all-to-all.",
    "Routing SWAPs are decomposed into cx, so the reported two-qubit count already includes routing overhead.",
    "Amplitude state preparation is excluded: its decomposition is identical for every arm at a given n and would dominate the body signal. The number is therefore a relative proxy, not a total device cost.",
    "Logical counts (1-qubit, 2-qubit, controlled-rotation, parameters, depth) come from the shared IR and are recorded for every evaluated candidate.",
    "Shot-noise (1,024 / 4,096) and the frozen depol_ro_v1 noisy profile belong to E4, which re-evaluates the best-validation selected circuit of every completed E2/E3 cell — it needs no LLM cells, so it runs on the classical matrix.",
  ], { x: 0.7, y: 1.7, w: 12.0, h: 5.0, size: 17, gap: 12 });
  footer(s, "llm_vqc/bench_v2/resources.py");
}
{ // A11
  const s = appendix("A11  API cost model and the cumulative cap", "Priced from a dated manifest and settled from returned tokens — no flat per-call charge");
  const rows = [
    [{ text: "Quantity", options: { bold: true, fill: { color: NAVY }, color: "FFFFFF" } },
     { text: "Value", options: { bold: true, fill: { color: NAVY }, color: "FFFFFF" } },
     { text: "Source", options: { bold: true, fill: { color: NAVY }, color: "FFFFFF" } }],
    ["Pinned model", "gpt-5-nano-2025-08-07", "OPENAI_MODEL, verified before any client is built"],
    ["Price (dated manifest)", "$0.05 / 1M input, $0.40 / 1M output", "configs/bench_v2/model_prices.json"],
    ["Cells to run", "220 (E1 60, E2 80, E3 80)", "protocol_v2.yaml"],
    ["Candidates per call", "3", "llm_arms.py BATCH_SIZE"],
    ["Expected total calls", "≈1,480", "ceil(B/3) + duplicate allowance, over 220 cells"],
    ["Expected cost per call", "≈$0.0004", "measured prompt size × manifest price"],
    ["Expected total cost", "≈$0.6", "1,480 × $0.0004"],
    ["Per-cell call guard", "40 calls (runaway guard — NOT the global cap)", "llm_providers.py"],
    ["Cumulative global cap", "$2.00, durable across cells, retries and restarts", "GlobalSpendLedger (SQLite, BEGIN IMMEDIATE)"],
    ["Actual measured spend", "$0.00 — 0 settled calls; account returned insufficient_quota", "runs/bench_v2/preflight_ledger.sqlite"],
  ];
  s.addTable(rows, { x: 0.7, y: 1.80, w: 11.9, colW: [2.9, 4.6, 4.4], fontSize: 13,
    fontFace: FONT, border: { type: "solid", color: RULE, pt: 0.75 }, rowH: 0.42,
    valign: "middle", margin: 0.06 });
  footer(s, "Cost is reserved before each request and settled from returned token counts; a claim that would exceed the cumulative cap is refused before the request is issued");
}
{ // A12/A13
  const s = appendix("A12, A13  Integrity checks and references");
  bullets(s, [
    "Break-tests: three safeguards were deliberately broken to confirm the tests catch them. Two exposed real blind spots — a plain `import torch.optim` evaded the AST check, and a self-consistent budget break evaded the ledger test — both are now closed.",
    "Completion contract: 17 machine-checked criteria; currently 10 pass. The failures are correct (missing LLM cells, E4/E5 markers, final report) and the checker was not modified.",
    "The self-consistent budget break that once evaded the ledger test is now closed twice over: the cumulative SQLite ledger is proved to survive a restart in a separate OS process, and a cap breach is refused before the request is issued rather than detected after it.",
    "Protocol content is hash-pinned in outputs/bench_v2/protocol_hashes.json. There was no external preregistration; the deck therefore says protocol-frozen throughout.",
    "Key references: Sim et al. 2019 (expressibility, entangling capability); Knipfer et al. 2026 and Sakka et al. 2026 (LLM-driven VQC design); DQAS; QuantumNAS; EVQE and MoG-VQE; TD-QAS; PWMCTS; BenchRL-QAS.",
  ], { x: 0.7, y: 1.7, w: 12.0, h: 5.0, size: 17, gap: 13 });
  footer(s, "docs/research/BREAK_TESTS.md · GOAL_CONTRACT.yaml · docs/research/LITERATURE_REVIEW_QAS.md");
}

pres.writeFile({ fileName: path.join(HERE, "20260731_GSoC_revised.pptx") })
  .then(() => console.log("deck written"));
