// bench_v2 weekly deck — 10 slides, all numbers traceable to
// data/*.csv (built by build_deck_figures.py from durable stores).
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

const HERE = __dirname;
const FIG = (name) => path.join(HERE, "figures", name);

const NAVY = "1E2761";
const ICE = "CADCFC";
const WHITE = "FFFFFF";
const DARKTEXT = "1A1A2E";
const MUTED = "5A5A6E";
const ACCENT = "C44E52";
const GREEN = "2C5F2D";
const AMBER = "B8860B";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.theme = { headFontFace: "Cambria", bodyFontFace: "Calibri" };

const W = 13.33, H = 7.5;

function footer(slide, text) {
  slide.addText(text, {
    x: 0.5, y: H - 0.42, w: W - 1.0, h: 0.3, fontSize: 8.5, color: MUTED,
    fontFace: "Calibri", align: "left", margin: 0,
  });
}

function titleBar(slide, title, kicker) {
  if (kicker) {
    slide.addText(kicker.toUpperCase(), {
      x: 0.55, y: 0.28, w: W - 1.1, h: 0.3, fontSize: 11, bold: true,
      color: ACCENT, fontFace: "Calibri", charSpacing: 2, margin: 0,
    });
  }
  slide.addText(title, {
    x: 0.55, y: kicker ? 0.55 : 0.35, w: W - 1.1, h: 0.75, fontSize: 30,
    bold: true, color: NAVY, fontFace: "Cambria", margin: 0,
  });
}

// ---------------------------------------------------------------- S1 ----
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addText("LLM-Guided VQC Design:\nFrom Pilot to Rigorous Benchmark", {
    x: 0.9, y: 1.7, w: 11.5, h: 2.0, fontSize: 40, bold: true, color: WHITE,
    fontFace: "Cambria", margin: 0,
  });
  s.addText(
    "Amplitude encoding, quantum readout, and budget-matched evaluation of\n" +
    "quantum architecture search (QAS)",
    { x: 0.9, y: 3.75, w: 11.0, h: 0.9, fontSize: 19, color: ICE,
      fontFace: "Calibri", margin: 0 });
  s.addText(
    [
      { text: "Tomoya Hatanaka", options: { bold: true, breakLine: true } },
      { text: "ML4SCI / Google Summer of Code 2026 — weekly update", options: { breakLine: true } },
      { text: "July 29, 2026  ·  branch research/rigorous-qas-benchmark-v2", options: {} },
    ],
    { x: 0.9, y: 5.5, w: 9.5, h: 1.2, fontSize: 14, color: ICE,
      fontFace: "Calibri", margin: 0 });
  s.addShape(pres.ShapeType.ellipse, {
    x: 11.1, y: 5.3, w: 1.5, h: 1.5, fill: { color: ICE, transparency: 82 },
    line: { color: ICE, width: 1 },
  });
  s.addText("|0⟩→Z", {
    x: 11.1, y: 5.3, w: 1.5, h: 1.5, align: "center", fontSize: 17,
    color: WHITE, fontFace: "Cambria", margin: 0,
  });
}

// ---------------------------------------------------------------- S2 ----
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  titleBar(s, "Does LLM guidance actually help quantum architecture search?",
           "Motivation");
  s.addShape(pres.ShapeType.roundRect, {
    x: 0.8, y: 1.6, w: 11.7, h: 1.5, rectRadius: 0.12,
    fill: { color: ICE, transparency: 45 }, line: { color: NAVY, width: 1 },
  });
  s.addText(
    "“Does LLM guidance improve VQC architecture search over random, evolutionary, " +
    "greedy, and reference ansätze under the same candidate-evaluation budget?”",
    { x: 1.1, y: 1.72, w: 11.1, h: 1.26, fontSize: 17, italic: true, color: NAVY,
      align: "center", fontFace: "Cambria", margin: 0 });

  const rows = [
    ["Prior agentic-VQC work motivates the direction",
     "Knipfer et al. 2026 and Sakka et al. 2026 show LLM agents can design VQCs — " +
     "but with no budget-matched non-LLM baselines and no seed-replicated statistics."],
    ["Anytime improvement alone is uninterpretable",
     "Random search also improves over iterations; only equal-budget comparison isolates " +
     "the value of the proposal strategy."],
    ["Our contribution: the controlled evaluation layer",
     "Same typed circuit grammar, same evaluator, same unique-candidate budget, paired seeds, " +
     "protected test — for every arm, LLM or classical."],
  ];
  rows.forEach(([head, body], i) => {
    const y = 3.5 + i * 1.15;
    s.addShape(pres.ShapeType.ellipse, {
      x: 0.85, y: y + 0.06, w: 0.42, h: 0.42, fill: { color: NAVY },
    });
    s.addText(String(i + 1), {
      x: 0.85, y: y + 0.06, w: 0.42, h: 0.42, align: "center", color: WHITE,
      bold: true, fontSize: 14, margin: 0 });
    s.addText([
      { text: head, options: { bold: true, color: DARKTEXT, breakLine: true, fontSize: 14.5 } },
      { text: body, options: { color: MUTED, fontSize: 12.5 } },
    ], { x: 1.5, y: y - 0.06, w: 11.0, h: 1.1, fontFace: "Calibri", margin: 0 });
  });
  footer(s, "Lit review + claim discipline: docs/research/LITERATURE_REVIEW_QAS.md (C1–C8)");
}

// ---------------------------------------------------------------- S3 ----
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  titleBar(s, "From a 2-seed pilot to a preregistered benchmark", "What changed");
  const header = { fill: { color: NAVY }, color: WHITE, bold: true, fontSize: 12.5 };
  const oldCell = { color: MUTED, fontSize: 12 };
  const newCell = { color: DARKTEXT, fontSize: 12 };
  const tableRows = [
    [{ text: "", options: header },
     { text: "July pilot (now: E0 provenance only)", options: header },
     { text: "bench_v2 (this benchmark)", options: header }],
    [{ text: "Replication", options: { bold: true, fontSize: 12 } },
     { text: "2 seeds", options: oldCell },
     { text: "10 paired replicates (5 for scaling), paired seeds per arm", options: newCell }],
    [{ text: "Budget", options: { bold: true, fontSize: 12 } },
     { text: "B = 4 proposals", options: oldCell },
     { text: "B = 16–24 unique candidate evaluations, exact ledgers", options: newCell }],
    [{ text: "Tasks", options: { bold: true, fontSize: 12 } },
     { text: "Gaussian peak only (n=3)", options: oldCell },
     { text: "signal_suite T1–T4; scaling n ∈ {3,4,6,8}", options: newCell }],
    [{ text: "Method scope", options: { bold: true, fontSize: 12 } },
     { text: "joint structure+θ only", options: oldCell },
     { text: "Track A (structure-only, shared trainer) + Track B (joint, verbatim θ)", options: newCell }],
    [{ text: "Baselines", options: { bold: true, fontSize: 12 } },
     { text: "random only", options: oldCell },
     { text: "random, evolutionary, greedy, 4 fixed reference ansätze, joint-evolutionary", options: newCell }],
    [{ text: "Claims", options: { bold: true, fontSize: 12 } },
     { text: "descriptive (integration demo)", options: oldCell },
     { text: "preregistered RQ1–RQ6, Holm-corrected paired tests, effect sizes", options: newCell }],
    [{ text: "Integrity", options: { bold: true, fontSize: 12 } },
     { text: "protected test, 2 committed seeds", options: oldCell },
     { text: "+ resource metrics, machine-checked 17-criterion goal contract, break-tests", options: newCell }],
  ];
  s.addTable(tableRows, {
    x: 0.6, y: 1.65, w: 12.1, colW: [1.7, 3.9, 6.5], fontFace: "Calibri",
    border: { type: "solid", color: "D8DDE8", pt: 0.75 },
    rowH: 0.52, valign: "middle", margin: 0.06,
  });
  footer(s, "Pilot preserved as E0; protocol frozen before the matrix: docs/research/BENCHMARK_V2_PROTOCOL.md");
}

// ---------------------------------------------------------------- S4 ----
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  titleBar(s, "Four controlled signal tasks, sized for amplitude encoding",
           "Benchmark design");
  s.addImage({ path: FIG("fig_tasks.png"), x: 0.55, y: 1.75, w: 12.2, h: 2.85 });
  const cards = [
    ["2^n features", "Every task generates exactly 2^n-point signals — the amplitude vector of n qubits. No padding, no truncation."],
    ["Nuisance randomization", "Amplitude, phase, baseline, widths, noise randomized so no fixed convention leaks the label."],
    ["Scaling profiles", "T1/T2 frozen for n ∈ {3,4,6,8} (E3, running); T3/T4 at n=5. Frequency cap N/4 keeps T2 below Nyquist."],
    ["Splits", "256 train / 256 val / 2048 protected test per replicate; deterministic hashes; disjointness enforced."],
  ];
  cards.forEach(([head, body], i) => {
    const x = 0.55 + i * 3.11;
    s.addShape(pres.ShapeType.roundRect, {
      x, y: 4.95, w: 2.92, h: 1.85, rectRadius: 0.1,
      fill: { color: ICE, transparency: 62 }, line: { color: "D8DDE8", width: 0.75 },
    });
    s.addText([
      { text: head, options: { bold: true, color: NAVY, fontSize: 12.5, breakLine: true } },
      { text: body, options: { color: DARKTEXT, fontSize: 10.5 } },
    ], { x: x + 0.15, y: 5.08, w: 2.62, h: 1.6, fontFace: "Calibri", margin: 0 });
  });
  footer(s, "Source: data/fig_tasks.csv (generated from frozen signal_suite_v1, data seed 1000)");
}

// ---------------------------------------------------------------- S5 ----
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  titleBar(s, "Two tracks separate architecture quality from angle luck",
           "Methodology");
  const track = (x, color, title, lines) => {
    s.addShape(pres.ShapeType.roundRect, {
      x, y: 1.7, w: 5.95, h: 3.4, rectRadius: 0.12,
      fill: { color, transparency: 88 }, line: { color, width: 1.5 },
    });
    s.addText(title, {
      x: x + 0.25, y: 1.85, w: 5.45, h: 0.45, bold: true, fontSize: 16.5,
      color, fontFace: "Cambria", margin: 0 });
    s.addText(lines.map((t, i) => ({
      text: t, options: { fontSize: 12, color: DARKTEXT, bullet: true,
                          breakLine: i < lines.length - 1, paraSpaceAfter: 6 },
    })), { x: x + 0.35, y: 2.4, w: 5.35, h: 2.6, fontFace: "Calibri", margin: 0,
           valign: "top" });
  };
  track(0.65, NAVY, "Track A — architecture search (primary)", [
    "Arms propose structure only, in one shared layered grammar",
    "One shared AdamW inner loop trains θ (40 epochs, frozen for every arm)",
    "θ-seed = f(task, data seed, search seed, structural hash) — same circuit ⇒ same result in every arm (global cache)",
    "Feedback to searchers: validation metric only",
  ]);
  track(6.75, "7A4E9E", "Track B — direct joint proposal (ablation)", [
    "Candidates carry gates, wires, AND numeric angles",
    "θ evaluated verbatim — structurally NO optimizer on this path (AST-enforced)",
    "Candidate identity includes θ: same structure + new angles = new candidate",
    "Mandatory classical rival: mixed discrete/continuous evolutionary search",
  ]);
  s.addShape(pres.ShapeType.roundRect, {
    x: 0.65, y: 5.35, w: 12.05, h: 1.15, rectRadius: 0.1,
    fill: { color: "FFF6E8" }, line: { color: AMBER, width: 1 },
  });
  s.addText([
    { text: "Why it matters:  ", options: { bold: true, color: DARKTEXT, fontSize: 13 } },
    { text: "a joint proposer can look good by guessing lucky angles rather than good architectures. " +
            "Only the split design attributes credit correctly — E5 additionally re-optimizes θ on frozen " +
            "LLM architectures at exactly matched objective-call budgets.",
      options: { color: DARKTEXT, fontSize: 12.5 } },
  ], { x: 0.95, y: 5.52, w: 11.5, h: 0.85, fontFace: "Calibri", margin: 0 });
  footer(s, "Protocol §5 (frozen): docs/research/BENCHMARK_V2_PROTOCOL.md");
}

// ---------------------------------------------------------------- S6 ----
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  titleBar(s, "Evaluation hygiene is enforced by construction, then attacked",
           "Reproducibility");
  const items = [
    ["Validation-only feedback", "SearchFeedback has no field that could hold a test value; arms never see test data."],
    ["Protected test, once", "“Protected-test RMSE” = test metric of the validation-selected candidate, evaluated once after selection freeze — never feedback."],
    ["Paired isolation", "data seed 1000+r, search seed 2000+r shared across arms; θ-seeds derived, never global."],
    ["Crash-resumable stores", "Per-cell SQLite + budget ledger; interrupt-and-resume proven equal to an uninterrupted run."],
    ["Machine-checked completion", "17-criterion goal contract; checker currently fails honestly (matrix incomplete) and rejects removed or interrupted cells."],
    ["Break-tests", "Deliberately broke 3 safeguards: 2 exposed real blind spots (now closed with stronger tests). No secrets or raw transcripts in artifacts."],
  ];
  items.forEach(([head, body], i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = 0.65 + col * 6.2, y = 1.75 + row * 1.62;
    s.addShape(pres.ShapeType.ellipse, {
      x, y: y + 0.05, w: 0.4, h: 0.4,
      fill: { color: i === 5 ? ACCENT : NAVY },
    });
    s.addText("✓", { x, y: y + 0.05, w: 0.4, h: 0.4, align: "center",
      color: WHITE, bold: true, fontSize: 13, margin: 0 });
    s.addText([
      { text: head, options: { bold: true, fontSize: 13.5, color: DARKTEXT, breakLine: true } },
      { text: body, options: { fontSize: 11, color: MUTED } },
    ], { x: x + 0.55, y: y - 0.05, w: 5.5, h: 1.5, fontFace: "Calibri", margin: 0 });
  });
  footer(s, "docs/research/BREAK_TESTS.md · GOAL_CONTRACT.yaml · scripts/check_goal_completion.py");
}

// ---------------------------------------------------------------- S7 ----
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  titleBar(s, "Completed: E0 provenance replay and E1 classical joint search",
           "Results — complete");
  s.addShape(pres.ShapeType.roundRect, {
    x: 0.6, y: 1.62, w: 3.6, h: 4.7, rectRadius: 0.1,
    fill: { color: ICE, transparency: 62 }, line: { color: NAVY, width: 1 },
  });
  s.addText([
    { text: "E0 — pilot replay", options: { bold: true, fontSize: 15, color: NAVY, breakLine: true, paraSpaceAfter: 8 } },
    { text: "All 21 committed real-LLM pilot candidates re-evaluated offline from stored operations + θ.", options: { fontSize: 11.5, color: DARKTEXT, breakLine: true, paraSpaceAfter: 8 } },
    { text: "Exact match", options: { bold: true, fontSize: 20, color: GREEN, breakLine: true } },
    { text: "validation AND test values, all 3 arms, both seeds", options: { fontSize: 11, color: MUTED, breakLine: true, paraSpaceAfter: 8 } },
    { text: "0 new API calls", options: { bold: true, fontSize: 20, color: GREEN, breakLine: true } },
    { text: "The July numbers are provenance-verified — and remain a 2-seed integration demo, not evidence.", options: { fontSize: 11, color: MUTED } },
  ], { x: 0.85, y: 1.8, w: 3.1, h: 4.4, fontFace: "Calibri", margin: 0 });
  s.addImage({ path: FIG("fig_e1_per_seed.png"), x: 4.45, y: 1.75, w: 8.35, h: 3.5 });
  s.addText([
    { text: "E1 (B=16, verbatim θ, 10 paired replicates): ", options: { bold: true, fontSize: 12.5, color: DARKTEXT } },
    { text: "random_joint and evolutionary_joint are statistically close — no Holm-corrected " +
            "difference (all p ≥ 0.28). T2 at n=5 appears difficult for both: test RMSE ≈ 0.29 " +
            "for every replicate, near the target-scale spread — verbatim-θ proposals rarely " +
            "beat it at this budget. LLM joint arms: pending budget authorization.",
      options: { fontSize: 12, color: MUTED } },
  ], { x: 4.55, y: 5.35, w: 8.1, h: 1.35, fontFace: "Calibri", margin: 0 });
  footer(s, "Sources: data/e0_replication_summary.csv · data/fig_e1_per_seed.csv · data/E1_stats_paired_tests.csv");
}

// ---------------------------------------------------------------- S8 ----
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  titleBar(s, "E2 classical matrix complete; E3 scaling is running now",
           "Results — main matrix");
  s.addImage({ path: FIG("fig_e2_per_seed.png"), x: 0.5, y: 1.55, w: 9.3, h: 2.45 });
  s.addImage({ path: FIG("fig_e2_anytime.png"), x: 0.5, y: 4.15, w: 9.3, h: 2.1 });

  const statusRows = [
    [{ text: "cells", options: { fill: { color: NAVY }, color: WHITE, bold: true, fontSize: 9.5 } },
     { text: "done", options: { fill: { color: NAVY }, color: WHITE, bold: true, fontSize: 9.5 } },
     { text: "state", options: { fill: { color: NAVY }, color: WHITE, bold: true, fontSize: 9.5 } }],
    ["E1 classical (60)", "60", { text: "complete", options: { color: GREEN, bold: true } }],
    ["E2 classical (280)", "280", { text: "complete", options: { color: GREEN, bold: true } }],
    ["E3 classical (120)", "running", { text: "running", options: { color: AMBER, bold: true } }],
    ["LLM cells (220)", "0", { text: "blocked*", options: { color: ACCENT, bold: true } }],
    ["E4 / E5", "—", { text: "pending", options: { color: MUTED, bold: true } }],
  ];
  s.addTable(statusRows, {
    x: 10.0, y: 1.6, w: 2.85, colW: [1.45, 0.6, 0.8], fontSize: 9.5,
    fontFace: "Calibri", border: { type: "solid", color: "D8DDE8", pt: 0.5 },
    rowH: 0.34, valign: "middle", margin: 0.04,
  });
  s.addText([
    { text: "Findings (10 paired reps, Holm-corrected):", options: { bold: true, fontSize: 10.5, color: DARKTEXT, breakLine: true, paraSpaceAfter: 4 } },
    { text: "Searched circuits beat shallow references on the regression tasks (T1: random vs StrongEnt-d2, p_Holm = 0.032).", options: { fontSize: 9.5, color: MUTED, bullet: true, breakLine: true, paraSpaceAfter: 4 } },
    { text: "The three classical strategies are statistically indistinguishable from each other on every task.", options: { fontSize: 9.5, color: MUTED, bullet: true, breakLine: true, paraSpaceAfter: 4 } },
    { text: "T4 (AUROC ≈ 0.73–0.75) is flat across arms.", options: { fontSize: 9.5, color: MUTED, bullet: true, breakLine: true, paraSpaceAfter: 4 } },
    { text: "This is the bar any LLM arm must beat.", options: { bold: true, fontSize: 9.5, color: NAVY, bullet: true } },
  ], { x: 10.0, y: 3.75, w: 2.9, h: 2.9, fontFace: "Calibri", margin: 0 });
  footer(s, "*LLM cells await LLM_API_BUDGET_USD (slide 10). Sources: data/fig_e2_per_seed.csv · data/fig_e2_anytime.csv · data/matrix_status.csv");
}

// ---------------------------------------------------------------- S9 ----
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  titleBar(s, "Selected circuits: resources measured, diagnostics descriptive",
           "Diagnostics");
  s.addImage({ path: FIG("fig_e2_resources.png"), x: 0.5, y: 1.65, w: 8.1, h: 3.05 });
  s.addImage({ path: FIG("fig_e2_diagnostics.png"), x: 8.75, y: 1.65, w: 4.15, h: 3.05 });
  s.addShape(pres.ShapeType.roundRect, {
    x: 0.6, y: 5.0, w: 12.1, h: 1.55, rectRadius: 0.1,
    fill: { color: ICE, transparency: 62 }, line: { color: "D8DDE8", width: 0.75 },
  });
  s.addText([
    { text: "Accuracy is bought with hardware cost: ", options: { bold: true, fontSize: 12.5, color: DARKTEXT } },
    { text: "searched circuits use ~3–10× the transpiled 2-qubit gates of the fixed references (line coupling, fixed seed) — the Pareto view is part of the final analysis. ", options: { fontSize: 12, color: MUTED } },
    { text: "Diagnostics characterize circuits; they are not assumed to predict task performance", options: { bold: true, fontSize: 12, color: NAVY } },
    { text: " (Sim et al. framing; RQ5 tests the association empirically, with circuit-size controls). Expressibility KL + Meyer–Wallach Q computed for the 28 best-validation selected circuits. Shot-noise and noisy-simulator robustness (E4): pending — runs after the full matrix.", options: { fontSize: 12, color: MUTED } },
  ], { x: 0.9, y: 5.18, w: 11.5, h: 1.25, fontFace: "Calibri", margin: 0 });
  footer(s, "Sources: data/fig_e2_resources.csv · data/fig_e2_diagnostics.csv (KL/Q per circuit)");
}

// --------------------------------------------------------------- S10 ----
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addText("Next steps", {
    x: 0.9, y: 0.55, w: 11.5, h: 0.7, fontSize: 32, bold: true, color: WHITE,
    fontFace: "Cambria", margin: 0 });
  const steps = [
    "Finish + verify E3 scaling (n ∈ {3,4,6,8}; running, resumable, ~120 classical cells)",
    "Run the real-LLM matrix (220 cells, both tracks) once budget is authorized",
    "Complete dependent analyses: E5 θ-isolation, E4 shot/noise robustness",
    "Regenerate final paper-style figures + statistical report from stores (goal checker must exit 0)",
    "Next mentor update: full budget-matched LLM-vs-classical comparison",
  ];
  s.addText(steps.map((t, i) => ({
    text: t, options: { fontSize: 15, color: ICE, bullet: true,
                        breakLine: i < steps.length - 1, paraSpaceAfter: 10 },
  })), { x: 1.0, y: 1.5, w: 11.3, h: 2.6, fontFace: "Calibri", margin: 0 });
  s.addShape(pres.ShapeType.roundRect, {
    x: 0.9, y: 4.35, w: 11.5, h: 2.3, rectRadius: 0.12,
    fill: { color: WHITE }, line: { color: ACCENT, width: 1.5 },
  });
  s.addText([
    { text: "Decision needed — API budget authorization", options: { bold: true, fontSize: 16, color: ACCENT, breakLine: true, paraSpaceAfter: 6 } },
    { text: "LLM_API_BUDGET_USD is not set, so all real-LLM cells are blocked by policy: an API key alone is not spending authorization, and a Claude Max subscription does not authorize experiment API spending.", options: { fontSize: 12.5, color: DARKTEXT, breakLine: true, paraSpaceAfter: 6 } },
    { text: "Suggested cap 120 (conservative $0.05/call reservation, hard 40-call/cell cap); expected actual cost ≈ $5–20 on a mini-class model. One-line resume command in docs/research/BLOCKED.md.", options: { fontSize: 12.5, color: DARKTEXT } },
  ], { x: 1.2, y: 4.55, w: 10.9, h: 1.95, fontFace: "Calibri", margin: 0 });
  s.addText("All slide numbers trace to durable stores · sources in docs/presentation/bench_v2_weekly/data/", {
    x: 0.9, y: 6.95, w: 11.5, h: 0.3, fontSize: 9, color: ICE, margin: 0 });
}

pres.writeFile({ fileName: path.join(HERE, "LLM_VQC_weekly_benchmark_v2.pptx") })
  .then(() => console.log("deck written"));
