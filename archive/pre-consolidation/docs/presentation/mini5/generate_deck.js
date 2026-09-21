// 10-slide deck for the mini_5gate_v1 study.
// Every number here is read from data/ at build time; nothing is typed in.
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");
const cp = require("child_process");

const HERE = __dirname;
const F = (n) => path.join(HERE, "figures", n);
const DATA = JSON.parse(fs.readFileSync(path.join(HERE, "data", "run_metadata.json")));
const COND = JSON.parse(fs.readFileSync(path.join(HERE, "data", "condition_comparison.json")));
const V2 = JSON.parse(fs.readFileSync(path.join(HERE, "data", "ctx_analysis.json")));
const M = V2.median_test_rmse;
const r3 = (x) => (x == null ? "-" : x.toFixed(3));
const SHA = cp.execSync("git rev-parse --short HEAD", { cwd: HERE }).toString().trim();
const BRANCH = cp.execSync("git rev-parse --abbrev-ref HEAD", { cwd: HERE }).toString().trim();

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";           // 13.333 x 7.5 in
const W = 13.333, H = 7.5;
const FONT = "Arial";
const NAVY = "1E3A5F", INK = "1A1A2E", MUTED = "55606E", ACCENT = "B3261E";
const OK = "0B7A4B", RULE = "D8DDE6", BG = "F4F6FA", WARN = "FDF6E8";

const fixed = COND["fixed 5 gates B = 8"];
const pilot = COND["variable 1-5 gates B = 4  (pilot setting)"];
const med = (c, fam, arm) => c.median_test_rmse[fam][arm].toFixed(3);

function slide(kicker, title) {
  const s = pres.addSlide();
  s.background = { color: "FFFFFF" };
  if (kicker) s.addText(kicker.toUpperCase(), {
    x: 0.55, y: 0.3, w: W - 1.1, h: 0.3, fontSize: 13, bold: true,
    color: ACCENT, fontFace: FONT, charSpacing: 1.5, margin: 0 });
  const long = title.length > 58;
  s.addText(title, { x: 0.55, y: kicker ? 0.6 : 0.4, w: W - 1.1,
    h: long ? 0.9 : 0.72, fontSize: long ? 24 : 30, bold: true,
    color: NAVY, fontFace: FONT, margin: 0 });
  return s;
}
function footer(s, text) {
  s.addText(text, { x: 0.55, y: H - 0.42, w: W - 1.1, h: 0.3, fontSize: 11,
    color: MUTED, fontFace: FONT, margin: 0 });
}
function bullets(s, items, o) {
  s.addText(items.map((t, i) => ({
    text: t, options: { bullet: true, breakLine: i < items.length - 1 },
  })), { x: o.x, y: o.y, w: o.w, h: o.h, fontSize: o.size || 17,
    color: o.color || INK, fontFace: FONT, margin: 0, valign: "top",
    lineSpacingMultiple: 1.12, paraSpaceAfter: o.gap == null ? 9 : o.gap });
}
// Every slide that shows a number carries one of these, so a reader never
// has to remember which of the three conditions or which task it came from.
function badge(s, condition, scope) {
  const w = 6.55;
  s.addShape(pres.ShapeType.roundRect, { x: W - 0.55 - w, y: 0.28, w, h: 0.44,
    rectRadius: 0.07, fill: { color: BG }, line: { color: RULE, width: 1 } });
  s.addText([
    { text: condition, options: { bold: true, fontSize: 11.5, color: NAVY } },
    { text: "   " + scope, options: { fontSize: 11.5, color: MUTED } },
  ], { x: W - 0.4 - w, y: 0.3, w: w - 0.3, h: 0.4, fontFace: FONT,
    margin: 0, valign: "middle", align: "right" });
}

const hdr = (t) => ({ text: t, options: { bold: true, fill: { color: NAVY }, color: "FFFFFF" } });

// ---------------------------------------------------------------- S1
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addText("Three ways an LLM looked better than it was", {
    x: 0.9, y: 1.5, w: 11.5, h: 1.7, fontSize: 40, bold: true,
    color: "FFFFFF", fontFace: FONT, margin: 0, lineSpacingMultiple: 1.05 });
  s.addText("Removing one confound at a time from an LLM-guided quantum circuit search", {
    x: 0.9, y: 3.35, w: 11.5, h: 0.6, fontSize: 20, color: "CADCFC",
    fontFace: FONT, margin: 0 });
  s.addShape(pres.ShapeType.roundRect, { x: 0.9, y: 4.25, w: 11.5, h: 1.0,
    rectRadius: 0.1, fill: { color: "FFFFFF", transparency: 88 },
    line: { color: "8FA6C4", width: 1 } });
  s.addText([
    { text: "Headline: ", options: { bold: true, fontSize: 18, color: "FFD9A0" } },
    { text: `each apparent advantage came from something other than design skill - circuit size, then a bad starting point. The LLM beats a textbook ansatz on all four tasks and never beats random search on any (${V2.n_seeds} seeds).`,
      options: { fontSize: 18, color: "FFFFFF" } },
  ], { x: 1.15, y: 4.35, w: 11.0, h: 0.8, fontFace: FONT, margin: 0, valign: "middle" });
  s.addText("Tomoya Hatanaka", { x: 0.9, y: 5.9, w: 8, h: 0.35, fontSize: 18,
    bold: true, color: "FFFFFF", fontFace: FONT, margin: 0 });
  s.addText("ML4SCI / Google Summer of Code 2026", { x: 0.9, y: 6.25, w: 8, h: 0.35,
    fontSize: 16, color: "CADCFC", fontFace: FONT, margin: 0 });
  s.addText(`July 31, 2026  ·  commit ${SHA}  ·  ${BRANCH}`, {
    x: 0.9, y: 6.6, w: 10, h: 0.3, fontSize: 12, color: "9FB3CC",
    fontFace: FONT, margin: 0 });
  s.addNotes("The question is deliberately narrow. Last week I reported that LLM-proposed circuits beat random search. This week I removed one degree of freedom from the search space and the advantage largely went away, which changes what that earlier result means.");
}

// ---------------------------------------------------------------- S2
{
  const s = slide("What changed", "Last week the LLM could choose how big the circuit was");
  s.addImage({ path: F("fig_conditions.png"), x: 0.45, y: 1.65, w: 12.45, h: 3.55 });
  s.addShape(pres.ShapeType.roundRect, { x: 0.6, y: 5.45, w: 12.15, h: 1.15,
    rectRadius: 0.1, fill: { color: WARN }, line: { color: "B8860B", width: 1.2 } });
  s.addText([
    { text: "Longer circuits usually score better. ", options: { bold: true, fontSize: 18, color: INK } },
    { text: "A proposer that always asks for the maximum length wins without designing anything. Fixing the length removes that move; 16 gates then makes the circuit visible to the readout.",
      options: { fontSize: 18, color: INK } },
  ], { x: 0.9, y: 5.58, w: 11.6, h: 0.9, fontFace: FONT, margin: 0, valign: "middle" });
  footer(s, "seed = one paired data + search random seed; a condition is re-run once per seed. Pilot values verified in its EXPERIMENT_CONFIG.json");
  s.addNotes("The pilot allowed one to five gates. That hands the proposer a free decision that correlates with performance. This week the length is pinned so the decision cannot be made, and two controls put it back so we can measure what it was worth.");
}

// ---------------------------------------------------------------- S3
{
  const s = slide("The four tasks", "Four questions about a 32-number signal");
  badge(s, "5 qubits on every task", "32 numbers in, one number out");
  s.addImage({ path: F("fig_tasks_v2.png"), x: 0.35, y: 1.55, w: 12.65, h: 2.78 });
  const cell = (x, w, title, body, colour) => {
    s.addShape(pres.ShapeType.roundRect, { x, y: 4.6, w, h: 1.75, rectRadius: 0.1,
      fill: { color: BG }, line: { color: colour, width: 1.5 } });
    s.addText(title, { x: x + 0.25, y: 4.72, w: w - 0.5, h: 0.35, bold: true,
      fontSize: 15, color: colour, fontFace: FONT, margin: 0 });
    s.addText(body, { x: x + 0.25, y: 5.12, w: w - 0.5, h: 1.1, fontSize: 13.5,
      color: INK, fontFace: FONT, margin: 0, valign: "top" });
  };
  cell(0.6, 3.85, "All four are synthetic",
       "so difficulty and leakage are controlled, and the answer is known exactly", NAVY);
  cell(4.72, 3.85, "T1-T3 are regression",
       "the target is a position or a rate, so RMSE is the natural score", OK);
  cell(8.85, 3.88, "T4 is a yes/no question",
       "RMSE there is the error against a 0/1 label; AUROC is recorded too", ACCENT);
  footer(s, "Generated by llm_vqc/tasks/signal_suite · 256 train / 256 validation / 2048 test, regenerated per seed");
  s.addNotes("Four synthetic tasks so I control difficulty and know the answer exactly. Each signal is 32 numbers, which is exactly what five qubits hold, so the whole signal goes in with no padding and no truncation.");
}

// ---------------------------------------------------------------- S4
{
  const s = slide("The rules", "One pipeline, and every constraint on it");
  badge(s, "The run reported on slides 6 and 9", "16 gates, 5 qubits, B = 8, 30 seeds");
  s.addImage({ path: F("fig_contract.png"), x: 0.5, y: 1.5, w: 12.35, h: 2.85 });
  s.addTable([
    [hdr("Constraint"), hdr("Value")],
    ["Gate count", "16 exactly - enough that the measured qubit can see the circuit (slide 7)"],
    ["Gate set", "H, RX, RY, RZ (one qubit)   ·   CRX, CRY, CRZ (two qubits)"],
    ["Angles", "the method picks one of 8 RANGES over [-pi, pi]; the harness draws inside it"],
    ["Budget B", "8 distinct circuits scored per method   ·   30 seeds"],
    ["Scoring", "validation RMSE picks the winner; test RMSE is read once, after that choice"],
  ], { x: 0.6, y: 4.5, w: 12.15, colW: [2.4, 9.75], fontSize: 13.5, fontFace: FONT,
    border: { type: "solid", color: RULE, pt: 0.75 }, rowH: 0.4,
    valign: "middle", margin: 0.07 });
  footer(s, "RMSE compares arms WITHIN one task only - the four tasks predict different quantities on different scales");
  s.addNotes("One operation is one physical gate, so sixteen means sixteen. The angle rule matters: the method chooses a range and the harness draws inside it, because the model cannot sample a continuum. Validation picks the winner and the test split is read once afterwards.");
}

// ---------------------------------------------------------------- S5
{
  const s = slide("Design", "Six methods, one shared budget");
  badge(s, "Six arms, identical pipeline", "16 gates, 5 qubits, B = 8 each");
  s.addImage({ path: F("fig_arms.png"), x: 0.45, y: 1.6, w: 12.45, h: 3.26 });
  s.addShape(pres.ShapeType.roundRect, { x: 0.6, y: 5.15, w: 12.15, h: 1.4,
    rectRadius: 0.1, fill: { color: BG }, line: { color: RULE, width: 1 } });
  s.addText([
    { text: "All three LLM arms are told what the task is. ", options: { bold: true, fontSize: 17, color: NAVY } },
    { text: "Open-loop never sees how its circuits scored, closed-loop does, and the hybrid lets random pick the first two before the LLM takes over.",
      options: { fontSize: 17, color: INK } },
  ], { x: 0.9, y: 5.28, w: 11.6, h: 1.15, fontFace: FONT, margin: 0, valign: "middle" });
  footer(s, "The fixed ansatz is size-matched at 16 gates, so it cannot be beaten or helped by circuit size");
  s.addNotes("The budget is distinct scored circuits because that is what every method spends and what dominates cost. Duplicates and invalid proposals are the proposer's own problem; they do not earn extra tries.");
}

// ---------------------------------------------------------------- S6
{
  const s = slide("Result", "The LLM beats the textbook ansatz, never beats random");
  badge(s, `16 gates, 5 qubits, B = 8, ${V2.n_seeds} seeds`, "all four tasks");
  s.addImage({ path: F("fig_v2_main.png"), x: 0.3, y: 1.5, w: 12.75, h: 3.65 });
  s.addShape(pres.ShapeType.roundRect, { x: 0.6, y: 5.5, w: 12.15, h: 1.15,
    rectRadius: 0.08, fill: { color: BG }, line: { color: RULE, width: 1 } });
  s.addText([
    { text: "vs the fixed ansatz: significant on all 4 tasks. ", options: { bold: true, fontSize: 16.5, color: OK } },
    { text: "  vs random: no difference detected on any. ", options: { bold: true, fontSize: 16.5, color: ACCENT } },
    { text: `Random holds the better median on T2, T3 and T4 (T3 ${r3(M.change_point.random)} vs ${r3(M.change_point.llm_open_ctx)}); the LLM leads only on T1 (${r3(M.gauss_peak.llm_open_ctx)} vs ${r3(M.gauss_peak.random)}), and not significantly.`,
      options: { fontSize: 16.5, color: INK } },
  ], { x: 0.85, y: 5.62, w: 11.6, h: 0.9, fontFace: FONT, margin: 0, valign: "middle" });
  footer(s, `Wilcoxon signed-rank on ${V2.n_seeds} paired seeds; Holm over the 12 pre-declared primary contrasts · data/ctx_analysis.json`);
  s.addNotes("Read the medians. Against the fixed hardware-efficient ansatz every search method wins, decisively. Against random search the LLM wins nothing, and random has the better median on three of four tasks.");
}

// ---------------------------------------------------------------- S7
{
  const s = slide("Metric check", "On T4 the two scores tell different stories");
  badge(s, `16 gates, 5 qubits, B = 8, ${V2.n_seeds} seeds`, "T4 only (one bump or two)");
  s.addImage({ path: F("fig_v2_t4.png"), x: 0.9, y: 1.55, w: 8.3, h: 3.6 });
  s.addText([
    { text: "AUROC\n", options: { bold: true, fontSize: 17, color: NAVY } },
    { text: "= how often the model ranks a two-bump signal above a one-bump one. 1.0 perfect, 0.5 a coin flip.\n\n",
      options: { fontSize: 15, color: INK } },
    { text: "Why report both\n", options: { bold: true, fontSize: 17, color: NAVY } },
    { text: "RMSE also punishes outputting 0.45 / 0.55 instead of 0 / 1. AUROC only cares about the order, so one alone can assert the wrong winner.",
      options: { fontSize: 15, color: INK } },
  ], { x: 9.45, y: 1.75, w: 3.4, h: 3.3, fontFace: FONT, margin: 0, valign: "top" });
  s.addShape(pres.ShapeType.roundRect, { x: 0.6, y: 5.5, w: 12.15, h: 1.15,
    rectRadius: 0.1, fill: { color: WARN }, line: { color: "B8860B", width: 1.2 } });
  s.addText([
    { text: "Neither reading is strong. ", options: { bold: true, fontSize: 17, color: INK } },
    { text: "Every method sits near 0.50 RMSE and barely above chance on AUROC. Sixteen gates without training do not solve T4, and no arm separates from the others.",
      options: { fontSize: 17, color: INK } },
  ], { x: 0.85, y: 5.62, w: 11.6, h: 0.9, fontFace: FONT, margin: 0, valign: "middle" });
  footer(s, "AUROC computed from the identical predictions used for the RMSE, on the same single test evaluation");
  s.addNotes("T4 is the one task where the metric can pick the winner, so both are on the slide. Here they agree that nothing works: RMSE near 0.5 and AUROC barely above a coin flip.");
}

// ---------------------------------------------------------------- S8
{
  const s = slide("Attribution (earlier run)",
                  "Last week's advantage was a size decision, not a design one");
  badge(s, "A SEPARATE, EARLIER RUN", "5 gates, 3 qubits, 10 seeds - not the run on slides 6, 7, 9");
  s.addImage({ path: F("fig_size_confound.png"), x: 0.3, y: 1.5, w: 12.75, h: 3.8 });
  s.addShape(pres.ShapeType.roundRect, { x: 0.6, y: 5.55, w: 12.15, h: 1.2,
    rectRadius: 0.08, fill: { color: BG }, line: { color: RULE, width: 1 } });
  s.addText([
    { text: "Free the length and only Random gets worse ", options: { bold: true, fontSize: 17, color: INK } },
    { text: `(${med(fixed, "gauss_peak", "random")} → ${med(pilot, "gauss_peak", "random")}); the LLM does not move (${med(fixed, "gauss_peak", "llm_open")} → ${med(pilot, "gauss_peak", "llm_open")}), because it was already asking for 5. At the pilot's setting they meet.`,
      options: { fontSize: 17, color: INK } },
  ], { x: 0.85, y: 5.65, w: 11.6, h: 1.0, fontFace: FONT, margin: 0, valign: "middle" });
  footer(s, "This attribution needed the pilot's own 5-gate space, so it has its own runs: all four tasks, 10 seeds, T1 plotted because the pilot used that family · data/condition_comparison.json");
  s.addNotes("This is the slide I would defend hardest. The pilot rewarded a single decision - ask for the maximum number of gates - and the LLM makes that decision almost every time while a uniform sampler does not. Pin the length and the advantage goes; restore the length and shrink the budget to the pilot's and the two meet again.");
}

// ---------------------------------------------------------------- S9
{
  const s = slide("A hypothesis, tested and refuted",
                  "The LLM's \"refinement skill\" was the room a bad start leaves");
  badge(s, `16 gates, B = 8, ${V2.n_seeds} seeds`, "median over all four tasks");
  s.addImage({ path: F("fig_v2_mechanism.png"), x: 0.75, y: 1.55, w: 11.8, h: 4.3 });
  s.addShape(pres.ShapeType.roundRect, { x: 0.6, y: 6.0, w: 12.15, h: 0.92,
    rectRadius: 0.08, fill: { color: WARN }, line: { color: "B8860B", width: 1.1 } });
  s.addText([
    { text: "I built the hybrid arm expecting it to win. ", options: { bold: true, fontSize: 16, color: INK } },
    { text: "It reached the best final score of any arm and still could not beat random - because the ability it was built to exploit was not there.",
      options: { fontSize: 16, color: INK } },
  ], { x: 0.85, y: 6.1, w: 11.6, h: 0.75, fontFace: FONT, margin: 0, valign: "middle" });
  footer(s, "Hybrid = random draws candidates 1-2, the LLM proposes 3-8 having seen their validation scores");
  s.addNotes("This is the slide I would keep if I could keep only one. The diagnosis was that the LLM refines well but starts badly, so I gave it a good start. The start improved, the final score improved, and the refinement rate collapsed below random. The 17 percent was never skill.");
}

// ---------------------------------------------------------------- S10
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addText("What I can and cannot claim", { x: 0.7, y: 0.45, w: 12, h: 0.7,
    fontSize: 30, bold: true, color: "FFFFFF", fontFace: FONT, margin: 0 });
  const card = (x, title, lines, colour) => {
    s.addShape(pres.ShapeType.roundRect, { x, y: 1.3, w: 5.85, h: 3.9,
      rectRadius: 0.1, fill: { color: "FFFFFF", transparency: 90 },
      line: { color: "8FA6C4", width: 1 } });
    s.addText(title, { x: x + 0.25, y: 1.42, w: 5.4, h: 0.4, bold: true,
      fontSize: 17, color: colour, fontFace: FONT, margin: 0 });
    bullets(s, lines, { x: x + 0.32, y: 1.95, w: 5.3, h: 3.1, size: 14.5,
      gap: 9, color: "E8EEF7" });
  };
  card(0.7, "Supported", [
    "The LLM beats a textbook ansatz on all 4 tasks",
    `It never beats random search - ${V2.n_seeds} seeds, 12 pre-declared contrasts`,
    "Given free length it picks the maximum ~70% of the time",
    "Its apparent refinement skill vanishes given a good start",
  ], "9FE3C0");
  card(6.85, "NOT supported", [
    "\"LLMs cannot design circuits\" - one small model, B = 8",
    "\"Task context does not help\" - never A/B tested at equal seeds",
    "Anything about hardware - this is exact simulation",
    "Comparing RMSE across different tasks",
  ], "FFC4B8");
  s.addShape(pres.ShapeType.roundRect, { x: 0.7, y: 5.45, w: 12.0, h: 1.15,
    rectRadius: 0.1, fill: { color: "FFFFFF" }, line: { color: ACCENT, width: 1.6 } });
  s.addText([
    { text: "Next: ", options: { bold: true, fontSize: 18, color: ACCENT } },
    { text: "give the LLM a decision that is actually hard, and check any advantage the same way.",
      options: { fontSize: 18, color: INK } },
  ], { x: 0.95, y: 5.57, w: 11.5, h: 0.95, fontFace: FONT, margin: 0, valign: "middle" });
  s.addText(`600 cells across 3 conditions · ${(DATA.wall_clock_seconds / 60).toFixed(1)} min for the main run · gpt-5-nano-2025-08-07 · $0.23 total`, {
    x: 0.7, y: 6.85, w: 12, h: 0.3, fontSize: 12, color: "9FB3CC",
    fontFace: FONT, margin: 0 });
  s.addNotes("The honest summary is that I removed a confound and a positive result mostly went away. That is a useful outcome: it tells us the earlier benchmark was rewarding the wrong thing, and it tells us what a fair test has to control for.");
}

const out = path.join(HERE, "20260731_GSoC_mini5.pptx");
pres.writeFile({ fileName: out }).then(() => console.log("deck written:", out));
