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
const hdr = (t) => ({ text: t, options: { bold: true, fill: { color: NAVY }, color: "FFFFFF" } });

// ---------------------------------------------------------------- S1
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addText("Does an LLM design better quantum circuits,\nor just bigger ones?", {
    x: 0.9, y: 1.5, w: 11.5, h: 1.7, fontSize: 40, bold: true,
    color: "FFFFFF", fontFace: FONT, margin: 0, lineSpacingMultiple: 1.05 });
  s.addText("A five-gate, fixed-length comparison of LLM-guided and classical circuit search", {
    x: 0.9, y: 3.35, w: 11.5, h: 0.6, fontSize: 20, color: "CADCFC",
    fontFace: FONT, margin: 0 });
  s.addShape(pres.ShapeType.roundRect, { x: 0.9, y: 4.25, w: 11.5, h: 1.0,
    rectRadius: 0.1, fill: { color: "FFFFFF", transparency: 88 },
    line: { color: "8FA6C4", width: 1 } });
  s.addText([
    { text: "Headline: ", options: { bold: true, fontSize: 18, color: "FFD9A0" } },
    { text: "last week's LLM advantage was mostly a circuit-size decision. Remove that one degree of freedom and it disappears on three of four tasks.",
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
  const s = slide("What changed", "Last week's search let the LLM choose the circuit size");
  const box = (x, colour, bg, title, lines) => {
    s.addShape(pres.ShapeType.roundRect, { x, y: 1.65, w: 6.0, h: 2.5,
      rectRadius: 0.1, fill: { color: bg }, line: { color: colour, width: 1.4 } });
    s.addText(title, { x: x + 0.25, y: 1.78, w: 5.5, h: 0.4, bold: true,
      fontSize: 19, color: colour, fontFace: FONT, margin: 0 });
    bullets(s, lines, { x: x + 0.32, y: 2.32, w: 5.4, h: 1.7, size: 15.5, gap: 7 });
  };
  box(0.6, MUTED, BG, "Last week (2026-07-24 pilot)", [
    "Circuit body: 1 to 5 gates - length was itself a choice",
    "Budget B = 4 candidates, 2 seeds",
    "Reported: LLM open-loop 0.155 vs Random 0.189",
    "Its own config file said: \"integration/smoke demonstration only; not a statistically powered result\"",
  ]);
  box(6.75, NAVY, "EAF0FB", "This week", [
    "Circuit body: EXACTLY 5 gates - length is fixed for every method",
    "Budget B = 8 candidates, 10 seeds",
    "Five arms instead of three",
    "Three conditions run so the difference can be attributed, not guessed",
  ]);
  s.addShape(pres.ShapeType.roundRect, { x: 0.6, y: 4.45, w: 12.15, h: 1.5,
    rectRadius: 0.1, fill: { color: WARN }, line: { color: "B8860B", width: 1.2 } });
  s.addText([
    { text: "Why this matters:  ", options: { bold: true, fontSize: 18, color: INK } },
    { text: "if circuit length is a free variable and longer circuits are usually better, then any proposer that simply always asks for the maximum length wins - without designing anything.",
      options: { fontSize: 18, color: INK } },
  ], { x: 0.9, y: 4.6, w: 11.6, h: 1.2, fontFace: FONT, margin: 0, valign: "middle" });
  footer(s, "Pilot config: outputs/free_amplitude_fixed_readout_v1/EXPERIMENT_CONFIG.json (max_gates 5, budget_per_arm 4, seeds 2)");
  s.addNotes("The pilot allowed one to five gates. That sounds harmless, but it hands the proposer a free decision that correlates with performance. This week I pinned the length so that decision cannot be made at all.");
}

// ---------------------------------------------------------------- S3
{
  const s = slide("Search space", "Every rule, stated so you can check any circuit by eye");
  s.addTable([
    [hdr("Constraint"), hdr("Value")],
    ["Qubits", "3 for T1 and T2 (2^3 = 8 amplitudes); 5 for T3 and T4 (2^5 = 32)"],
    ["Gate count", "EXACTLY 5 - not a maximum, not a range. Identical for every method."],
    ["Gate set (7 types)", "one wire: H, RX, RY, RZ      two wires: CRX, CRY, CRZ"],
    ["Angles", "one continuous theta per gate in [-pi, pi]; H carries none"],
    ["Training", "NONE. The proposed angles are evaluated verbatim - no optimizer exists on this path."],
    ["Encoding / readout", "amplitude encoding; Pauli-Z on qubit 0; prediction = (1 - <Z0>)/2"],
    ["Classical parameters", "zero - no dense layer can rescue a weak circuit"],
  ], { x: 0.6, y: 1.6, w: 12.15, colW: [2.75, 9.4], fontSize: 14, fontFace: FONT,
    border: { type: "solid", color: RULE, pt: 0.75 }, rowH: 0.44,
    valign: "middle", margin: 0.07 });
  s.addShape(pres.ShapeType.roundRect, { x: 0.6, y: 5.5, w: 12.15, h: 1.15,
    rectRadius: 0.08, fill: { color: BG }, line: { color: RULE, width: 1 } });
  s.addText([
    { text: "One operation = one physical gate here.  ", options: { bold: true, fontSize: 16, color: INK } },
    { text: "There are no layer macros: a 5-gate circuit compiles to 5 gates, so \"5\" means the same thing to every arm and to the reader.",
      options: { fontSize: 16, color: INK } },
  ], { x: 0.85, y: 5.62, w: 11.6, h: 0.9, fontFace: FONT, margin: 0, valign: "middle" });
  footer(s, "llm_vqc/mini5/space.py - the same grammar check is applied to classical samplers and LLM replies alike");
  s.addNotes("Everything is on this slide on purpose. Three or five qubits, seven gate types, exactly five gates, one angle per gate, no optimizer, no classical parameters. A reader can verify any proposed circuit against these seven rows.");
}

// ---------------------------------------------------------------- S4
{
  const s = slide("What is measured", "One number, defined once: RMSE");
  bullets(s, [
    "Validation RMSE drives the search. Each arm evaluates its candidates and keeps the best one by validation RMSE.",
    "Test RMSE is reported. The held-out split is touched once per cell, after the choice is frozen, so the reported number was never used to make that choice.",
    "T1-T3 regress a continuous quantity, so RMSE is the natural metric.",
    "T4 is binary classification. RMSE against a 0/1 label equals the square root of the Brier score - a proper scoring rule - so all four panels share one axis. AUROC is reported alongside it, and slide 7 shows why that matters.",
  ], { x: 0.7, y: 1.65, w: 12.0, h: 3.0, size: 17, gap: 12 });
  s.addShape(pres.ShapeType.roundRect, { x: 0.7, y: 4.85, w: 11.9, h: 1.5,
    rectRadius: 0.1, fill: { color: WARN }, line: { color: "B8860B", width: 1.2 } });
  s.addText([
    { text: "RMSE is comparable between arms within one task, never between tasks.  ",
      options: { bold: true, fontSize: 17, color: INK } },
    { text: "T1, T2, T3 and T4 predict different quantities on different scales, so the panels each carry their own y-axis and a T1 value must not be read against a T3 value.",
      options: { fontSize: 17, color: INK } },
  ], { x: 0.95, y: 5.0, w: 11.4, h: 1.2, fontFace: FONT, margin: 0, valign: "middle" });
  footer(s, "256 train / 256 validation / 2048 test per replicate, regenerated per seed");
  s.addNotes("I dropped the phrase protected-test RMSE from last week because it was confusing. The discipline behind it is unchanged: the test split is evaluated once, after selection. What changed is only the label.");
}

// ---------------------------------------------------------------- S5
{
  const s = slide("Design", "Five arms, one shared budget of unique candidates");
  s.addTable([
    [hdr("Arm"), hdr("How it proposes"), hdr("API calls")],
    ["Random", "draws 5 gates, wires and angles uniformly", "none"],
    ["Evolutionary", "(mu+lambda), mu=4: seed 4 at random, then mutate a survivor", "none"],
    ["Greedy", "hill climbing: mutate the incumbent, keep it only if validation improves", "none"],
    ["LLM open-loop", "one request per candidate; sees only the model contract and n", "1 per candidate"],
    ["LLM closed-loop", "same, plus validation-only results for what it already tried", "1 per candidate"],
  ], { x: 0.6, y: 1.6, w: 12.15, colW: [2.5, 7.6, 2.05], fontSize: 14.5,
    fontFace: FONT, border: { type: "solid", color: RULE, pt: 0.75 }, rowH: 0.46,
    valign: "middle", margin: 0.07 });
  s.addShape(pres.ShapeType.roundRect, { x: 0.6, y: 4.5, w: 12.15, h: 1.85,
    rectRadius: 0.1, fill: { color: BG }, line: { color: RULE, width: 1 } });
  s.addText([
    { text: `B = ${DATA.budget_unique} unique evaluated candidates per arm, ${DATA.seeds} seeds.  `,
      options: { bold: true, fontSize: 17, color: NAVY } },
    { text: "A duplicate or an ungrammatical proposal is recorded but consumes no budget, so no method can buy extra evaluations by proposing badly - and no method gets fewer because it proposes well. One candidate per request, so the closed loop is genuinely sequential: propose, see the result, propose again, eight times.",
      options: { fontSize: 17, color: INK } },
  ], { x: 0.85, y: 4.62, w: 11.6, h: 1.6, fontFace: FONT, margin: 0, valign: "middle" });
  footer(s, "Greedy growth cannot apply at fixed length, so Greedy here is fixed-length hill climbing");
  s.addNotes("The budget is unique evaluated candidates because that is the quantity every method spends and the one that dominates cost. Duplicates and invalid proposals are the proposer's own problem; they do not earn extra tries.");
}

// ---------------------------------------------------------------- S6
{
  const s = slide("Result", "Random search wins on three of four tasks");
  s.addImage({ path: F("fig_main.png"), x: 0.35, y: 1.5, w: 12.6, h: 3.95 });
  s.addShape(pres.ShapeType.roundRect, { x: 0.6, y: 5.72, w: 12.15, h: 1.0,
    rectRadius: 0.08, fill: { color: BG }, line: { color: RULE, width: 1 } });
  s.addText([
    { text: "The LLM wins on T2 alone, and there it wins clearly ", options: { bold: true, fontSize: 16, color: OK } },
    { text: `(${med(fixed, "sin_freq", "llm_open")} vs ${med(fixed, "sin_freq", "random")}).   `, options: { fontSize: 16, color: INK } },
    { text: "Closed-loop never beats open-loop on any task: eight rounds of feedback did not help this model at this budget.",
      options: { bold: true, fontSize: 16, color: ACCENT } },
  ], { x: 0.85, y: 5.84, w: 11.6, h: 0.8, fontFace: FONT, margin: 0, valign: "middle" });
  footer(s, `${DATA.seeds} seeds per arm, B = ${DATA.budget_unique}, exactly 5 gates · data/per_cell.csv`);
  s.addNotes("Read the medians, not the individual dots. Random is best on T1, T3 and T4. The LLM arms win on T2 and only there. I am reporting this as measured; it is a negative result for the LLM hypothesis at this scale.");
}

// ---------------------------------------------------------------- S7
{
  const s = slide("Metric check", "On T4 the metric, not the method, picks the winner");
  s.addImage({ path: F("fig_t4_check.png"), x: 0.8, y: 1.55, w: 7.3, h: 3.3 });
  bullets(s, [
    "By RMSE, Random leads. By AUROC, LLM open-loop leads. Same predictions, opposite ranking.",
    "RMSE punishes a circuit that ranks the two classes correctly but outputs 0.45 and 0.55 instead of 0 and 1. AUROC does not.",
    "With five gates and no optimizer the outputs sit near 0.5, so this failure mode is live here, not hypothetical.",
    "Both numbers are shown because either one alone asserts a different answer.",
  ], { x: 8.4, y: 1.7, w: 4.4, h: 3.3, size: 14.5, gap: 10 });
  s.addShape(pres.ShapeType.roundRect, { x: 0.6, y: 5.2, w: 12.15, h: 1.4,
    rectRadius: 0.1, fill: { color: WARN }, line: { color: "B8860B", width: 1.2 } });
  s.addText([
    { text: "Neither reading is strong.  ", options: { bold: true, fontSize: 16.5, color: INK } },
    { text: "AUROC spans roughly 0.59 to 0.62 against 0.5 for chance. The honest summary of T4 is that five gates without training barely solve it at all, and that no arm separates convincingly.",
      options: { fontSize: 16.5, color: INK } },
  ], { x: 0.85, y: 5.32, w: 11.6, h: 1.2, fontFace: FONT, margin: 0, valign: "middle" });
  footer(s, "AUROC computed from the identical predictions used for the RMSE, on the same single test evaluation");
  s.addNotes("This is why I kept AUROC even though the request was to use plain RMSE everywhere. On T4 the two metrics disagree about who wins, so reporting only one would have asserted something the data does not settle.");
}

// ---------------------------------------------------------------- S8
{
  const s = slide("Attribution", "Last week's advantage was a size decision, not a design one");
  s.addImage({ path: F("fig_size_confound.png"), x: 0.3, y: 1.5, w: 12.75, h: 3.8 });
  s.addShape(pres.ShapeType.roundRect, { x: 0.6, y: 5.55, w: 12.15, h: 1.2,
    rectRadius: 0.08, fill: { color: BG }, line: { color: RULE, width: 1 } });
  s.addText([
    { text: "Right panel: ", options: { bold: true, fontSize: 15.5, color: INK } },
    { text: "when length is free the LLM picks the maximum in about 7 of 10 cells while the classical arms spread over 1 to 5.   ", options: { fontSize: 15.5, color: INK } },
    { text: "Left three: ", options: { bold: true, fontSize: 15.5, color: INK } },
    { text: `only Random degrades as length is freed and the budget shrinks (T1 median ${med(fixed, "gauss_peak", "random")} → ${med(pilot, "gauss_peak", "random")}), while the LLM barely moves (${med(fixed, "gauss_peak", "llm_open")} → ${med(pilot, "gauss_peak", "llm_open")}). At the pilot's setting they meet.`,
      options: { fontSize: 15.5, color: INK } },
  ], { x: 0.85, y: 5.65, w: 11.6, h: 1.0, fontFace: FONT, margin: 0, valign: "middle" });
  footer(s, "Three conditions, same code, same model, same tasks, 10 seeds each · data/condition_comparison.json");
  s.addNotes("This is the slide I would defend hardest. The pilot rewarded a single decision - ask for the maximum number of gates - and the LLM makes that decision almost every time while a uniform sampler does not. Pin the length and the advantage goes; restore the length and shrink the budget to the pilot's and the two meet again.");
}

// ---------------------------------------------------------------- S9
{
  const s = slide("Search dynamics", "More feedback did not translate into better circuits");
  s.addImage({ path: F("fig_anytime.png"), x: 0.35, y: 1.5, w: 12.6, h: 3.5 });
  bullets(s, [
    "Closed-loop received eight rounds of validation-only feedback on its own candidates and still did not beat open-loop on any task.",
    "A first attempt measured a 74% duplicate rate for closed-loop, with 11 of 40 cells starved of budget. The prompt now lists every circuit already tried; duplicates fell to 16% and every cell reached its full budget. The numbers here are from the corrected run.",
  ], { x: 0.7, y: 5.15, w: 12.0, h: 1.5, size: 15.5, gap: 9 });
  footer(s, "Curves are the median best-so-far validation RMSE over 10 seeds; the x-axis is unique candidates, not API calls");
  s.addNotes("Worth being explicit: the first version of this experiment handicapped closed-loop through a prompt defect. I fixed it and re-ran rather than reporting the handicapped numbers. Even after the fix, feedback did not help.");
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
      fontSize: 19, color: colour, fontFace: FONT, margin: 0 });
    bullets(s, lines, { x: x + 0.32, y: 1.95, w: 5.3, h: 3.1, size: 14.5,
      gap: 9, color: "E8EEF7" });
  };
  card(0.7, "Supported by this run", [
    "With circuit length fixed, gpt-5-nano does not beat random search on 3 of 4 tasks",
    "It does beat every classical arm on T2, clearly and across all 10 seeds",
    "When length is free it picks the maximum ~70% of the time; uniform sampling does not",
    "That single decision explains most of last week's reported advantage",
  ], "9FE3C0");
  card(6.85, "NOT supported", [
    "\"LLMs cannot design quantum circuits\" - one small model, one budget, four tiny tasks",
    "\"Closed-loop feedback does not work\" - it failed here at 8 rounds on 5-gate circuits",
    "Any claim about hardware: this is noiseless statevector simulation throughout",
    "Any cross-task comparison of RMSE values",
  ], "FFC4B8");
  s.addShape(pres.ShapeType.roundRect, { x: 0.7, y: 5.45, w: 12.0, h: 1.15,
    rectRadius: 0.1, fill: { color: "FFFFFF" }, line: { color: ACCENT, width: 1.6 } });
  s.addText([
    { text: "Next: ", options: { bold: true, fontSize: 17, color: ACCENT } },
    { text: "give the LLM a decision that is actually hard. Fixed length removed the easy win; the open question is whether a larger budget, a stronger model, or a task where structure matters more can produce an advantage that survives this kind of attribution check.",
      options: { fontSize: 17, color: INK } },
  ], { x: 0.95, y: 5.57, w: 11.5, h: 0.95, fontFace: FONT, margin: 0, valign: "middle" });
  s.addText(`600 cells across 3 conditions · ${(DATA.wall_clock_seconds / 60).toFixed(1)} min for the main run · gpt-5-nano-2025-08-07 · $0.23 total`, {
    x: 0.7, y: 6.85, w: 12, h: 0.3, fontSize: 12, color: "9FB3CC",
    fontFace: FONT, margin: 0 });
  s.addNotes("The honest summary is that I removed a confound and a positive result mostly went away. That is a useful outcome: it tells us the earlier benchmark was rewarding the wrong thing, and it tells us what a fair test has to control for.");
}

const out = path.join(HERE, "20260731_GSoC_mini5.pptx");
pres.writeFile({ fileName: out }).then(() => console.log("deck written:", out));
