"""Build the redesigned 10-slide robustness deck (.pptx + .pdf).

Structure and explanatory style follow the previous meeting deck (numbered
section kicker, step-by-step definitions, tinted definition boxes, an
explicit "Reading" note under every figure). Content is the current
robustness study only. Every number is read from the committed result
tables; nothing is typed by hand.

QA performed on every build:
  - exactly 10 slides, nothing off-canvas, no estimated text overflow
  - no coding/version identifiers in visible text (shared pattern list)
  - every technical term is defined on or before the slide it first appears
  - both 3 x 3 Hamiltonian grids present, six categories per cell
  - the next action names both the target accuracy and the minimum budget

Usage:
  python scripts/qae/build_qae_robustness_slides.py          # build + PDF
  python scripts/qae/build_qae_robustness_slides.py --check  # audit only
"""
import argparse
import importlib.util
import json
import re
import subprocess
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path("outputs/qae_robustness")
FIGURES = ROOT / "figures"
DECK_DIR = ROOT / "deck"
DECK_NAME = "20260904_GSoC"

# Shared prohibited-string list and geometry audit from the first build.
_spec = importlib.util.spec_from_file_location(
    "qae_deck_common", Path("scripts/qae/build_qae_robustness_deck.py"))
_common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_common)
FORBIDDEN_PATTERNS = _common.FORBIDDEN_PATTERNS
geometry_audit = _common.audit
extract_text = _common.extract_text

SLIDE_W, SLIDE_H = Inches(13.333), Inches(7.5)

# Palette of the previous deck.
NAVY = RGBColor(0x1F, 0x2A, 0x44)
KICKER = RGBColor(0xC5, 0x22, 0x1F)
INK = RGBColor(0x20, 0x21, 0x24)
MUTED = RGBColor(0x5F, 0x63, 0x68)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BLUE, BLUE_FILL = RGBColor(0x1A, 0x73, 0xE8), RGBColor(0xE8, 0xF0, 0xFE)
ORANGE, ORANGE_FILL = RGBColor(0xE8, 0x71, 0x0A), RGBColor(0xFE, 0xF3, 0xE0)
GREEN, GREEN_FILL = RGBColor(0x18, 0x80, 0x38), RGBColor(0xE6, 0xF4, 0xEA)
RED, RED_FILL = RGBColor(0xD9, 0x30, 0x25), RGBColor(0xFC, 0xE8, 0xE6)
AMBER, AMBER_FILL = RGBColor(0xF9, 0xAB, 0x00), RGBColor(0xFE, 0xF7, 0xE0)
GREY, GREY_FILL = RGBColor(0x9A, 0xA0, 0xA6), RGBColor(0xF1, 0xF3, 0xF4)
PURPLE = RGBColor(0x84, 0x30, 0xCE)
FONT = "Helvetica Neue"

MODEL_A = "gpt-5.4-mini"
MODEL_B = "gpt-4.1-mini"

# ------------------------------------------------------------ primitives ---


def _flat(shape):
    shape.shadow.inherit = False
    style = shape._element.find(
        "{http://schemas.openxmlformats.org/presentationml/2006/main}style")
    if style is not None:
        shape._element.remove(style)
    return shape


def rect(slide, left, top, width, height, color):
    shape = _flat(slide.shapes.add_shape(1, left, top, width, height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    return shape


def box(slide, left, top, width, height, fill, border, border_pt=1.25):
    shape = _flat(slide.shapes.add_shape(5, left, top, width, height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = border
    shape.line.width = Pt(border_pt)
    return shape


def textbox(slide, left, top, width, height, align=PP_ALIGN.LEFT):
    frame = slide.shapes.add_textbox(left, top, width, height).text_frame
    frame.word_wrap = True
    frame.paragraphs[0].alignment = align
    return frame


def write(frame, lines, *, size=12, color=INK, bold=False, space_after=4,
          line_spacing=1.0):
    """lines: str | (str, opts) | list of runs [(text, opts), ...] per paragraph."""
    first = True
    for line in lines:
        if isinstance(line, list):
            runs, opts = line, {}
        else:
            text, opts = (line if isinstance(line, tuple) else (line, {}))
            runs = [(text, opts)]
        paragraph = frame.paragraphs[0] if first else frame.add_paragraph()
        first = False
        paragraph.space_after = Pt(opts.get("space_after", space_after))
        paragraph.line_spacing = opts.get("line_spacing", line_spacing)
        if "align" in opts:
            paragraph.alignment = opts["align"]
        for text, ropts in runs:
            run = paragraph.add_run()
            run.text = text
            font = run.font
            font.name = FONT
            font.size = Pt(ropts.get("size", opts.get("size", size)))
            font.bold = ropts.get("bold", opts.get("bold", bold))
            font.italic = ropts.get("italic", False)
            font.color.rgb = ropts.get("color", opts.get("color", color))
    return frame


def header(slide, kicker, title, subtitle=None):
    frame = textbox(slide, Inches(0.55), Inches(0.24), Inches(12.2), Inches(0.28))
    write(frame, [(kicker.upper(), {"size": 11, "bold": True, "color": KICKER})],
          space_after=0)
    frame = textbox(slide, Inches(0.55), Inches(0.56), Inches(12.2), Inches(0.5))
    write(frame, [(title, {"size": 24, "bold": True, "color": NAVY})], space_after=0)
    if subtitle:
        frame = textbox(slide, Inches(0.55), Inches(1.1), Inches(12.2), Inches(0.38))
        write(frame, [(subtitle, {"size": 11, "color": MUTED})], space_after=0)


def footer(slide, text):
    frame = textbox(slide, Inches(0.55), Inches(7.0), Inches(12.2), Inches(0.35))
    write(frame, [(text, {"size": 8.5, "color": MUTED})], space_after=0)


def picture(slide, name, left, top, width=None, height=None):
    path = FIGURES / f"{name}.png"
    if not path.exists():
        raise SystemExit(f"missing figure {path}; run the grid figure builder first")
    if width is not None:
        return slide.shapes.add_picture(str(path), left, top, width=width)
    return slide.shapes.add_picture(str(path), left, top, height=height)


def titled_box(slide, left, top, width, height, fill, border, title, lines,
               title_color=None, title_size=12, body_size=10.5, space_after=3):
    box(slide, left, top, width, height, fill, border)
    frame = textbox(slide, left + Inches(0.14), top + Inches(0.08),
                    width - Inches(0.28), height - Inches(0.16))
    write(frame, [(title, {"size": title_size, "bold": True,
                           "color": title_color or border, "space_after": 3})]
          + [(line if isinstance(line, (tuple, list)) else
              (line, {"size": body_size})) for line in lines],
          size=body_size, space_after=space_after)
    return frame


def chip(slide, left, top, color, hollow=False):
    shape = _flat(slide.shapes.add_shape(9, left, top, Inches(0.16), Inches(0.16)))
    shape.fill.solid()
    shape.fill.fore_color.rgb = WHITE if hollow else color
    shape.line.color.rgb = color
    shape.line.width = Pt(1.75 if hollow else 0.75)
    return shape


TABLE_ROW_HEIGHT = Inches(0.34)


def table(slide, rows, left, top, col_widths, row_height=TABLE_ROW_HEIGHT,
          header_size=10, body_size=9.5, colors=None, align_first_left=True):
    width = sum(col_widths, Emu(0))
    shape = slide.shapes.add_table(len(rows), len(rows[0]), left, top, width,
                                   row_height * len(rows)).table
    for index, w in enumerate(col_widths):
        shape.columns[index].width = w
    for r, row in enumerate(rows):
        shape.rows[r].height = row_height
        for c, value in enumerate(row):
            cell = shape.cell(r, c)
            cell.text = str(value)
            cell.margin_left = cell.margin_right = Inches(0.07)
            cell.margin_top = cell.margin_bottom = Emu(9525)
            for paragraph in cell.text_frame.paragraphs:
                paragraph.alignment = (PP_ALIGN.LEFT if (c == 0 and align_first_left)
                                       else PP_ALIGN.CENTER)
                for run in paragraph.runs:
                    run.font.size = Pt(header_size if r == 0 else body_size)
                    run.font.bold = r == 0
                    run.font.name = FONT
                    run.font.color.rgb = INK
                    if colors and (r, c) in colors:
                        run.font.color.rgb = colors[(r, c)]
                        run.font.bold = True
            cell.fill.solid()
            cell.fill.fore_color.rgb = (RGBColor(0xE8, 0xEA, 0xED) if r == 0 else
                                        (WHITE if r % 2 else RGBColor(0xF8, 0xF9, 0xFA)))
    return shape


# --------------------------------------------------------------- numbers ---


def _fmt(value, digits=4, sign=True):
    return f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}"


def _p(value):
    return "p < 0.001" if value < 0.001 else f"p = {value:.3f}"


def stat(summary, key, contrast):
    return summary["conditions"][key]["contrasts"][contrast]


def mean_of(summary, key, method):
    return summary["conditions"][key]["per_method"][method]["mean"]


def valid_share(summary, key):
    return summary["conditions"][key]["llm_proposal_validity"]["valid_fraction"]


def arm_validity(key, method):
    quality = json.loads((ROOT / key / "proposal_quality.json").read_text())
    entry = quality[method]
    return 1 - entry["fallback_evaluations"] / entry["evaluations"]


def contrast_line(summary, key, contrast):
    entry = stat(summary, key, contrast)
    lo, hi = entry["bootstrap_95ci"]
    return (f"{_fmt(entry['mean_paired_gain'])}  [{_fmt(lo)}, {_fmt(hi)}]  ·  "
            f"{entry['wins_b']}/{entry['n']} seeds  ·  "
            f"{_p(entry['wilcoxon_exact_two_sided_p'])}")


def verdict(summary, key, contrast):
    entry = stat(summary, key, contrast)
    if entry["wilcoxon_exact_two_sided_p"] >= 0.05:
        return "not detected"
    return "holds" if entry["mean_paired_gain"] > 0 else "reverses"


OPEN_RANDOM = "LLM-Open_minus_Random"
CLOSED_OPEN = "LLM-Closed_minus_LLM-Open"
CLOSED_RANDOM = "LLM-Closed_minus_Random"
GREEDY_RANDOM = "Greedy_minus_Random"
VARIED = ["budget_b4", "budget_b16", "qubits_n6", "qubits_n8", "hamiltonian_xxz",
          "model_alt"]
NAMES = {"budget_b4": "B = 4", "budget_b16": "B = 16", "qubits_n6": "6 qubits",
         "qubits_n8": "8 qubits", "hamiltonian_xxz": "the XXZ chain",
         "model_alt": "Model B"}


# ---------------------------------------------------------------- slides ---


def slide_01_title(deck, summary):
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    rect(slide, Emu(0), Emu(0), SLIDE_W, SLIDE_H, NAVY)
    frame = textbox(slide, Inches(0.9), Inches(1.05), Inches(11.5), Inches(1.9))
    write(frame, [("Does the quantum-autoencoder search result",
                   {"size": 36, "bold": True, "color": WHITE, "space_after": 0}),
                  ("hold up when one thing changes?",
                   {"size": 36, "bold": True, "color": WHITE})], space_after=6)
    frame = textbox(slide, Inches(0.9), Inches(2.95), Inches(11.5), Inches(0.9))
    write(frame, [("A controlled robustness study of the previous result: change "
                   "exactly one factor at a time — evaluation budget, qubit count, "
                   "Hamiltonian, language model — and see which findings survive.",
                   {"size": 16, "color": RGBColor(0xC8, 0xD4, 0xEC)})])
    box(slide, Inches(0.9), Inches(3.95), Inches(11.5), Inches(1.45),
        RGBColor(0x2A, 0x3A, 0x5C), RGBColor(0x5B, 0x8D, 0xEF))
    frame = textbox(slide, Inches(1.12), Inches(4.07), Inches(11.1), Inches(1.25))
    write(frame, [[("The question:  ", {"bold": True, "color": RGBColor(0xF9, 0xAB, 0x00)}),
                   ("last time, physics-informed language-model proposals beat random "
                    "and greedy circuit search by a wide margin, and a closed feedback "
                    "loop added a small extra gain — at 4 qubits, one Hamiltonian, one "
                    "budget, one model. Which of those two findings is a property of "
                    "the method, and which was a property of that one setting? "
                    "We answer by moving exactly one declared factor per condition.",
                    {"color": WHITE})]], size=13, space_after=0)
    frame = textbox(slide, Inches(0.9), Inches(5.75), Inches(11.5), Inches(1.1))
    write(frame, [("Tomoya Hatanaka", {"size": 15, "bold": True, "color": WHITE}),
                  ("ML4SCI / Google Summer of Code 2026",
                   {"size": 12, "color": RGBColor(0xC8, 0xD4, 0xEC)}),
                  ("September 4, 2026",
                   {"size": 11, "color": RGBColor(0x9A, 0xA8, 0xC8)})], space_after=2)


def slide_02_baseline(deck, summary):
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    header(slide, "1 · Previous baseline",
           "Where we left off: the previous baseline result",
           "Quantum autoencoder (QAE): compress ground states into fewer qubits — "
           "4 qubits · transverse-field Ising chain · budget B = 8 · 12 paired seeds")
    # task flow strip
    flow = [
        ("Input  |ψ(h)⟩", "ground state of the 4-qubit Ising chain\n"
         "H = −Σ Zᵢ Zᵢ₊₁ − h Σ Xᵢ,  h ∈ [0.2, 2.0]", BLUE_FILL, BLUE),
        ("Encoder  Uθ", "the circuit we search:\n16 gates, angles trained", ORANGE_FILL, ORANGE),
        ("Latent  q0, q1", "the 2 qubits we KEEP\n(the compressed code)", GREEN_FILL, GREEN),
        ("Trash  q2, q3", "should end in the SAME state |00⟩\nfor every input", RED_FILL, RED),
    ]
    x = 0.55
    widths = [3.55, 2.55, 2.55, 2.75]
    for (title, body, fill, border), w in zip(flow, widths, strict=True):
        titled_box(slide, Inches(x), Inches(1.55), Inches(w), Inches(0.92), fill, border,
                   title, [(body, {"size": 9, "color": INK})], title_size=11.5, body_size=9,
                   space_after=0)
        if w != widths[-1]:
            frame = textbox(slide, Inches(x + w + 0.02), Inches(1.78), Inches(0.3), Inches(0.4))
            write(frame, [("→", {"size": 16, "color": MUTED})], space_after=0)
        x += w + 0.35
    # metric definition
    box(slide, Inches(0.55), Inches(2.6), Inches(12.2), Inches(0.62), BLUE_FILL, BLUE)
    frame = textbox(slide, Inches(0.7), Inches(2.66), Inches(11.95), Inches(0.55))
    write(frame, [[("Metric — trash fidelity:  ", {"bold": True, "color": BLUE}),
                   ("F_trash = ⟨00| ρ_trash |00⟩ = P(q2 q3 = 00), the probability that "
                    "the trash qubits end in |00⟩.  ", {}),
                   ("Higher is better.  ", {"bold": True, "color": GREEN}),
                   ("Reported on a held-out test set (64 unseen h values never used for "
                    "training, selection or feedback), read once after selection.", {})]],
          size=10.5, space_after=0)
    # figure
    picture(slide, "baseline_cell", Inches(0.45), Inches(3.32), width=Inches(6.0))
    # methods
    methods = [
        ("Random", GREY, "B independent uniform circuit draws; no physics, no feedback."),
        ("Greedy", AMBER, "B/2 random starts (exploration, no feedback), then B/2 "
                          "single-gate refinements of the best one (refinement, uses "
                          "validation feedback); a change is kept only if validation "
                          "improves."),
        ("LLM-Open", BLUE, "open-loop: the language model writes all B physics-informed "
                           "proposals BEFORE any score is seen."),
        ("LLM-Closed", RED, "closed-loop: B/2 proposals, then B/2 free redesigns of the "
                            "current best circuit, each given its validation score."),
    ]
    box(slide, Inches(6.65), Inches(3.32), Inches(6.1), Inches(1.95), GREY_FILL, GREY)
    frame = textbox(slide, Inches(6.8), Inches(3.38), Inches(5.85), Inches(1.85))
    lines = [[("The four methods — each may train and score exactly B candidate "
               "circuits per seed (B = the evaluation budget)", {"bold": True})]]
    for name, color, body in methods:
        lines.append([(f"{name}  ", {"bold": True, "color": color}), (body, {})])
    write(frame, lines, size=9.5, space_after=2)
    # stats
    box(slide, Inches(6.65), Inches(5.37), Inches(6.1), Inches(1.5), AMBER_FILL, AMBER)
    frame = textbox(slide, Inches(6.8), Inches(5.42), Inches(5.85), Inches(1.42))
    write(frame, [
        [("What the baseline found", {"bold": True, "color": ORANGE})],
        [("1  Semantics is decisive.  ", {"bold": True, "color": BLUE}),
         ("LLM-Open − Random  " + contrast_line(summary, "reference", OPEN_RANDOM), {})],
        [("2  The closed loop adds a little more.  ", {"bold": True, "color": RED}),
         ("LLM-Closed − LLM-Open  " + contrast_line(summary, "reference", CLOSED_OPEN), {})],
        [("3  Non-semantic refinement is flat.  ", {"bold": True, "color": ORANGE}),
         ("Greedy − Random  " + contrast_line(summary, "reference", GREEDY_RANDOM), {})],
    ], size=9.5, space_after=2)
    footer(slide, "Reading: dots = the 12 paired seeds (12 independent draws of training/"
                  "validation data, shared by all methods); marker = mean; bar = bootstrap "
                  "95% confidence interval (CI). Differences are per-seed paired; "
                  "p = exact paired Wilcoxon test. Model A = the reference model of the "
                  "baseline (" + MODEL_A + ").")


def slide_03_protocol(deck, summary):
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    header(slide, "2 · Controlled protocol",
           "One factor at a time; everything else frozen",
           "Every condition differs from the baseline in exactly one declared factor, and "
           "an automated check refuses any condition that does not")
    titled_box(slide, Inches(0.55), Inches(1.6), Inches(5.55), Inches(3.45), GREEN_FILL, GREEN,
               "FROZEN for every condition", [
        "•  the task: quantum autoencoder, trash fidelity metric, 12 paired seeds",
        "•  the four methods, exactly as defined on the previous slide",
        "•  the language-model workflow: prompt wording, output format, sampling "
        "temperature, retry policy (only stated Hamiltonian facts are substituted)",
        "•  the circuit trainer: same optimizer, epochs, initialisation, data splits",
        "•  the selection rule: lowest validation loss; test set read once afterwards",
        "•  the gate set: RX, RY, RZ rotations and CNOT (controlled-NOT) gates",
        "•  random-number streams and all statistical conventions",
    ], body_size=10)
    titled_box(slide, Inches(6.35), Inches(1.6), Inches(6.4), Inches(3.45), BLUE_FILL, BLUE,
               "VARIED — one factor per condition", [])
    rows = [["Factor", "Levels tested", "All other settings"],
            ["Evaluation budget B", "4 · 8 · 16", "4 qubits, Ising, Model A"],
            ["Qubit count", "4 · 6 · 8", "B = 8, Ising, Model A"],
            ["Hamiltonian", "Ising · XXZ", "4 qubits, B = 8, Model A"],
            ["Language model", "Model A · Model B", "4 qubits, Ising, B = 8"]]
    table(slide, rows, Inches(6.55), Inches(2.05),
          [Inches(1.85), Inches(1.75), Inches(2.4)], row_height=Inches(0.36))
    frame = textbox(slide, Inches(6.55), Inches(3.98), Inches(6.0), Inches(1.0))
    write(frame, [
        [("Model A", {"bold": True, "color": BLUE}),
         (f" = the reference model of the baseline ({MODEL_A}, March 2026 snapshot).  ", {}),
         ("Model B", {"bold": True, "color": PURPLE}),
         (f" = the alternative model ({MODEL_B}, April 2025 snapshot): same prompt "
          "bytes, temperature, retries and token limit; only the model name changes.", {})],
        [("XXZ", {"bold": True, "color": BLUE}),
         (" = the second Hamiltonian family, an XXZ Heisenberg chain "
          "H = Σ (Xᵢ Xᵢ₊₁ + Yᵢ Yᵢ₊₁ + Δ Zᵢ Zᵢ₊₁), Δ ∈ [0.2, 2.0], declared before "
          "the run. The prompt changes only in the stated physics facts.", {})],
    ], size=9.5, space_after=3)
    box(slide, Inches(0.55), Inches(5.2), Inches(12.2), Inches(1.6), AMBER_FILL, AMBER)
    frame = textbox(slide, Inches(0.7), Inches(5.26), Inches(11.95), Inches(1.5))
    write(frame, [
        [("Scaling rules, fixed before the run:  ", {"bold": True, "color": ORANGE}),
         ("(1) circuit width = 3 trainable rotations + 1 CNOT per qubit for every "
          "method — so 4 qubits reproduces the baseline's 16-gate contract exactly; "
          "(2) half the qubits are latent, half are trash; (3) the two adaptive methods "
          "(Greedy, LLM-Closed) always split B evenly: B/2 exploration (proposals made "
          "without feedback) + B/2 refinement (proposals that use validation "
          "feedback).", {})],
        [("Fairness check:  ", {"bold": True, "color": ORANGE}),
         ("each condition stores a fingerprint of every frozen component above; the "
          "run refuses to start if the fingerprint differs from the baseline's or if "
          "more than one factor moves. The baseline cell is reused rather than re-run; "
          "tests prove the study code reproduces its data, prompts and trainer "
          "exactly.", {})],
    ], size=10, space_after=4)
    footer(slide, "Reused baseline cell: 4 qubits · Ising · B = 8 · Model A. "
                  "Not part of this study, by design: any change to the prompt's "
                  "physical content.")


def slide_04_reading(deck, summary):
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    header(slide, "3 · How to read the result grids",
           "How to read the two result grids",
           "One grid per Hamiltonian · one cell per (qubit count, budget) · six "
           "categories per cell — the next two slides use exactly this layout")
    coverage = json.loads((ROOT / "grid_cells.json").read_text())["coverage"]

    def cell_text(family, n, budget):
        present = coverage[family][f"{n}q_B{budget}"]
        if not present:
            return "not run"
        return "6 categories" if len(present) == 6 else "4 categories\n(Model A only)"

    for i, (family, label) in enumerate((("TFIM", "Ising grid (next slide)"),
                                         ("XXZ", "XXZ grid (slide after)"))):
        left = Inches(0.55 + i * 3.35)
        frame = textbox(slide, left, Inches(1.62), Inches(3.1), Inches(0.3))
        write(frame, [(label, {"size": 10.5, "bold": True, "color": NAVY})], space_after=0)
        rows = [["", "B = 4", "B = 8", "B = 16"]]
        colors = {}
        for r, n in enumerate((4, 6, 8), start=1):
            row = [f"{n} qubits"]
            for c, budget in enumerate((4, 8, 16), start=1):
                text = cell_text(family, n, budget)
                row.append(text)
                colors[(r, c)] = GREY if text == "not run" else GREEN
            rows.append(row)
        table(slide, rows, left, Inches(1.95),
              [Inches(0.8), Inches(0.75), Inches(0.8), Inches(0.75)],
              row_height=Inches(0.5), header_size=9, body_size=8, colors=colors)
    frame = textbox(slide, Inches(0.55), Inches(4.05), Inches(6.2), Inches(1.0))
    write(frame, [
        [("Rows", {"bold": True}), (" = qubit count (4, 6, 8; half latent, half trash).  ", {}),
         ("Columns", {"bold": True}), (" = evaluation budget B (4, 8, 16).", {})],
        [("Why cells are missing: ", {"bold": True, "color": KICKER}),
         ("the study was designed one factor at a time from the baseline cell, so the "
          "off-axis cells were not run. They are shown grey, never estimated. Model B "
          "was run only in the baseline cell.", {})],
    ], size=9.5, space_after=3)

    box(slide, Inches(7.05), Inches(1.6), Inches(5.7), Inches(3.5), GREY_FILL, GREY)
    frame = textbox(slide, Inches(7.2), Inches(1.66), Inches(5.4), Inches(0.3))
    write(frame, [("The six categories in every cell, left to right",
                   {"size": 11, "bold": True, "color": NAVY})], space_after=0)
    categories = [
        ("Random", GREY, False, "no physics, no feedback"),
        ("Greedy", AMBER, False, "no physics; single-gate refinement of the best start"),
        ("LLM-Open · Model A", BLUE, False, f"open-loop proposals, {MODEL_A}"),
        ("LLM-Closed · Model A", RED, False, f"closed-loop redesigns, {MODEL_A}"),
        ("LLM-Open · Model B", BLUE, True, f"open-loop proposals, {MODEL_B}"),
        ("LLM-Closed · Model B", RED, True, f"closed-loop redesigns, {MODEL_B}"),
    ]
    for j, (name, color, hollow, body) in enumerate(categories):
        top = Inches(2.05 + j * 0.49)
        chip(slide, Inches(7.28), top + Inches(0.05), color, hollow)
        frame = textbox(slide, Inches(7.52), top, Inches(5.1), Inches(0.46))
        write(frame, [[(f"{j + 1}  {name}", {"bold": True, "color": color}),
                       (f"  —  {body}", {"color": INK})]], size=9.5, space_after=0)
    box(slide, Inches(0.55), Inches(5.2), Inches(12.2), Inches(1.6), AMBER_FILL, AMBER)
    frame = textbox(slide, Inches(0.7), Inches(5.26), Inches(11.95), Inches(1.5))
    write(frame, [
        [("Encoding in every cell:  ", {"bold": True, "color": ORANGE}),
         ("small dots = the 12 paired seeds; large marker = mean; vertical bar = "
          "bootstrap 95% CI; the number above = mean. Model A = filled markers, "
          "Model B = hollow markers of the same colour. Thick frame = the baseline "
          "cell. All cells share the same vertical axis.", {})],
        [("Metric:  ", {"bold": True, "color": ORANGE}),
         ("held-out test trash fidelity F_trash, higher is better; 1.0 means the "
          "trash qubits are exactly |00⟩ for every test state.", {})],
        [("Comparisons:  ", {"bold": True, "color": ORANGE}),
         ("all four methods in a cell share the same seeds and data, so differences "
          "are read per seed (paired), never as raw means.", {})],
    ], size=10, space_after=4)
    footer(slide, "Random and Greedy do not depend on the language model, so they appear "
                  "once per cell; the two LLM methods are split by model, giving six "
                  "categories.")


def _grid_slide(deck, summary, family, kicker, title, subtitle, foot):
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    header(slide, kicker, title, subtitle)
    picture(slide, f"grid_{family.lower()}", Inches(0.95), Inches(1.5), height=Inches(5.45))
    footer(slide, foot)
    return slide


def slide_05_grid_ising(deck, summary):
    n4 = mean_of(summary, "reference", "LLM-Open")
    n8 = mean_of(summary, "qubits_n8", "LLM-Open")
    r4 = mean_of(summary, "reference", "Random")
    r8 = mean_of(summary, "qubits_n8", "Random")
    return _grid_slide(
        deck, summary, "TFIM", "4 · Results — Hamiltonian 1",
        "Ising chain: the semantic methods stay on top in every cell run",
        "H = −Σ Zᵢ Zᵢ₊₁ − h Σ Xᵢ, h ∈ [0.2, 2.0]  ·  5 of 9 cells run (one factor at a "
        "time)  ·  Model B only in the baseline cell",
        f"Reading: going down the middle column, Random falls from {r4:.3f} to {r8:.3f} "
        f"while LLM-Open (Model A) falls only from {n4:.3f} to {n8:.3f}. Every number "
        "is a mean over 12 paired seeds; grey cells were not run.")


def slide_06_grid_xxz(deck, summary):
    xo = mean_of(summary, "hamiltonian_xxz", "LLM-Open")
    xc = mean_of(summary, "hamiltonian_xxz", "LLM-Closed")
    xr = mean_of(summary, "hamiltonian_xxz", "Random")
    return _grid_slide(
        deck, summary, "XXZ", "5 · Results — Hamiltonian 2",
        "XXZ chain: the same picture in the one cell that was run",
        "H = Σ (Xᵢ Xᵢ₊₁ + Yᵢ Yᵢ₊₁ + Δ Zᵢ Zᵢ₊₁), Δ ∈ [0.2, 2.0]  ·  1 of 9 cells run "
        "(the baseline's qubit count and budget)  ·  Model A only",
        f"Reading: at 4 qubits, B = 8 the ordering is LLM-Closed {xc:.3f} > LLM-Open "
        f"{xo:.3f} > Random {xr:.3f}, with Random's seeds spread widely. The other eight "
        "cells were not run and are shown grey, not estimated.")


def slide_07_summary(deck, summary):
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    header(slide, "6 · Cross-grid summary",
           "What consistently helps, and what is fragile",
           "Rows = run cells; panels = paired per-seed differences (positive = the first "
           "method wins). Filled marker = p < 0.05; dashed line = the baseline's value.")
    picture(slide, "forest_defined", Inches(0.45), Inches(1.55), width=Inches(12.45))

    def tally(contrast):
        same = sum(1 for k in VARIED
                   if (stat(summary, k, contrast)["mean_paired_gain"] > 0)
                   == (stat(summary, "reference", contrast)["mean_paired_gain"] > 0))
        holds = [k for k in VARIED if verdict(summary, k, contrast) == "holds"]
        reverses = [k for k in VARIED if verdict(summary, k, contrast) == "reverses"]
        return same, holds, reverses

    open_same, open_holds, _ = tally(OPEN_RANDOM)
    closed_same, closed_holds, closed_rev = tally(CLOSED_OPEN)
    cr_same, cr_holds, _ = tally(CLOSED_RANDOM)
    _g_same, g_holds, g_rev = tally(GREEDY_RANDOM)
    n = len(VARIED)
    boxes = [
        (GREEN_FILL, GREEN, "ROBUST — physics-informed proposals beat non-semantic search",
         f"LLM-Open − Random keeps its sign in {open_same}/{n} varied conditions and is "
         f"significant in {len(open_holds)}/{n}; LLM-Closed − Random is significant in "
         f"{len(cr_holds)}/{n}. The gap is largest at 8 qubits, where the non-semantic "
         "methods degrade fastest."),
        (RED_FILL, RED, "FRAGILE — the closed loop's edge over the open batch",
         f"LLM-Closed − LLM-Open keeps the baseline's sign in only {closed_same}/{n} "
         "varied conditions: significantly positive at "
         f"{', '.join(NAMES[k] for k in closed_holds)}, significantly negative at "
         f"{', '.join(NAMES[k] for k in closed_rev)}."),
        (AMBER_FILL, AMBER, "NEVER HELPS — single-gate greedy refinement",
         f"Greedy − Random is significantly positive in {len(g_holds)}/{n} conditions"
         + (f" and significantly negative at {', '.join(NAMES[k] for k in g_rev)}: spending "
            "half of a small budget on single-gate steps costs breadth." if g_rev else ".")),
    ]
    for i, (fill, border, title, body) in enumerate(boxes):
        left = Inches(0.55 + i * 4.1)
        titled_box(slide, left, Inches(5.15), Inches(3.95), Inches(1.7), fill, border,
                   title, [(body, {"size": 9.5})], title_size=10.5, body_size=9.5)
    footer(slide, "\"Significant\" = exact paired Wilcoxon p < 0.05 over 12 seeds; "
                  "\"keeps its sign\" = the mean paired difference has the same sign as "
                  "in the baseline. A non-significant cell is not evidence of no effect.")


def slide_08_interpretation(deck, summary):
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    header(slide, "7 · Interpretation",
           "What the pattern implies — and what it does not",
           "Four factors, read one at a time; the right-hand panel shows a failure mode "
           "that runs across all of them")
    co = {k: stat(summary, k, CLOSED_OPEN)["mean_paired_gain"]
          for k in ("budget_b4", "reference", "budget_b16", "qubits_n6", "qubits_n8",
                    "hamiltonian_xxz", "model_alt")}
    orr = {k: stat(summary, k, OPEN_RANDOM) for k in co}
    items = [
        (BLUE_FILL, BLUE, "Budget B",
         f"The closed loop's edge over the open batch is largest at the smallest budget "
         f"({_fmt(co['budget_b4'])} at B = 4), small at B = 8 ({_fmt(co['reference'])}) "
         f"and gone at B = 16 ({_fmt(co['budget_b16'])}). Free redesign is worth most when "
         "the open batch is thinnest; a wide open batch buys the same breadth without "
         "feedback."),
        (GREEN_FILL, GREEN, "Qubit count",
         f"Absolute fidelity falls for every method as the register widens, but the "
         "semantic advantage grows: LLM-Open − Random "
         f"{_fmt(orr['reference']['mean_paired_gain'])} → "
         f"{_fmt(orr['qubits_n6']['mean_paired_gain'])} → "
         f"{_fmt(orr['qubits_n8']['mean_paired_gain'])}. The closed loop reverses "
         f"({_fmt(co['qubits_n6'])} at 6, {_fmt(co['qubits_n8'])} at 8 qubits)."),
        (AMBER_FILL, AMBER, "Hamiltonian family",
         f"On the XXZ chain the closed loop wins clearly ({_fmt(co['hamiltonian_xxz'])}, "
         f"{stat(summary, 'hamiltonian_xxz', CLOSED_OPEN)['wins_b']}/12 seeds). LLM-Open − Random "
         f"is {_fmt(orr['hamiltonian_xxz']['mean_paired_gain'])} on average but only "
         f"{orr['hamiltonian_xxz']['wins_b']}/12 seeds: Random occasionally finds a very good "
         "circuit here, so the mean overstates a per-seed effect. Two families are two points."),
        (RED_FILL, RED, "Language model",
         f"With Model B the semantic advantage is essentially unchanged "
         f"({_fmt(orr['model_alt']['mean_paired_gain'])} vs "
         f"{_fmt(orr['reference']['mean_paired_gain'])}), "
         f"even though only {valid_share(summary, 'model_alt') * 100:.0f}% of its "
         "evaluated proposals satisfied the gate-count contract. The closed loop does not "
         f"keep its edge ({_fmt(co['model_alt'])})."),
    ]
    for i, (fill, border, title, body) in enumerate(items):
        left = Inches(0.55 + (i % 2) * 3.75)
        top = Inches(1.6 + (i // 2) * 2.62)
        titled_box(slide, left, top, Inches(3.6), Inches(2.5), fill, border, title,
                   [(body, {"size": 9.5})], title_size=11.5, body_size=9.5)
    picture(slide, "compliance_by_arm", Inches(8.15), Inches(1.55), height=Inches(3.5))
    box(slide, Inches(8.15), Inches(5.12), Inches(4.7), Inches(1.75), GREY_FILL, GREY)
    frame = textbox(slide, Inches(8.3), Inches(5.17), Inches(4.4), Inches(1.65))
    write(frame, [
        [("A separate failure mode: the gate-count contract  ", {"bold": True, "color": NAVY}),
         ("= every proposal must contain exactly 3 rotations and 1 CNOT per qubit; "
          "a proposal that does not is replaced by a flagged random circuit and costs "
          "one of the B evaluations.", {})],
        [("Which call breaks depends on the condition, not the method: ", {"bold": True}),
         (f"with Model B the open batch fell to "
          f"{arm_validity('model_alt', 'LLM-Open') * 100:.0f}% valid (closed loop "
          f"{arm_validity('model_alt', 'LLM-Closed') * 100:.0f}%); at 6 qubits the closed "
          f"loop fell to {arm_validity('qubits_n6', 'LLM-Closed') * 100:.0f}% (open batch "
          f"{arm_validity('qubits_n6', 'LLM-Open') * 100:.0f}%). ", {}),
         ("So part of an apparent method effect at larger registers or with a weaker "
          "model is a proposal-validity effect.", {"bold": True, "color": KICKER})],
    ], size=9, space_after=3)
    footer(slide, "Not claimed: that the closed loop is bad in general, or that any one "
                  "ingredient causes the semantic advantage. Greedy and LLM-Closed differ in "
                  "search freedom as well as physics knowledge, by design.")


def slide_09_limits_next(deck, summary):
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    header(slide, "8 · Limitations and next action",
           "What this study cannot say, and the concrete next step",
           "Limitations of the evidence on the left; the next experiment, phrased as a "
           "measurable target, on the right")
    titled_box(slide, Inches(0.55), Inches(1.6), Inches(5.9), Inches(5.25), GREY_FILL, GREY,
               "Limitations", [
        "•  12 paired seeds per cell: enough for large effects, underpowered for small "
        "ones — a \"not detected\" cell is not evidence of absence.",
        "•  One factor at a time: no interaction was measured (for example a large "
        "budget at 8 qubits); 4 of 9 Ising cells and 8 of 9 XXZ cells are empty.",
        "•  Two Hamiltonian families and two language models are two points each, not "
        "a survey; Model B was run only in the baseline cell.",
        "•  Noiseless state-vector simulation: no sampling noise, no device connectivity.",
        "•  The circuit width grows with the qubit count by a fixed rule, so \"more "
        "qubits\" also means \"more gates\"; the two were not separated.",
        "•  Greedy and LLM-Closed differ in search freedom as well as in physics "
        "knowledge, so that pair is not an information-matched comparison.",
        "•  Simulation cost, not model cost, is the binding constraint: state-vector "
        "training grows exponentially with the qubit count.",
    ], title_color=NAVY, body_size=11, space_after=8)
    box(slide, Inches(6.7), Inches(1.6), Inches(6.05), Inches(5.25), AMBER_FILL, AMBER, 1.75)
    frame = textbox(slide, Inches(6.88), Inches(1.68), Inches(5.7), Inches(5.1))
    write(frame, [
        [("NEXT ACTION — target accuracy and minimum required budget",
          {"bold": True, "color": ORANGE, "size": 12})],
        [("So far every comparison asks \"which method is best at a fixed budget B?\". "
          "The practical question is the reverse:", {})],
        [("1  Target accuracy.  ", {"bold": True, "color": KICKER}),
         ("Fix, before any run, the held-out trash fidelity the compressed state must "
          "reach for its downstream use — a target F_target (candidate levels 0.95 and "
          "0.99, to be chosen with the application in mind).", {})],
        [("2  Minimum required budget.  ", {"bold": True, "color": KICKER}),
         ("For each method and condition, measure B_min = the smallest budget at which "
          "the selected circuit reaches F_target in at least 10 of the 12 paired seeds. "
          "Sweep B ∈ {2, 4, 8, 16, 32} at 4 and 8 qubits for both Hamiltonians and both "
          "models.", {})],
        [("What already exists: ", {"bold": True}),
         ("Random's candidates are nested across budgets, so its B_min for B ≤ 16 can be "
          "read from the logged runs without new simulation; the LLM methods need new "
          "runs because their batches change with B.", {})],
        [("Deliverable: ", {"bold": True}),
         ("one table of B_min per (method, qubit count, Hamiltonian, model) at each "
          "F_target, pre-registered like this study, with the same paired-seed "
          "statistics.", {})],
    ], size=11, space_after=8)
    footer(slide, "Also still open, in order: a full factorial grid (interactions), more "
                  "Hamiltonian families and models, sampling noise and hardware topology.")


def slide_10_conclusion(deck, summary):
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    header(slide, "9 · Conclusion", "What is robust, what is fragile, what is open")
    n = len(VARIED)
    open_same = sum(1 for k in VARIED
                    if stat(summary, k, OPEN_RANDOM)["mean_paired_gain"] > 0)
    open_sig = sum(1 for k in VARIED if verdict(summary, k, OPEN_RANDOM) == "holds")
    closed_pos = sum(1 for k in VARIED if verdict(summary, k, CLOSED_OPEN) == "holds")
    closed_neg = sum(1 for k in VARIED if verdict(summary, k, CLOSED_OPEN) == "reverses")
    worst = min(VARIED + ["reference"], key=lambda k: valid_share(summary, k))
    items = [
        (GREEN_FILL, GREEN, "Robust: physics-informed proposals beat non-semantic search",
         f"Same sign in {open_same} of {n} varied conditions (significant in {open_sig}) — "
         "a different budget, a wider register, a different Hamiltonian, a different "
         "language model. This is no longer a property of one setting."),
        (RED_FILL, RED, "Fragile: the closed loop's edge over the open batch",
         f"Significantly positive in {closed_pos} of {n} varied conditions and "
         f"significantly negative in {closed_neg}: it helps at small budgets and on the "
         "XXZ chain, vanishes at a large budget, and reverses at 6 and 8 qubits. The "
         "claim must be restated as budget- and width-dependent."),
        (AMBER_FILL, AMBER, "New and measurable: proposal validity is its own bottleneck",
         f"As few as {valid_share(summary, worst) * 100:.0f}% of evaluated proposals met "
         "the gate-count contract in the weakest cell, and each invalid proposal is spent "
         "as a random draw. Fixing generation validity is cheaper than richer feedback."),
        (BLUE_FILL, BLUE, "Open: the budget the application actually needs",
         "Next: fix a target accuracy and measure the minimum budget each method needs "
         "to reach it — the question that decides whether any of this is worth using."),
    ]
    for i, (fill, border, title, body) in enumerate(items):
        top = Inches(1.5 + i * 1.22)
        titled_box(slide, Inches(0.55), top, Inches(12.2), Inches(1.1), fill, border,
                   title, [(body, {"size": 11})], title_size=13, body_size=11)
    box(slide, Inches(0.55), Inches(6.48), Inches(12.2), Inches(0.5), NAVY, NAVY)
    frame = textbox(slide, Inches(0.7), Inches(6.53), Inches(11.95), Inches(0.42))
    write(frame, [[("One-line message:  ", {"bold": True, "color": AMBER}),
                   ("the language model's advantage is in the proposals it writes, and "
                    "that advantage survived every single-factor change; the feedback "
                    "loop's advantage did not.", {"color": WHITE})]], size=11.5,
          space_after=0)
    frame = textbox(slide, Inches(0.55), Inches(7.05), Inches(12.2), Inches(0.3))
    write(frame, [("All figures and statistics come from the committed result tables via "
                   "the committed build scripts; 12 paired seeds per cell; held-out test "
                   "read once after validation-only selection.",
                   {"size": 8.5, "color": MUTED})], space_after=0)


# ------------------------------------------------------------------ QA ----

# term -> (regex that detects a USE, regex that detects the DEFINITION)
GLOSSARY = {
    "QAE": (r"\bQAE\b", r"[Qq]uantum autoencoder \(QAE\)"),
    "trash fidelity": (r"F_trash|trash fidelity", r"trash fidelity:\s+F_trash ="),
    "open-loop": (r"open-loop", r"open-loop: the language model writes"),
    "closed-loop": (r"closed-loop", r"closed-loop: B/2 proposals"),
    "budget B": (r"\bB\s*=\s*\d|budget B\b|\bB/2\b", r"B = the evaluation budget"),
    "Ising": (r"\bIsing\b", r"H = −Σ Zᵢ Zᵢ₊₁ − h Σ Xᵢ"),
    "XXZ": (r"\bXXZ\b", r"XXZ Heisenberg chain\s+H = Σ"),
    "Model A": (r"\bModel A\b", r"Model A = the reference model|Model A\b.*= the reference model"),
    "Model B": (r"\bModel B\b", r"Model B\b.*= the alternative model"),
    "CI": (r"\bCI\b", r"confidence interval \(CI\)"),
    "seeds": (r"\bseeds?\b", r"12 paired seeds\s*\(12 independent draws"),
    "Wilcoxon": (r"Wilcoxon", r"p = exact paired Wilcoxon test"),
    "CNOT": (r"\bCNOT\b", r"CNOT \(controlled-NOT\)"),
    "latent": (r"\blatent\b", r"Latent\s+q0, q1"),
    "trash qubits": (r"\btrash\b", r"Trash\s+q2, q3"),
    "held-out": (r"held-out", r"held-out test set \(64 unseen h values"),
    "gate-count contract": (r"gate-count contract", r"exactly 3 rotations and 1 CNOT per qubit"),
    "exploration/refinement": (r"\bexploration\b|\brefinement\b",
                               r"\(exploration, no feedback\)"),
    "one factor at a time": (r"one factor at a time|one-factor-at-a-time",
                             r"exactly one declared factor per condition"),
    "F_target": (r"F_target", r"a target F_target"),
    "B_min": (r"B_min", r"B_min = the smallest budget"),
}


def slide_texts(path: Path) -> list[str]:
    deck = Presentation(str(path))
    texts = [""] * len(deck.slides._sldIdLst)
    for index, text in extract_text(path):
        texts[index - 1] += text + "\n"
    return texts


def glossary_audit(texts: list[str]) -> list[str]:
    problems = []
    for term, (use, definition) in GLOSSARY.items():
        first_use = next((i for i, t in enumerate(texts) if re.search(use, t)), None)
        defined = next((i for i, t in enumerate(texts) if re.search(definition, t)), None)
        if first_use is None:
            continue
        if defined is None:
            problems.append(f"'{term}' is used on slide {first_use + 1} but never defined")
        elif defined > first_use:
            problems.append(f"'{term}' is used on slide {first_use + 1} before its "
                            f"definition on slide {defined + 1}")
    return problems


def structure_audit(path: Path, texts: list[str]) -> list[str]:
    problems = []
    deck = Presentation(str(path))
    expected = ["question", "previous baseline", "controlled protocol",
                "how to read the result grids", "results — hamiltonian 1",
                "results — hamiltonian 2", "cross-grid summary", "interpretation",
                "limitations and next action", "conclusion"]
    for index, (needle, text) in enumerate(zip(expected, texts, strict=True), start=1):
        if needle.lower() not in text.lower():
            problems.append(f"slide {index}: expected the section '{needle}'")
    # the two grids, six categories each
    for index, _figure in ((5, "grid_tfim"), (6, "grid_xxz")):
        slide = deck.slides[index - 1]
        pictures = [s for s in slide.shapes if s.shape_type == 13]
        if len(pictures) != 1:
            problems.append(f"slide {index}: expected exactly one grid picture")
    coverage = json.loads((ROOT / "grid_cells.json").read_text())
    if coverage["categories"] != [
        "Random", "Greedy", "LLM-Open Model A", "LLM-Closed Model A",
        "LLM-Open Model B", "LLM-Closed Model B"]:
        problems.append("grid figures do not use the six required categories in order")
    for family in ("TFIM", "XXZ"):
        if len(coverage["coverage"][family]) != 9:
            problems.append(f"{family} grid does not have 9 cells")
    # next action must name both items
    next_text = texts[8].lower()
    for phrase in ("target accuracy", "minimum required budget"):
        if phrase not in next_text:
            problems.append(f"slide 9: next action does not mention '{phrase}'")
    return problems


def render_pdf(pptx_path: Path) -> Path:
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir",
                    str(pptx_path.parent), str(pptx_path)],
                   check=True, capture_output=True, timeout=600)
    return pptx_path.with_suffix(".pdf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--no-pdf", action="store_true")
    args = parser.parse_args()
    DECK_DIR.mkdir(parents=True, exist_ok=True)
    pptx_path = DECK_DIR / f"{DECK_NAME}.pptx"

    if not args.check:
        summary = json.loads((ROOT / "summary.json").read_text())
        deck = Presentation()
        deck.slide_width, deck.slide_height = SLIDE_W, SLIDE_H
        for builder in (slide_01_title, slide_02_baseline, slide_03_protocol,
                        slide_04_reading, slide_05_grid_ising, slide_06_grid_xxz,
                        slide_07_summary, slide_08_interpretation,
                        slide_09_limits_next, slide_10_conclusion):
            builder(deck, summary)
        deck.save(str(pptx_path))
        print("wrote", pptx_path)

    texts = slide_texts(pptx_path)
    problems = geometry_audit(pptx_path) + glossary_audit(texts) + \
        structure_audit(pptx_path, texts)
    if problems:
        print("\nAUDIT FAILED:")
        for problem in problems:
            print("  -", problem)
    else:
        print("audit passed: 10 slides, required structure, both grids with six "
              "categories, next action complete, every term defined before use, "
              "nothing off-canvas, no prohibited strings")
    if not args.no_pdf and not args.check:
        print("wrote", render_pdf(pptx_path))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
