#!/usr/bin/env python
"""Diagrams that replace prose on the setup slides.

A slide should not ask the audience to read a paragraph while the speaker
talks over it. These three figures carry what used to be bullet lists:
the model contract, the three run conditions, and the five arms. Each
defines its own terms in place, so a reader meeting "closed-loop" or
"budget B" for the first time can resolve it without a glossary slide.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
FIG = REPO / "docs" / "presentation" / "mini5" / "figures"

NAVY, INK, MUTED, ACCENT = "#1E3A5F", "#1A1A2E", "#55606E", "#B3261E"
GREY, GOLD, GREEN, BLUE, PINK = "#7F7F7F", "#E69F00", "#009E73", "#0072B2", "#CC79A7"
BG, RULE = "#F4F6FA", "#D8DDE6"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "text.color": INK, "figure.facecolor": "white",
})


def _save(fig, stem):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{stem}.png", bbox_inches="tight", dpi=170,
                facecolor="white")
    plt.close(fig)
    print(f"  wrote figures/{stem}.png")


def fig_contract():
    """The whole model contract as a pipeline, with each term defined where
    it appears."""
    fig, ax = plt.subplots(figsize=(13.0, 3.0))
    ax.axis("off")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 3)

    steps = [
        ("Signal", "8 or 32 numbers\n(one task sample)", BG, RULE),
        ("Amplitude\nencoding", "the numbers become\nthe qubit state\ndirectly", BG, RULE),
        ("5-gate body", "THE ONLY PART\nTHAT DIFFERS\nbetween methods", "#FDF0E4", GOLD),
        ("Measure Z\non qubit 0", "one number\nout of the circuit", BG, RULE),
        ("Prediction", "$\\hat{y}=(1-\\langle Z_0\\rangle)/2$\nin [0, 1]", "#E8F5EE", GREEN),
    ]
    w, gap = 2.28, 0.35
    for i, (title, sub, face, edge) in enumerate(steps):
        x = 0.15 + i * (w + gap)
        ax.add_patch(mp.FancyBboxPatch((x, 0.85), w, 1.5,
                                       boxstyle="round,pad=0.02,rounding_size=0.08",
                                       facecolor=face, edgecolor=edge, linewidth=1.8))
        ax.text(x + w / 2, 1.98, title, ha="center", va="center",
                fontsize=13, weight="bold", linespacing=1.15)
        ax.text(x + w / 2, 1.32, sub, ha="center", va="center",
                fontsize=10.5, color=MUTED, linespacing=1.35)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + w + gap - 0.05, 1.6), xytext=(x + w + 0.05, 1.6),
                        arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 1.6})

    ax.text(6.5, 0.42, "No training anywhere: the angles a method proposes are used exactly as written.   "
                       "No classical layers: nothing outside the circuit can fix a bad circuit.",
            ha="center", fontsize=12, color=ACCENT, weight="bold")
    _save(fig, "fig_contract")


def fig_conditions():
    """Three runs, one picture. Replaces two columns of bullets."""
    fig, ax = plt.subplots(figsize=(12.6, 3.6))
    ax.axis("off")
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0, 3.6)

    def card(x, w, title, gates, budget, seeds, note, face, edge, tcol):
        ax.add_patch(mp.FancyBboxPatch((x, 0.5), w, 2.7,
                                       boxstyle="round,pad=0.02,rounding_size=0.1",
                                       facecolor=face, edgecolor=edge, linewidth=2))
        ax.text(x + w / 2, 2.92, title, ha="center", fontsize=12.5,
                weight="bold", color=tcol)
        for k, (label, value) in enumerate(
            [("gates per circuit", gates), ("budget B", budget), ("seeds", seeds)]
        ):
            y = 2.42 - k * 0.55
            ax.text(x + 0.25, y, label, ha="left", fontsize=10.5, color=MUTED)
            ax.text(x + w - 0.25, y, value, ha="right", fontsize=13,
                    weight="bold", color=INK)
        ax.text(x + w / 2, 0.78, note, ha="center", fontsize=10, color=MUTED,
                style="italic")

    card(0.1, 3.2, "LAST WEEK (pilot)", "1 to 5", "4", "2", "different task generator",
         "#F0F0F2", GREY, GREY)
    ax.annotate("", xy=(3.72, 1.85), xytext=(3.42, 1.85),
                arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 2})
    card(3.85, 2.7, "PRIMARY", "5 fixed", "8", "10", "slides 6, 7, 9",
         "#EAF0FB", NAVY, NAVY)
    card(6.75, 2.7, "CONTROL A", "1 to 5", "8", "10", "slide 8", BG, RULE, MUTED)
    card(9.65, 2.7, "CONTROL B", "1 to 5", "4", "10", "slide 8", BG, RULE, MUTED)
    ax.add_patch(mp.FancyBboxPatch((6.7, 0.42), 5.8, 2.86,
                                   boxstyle="round,pad=0.02,rounding_size=0.1",
                                   facecolor="none", edgecolor=ACCENT,
                                   linewidth=1.6, linestyle=(0, (5, 3))))
    ax.text(9.6, 0.1, "controls: rebuild the pilot's freedom to test what it bought",
            ha="center", fontsize=11, color=ACCENT, weight="bold")
    _save(fig, "fig_conditions")


def fig_arms():
    """Five arms, what each one does, and which ones cost money."""
    fig, ax = plt.subplots(figsize=(13.1, 3.3))
    ax.axis("off")
    ax.set_xlim(0, 13.1)
    ax.set_ylim(0, 3.3)

    arms = [
        ("Random", GREY, "draw 5 gates\nat random", "no"),
        ("Evolutionary", GOLD, "keep the best,\nmutate them", "no"),
        ("Greedy", GREEN, "mutate the best;\nkeep only if\nit improves", "no"),
        ("LLM\nopen-loop", BLUE, "ask the model,\nshow it nothing", "1 call each"),
        ("LLM\nclosed-loop", PINK, "ask the model,\nshow it its own\nscores so far", "1 call each"),
    ]
    w, gap = 2.24, 0.35
    for i, (name, colour, how, api) in enumerate(arms):
        x = 0.15 + i * (w + gap)
        ax.add_patch(mp.FancyBboxPatch((x, 0.75), w, 2.15,
                                       boxstyle="round,pad=0.02,rounding_size=0.09",
                                       facecolor="white", edgecolor=colour, linewidth=2.2))
        ax.add_patch(mp.Rectangle((x, 2.55), w, 0.35, facecolor=colour,
                                  edgecolor=colour))
        ax.text(x + w / 2, 2.72, name.replace("\n", " "), ha="center", va="center",
                fontsize=11.5, weight="bold", color="white")
        ax.text(x + w / 2, 1.85, how, ha="center", va="center", fontsize=11,
                color=INK, linespacing=1.4)
        ax.text(x + w / 2, 1.02, f"API: {api}", ha="center", fontsize=10,
                color=ACCENT if api != "no" else MUTED)

    ax.text(6.55, 0.35,
            "All five get the same budget B: the same number of DISTINCT circuits actually scored. "
            "Repeats and malformed proposals are free but earn nothing.",
            ha="center", fontsize=11.5, color=INK)
    _save(fig, "fig_arms")


if __name__ == "__main__":
    fig_contract()
    fig_conditions()
    fig_arms()
