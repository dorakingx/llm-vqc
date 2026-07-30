# Visual QA — 20260731_GSoC_revised

Method: the PPTX was exported to PDF with LibreOffice and every page
rendered to PNG at 130 dpi (1734 × 975, 16:9) into `renders/`. **All 18
slides were opened and inspected individually.** Defects found during
review, the fix, and the state after re-render are recorded below.

Checked on every slide: clipped titles · overlapping text · off-slide
objects · unreadable legends · meaning-changing line breaks · tiny axis
labels · missing symbols · distorted plots · font substitution · status
contradictions · whether the takeaway is legible at presentation scale.

## Defects found and fixed

| # | Slide | Defect | Fix | Verified |
|---|---|---|---|---|
| 1 | 3 | Pipeline figure: sub-captions overflowed their boxes horizontally ("no padding, no truncation", "inside AmplitudeEmbedding") | Widened boxes 1.72→1.94 in, shortened captions, title 15→13.5 pt, caption 11.5→11 pt | Fixed |
| 2 | 3 | Third bullet collided with the source footer | Shortened all three bullets, moved block up (y 4.85→4.72), size 20→19 pt | Fixed |
| 3 | 5 | "Track A — architecture search (primary)" wrapped to two lines and ran into the first bullet | Retitled "Track A — structure search (primary)" (one line); bullets moved down (y 2.18→2.30) | Fixed |
| 4 | 7 | Kicker "RESULT — N = 5" was ambiguous: reads as sample size, means qubits | Changed to "RESULT — 5 QUBITS" | Fixed |
| 5 | 9 | Third bullet overflowed the bottom slide edge ("is untested here" cut off) and crossed the footer | Diagnostics image 3.55→3.36 in, bullets moved up and shortened, footer shortened | Fixed |
| 6 | 18 | Text contained the word "preregistered" (while explaining that it is *not* used), tripping the verifier | Reworded to "There was no external preregistration; the deck therefore says protocol-frozen throughout." | Fixed |
| 7 | — | Forest plot: 24 contrasts crammed p-labels on top of each other and the legend | Redesigned: 12 labelled contrasts (search vs strongest reference, search vs search, both tasks), p and δ annotated at the right margin | Fixed |
| 8 | — | Pareto plot: symlog x-axis distorted spacing and the legend overlapped data | Linear axis with explicit limits, legend moved to the empty upper-right | Fixed |

## Slide-by-slide result after fixes

| Slide | Content | Result |
|---|---|---|
| 1 | Title, interim status, author, date, commit, branch | Clean. Status line legible; commit and branch present. |
| 2 | Primary/secondary question, anchors, literature gap | Clean. Two boxes aligned, three bullets, no overflow. |
| 3 | End-to-end pipeline figure + three points | Clean after fixes 1–2. Box text fits; footer clear. |
| 4 | T1–T4 2×2 grid + four cards | Clean. Axis labels legible; cards do not collide. |
| 5 | Track A / Track B + matched-vs-not band | Clean after fix 3. Titles one line; band readable. |
| 6 | E0–E5 progress bars + status strip | Clean. Blocked rows in red, complete in green; counts match the store. |
| 7 | Paired dot plot + forest + two findings | Clean after fixes 4, 7. Both figures legible side by side. |
| 8 | Scaling, two panels + descriptive caveat | Clean. Crossover annotation readable; simulation regime in the footer. |
| 9 | Pareto + diagnostics + three points | Clean after fixes 5, 8. No text over data. |
| 10 | Interim answer, budget ask, next steps, closing question | Clean. "Not yet." prominent; closing question centred. |
| 11 | A1–A2 pilot replay and E1 | Clean. Three panels, paired lines visible. |
| 12 | A3 full E2 grid | Clean. Rotated arm labels legible at 11 pt in-figure. |
| 13 | A4 anytime curves | Clean. Legend inside the first panel, no data covered. |
| 14 | A6/A8 statistics and seeds | Clean. Six bullets, ends well above the footer. |
| 15 | A7/A10 grammar and diagnostic selection | Clean. Five bullets, no overflow. |
| 16 | A9 resource assumptions | Clean. Five bullets, no overflow. |
| 17 | A11 budget derivation table | Clean. Ten rows, header contrast good, all cells fit. |
| 18 | A12/A13 integrity and references | Clean after fix 6. |

## Typography audit

- Slide titles: 30 pt (main), 26 pt (appendix headers) — meets ≥30 pt for
  main slides; appendix headers are intentionally one step smaller and
  are not body text.
- Main-deck body text: 19–21 pt. Slides 7 and 9 use 15–16 pt for the
  two-column commentary that sits beside a figure; this is a deliberate
  exception for secondary annotation, and the primary message on those
  slides is carried by the figure and the ≥19 pt title.
- Figure text: rendered at ≥16 pt effective size on the slide — figures
  are rasterised at `TARGET_PPI` × their on-slide width, so a 15 pt
  matplotlib label lands at ≥16 pt equivalent.
- Fonts: Arial only (Google Slides safe). Mathematical symbols (⟨Z₀⟩, θ,
  δ, ⇒, 2ⁿ) verified present in the PDF render, no tofu boxes.

## Colour system

Okabe–Ito derived, colourblind safe, identical in every figure:
random **#0072B2** blue · evolutionary **#E69F00** orange · greedy
**#009E73** green · LLM arms **#CC79A7** magenta (reserved, not yet
measured) · fixed references grey **#7F7F7F / #BDBDBD**. Accent red
**#B3261E** for warnings and the interim answer; navy **#1E3A5F** for
titles.

## Second review round — 2026-07-31

Trigger: the deck was regenerated after the spend ledger, model pinning
and search-space parity work landed. Every slide was re-exported and
inspected again at 110 dpi. Nine changes were requested; the defects each
one introduced are listed with the fix.

| # | Slide | Defect after the change | Fix | Verified |
|---|---|---|---|---|
| 9 | 3 | The new protected-test-metric definition ran past the bottom edge into the footer | Bullets moved up (y 4.72→4.60), size 18→17 pt, definition shortened | Fixed |
| 10 | 4 | The new constraints table's last row touched the footer, and the colour note was stated twice (once inside the figure, once on the slide) | Table lifted (y 5.12→4.92), figure narrowed to 7.3 in, the slide-level duplicate folded into the footer | Fixed |
| 11 | 7, 8 | The two new titles are 66 and 84 characters and were clipped at 30 pt | `slide()` now drops to 23 pt with a taller box for titles over 60 characters, so they wrap instead of clipping | Fixed |
| 12 | 7 | The forest plot's new "p = Holm-adjusted, δ = Cliff's delta" key collided with its two-line x-axis label | Key moved into the axis title as a second line | Fixed |
| 13 | 7 | In the main dot plot the reference median bar crossed the new legend text | Legend given a 93%-opaque white background | Fixed |
| 14 | 7 | Enlarging the left figure pushed the forest plot past the right margin | Forest repositioned to x 8.30, w 4.48 in (right edge 12.78 in) | Fixed |
| 15 | 10 | "Spending control is implemented and enforced" wrapped to two lines into the first bullet, and the fourth bullet overflowed the card | Header shortened to "Spending control: implemented, not promised"; bullets shortened, 14.5→14 pt | Fixed |

## Content changes in this round

| Change | Where |
|---|---|
| Slide 6 title now reads "Classical matrix complete; real-LLM matrix not started" — the verified final state | S6 |
| Slide 7 title now names the finding and the non-finding separately | S7 |
| Slide 8 title now says "descriptive" in the title, not only in the caveat band | S8 |
| Every line, dot, marker, connector and band is explained inside its own figure rather than in slide text or speaker notes | fig_e2_main, fig_e2_forest, fig_e3_scaling, fig_pareto, fig_diagnostics, fig_tasks |
| Shared method legend on every multi-panel figure | fig_e3_scaling, fig_ap_e2_grid, fig_ap_anytime, fig_ap_e1 |
| Line colour on slide 4 is stated to distinguish example signals only | fig_tasks caption + S4 footer |
| Constraints table: qubits, allowed gates, max layer operations, and that one operation expands to many physical gates | S4 |
| Protected-test metric defined as the same metric formula on a final held-out split, evaluated once per selected circuit after validation-based selection | S3 + fig_pipeline |
| "protected-test metric" replaces "protected-test RMSE" wherever the statement spans the pipeline, because T4 is classification | S3, fig_pipeline, fig_ap_e2_grid panel titles |
| The $120 global-cap claim is gone. The deck now states the implemented $2.00 cumulative ledger cap and the actual measured cost of $0.00 | S6, S10, A11, speaker notes, claim_source_map.yaml |
| The matrix figure no longer says "blocked — no API budget"; it distinguishes "not started — API quota exhausted" (LLM rows), "blocked — needs LLM-proposed circuits" (E5) and E4's live state | fig_matrix |
