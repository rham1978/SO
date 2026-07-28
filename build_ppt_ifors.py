#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_ppt_ifors.py
════════════════════════════════════════════════════════════════════════
Construye una versión EDITABLE en PowerPoint del deck IFORS 2026
(Araneda & Carrasco), espejando la estructura del Beamer .tex y aplicando
las recomendaciones del relato (SMAC-SK como protagonista).

Figuras propias (Pareto, convergencia, costo, heatmap) se embeben desde
reporte_figs/. Las fotos del .tex que no están aquí (listas_espera.png,
proceso_atencion.png, macro.png, etc.) se dejan como PLACEHOLDER para que
las pegues tú.

Uso:  python build_ppt_ifors.py [--out IFORS2026_Araneda.pptx]
"""
from __future__ import annotations
import argparse
import os

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

FIG = "reporte_figs"
NAVY = RGBColor(0x17, 0x3F, 0x8A)
BLUE = RGBColor(0x00, 0x76, 0xDE)
GREEN = RGBColor(0x1A, 0x6B, 0x1A)
RED = RGBColor(0xC0, 0x39, 0x2B)
GRAY = RGBColor(0x66, 0x66, 0x66)
LIGHT = RGBColor(0xEE, 0xF2, 0xF7)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
SW, SH = prs.slide_width, prs.slide_height
BLANK = prs.slide_layouts[6]


def slide():
    return prs.slides.add_slide(BLANK)


def title(s, text, sub=None):
    tb = s.shapes.add_textbox(Inches(0.45), Inches(0.25), Inches(12.4), Inches(1.0))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = text
    p.font.size = Pt(28); p.font.bold = True; p.font.color.rgb = NAVY
    if sub:
        p2 = tf.add_paragraph(); p2.text = sub
        p2.font.size = Pt(13); p2.font.color.rgb = GRAY
    return s


def bullets(s, items, left=0.7, top=1.5, w=12.0, h=5.4, size=18):
    tb = s.shapes.add_textbox(Inches(left), Inches(top), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    for i, it in enumerate(items):
        txt, lvl = it if isinstance(it, tuple) else (it, 0)
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ("• " if lvl == 0 else "– ") + txt
        p.level = lvl
        p.font.size = Pt(size - 3 * lvl)
        p.space_after = Pt(8)
        if lvl == 0:
            p.font.bold = True
    return s


def img(s, path, left, top, w=None, h=None):
    kw = {}
    if w: kw["width"] = Inches(w)
    if h: kw["height"] = Inches(h)
    if os.path.exists(path):
        s.shapes.add_picture(path, Inches(left), Inches(top), **kw)
    else:
        placeholder(s, f"[FIGURA: {os.path.basename(path)} — pegar aquí]",
                    left, top, w or 8.0, h or 4.5)


def placeholder(s, text, left, top, w, h):
    from pptx.enum.shapes import MSO_SHAPE
    sh = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(left), Inches(top),
                            Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = LIGHT
    sh.line.color.rgb = GRAY; sh.line.dash_style = 2
    tf = sh.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = text; p.alignment = PP_ALIGN.CENTER
    p.font.size = Pt(14); p.font.color.rgb = GRAY; p.font.italic = True
    return sh


def table(s, data, left, top, w, h, fontsize=11, highlight_last=False):
    rows, cols = len(data), len(data[0])
    t = s.shapes.add_table(rows, cols, Inches(left), Inches(top),
                           Inches(w), Inches(h)).table
    for r in range(rows):
        for c in range(cols):
            cell = t.cell(r, c); cell.text = str(data[r][c])
            for para in cell.text_frame.paragraphs:
                para.font.size = Pt(fontsize)
                para.alignment = PP_ALIGN.CENTER if c > 0 else PP_ALIGN.LEFT
                if r == 0:
                    para.font.bold = True; para.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            if r == 0:
                cell.fill.solid(); cell.fill.fore_color.rgb = NAVY
            elif highlight_last and r == rows - 1:
                cell.fill.solid(); cell.fill.fore_color.rgb = RGBColor(0xDD, 0xE6, 0xF5)
    return t


def framework_diagram(s):
    """3-bloque DES → SMAC-SK → Output, con flecha de re-evaluación."""
    from pptx.enum.shapes import MSO_SHAPE
    ys, hb, wb = 2.6, 1.9, 3.0
    xs = [1.3, 5.15, 9.0]
    labels = [("DES\nEvaluator", "Black-box stochastic\nevaluator", LIGHT, NAVY),
              ("SMAC-SK", "Core contribution\n(recommended method)", RGBColor(0xE9, 0xF7, 0xEE), GREEN),
              ("Prescriptive\nOutput", "Applied to\nCRS Cordillera", LIGHT, NAVY)]
    for x, (t1, t2, fc, ec) in zip(xs, labels):
        sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(ys),
                                Inches(wb), Inches(hb))
        sh.fill.solid(); sh.fill.fore_color.rgb = fc
        sh.line.color.rgb = ec; sh.line.width = Pt(2.2)
        tf = sh.text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]; p.text = t1; p.alignment = PP_ALIGN.CENTER
        p.font.size = Pt(20); p.font.bold = True; p.font.color.rgb = ec
        cap = s.shapes.add_textbox(Inches(x), Inches(ys + hb + 0.05), Inches(wb), Inches(0.8))
        cp = cap.text_frame; cp.word_wrap = True
        q = cp.paragraphs[0]; q.text = t2; q.alignment = PP_ALIGN.CENTER
        q.font.size = Pt(11); q.font.italic = True; q.font.color.rgb = GRAY
    # arrows
    for x0, x1 in [(xs[0] + wb, xs[1]), (xs[1] + wb, xs[2])]:
        con = s.shapes.add_connector(2, Inches(x0), Inches(ys + hb / 2),
                                     Inches(x1), Inches(ys + hb / 2))
        con.line.color.rgb = NAVY; con.line.width = Pt(3)
    lab = s.shapes.add_textbox(Inches(xs[0] + wb), Inches(ys + hb / 2 - 0.5), Inches(2.0), Inches(0.4))
    lab.text_frame.paragraphs[0].text = "f(x) ± σ(x)"
    lab.text_frame.paragraphs[0].font.size = Pt(12)
    lab2 = s.shapes.add_textbox(Inches(xs[1] + wb), Inches(ys + hb / 2 - 0.5), Inches(2.0), Inches(0.4))
    lab2.text_frame.paragraphs[0].text = "optimal x*"
    lab2.text_frame.paragraphs[0].font.size = Pt(12)
    note = s.shapes.add_textbox(Inches(0.7), Inches(6.4), Inches(12), Inches(0.6))
    note.text_frame.paragraphs[0].text = ("Benchmark: 7 algorithms × 15 seeds, ~150 simulator "
                                          "evaluations, Common Random Numbers (CRN); incumbent re-evaluated n=50.")
    note.text_frame.paragraphs[0].font.size = Pt(12); note.text_frame.paragraphs[0].font.italic = True


# ════════════════════════════════════════════════════════════════════════
# SLIDES
# ════════════════════════════════════════════════════════════════════════

# 1 Title
s = slide()
tb = s.shapes.add_textbox(Inches(0.8), Inches(2.1), Inches(11.7), Inches(2.4))
tf = tb.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "A Prescriptive Simulation–Hybrid Optimization Framework for Capacity Planning in Minimally Invasive Gynecology"
p.font.size = Pt(30); p.font.bold = True; p.font.color.rgb = NAVY
for line in ["Raúl H. Araneda, Rodrigo A. Carrasco",
             "Department of Industrial and Systems Engineering — UC Chile",
             "IFORS 2026 — Vienna, Austria   ·   Presented by Raúl H. Araneda",
             "Research partially funded by Fondecyt Regular 1231245"]:
    q = tf.add_paragraph(); q.text = line; q.font.size = Pt(15); q.font.color.rgb = GRAY

# 2 Outline
s = slide(); title(s, "Outline")
bullets(s, ["Motivation", "Research Problem", "Literature Review", "Objective",
            "Proposed Framework", "Results", "Discussion", "References"], size=20)

# 3 Motivation
s = slide(); title(s, "Waiting kills — and Chile is no exception", "Motivation")
bullets(s, [
    "Prolonged waits deteriorate patient health and increase mortality by 3.4% (Martinez et al., 2019).",
    "Prolonged waits reduce wages and labor productivity (OECD, 2026).",
], top=1.4, w=12.2, size=17, h=1.7)
img(s, f"{FIG}/listas_espera.png", 3.85, 3.15, w=5.6, h=3.9)

# 4 Case Study I
s = slide(); title(s, "Inside one clinic: where the bottleneck lives", "Case Study")
bullets(s, [
    ("Institutional Relevance", 0),
    ("CRSCO is a medium-complexity public ambulatory center, a key referral hub for Macul and Peñalolén.", 1),
    ("Demand", 0),
    ("More than 30,000 consultations in 2025 — growing pressure on specialized services.", 1),
    ("Gynecology Department Burden", 0),
    ("30% of the total waiting list; 3,859 pending cases across 11 specialized clinics.", 1),
], size=17)

# 5 Case Study II
s = slide(); title(s, "The victim: 318 cases, up to 1,164 days waiting", "Case Study")
bullets(s, [
    ("Waiting Time Metrics", 0),
    ("Average waiting time: 205 days; maximum recorded: 1,164 days.", 1),
    ("Minimally Invasive Gynecology (CMI)", 0),
    ("Fifth largest subspecialty list — 318 active cases, average wait 81 days.", 1),
], size=17)

# 6 Problem Definition
s = slide(); title(s, "The question: can we make fewer patients wait?", "Problem Definition")
img(s, f"{FIG}/proceso_atencion.png", 1.8, 1.3, w=9.6, h=3.9)
bullets(s, [
    "Absence of an analytical framework able to represent and support, in an integrated way, "
    "capacity-planning decisions under uncertainty.",
    "Research question: how do I optimize this process so fewer patients wait?",
], top=5.45, w=12.4, size=15, h=1.6)

# 6b System model (BPMN) — the modeling that reveals the challenges
s = slide(); title(s, "The scene: modeling the gynecology macro-process",
                   "BPMN model of the system — the source of the five challenges")
img(s, f"{FIG}/macro.png", 1.0, 1.5, w=11.3, h=5.3)

# 7 Challenges
s = slide(); title(s, "The five main pillars",
                   "Challenges that emerge from the model — no off-the-shelf method handles all five")
bullets(s, [
    "Stochastic demand: patient arrivals, no-shows, cancellations.",
    "Mixed integer–continuous decision space.",
    "Structural heteroscedastic noise.",
    "Expensive evaluations.",
    "No gradient available.",
], size=19)

# 8 Literature Review
s = slide(); title(s, "Fundamental pillars", "Literature Review")
bullets(s, [
    "DES is a central tool for representing patient flows, queues, operational rules (Monks & Harper, 2025).",
    "Simulation Optimization (SO): optimize a black-box stochastic objective via simulation (Amaran et al., 2016).",
    "Derivative-Free Optimization (DFO): minimize when the gradient is unavailable (Conn et al., 2009).",
    "Metaheuristics (SA, GA, OptQuest): common in healthcare SO but lack formal guarantees under "
    "heteroscedastic noise (Wang & Demeulemeester, 2023; Nazri & Yusoh, 2025).",
], size=16)

# 9 Methodological Gap
s = slide(); title(s, "The gap: nobody covers all five at once", "The Methodological Gap")
table(s, [
    ["", "Stochastic DES", "Hetero-scedasticity", "Mixed-integer", "Adaptive Repl.", "Conv. Guarantees"],
    ["DFO continuous", "✓", "✗", "✗", "Partial", "✓"],
    ["SO integer", "✓", "✗", "Partial", "Partial", "✓"],
    ["Surrogate-based (RF/GP)", "✓", "Partial", "✓", "Partial", "✗"],
    ["Surrogate-based (SK)", "✓", "✓", "✗", "Partial", "✗"],
    ["Metaheuristics", "✓", "✗", "✓", "✗", "✗"],
    ["Proposed framework", "✓", "✓", "✓", "✓", "✓"],
], 0.5, 1.7, 12.3, 4.0, fontsize=12, highlight_last=True)
bullets(s, [("✓ satisfies   ✗ does not satisfy   Partial = partial. The empty space is the gap this work targets.", 0)],
        top=6.1, size=12, h=0.8)

# 10 Objective
s = slide(); title(s, "The suspect: SMAC-SK", "Objective")
bullets(s, [
    "Design, formulate, and evaluate a prescriptive decision-support framework for capacity "
    "planning in Minimally Invasive Gynecology.",
    ("Given the methodological gap, the proposed instrument is SMAC-SK "
     "(SMAC with a Stochastic-Kriging surrogate) — my workhorse for the rest of the talk.", 0),
], top=1.8, w=12.2, size=19)

# 11 Proposed Framework
s = slide(); title(s, "How SMAC-SK works: simulate → optimize → prescribe", "Proposed Framework")
framework_diagram(s)

# 12 Scheduling
s = slide(); title(s, "The levers we can actually pull", "Specialist hour scheduling")
img(s, f"{FIG}/diagrama_prog.png", 0.6, 1.5, w=6.0, h=5.0)
img(s, f"{FIG}/grafico_dist_por.png", 6.9, 1.5, w=6.0, h=5.0)

# 14 Results: Optimized vs Manual  (SMAC-SK protagonist)
s = slide(); title(s, "Caught: algorithms beat every manual policy",
                   "SMAC-SK (green star) is the recommended method")
img(s, f"{FIG}/fig_pareto.png", 2.35, 1.5, h=5.6)

# 14b Pairwise hypothesis tests (all vs all)
s = slide(); title(s, "Who really differs? All-vs-all tests",
                   "Welch t-test, Holm-corrected — top algorithms are statistically tied")
img(s, f"{FIG}/fig_pairwise_tts.png", 2.0, 1.55, h=5.6)

# 15 Parameter Comparison
s = slide(); title(s, "What SMAC-SK does differently", "Parameter comparison")
img(s, f"{FIG}/fig_variables_heatmap.png", 1.7, 1.6, h=5.6)

# 16 Convergence
s = slide(); title(s, "The proof it was not luck: SK earns its place",
                   "Green = SMAC-SK; compare against SMAC without SK — the SK term drives the gain")
img(s, f"{FIG}/fig_convergencia.png", 0.5, 1.7, w=12.3)

# 17 Computational cost
s = slide(); title(s, "The price: why sample-efficiency matters", "Computational cost")
img(s, f"{FIG}/fig_tiempo_exec.png", 2.7, 1.6, w=8.0)

# 18 Discussion
s = slide(); title(s, "Case closed — but the guarantees do not transfer", "Discussion")
bullets(s, [
    "Main result: SMAC-SK reduced TTS from 263 to 177 days (−33%) while patients served "
    "rose from 1,148 to ~2,180 (+90%).   [verificar 164 vs 177 del deck]",
    "No trade-off: algorithmic optimization Pareto-dominates all manual configurations — "
    "wait time and throughput improve simultaneously.",
    "Existing guarantees do not transfer: all methods converge empirically, yet their asymptotic "
    "results assume homoscedastic noise and smoothness — both violated by the DES evaluator.",
    "Open problem: no existing method natively handles structural heteroscedasticity with "
    "mixed-integer variables and formal convergence guarantees — the gap the doctoral work addresses.",
], size=15)

# 19 Future Work
s = slide(); title(s, "The open case: theory the thesis will build", "Future Work")
bullets(s, [
    "Algorithm design: a DFO method for mixed-integer variables with an adaptive "
    "replication-allocation mechanism driven by local variance estimation σ̂²(x).",
    "Convergence theory: sufficient regularity conditions on the heteroscedastic noise "
    "structure under which almost-sure convergence to first-order stationary points is establishable.",
], top=1.8, size=18)

# 20 Closing
s = slide()
tb = s.shapes.add_textbox(Inches(1.0), Inches(2.6), Inches(11.3), Inches(2.2))
tf = tb.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]; p.text = "Thank you"
p.font.size = Pt(40); p.font.bold = True; p.font.color.rgb = NAVY
q = tf.add_paragraph(); q.text = "Raúl H. Araneda — raul.araneda@uc.cl   ·   IFORS 2026, Vienna"
q.font.size = Pt(16); q.font.color.rgb = GRAY

# Appendix F — benchmark methods (key table for Q&A)
s = slide(); title(s, "Appendix — Benchmark Methods")
table(s, [
    ["Method", "Surrogate", "Variables", "Heterosc.", "Adapt. Rep.", "Convergence"],
    ["SMAC-GP+EI", "Gaussian Process", "Mixed", "✗", "✗", "Asymptotic*"],
    ["SMAC-RF", "Random Forest", "Mixed", "✗", "✗", "None (emp.)"],
    ["SMAC-SK ★", "Stochastic Kriging", "Mixed", "Part.", "✗", "Asymptotic*"],
    ["SK-Adaptive", "Stochastic Kriging", "Cont.†", "Part.", "Part.", "Asymptotic*"],
    ["SK-KGCP", "Stochastic Kriging", "Cont.†", "✗", "✗", "Asymptotic*"],
    ["SPSA", "—", "Cont.†", "✗", "✗", "Asymptotic*"],
    ["ASTRO-DF", "Quadratic model", "Cont.†", "✓", "✓", "Almost sure"],
], 0.5, 1.6, 12.3, 4.2, fontsize=11)
bullets(s, [("* assumes homoscedastic noise — not satisfied here.   † integers via continuous relaxation + rounding.", 0)],
        top=6.0, size=11, h=0.8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="IFORS2026_Araneda.pptx")
    args = ap.parse_args()
    prs.save(args.out)
    print(f"OK: {args.out}  ({len(prs.slides._sldIdLst)} slides)")


if __name__ == "__main__":
    main()
