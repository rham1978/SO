#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig_diagramas_metodos.py
════════════════════════════════════════════════════════════════════════
Genera un diagrama de flujo (PNG) por cada método de optimización del
benchmark IFORS, con el mismo nivel de detalle:

  M4    SMAC-GP+EI      (BO global, surrogate GP)
  M4RF  SMAC-RF         (BO global, surrogate Random Forest)
  M7    SMAC-SK         (BO global, surrogate Stochastic Kriging — propio)
  M8    SK-Adaptive     (SK + replicación adaptativa)
  M10   SK-KGCP         (SK + Knowledge Gradient)
  M13   SPSA            (gradiente por perturbación simultánea)
  M11   ASTRO-DF        (trust-region derivative-free — falló)

Uso:  python fig_diagramas_metodos.py [--out reporte_figs]
"""
from __future__ import annotations
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BLUE, RED, GREEN, GREY = "#1f77b4", "#d62728", "#2e7d32", "#555555"
ORANGE, PURPLE = "#e07b00", "#7b3fa0"


def _box(ax, xy, w, h, text, fc, ec, fs=10.5, tc="#111111", lw=1.6):
    x, y = xy
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle="round,pad=0.02,rounding_size=0.04",
                       fc=fc, ec=ec, lw=lw, zorder=2)
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            color=tc, zorder=3, wrap=True)


def _arrow(ax, p0, p1, color=GREY, lw=2.0, style="-|>", rad=0.0):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=18,
                        color=color, lw=lw, zorder=1,
                        connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)


def _note(ax, xy, text, color=GREY, fs=9.0, ha="left"):
    ax.text(xy[0], xy[1], text, ha=ha, va="center", fontsize=fs,
            color=color, style="italic", zorder=3)


def flujo(out, fname, title, subtitle, accent, steps, notes,
          loop_label="presupuesto NO agotado", final=None):
    """
    steps: lista de (texto, tipo) donde tipo ∈ {start, proc, eval, model, acq, decide, end}
    notes: dict idx -> (texto, lado)   lado ∈ {'L','R'}
    """
    n = len(steps)
    fig_h = 1.05 * n + 2.2
    fig, ax = plt.subplots(figsize=(13.5, fig_h))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    cmap = {
        "start":  ("#eef2f7", "#3a4a5a"),
        "proc":   ("#eaf2fb", BLUE),
        "eval":   ("#fde9e9", RED),
        "model":  ("#eafaf0", GREEN),
        "acq":    ("#f3ecfb", PURPLE),
        "decide": ("#fff4e3", ORANGE),
        "end":    ("#e9f7ee", GREEN),
    }

    cx = 4.4
    w, h = 5.0, 0.74
    gap = 1.05
    y_top = fig_h - 1.85
    ys = [y_top - i * gap for i in range(n)]

    # título
    ax.text(cx, fig_h - 0.45, title, ha="center", va="center",
            fontsize=16, fontweight="bold", color=accent)
    ax.text(cx, fig_h - 0.92, subtitle, ha="center", va="center",
            fontsize=10.5, color="#444444")

    for i, ((txt, typ), y) in enumerate(zip(steps, ys)):
        fc, ec = cmap.get(typ, cmap["proc"])
        bw = w + (0.9 if typ == "decide" else 0)
        _box(ax, (cx, y), bw, h, txt, fc, ec)
        if i < n - 1:
            _arrow(ax, (cx, y - h / 2), (cx, ys[i + 1] + h / 2), color="#444444")

    # notas laterales (todas a la derecha, para no chocar con el loop)
    for idx, (txt, _side) in notes.items():
        y = ys[idx]
        _note(ax, (cx + w / 2 + 0.35, y), txt, ha="left")

    # flecha de realimentación (loop) por la IZQUIERDA: de la decisión
    # (penúltimo paso) de vuelta al primer paso iterativo (índice 1)
    y_dec = ys[-2]
    y_back = ys[1]
    x_loop = cx - (w + 0.9) / 2 - 0.55
    _arrow(ax, (cx - (w + 0.9) / 2, y_dec), (x_loop, y_dec), color=accent, lw=2.0)
    _arrow(ax, (x_loop, y_dec), (x_loop, y_back), color=accent, lw=2.0)
    _arrow(ax, (x_loop, y_back), (cx - w / 2, y_back), color=accent, lw=2.0)
    ax.text(x_loop - 0.12, (y_dec + y_back) / 2, loop_label, rotation=90,
            ha="right", va="center", fontsize=8.8, color=accent, style="italic")

    fig.tight_layout()
    p = os.path.join(out, fname)
    fig.savefig(p, dpi=190, facecolor="white", bbox_inches="tight")
    plt.close()
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reporte_figs/metodos")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    outs = []

    # ── M4 · SMAC-GP+EI ───────────────────────────────────────────────────
    outs.append(flujo(
        args.out, "diag_M4_SMAC_GP.png",
        "M4 · SMAC-GP+EI", "Optimización bayesiana global (BlackBoxFacade de SMAC3)",
        BLUE,
        steps=[
            ("Diseño inicial Sobol  →  k puntos iniciales", "start"),
            ("Evaluar simulador: n=3 réplicas (CRN)  →  media TTS", "eval"),
            ("Ajustar surrogate: Proceso Gaussiano (Matérn)\nKy = K(X,X) + σ²·I  (homocedástico)", "model"),
            ("Maximizar Expected Improvement\nEI(x)=E[max(μ*−f(x),0)]", "acq"),
            ("¿Presupuesto (~150 evals) agotado?", "decide"),
            ("Incumbente → re-evaluar n=50 (TTS insesgado)", "end"),
        ],
        notes={
            1: ("variables: ConfigSpace mixto\n(Integer + Float nativos)", "R"),
            2: ("GP cuantifica incertidumbre suave;\nsufre con d alto y mucho ruido", "L"),
            3: ("propone el punto de mayor\nmejora esperada (greedy)", "R"),
        },
    ))

    # ── M4RF · SMAC-RF ────────────────────────────────────────────────────
    outs.append(flujo(
        args.out, "diag_M4RF_SMAC_RF.png",
        "M4RF · SMAC-RF", "Optimización bayesiana global (HPO Facade de SMAC3 — configuración por defecto)",
        BLUE,
        steps=[
            ("Diseño inicial Sobol  →  k puntos iniciales", "start"),
            ("Evaluar simulador: n=3 réplicas (CRN)  →  media TTS", "eval"),
            ("Ajustar surrogate: Random Forest\n(μ, σ² por varianza entre árboles)", "model"),
            ("Maximizar Expected Improvement (EI)", "acq"),
            ("¿Presupuesto agotado?", "decide"),
            ("Incumbente → re-evaluar n=50", "end"),
        ],
        notes={
            1: ("variables: ConfigSpace mixto\n(RF maneja mixto/condicional nativo)", "R"),
            2: ("más barato que GP; incertidumbre\nmenos suave (escalonada)", "L"),
        },
    ))

    # ── M7 · SMAC-SK ──────────────────────────────────────────────────────
    outs.append(flujo(
        args.out, "diag_M7_SMAC_SK.png",
        "M7 · SMAC-SK  (propuesta propia)", "SMAC3 con surrogate Stochastic Kriging — el SK NO viene en SMAC, se programó a medida",
        GREEN,
        steps=[
            ("Diseño inicial LHS  →  k puntos", "start"),
            ("Evaluar simulador: n=3 réplicas\n→ media TTS  +  σ²(x) al _variance_store", "eval"),
            ("Ajustar SK (Matérn 5/2)\nKy = K(X,X) + diag(σ²(x)/n)  (heterocedástico)", "model"),
            ("Maximizar EI sobre (μ, σ²_modelo)\n1000 challengers", "acq"),
            ("¿Presupuesto agotado?", "decide"),
            ("Incumbente → re-evaluar n=50", "end"),
        ],
        notes={
            1: ("canal lateral: la varianza viaja\npor _variance_store (SMAC no\npasa σ² en su API)", "R"),
            2: ("puntos ruidosos (σ² alto) pesan\nMENOS → robusto a saturación", "L"),
        },
    ))

    # ── M8 · SK-Adaptive ──────────────────────────────────────────────────
    outs.append(flujo(
        args.out, "diag_M8_SK_Adaptive.png",
        "M8 · SK-Adaptive", "Stochastic Kriging + asignación adaptativa de réplicas",
        PURPLE,
        steps=[
            ("Diseño inicial LHS", "start"),
            ("Política de réplicas n(x):\nwarmup → n_min;  luego n=n_min+round(pct·(n_max−n_min))", "decide"),
            ("Evaluar simulador: n(x) réplicas → media + σ²", "eval"),
            ("Ajustar SK heterocedástico (= M7)", "model"),
            ("Maximizar EI → candidato", "acq"),
            ("¿Presupuesto agotado?", "decide"),
            ("Incumbente → re-evaluar n=50", "end"),
        ],
        notes={
            1: ("pct = percentil de σ²(x) entre\nvarianzas conocidas (k-NN)", "R"),
            2: ("zona ruidosa → más réplicas\nzona estable → menos (ahorro)", "L"),
        },
    ))

    # ── M10 · SK-KGCP ─────────────────────────────────────────────────────
    outs.append(flujo(
        args.out, "diag_M10_SK_KGCP.png",
        "M10 · SK-KGCP", "Stochastic Kriging + Knowledge Gradient (lookahead 1 paso)",
        PURPLE,
        steps=[
            ("Diseño inicial LHS", "start"),
            ("Evaluar simulador (réplicas adaptativas) → media + σ²", "eval"),
            ("Ajustar SK heterocedástico", "model"),
            ("Adquisición KGCP (Monte Carlo, n_mc=64, n_cand=500)\n"
             "KG(x)=E_Z[max_x'(μ(x')+σ_kg·Z)] − max μ", "acq"),
            ("¿Presupuesto agotado?", "decide"),
            ("Incumbente → re-evaluar n=50", "end"),
        ],
        notes={
            2: ("σ_kg(x,x')=K(x,x') /\n√(K(x,x)+σ²_sim(x)/n)", "R"),
            3: ("mide la mejora del INCUMBENTE\nglobal, no solo en x (vs EI greedy)", "L"),
        },
    ))

    # ── M13 · SPSA ────────────────────────────────────────────────────────
    outs.append(flujo(
        args.out, "diag_M13_SPSA.png",
        "M13 · SPSA", "Aproximación estocástica por gradiente — 2 evals/iter, independiente de d",
        ORANGE,
        steps=[
            ("x0 = centro del espacio  [0,1]¹²", "start"),
            ("Δk ∈ {−1,+1}¹²  (Bernoulli ±1)   ·   ck = step/(k+1)^γ", "proc"),
            ("Evaluar:  y⁺=F̄(xk+ck·Δk, n_reps)   y⁻=F̄(xk−ck·Δk, n_reps)", "eval"),
            ("Gradiente:  ĝk,i = (y⁺−y⁻)/(2·ck·Δk,i)", "model"),
            ("Paso:  xk+1 = clip( xk − ak·ĝk ,  [0,1]¹² ),  ak=α/(k+A)^α", "proc"),
            ("¿Iteración < máx?", "decide"),
            ("Incumbente → re-evaluar n=50", "end"),
        ],
        notes={
            1: ("perturba TODAS las dims a la vez\n→ costo no crece con d=12", "R"),
            2: ("solo 2 evaluaciones del simulador\npor iteración (vs d+1 trust-region)", "L"),
            3: ("defaults SimOpt: α=0.602, γ=0.101,\nstep=0.1, n_reps=30, A=10", "R"),
        },
        loop_label="iteración siguiente",
    ))

    # ── M11 · ASTRO-DF (falló) ────────────────────────────────────────────
    outs.append(flujo(
        args.out, "diag_M11_ASTRODF.png",
        "M11 · ASTRO-DF  (no completó)", "Trust-region derivative-free con muestreo adaptativo (Shashaani et al., 2018)",
        RED,
        steps=[
            ("x0, región de confianza Δ0", "start"),
            ("Construir modelo lineal poisado en B(xk;Δ)\ncon muestreo adaptativo Ñ(x) ∝ σ̂/Δ²", "model"),
            ("Paso de Cauchy → candidato x̃ = xk + sk", "proc"),
            ("Evaluar x̃ (muestreo adaptativo)", "eval"),
            ("ρ̂ = (descenso observado)/(descenso del modelo)", "proc"),
            ("¿ρ̂ ≥ η1 ?   éxito→aceptar+expandir Δ ;  fallo→contraer Δ", "decide"),
            ("Incumbente  (★ deadlock: futures unfinished)", "end"),
        ],
        notes={
            1: ("más réplicas cerca del óptimo\n(Δ pequeño), menos lejos", "R"),
            2: ("convergencia wp1 garantizada\n(Teorema del paper)", "L"),
        },
        loop_label="iteración siguiente",
    ))

    print("Diagramas generados:")
    for p in outs:
        print("  ", p)


if __name__ == "__main__":
    main()
