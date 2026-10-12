#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig_stochastic_kriging.py
════════════════════════════════════════════════════════════════════════
Diagrama esquemático de cómo funciona Stochastic Kriging (Ankenman,
Nelson & Staum 2010) por sí solo: separa las dos fuentes de varianza
(extrínseca/espacial e intrínseca/simulación) y produce media + incertidumbre.

Uso:  python fig_stochastic_kriging.py [--out reporte_figs/metodos]
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


def _box(ax, cx, cy, w, h, text, fc, ec, fs=11, tc="#111111", lw=1.8):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.05",
                 fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=tc, zorder=3)


def _arrow(ax, p0, p1, color=GREY, lw=2.0, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=18,
                 color=color, lw=lw, zorder=1,
                 connectionstyle=f"arc3,rad={rad}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reporte_figs/metodos")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13.5, 9.2))
    ax.set_xlim(0, 13.5); ax.set_ylim(0, 9.2); ax.axis("off")

    ax.text(6.75, 8.85, "Stochastic Kriging — ¿cómo funciona por sí solo?",
            ha="center", fontsize=17, fontweight="bold", color=PURPLE)
    ax.text(6.75, 8.42, "Metamodelo de la superficie media de un simulador ruidoso "
            "(Ankenman, Nelson & Staum, 2010)",
            ha="center", fontsize=10.5, color="#444444")

    # 1) Simulador
    _box(ax, 6.75, 7.55, 9.6, 0.85,
         "Simulador estocástico: en cada punto muestreado xᵢ se corren nᵢ réplicas\n"
         "Y_j(xᵢ) = y(xᵢ) + ε_j(xᵢ)      (cada réplica da un valor distinto)",
         "#fde9e9", RED, fs=10.5)

    # 2) dos salidas: media y varianza muestral
    _box(ax, 3.5, 6.05, 5.0, 0.92,
         "Media muestral\nȲ(xᵢ) = (1/nᵢ) Σ_j Y_j(xᵢ)", "#eafaf0", GREEN, fs=10.5)
    _box(ax, 10.0, 6.05, 5.2, 0.92,
         "Varianza muestral (requiere nᵢ ≥ 2)\nV̂(xᵢ) = var réplica a réplica", "#fff4e3", ORANGE, fs=10.5)
    _arrow(ax, (5.4, 7.12), (3.7, 6.55), color="#444444")
    _arrow(ax, (8.1, 7.12), (9.8, 6.55), color="#444444")

    # 3) dos fuentes de varianza
    _box(ax, 3.5, 4.35, 5.2, 1.15,
         "Incertidumbre EXTRÍNSECA (espacial)\n"
         "M(x) ~ Proceso Gaussiano\n"
         "Σ_M = τ²·R(x − x'; θ)", "#eaf2fb", BLUE, fs=10.5)
    _box(ax, 10.0, 4.35, 5.2, 1.15,
         "Ruido INTRÍNSECO (de simulación)\n"
         "Σ_ε = diag( V(x₁)/n₁ , … , V(x_k)/n_k )\n"
         "diagonal y VARIABLE por punto (heterocedástico)", "#fff4e3", ORANGE, fs=10.0)
    _arrow(ax, (3.5, 5.59), (3.5, 4.93), color=BLUE)
    _arrow(ax, (10.0, 5.59), (10.0, 4.93), color=ORANGE)

    # 4) matriz combinada + MLE
    _box(ax, 6.75, 2.75, 9.8, 1.05,
         "Matriz de covarianza:   Σ = Σ_M + Σ_ε\n"
         "Ajuste de τ², θ, β₀ por máxima verosimilitud (MLE)", "#f3ecfb", PURPLE, fs=11.5)
    _arrow(ax, (3.5, 3.77), (5.3, 3.28), color=BLUE)
    _arrow(ax, (10.0, 3.77), (8.2, 3.28), color=ORANGE)

    # 5) predicción
    _box(ax, 6.75, 1.05, 11.6, 1.15,
         "Predicción en un punto nuevo x₀:\n"
         "ŷ(x₀) = β₀ + Σ_M(x₀,·)·Σ⁻¹·(Ȳ − β₀·1)        ← media suave\n"
         "MSE(x₀) = τ² − Σ_M(x₀,·)·Σ⁻¹·Σ_M(·,x₀)        ← incertidumbre del metamodelo (alimenta EI / KG)",
         "#e9f7ee", GREEN, fs=10.5)
    _arrow(ax, (6.75, 2.22), (6.75, 1.63), color=GREEN)

    # nota lateral de contraste con GP estándar
    ax.text(0.15, 4.35,
            "vs. GP\nhomocedástico:\nΣ = Σ_M + σ²·I\n(un único ruido\npara todo x)",
            ha="left", va="center", fontsize=9.0, color=GREY, style="italic")

    fig.tight_layout()
    p = os.path.join(args.out, "diag_stochastic_kriging.png")
    fig.savefig(p, dpi=190, facecolor="white", bbox_inches="tight")
    plt.close()
    print("OK:", p)


if __name__ == "__main__":
    main()
