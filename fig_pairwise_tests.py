#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig_pairwise_tests.py
════════════════════════════════════════════════════════════════════════
Pruebas de hipótesis TODOS contra TODOS sobre el TTS (no solo vs Current).

  · Welch's t-test (no asume varianzas iguales — apropiado bajo
    heterocedasticidad) para cada par de configuraciones.
  · Corrección de Holm sobre las C(n,2) comparaciones (control FWER).
  · Matriz/heatmap de significancia + tamaño de efecto (Cohen's d).

Salidas:
  reporte_figs/fig_pairwise_tts.png   — heatmap de significancia (Holm)
  reporte_figs/pairwise_tts.csv       — Δmedia, t, p, p_holm, d por par

Uso:  python fig_pairwise_tests.py
"""
from __future__ import annotations
import json
import os
import csv
import itertools

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

import generar_reporte_ppt as g

JSON = "comparacion_manual/comparacion_manual.json"
OUT = "reporte_figs"


def cohen_d(a, b):
    na, nb = len(a), len(b)
    sa, sb = np.var(a, ddof=1), np.var(b, ddof=1)
    sp = np.sqrt(((na - 1) * sa + (nb - 1) * sb) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / sp if sp > 0 else 0.0


def holm(pvals):
    """Corrección de Holm-Bonferroni. Devuelve p ajustados en orden original."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def main():
    d = json.load(open(JSON, encoding="utf-8"))
    esc = [x for x in d["escenarios"] if np.isfinite(np.mean(x.get("tts_raw", [np.nan])))]
    esc.sort(key=lambda x: np.mean(x["tts_raw"]))   # por TTS ascendente

    names = [g.disp(x["escenario"]) for x in esc]
    tipos = [x["tipo"] for x in esc]
    data = [np.asarray(x["tts_raw"], float) for x in esc]
    n = len(esc)

    # ── todas las comparaciones por pares ─────────────────────────────
    pairs = list(itertools.combinations(range(n), 2))
    raw_p, rows = [], []
    for i, j in pairs:
        t, p = stats.ttest_ind(data[i], data[j], equal_var=False)  # Welch
        raw_p.append(p)
    p_holm = holm(np.array(raw_p))

    P = np.full((n, n), np.nan)     # p ajustado
    D = np.full((n, n), np.nan)     # diferencia de medias (fila - col)
    for (i, j), p, pa in zip(pairs, raw_p, p_holm):
        di = cohen_d(data[i], data[j])
        dm = float(np.mean(data[i]) - np.mean(data[j]))
        P[i, j] = pa; P[j, i] = pa
        D[i, j] = dm; D[j, i] = -dm
        rows.append([names[i], names[j], round(dm, 2), round(p, 5),
                     round(pa, 5), "yes" if pa < 0.05 else "no", round(di, 2)])

    with open(os.path.join(OUT, "pairwise_tts.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["A", "B", "mean_diff_A_minus_B", "p_welch", "p_holm",
                    "significant_0.05", "cohens_d"])
        w.writerows(rows)

    # ── heatmap de significancia ──────────────────────────────────────
    # categorías: diagonal, ns (≥0.05), *, **, ***
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                M[i, j] = 0
            else:
                pa = P[i, j]
                M[i, j] = (4 if pa < 1e-3 else 3 if pa < 1e-2 else
                           2 if pa < 5e-2 else 1)
    cmap = ListedColormap(["#dddddd", "#f2dede", "#fde9c8", "#bfe3c0", "#2e7d32"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)

    fig, ax = plt.subplots(figsize=(10.5, 8.6))
    ax.imshow(M, cmap=cmap, norm=norm, aspect="equal")

    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", color="#999", fontsize=11)
                continue
            pa = P[i, j]
            sym = ("***" if pa < 1e-3 else "**" if pa < 1e-2 else
                   "*" if pa < 5e-2 else "ns")
            txt = f"{sym}\nΔ{D[i, j]:+.0f}"
            tc = "white" if M[i, j] == 4 else "#222"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.5, color=tc)

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    cols = ["#1f77b4" if t == "optimo" else "#d62728" for t in tipos]
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=10)
    ax.set_yticklabels(names, fontsize=10)
    for tick, c in zip(ax.get_xticklabels(), cols): tick.set_color(c)
    for tick, c in zip(ax.get_yticklabels(), cols): tick.set_color(c)

    ax.set_title("Pairwise hypothesis tests on TTS — all vs all\n"
                 "Welch's t-test, Holm-corrected   (Δ = row mean − col mean, days)",
                 fontsize=12, fontweight="bold")

    from matplotlib.patches import Patch
    leg = [Patch(color="#2e7d32", label="*** p<0.001"),
           Patch(color="#bfe3c0", label="** p<0.01"),
           Patch(color="#fde9c8", label="* p<0.05"),
           Patch(color="#f2dede", label="ns (≥0.05)")]
    ax.legend(handles=leg, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              fontsize=9, framealpha=0.95, title="Holm-adjusted")
    ax.text(1.02, 0.0, "blue = algorithm\nred = manual", transform=ax.transAxes,
            fontsize=9, color="#444", va="bottom")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_pairwise_tts.png"), dpi=200, facecolor="white")
    plt.close()

    # resumen consola
    nsig = sum(1 for r in rows if r[5] == "yes")
    print(f"OK: fig_pairwise_tts.png  ({len(pairs)} pares, {nsig} significativos Holm<0.05)")
    print("Pares NO significativos (empates estadísticos):")
    for r in rows:
        if r[5] == "no":
            print(f"    {r[0]:12} vs {r[1]:12}  Δ={r[2]:+6.1f} d  p_holm={r[4]}")


if __name__ == "__main__":
    main()
