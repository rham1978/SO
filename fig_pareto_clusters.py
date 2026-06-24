#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig_pareto_clusters.py
════════════════════════════════════════════════════════════════════════
Versión "clusterizada y anonimizada" de la figura Pareto para IFORS:
  · sin nombres ni días en cada punto → etiquetas genéricas Alg 1..N / Man 1..M
  · dos nubes/clusters: Manual (rojo) y Algorithms (azul)
  · SMAC-SK marcado (estrella verde + halo)
  · sin la palabra "optimized" en el título

Guarda fig_pareto.png (lo que incluye el deck) y fig_pareto_clusters.png.
Imprime el mapeo Alg N / Man N → nombre real.

Uso:  python fig_pareto_clusters.py
"""
from __future__ import annotations
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

import generar_reporte_ppt as g

BLUE, RED, GREEN = "#1f77b4", "#d62728", "#2e7d32"
JSON = "comparacion_manual/comparacion_manual.json"
OUT = "reporte_figs"


def cluster_polygon(P, expand=1.28, round_n=24):
    """Polígono convexo expandido (suavizado) que encierra los puntos P (n,2)."""
    P = np.asarray(P, float)
    c = P.mean(axis=0)
    if len(P) >= 3:
        hull = ConvexHull(P)
        verts = P[hull.vertices]
    else:
        # pocos puntos: caja
        mn, mx = P.min(0), P.max(0)
        verts = np.array([[mn[0], mn[1]], [mx[0], mn[1]], [mx[0], mx[1]], [mn[0], mx[1]]])
    # expandir respecto al centroide
    verts = c + (verts - c) * expand
    return verts


def main():
    d = json.load(open(JSON, encoding="utf-8"))
    EXCLUDE = getattr(g, "EXCLUDE", set())
    esc = [x for x in d["escenarios"] if g.modkey(x["escenario"]) not in EXCLUDE
           and np.isfinite(x["atenciones_media"])]

    manual = [x for x in esc if x["tipo"] == "manual"]
    algos = [x for x in esc if x["tipo"] == "optimo"]
    # orden por TTS ascendente (mejor primero)
    manual.sort(key=lambda x: x["tts_media"])
    algos.sort(key=lambda x: x["tts_media"])

    fig, ax = plt.subplots(figsize=(12.5, 8.0))

    # ── clusters (nubes) ──────────────────────────────────────────────
    for grp, col, name, ytag in [
            (algos, BLUE, "Algorithms", "low"),
            (manual, RED, "Manual", "high")]:
        P = np.array([[x["atenciones_media"], x["tts_media"]] for x in grp])
        poly = cluster_polygon(P, expand=1.30)
        ax.add_patch(Polygon(poly, closed=True, facecolor=col, alpha=0.10,
                             edgecolor=col, lw=2.0, ls="--", zorder=1))
        cx = poly[:, 0].mean()
        cy = poly[:, 1].max() + 4 if ytag == "high" else poly[:, 1].min() - 4
        ax.text(cx, cy, name, color=col, fontsize=15, fontweight="bold",
                ha="center", va="bottom" if ytag == "high" else "top", zorder=6)

    # ── puntos + etiquetas genéricas (con repulsión) ─────────────────
    from adjustText import adjust_text
    alg_map, man_map = {}, {}
    texts, xs, ys = [], [], []
    for i, x in enumerate(algos, 1):
        tag = f"Alg {i}"
        alg_map[tag] = (g.disp(x["escenario"]), x["tts_media"], x["atenciones_media"])
        at, tts = x["atenciones_media"], x["tts_media"]
        if g.modkey(x["escenario"]) == "M7":   # SMAC-SK protagonista
            ax.scatter(at, tts, s=620, marker="*", color=GREEN, zorder=5,
                       edgecolor="white", linewidth=1.6)
            ax.scatter(at, tts, s=1700, marker="o", facecolor="none",
                       edgecolor=GREEN, linewidth=1.8, alpha=0.5, zorder=4)
            texts.append(ax.text(at, tts, f"{tag} = SMAC-SK", color=GREEN,
                                 fontsize=12, fontweight="bold", zorder=7))
        else:
            ax.scatter(at, tts, s=180, marker="o", color=BLUE, zorder=3,
                       edgecolor="white", linewidth=1.3)
            texts.append(ax.text(at, tts, tag, color="#10324f",
                                 fontsize=11, fontweight="bold", zorder=7))
        xs.append(at); ys.append(tts)
    for i, x in enumerate(manual, 1):
        tag = f"Man {i}"
        man_map[tag] = (x["escenario"], x["tts_media"], x["atenciones_media"])
        at, tts = x["atenciones_media"], x["tts_media"]
        ax.scatter(at, tts, s=180, marker="s", color=RED, zorder=3,
                   edgecolor="white", linewidth=1.3)
        texts.append(ax.text(at, tts, tag, color="#5a1414",
                             fontsize=11, fontweight="bold", zorder=7))
        xs.append(at); ys.append(tts)
    adjust_text(texts, x=xs, y=ys, ax=ax,
                expand=(2.2, 2.6), force_text=(0.8, 1.1),
                only_move={"text": "xy", "static": "xy"},
                arrowprops=dict(arrowstyle="-", color="#888888", lw=0.8))

    # (sin leyenda — los clusters y la estrella verde ya identifican los grupos)

    # ── "better" arrow ────────────────────────────────────────────────
    ax.annotate("", xy=(0.60, 0.10), xytext=(0.40, 0.25), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=2.4))
    ax.text(0.39, 0.28, "better\n(less wait + more patients)", transform=ax.transAxes,
            color=GREEN, fontsize=12, fontweight="bold", ha="center")

    ax.set_xlabel("Patients served (total attentions)   →   more = better", fontsize=12)
    ax.set_ylabel("TTS — time in system [days]   ←   less = better", fontsize=12)
    ax.set_title("TTS vs throughput", fontsize=15, fontweight="bold")
    ax.margins(x=0.16, y=0.18)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    for fname in ("fig_pareto.png", "fig_pareto_clusters.png"):
        fig.savefig(os.path.join(OUT, fname), dpi=220, facecolor="white")
    plt.close()

    print("Mapeo de etiquetas:")
    print("  ALGORITMOS (orden por TTS ascendente):")
    for k, (name, tts, at) in alg_map.items():
        star = "  ★" if name == "SMAC-SK" else ""
        print(f"    {k:6} = {name:12}  (TTS {tts:.0f} d, {at:.0f} patients){star}")
    print("  MANUALES:")
    for k, (name, tts, at) in man_map.items():
        print(f"    {k:6} = {name:14}(TTS {tts:.0f} d, {at:.0f} patients)")


if __name__ == "__main__":
    main()
