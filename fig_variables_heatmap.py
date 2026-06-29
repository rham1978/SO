#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig_variables_heatmap.py
════════════════════════════════════════════════════════════════════════
Regenera la figura "Decision variables by configuration" como heatmap:
  · filas  = 12 variables de decisión
  · cols   = escenarios manuales (rojo) + modelos de optimización (azul)
  · texto  = valor real de la variable
  · color  = posición normalizada en el rango admisible (azul=min, rojo=max)

Usa los datos actuales de comparacion_manual/comparacion_manual.json, de
modo que incluye SMAC-RF (M4RF), que faltaba en la versión anterior.

Uso:
    python fig_variables_heatmap.py
    python fig_variables_heatmap.py --json otra.json --out figura.png
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Variables de decisión: (min, max, tipo, etiqueta en inglés) — mismo orden que el PPT
VARIABLES = {
    "horas_especialista_1ra":   (8, 30, "int",   "First-consult slots/week"),
    "horas_control_post":       (20, 70, "int",  "Post-control slots/week"),
    "cupos_laboratorio_ugd":    (20, 100, "int", "Lab slots (UGD)"),
    "cupos_ecografia_matrona":  (10, 50, "int",  "Ultrasound slots (midwife)"),
    "cupos_ecografia_ugd":      (10, 50, "int",  "Ultrasound slots (UGD)"),
    "dias_publicacion":         (1, 10, "int",   "Scheduling lead time (days)"),
    "num_matronas":             (1, 4, "int",    "# Midwives"),
    "num_agentes_ugd":          (1, 4, "int",    "# UGD agents"),
    "pct_bloqueo_1ra":          (0.05, 0.5, "float", "First-consult blocking %"),
    "pct_consultas_vacias":     (0.05, 0.5, "float", "Empty-consult %"),
    "pct_no_contactabilidad":   (0.05, 0.5, "float", "Non-contactability %"),
    "pct_bloqueo_post_control": (0.05, 0.5, "float", "Post-control blocking %"),
}

SHORT = {
    "M4": "SMAC-GP+EI", "M4RF": "SMAC-RF", "M7": "SMAC-SK", "M8": "SK-Adaptive",
    "M9": "SK-REVI", "M10": "SK-KGCP", "M11": "ASTRO-DF", "M12": "STRONG",
    "M13": "SPSA", "M14": "ALOE", "SA": "SA", "RS": "Random Search",
}

# Orden de los modelos de optimización (familia SMAC, luego SK, luego gradiente)
OPT_ORDER = ["M13", "M4", "M8", "M7", "M4RF", "M10", "M11"]
# Orden de los escenarios manuales
MAN_ORDER = ["Current", "Management", "Mgmt+Cap", "Mgmt+Cap v2"]

RED, BLUE = "#d62728", "#1f77b4"


def modkey(nombre):
    return nombre.replace("ÓPTIMO", "").strip() if "ÓPTIMO" in nombre else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="comparacion_manual/comparacion_manual.json")
    ap.add_argument("--out",  default="reporte_figs/fig_variables_heatmap.png")
    args = ap.parse_args()

    d = json.load(open(args.json, encoding="utf-8"))
    vd = d["variables_decision"]

    # ── Construir el orden de columnas y sus etiquetas/tipos ──────────────
    man_cols = [c for c in MAN_ORDER if c in vd]
    # mapear claves de optimización (M4 → 'ÓPTIMO M4') presentes en los datos
    opt_keys = []
    for mk in OPT_ORDER:
        nombre = f"ÓPTIMO {mk}"
        if nombre in vd:
            opt_keys.append((mk, nombre))

    col_names   = man_cols + [SHORT[mk] for mk, _ in opt_keys]
    col_sources = man_cols + [nombre for _, nombre in opt_keys]
    col_types   = ["manual"] * len(man_cols) + ["optimo"] * len(opt_keys)
    n_man = len(man_cols)

    var_keys = list(VARIABLES.keys())
    row_labels = [VARIABLES[k][3] for k in var_keys]

    # ── Matriz de valores normalizados (0..1 en rango admisible) ──────────
    nrow, ncol = len(var_keys), len(col_sources)
    norm = np.full((nrow, ncol), np.nan)
    raw  = np.full((nrow, ncol), np.nan)
    for j, src in enumerate(col_sources):
        cfg = vd.get(src, {})
        for i, vk in enumerate(var_keys):
            lo, hi, _typ, _lab = VARIABLES[vk]
            v = cfg.get(vk, np.nan)
            if v is not None and np.isfinite(v):
                raw[i, j] = v
                norm[i, j] = (float(v) - lo) / (hi - lo)

    # ── Figura ────────────────────────────────────────────────────────────
    cmap = plt.get_cmap("Reds")  # white=min (low), red=max (high)
    fig, ax = plt.subplots(figsize=(16, 9))
    im = ax.imshow(norm, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    # sin texto en celdas

    # ejes
    ax.set_xticks(range(ncol))
    ax.set_yticks(range(nrow))
    ax.set_yticklabels(row_labels, fontsize=11)
    ax.set_xticklabels([])  # las etiquetas de columna se ponen arriba, rotadas

    # etiquetas de columna arriba, rotadas, coloreadas por tipo
    for j, (name, typ) in enumerate(zip(col_names, col_types)):
        ax.text(j - 0.1, -0.62, name, rotation=35, ha="left", va="bottom",
                fontsize=12, fontweight="bold",
                color=RED if typ == "manual" else BLUE)

    # línea divisoria manual / optimización
    ax.axvline(n_man - 0.5, color="black", lw=2.5)

    # cabeceras de grupo + subrayado (bien por encima de las etiquetas rotadas)
    x_man = (n_man - 1) / 2.0
    x_opt = n_man + (ncol - n_man - 1) / 2.0
    ax.text(x_man, -3.05, "MANUAL", ha="center", va="bottom",
            fontsize=16, fontweight="bold", color=RED)
    ax.text(x_opt, -3.05, "ALGORITHMS", ha="center", va="bottom",
            fontsize=16, fontweight="bold", color=BLUE)
    ax.plot([-0.3, n_man - 0.7], [-2.8, -2.8], color=RED, lw=2.5,
            clip_on=False)
    ax.plot([n_man - 0.3, ncol - 0.3], [-2.8, -2.8], color=BLUE, lw=2.5,
            clip_on=False)

    ax.set_ylim(nrow - 0.5, -3.4)   # dejar espacio arriba para cabeceras
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # colorbar
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cb.set_ticks([0, 0.5, 1.0])
    cb.set_ticklabels(["0", "0.5", "1"])

    # título y nota al pie
    ncfg = ncol
    fig.suptitle(f"Decision variables by configuration  ({ncfg} configs)",
                 fontsize=18, fontweight="bold", y=0.99)
    fig.text(0.5, 0.02,
             "cell text = actual value   ·   color = position within each "
             "variable's admissible range (white = min, red = max)",
             ha="center", fontsize=11, color="#333333")

    fig.tight_layout(rect=[0, 0.04, 1, 0.93])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor="white")
    plt.close()
    print(f"OK: {args.out}  ({ncol} columns: {n_man} manual + {ncol - n_man} optimization)")
    print("Columns:", col_names)


if __name__ == "__main__":
    main()
