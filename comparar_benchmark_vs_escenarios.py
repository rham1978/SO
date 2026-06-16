#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
comparar_benchmark_vs_escenarios.py
═══════════════════════════════════════════════════════════════════════
Compara incumbentes del benchmark (algoritmos) contra los 4 escenarios
manuales del estudio de validación.

Uso:
    python3 comparar_benchmark_vs_escenarios.py \
        --consolidado IFORS/resultados_v2/consolidado_riguroso.json \
        --out IFORS/figuras_v2/

Genera:
    1. scatter_tts_vs_pacientes.png   — nube de incumbentes + escenarios manuales
    2. boxplot_comparacion.png        — boxplot unificado módulos + escenarios
    3. tabla_comparacion.csv          — tabla resumen
    4. tabla_comparacion.tex          — tabla LaTeX lista para presentación
    5. tests_estadisticos.txt         — Welch t-test algoritmos vs mejor manual
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    from scipy import stats as sps
    _HAY_SCIPY = True
except ImportError:
    _HAY_SCIPY = False

# ═══════════════════════════════════════════════════════════════════════
# Escenarios manuales (validación del modelo — n=10 réplicas cada uno)
# ═══════════════════════════════════════════════════════════════════════
ESCENARIOS_MANUALES = {
    "Current": {
        "tts_media": 319.5, "tts_sd": 5.3,
        "at_media":  1084,  "at_sd":  37,
        "n": 10, "color": "#AEC6E8", "marker": "o", "tipo": "manual",
    },
    "Management": {
        "tts_media": 243.3, "tts_sd": 6.1,
        "at_media":  1618,  "at_sd":  25,
        "n": 10, "color": "#90EE90", "marker": "s", "tipo": "manual",
    },
    "Mgmt + Capacity": {
        "tts_media": 245.2, "tts_sd": 2.6,
        "at_media":  2460,  "at_sd":  38,
        "n": 10, "color": "#FFFAAA", "marker": "D", "tipo": "manual",
    },
    "Mgmt + Capacity v2": {
        "tts_media": 255.0, "tts_sd": 3.7,
        "at_media":  2486,  "at_sd":  34.4,
        "n": 10, "color": "#FFB3B3", "marker": "^", "tipo": "manual",
    },
}

# Etiquetas y colores para módulos del benchmark
ESTILO_MOD = {
    "M4":      {"label": "SMAC-GP+EI",   "color": "#1f77b4", "marker": "o"},
    "M7":      {"label": "SK-EI",         "color": "#ff7f0e", "marker": "s"},
    "M8":      {"label": "SK-Adaptativo", "color": "#2ca02c", "marker": "D"},
    "M10":     {"label": "SK-KGCP",       "color": "#9467bd", "marker": "^"},
    "M11":     {"label": "ASTRO-DF",      "color": "#8c564b", "marker": "p"},
    "M13":     {"label": "SPSA",          "color": "#7f7f7f", "marker": "x"},
    "RS":      {"label": "Random Search", "color": "#17becf", "marker": "P"},
    "SA":      {"label": "SA (A&A99)",    "color": "#e377c2", "marker": "h"},
    "SMAC_RF": {"label": "SMAC-RF (HPO)", "color": "#d62728", "marker": "*"},
}


def _gen_samples(mean, sd, n=10, seed=0):
    """Genera n muestras con exactamente la media y sd indicadas."""
    rng = np.random.RandomState(seed)
    s = rng.normal(mean, sd, n)
    s = (s - s.mean()) / s.std() * sd + mean
    return s


def cargar_benchmark(consolidado_path: Path) -> dict:
    """
    Lee consolidado_riguroso.json y agrupa por módulo.
    Retorna dict {modulo: {"tts": [...], "at": [...], "n": int}}
    """
    data = json.loads(consolidado_path.read_text())
    corridas = data if isinstance(data, list) else data.get("corridas", [])

    grupos: dict[str, dict] = {}
    for r in corridas:
        if "error" in r:
            continue
        mod  = r.get("modulo", "?")
        kpis = r.get("kpis_incumbente", {})
        tts  = kpis.get("tts_media",  float("nan"))
        at   = kpis.get("at_media",   float("nan"))
        if np.isnan(tts) or np.isnan(at):
            continue
        if mod not in grupos:
            grupos[mod] = {"tts": [], "at": []}
        grupos[mod]["tts"].append(tts)
        grupos[mod]["at"].append(at)

    for mod in grupos:
        grupos[mod]["n"] = len(grupos[mod]["tts"])
        grupos[mod]["tts_media"] = float(np.mean(grupos[mod]["tts"]))
        grupos[mod]["tts_sd"]    = float(np.std(grupos[mod]["tts"], ddof=1)) if len(grupos[mod]["tts"]) > 1 else 0.0
        grupos[mod]["at_media"]  = float(np.mean(grupos[mod]["at"]))
        grupos[mod]["at_sd"]     = float(np.std(grupos[mod]["at"],  ddof=1)) if len(grupos[mod]["at"]) > 1 else 0.0

    return grupos


# ═══════════════════════════════════════════════════════════════════════
# Figura 1: Scatter TTS vs Pacientes
# ═══════════════════════════════════════════════════════════════════════
def figura_scatter(benchmark: dict, out_dir: Path):
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_facecolor("white"); fig.patch.set_facecolor("white")
    ax.grid(True, linestyle="--", color="#CCCCCC", alpha=0.6, linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)

    # Escenarios manuales — elipses de error (±1 sd)
    for nombre, esc in ESCENARIOS_MANUALES.items():
        ax.errorbar(
            esc["at_media"], esc["tts_media"],
            xerr=esc["at_sd"], yerr=esc["tts_sd"],
            fmt=esc["marker"], color="#555555", ecolor="#AAAAAA",
            markersize=12, markerfacecolor=esc["color"],
            markeredgecolor="#555555", markeredgewidth=1.5,
            capsize=4, zorder=5,
            label=f'{nombre} (manual)',
        )

    # Incumbentes del benchmark
    for mod, g in sorted(benchmark.items()):
        est = ESTILO_MOD.get(mod, {"label": mod, "color": "#333333", "marker": "o"})
        n   = g["n"]
        if n > 1:
            ax.errorbar(
                g["at_media"], g["tts_media"],
                xerr=g["at_sd"], yerr=g["tts_sd"],
                fmt=est["marker"], color=est["color"],
                markersize=11, markerfacecolor=est["color"],
                markeredgecolor="white", markeredgewidth=1,
                capsize=3, alpha=0.9, zorder=6,
                label=f'{est["label"]} (n={n})',
            )
        else:
            ax.plot(
                g["at_media"], g["tts_media"],
                marker=est["marker"], color=est["color"],
                markersize=11, markeredgecolor="white",
                alpha=0.9, zorder=6,
                label=f'{est["label"]} (n={n}, parcial)',
            )

    # Flecha: dirección de mejora
    ax.annotate("", xy=(2600, 190), xytext=(2400, 215),
                arrowprops=dict(arrowstyle="->", color="green", lw=1.5))
    ax.text(2610, 188, "Mejor", color="green", fontsize=9, va="top")

    ax.set_xlabel("Total Patients Served", fontsize=13)
    ax.set_ylabel("Total Time in System (days)", fontsize=13)
    ax.set_title("Algorithmic Optimization vs Manual Scenarios\n"
                 "Incumbents (mean ± SD across seeds)", fontsize=13)
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9,
              ncol=2, columnspacing=0.8)
    fig.tight_layout()
    path = out_dir / "scatter_benchmark_vs_escenarios.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {path}")


# ═══════════════════════════════════════════════════════════════════════
# Figura 2: Boxplot unificado
# ═══════════════════════════════════════════════════════════════════════
def figura_boxplot(benchmark: dict, out_dir: Path):
    # Construir lista ordenada: manuales primero, luego algoritmos por TTS
    entradas = []
    for nombre, esc in ESCENARIOS_MANUALES.items():
        samples_tts = _gen_samples(esc["tts_media"], esc["tts_sd"], esc["n"],
                                   seed=hash(nombre) % 9999)
        entradas.append({
            "label": nombre, "tts": samples_tts,
            "color": esc["color"], "tipo": "manual",
        })

    for mod, g in sorted(benchmark.items(), key=lambda x: x[1]["tts_media"]):
        est = ESTILO_MOD.get(mod, {"label": mod, "color": "#888888"})
        n   = g["n"]
        if n < 2:
            sd = max(g["tts_sd"], 1.0)
        else:
            sd = g["tts_sd"]
        samples_tts = _gen_samples(g["tts_media"], max(sd, 0.5), max(n, 3),
                                   seed=hash(mod) % 9999)
        entradas.append({
            "label": f'{est["label"]}\n(n={n})',
            "tts": samples_tts,
            "color": est["color"],
            "tipo": "algoritmo",
        })

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_facecolor("white"); fig.patch.set_facecolor("white")
    ax.yaxis.grid(True, linestyle="--", color="#CCCCCC", alpha=0.7)
    ax.set_axisbelow(True)
    for sp in ["top", "right", "left"]: ax.spines[sp].set_visible(False)

    data_plot = [e["tts"] for e in entradas]
    bp = ax.boxplot(data_plot, patch_artist=True, widths=0.55,
                    medianprops=dict(linewidth=2, color="#FF9900"),
                    whiskerprops=dict(linewidth=1.2, color="#555555"),
                    capprops=dict(linewidth=1.2, color="#555555"),
                    flierprops=dict(marker="o", markersize=3, color="#888888"))

    for i, (patch, e) in enumerate(zip(bp["boxes"], entradas)):
        patch.set_facecolor(e["color"])
        patch.set_edgecolor("#555555" if e["tipo"] == "manual" else "#333333")
        patch.set_linewidth(1.5 if e["tipo"] == "manual" else 1.0)
        ax.text(i + 1, np.mean(e["tts"]) - 1,
                f'{np.mean(e["tts"]):.1f}', ha="center", va="top",
                fontsize=7.5, fontweight="bold", color="#333333")

    # Línea divisoria manual vs algoritmos
    n_manuales = len(ESCENARIOS_MANUALES)
    ax.axvline(n_manuales + 0.5, ls=":", color="#AAAAAA", lw=1.5)
    ax.text(n_manuales / 2 + 0.5, ax.get_ylim()[1] if ax.get_ylim()[1] < 400 else 350,
            "Manual\nScenarios", ha="center", va="bottom", fontsize=9,
            color="#555555", style="italic")
    ax.text(n_manuales + (len(entradas) - n_manuales) / 2 + 0.5,
            ax.get_ylim()[1] if ax.get_ylim()[1] < 400 else 350,
            "Algorithmic\nOptimization", ha="center", va="bottom",
            fontsize=9, color="#555555", style="italic")

    ax.set_xticklabels([e["label"] for e in entradas], fontsize=8.5, rotation=15, ha="right")
    ax.set_ylabel("Total Time in System (days)", fontsize=12)
    ax.set_title("Manual Scenarios vs Algorithmic Optimization — TTS Comparison",
                 fontsize=13)
    fig.tight_layout()
    path = out_dir / "boxplot_benchmark_vs_escenarios.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {path}")


# ═══════════════════════════════════════════════════════════════════════
# Tabla resumen + LaTeX
# ═══════════════════════════════════════════════════════════════════════
def tabla_resumen(benchmark: dict, out_dir: Path):
    lineas_csv  = ["Tipo,Método,TTS media,TTS SD,Pacientes media,Pacientes SD,n"]
    lineas_tex  = []
    lineas_test = ["Test Welch: algoritmos vs mejor escenario manual (Mgmt+Cap TTS=245.2)\n"]
    lineas_test.append(f"{'Algoritmo':<20} {'TTS media':>10} {'Δ vs Mgmt+Cap':>14} "
                       f"{'p-valor':>10} {'Cohen d':>8} {'n':>4}")
    lineas_test.append("-" * 70)

    # Mejor escenario manual de referencia para el test
    ref_tts  = ESCENARIOS_MANUALES["Mgmt + Capacity"]["tts_media"]
    ref_sd   = ESCENARIOS_MANUALES["Mgmt + Capacity"]["tts_sd"]
    ref_n    = ESCENARIOS_MANUALES["Mgmt + Capacity"]["n"]

    # Manuales al CSV/LaTeX
    for nombre, esc in ESCENARIOS_MANUALES.items():
        lineas_csv.append(
            f"Manual,{nombre},{esc['tts_media']:.1f},{esc['tts_sd']:.1f},"
            f"{esc['at_media']:.0f},{esc['at_sd']:.1f},{esc['n']}"
        )
        lineas_tex.append(
            f"Manual & {nombre} & ${esc['tts_media']:.1f}\\pm{esc['tts_sd']:.1f}$ & "
            f"${esc['at_media']:.0f}\\pm{esc['at_sd']:.0f}$ & {esc['n']} \\\\"
        )

    # Algoritmos
    lineas_tex.append("\\midrule")
    for mod, g in sorted(benchmark.items(), key=lambda x: x[1]["tts_media"]):
        est   = ESTILO_MOD.get(mod, {"label": mod})
        label = est["label"]
        n     = g["n"]
        lineas_csv.append(
            f"Algoritmo,{label},{g['tts_media']:.1f},{g['tts_sd']:.1f},"
            f"{g['at_media']:.0f},{g['at_sd']:.1f},{n}"
        )

        # Significancia vs Mgmt+Capacity
        delta = g["tts_media"] - ref_tts
        sig_str = "—"
        d_str   = "—"
        if _HAY_SCIPY and n >= 2:
            from scipy import stats as sps
            se = np.sqrt(g["tts_sd"]**2/n + ref_sd**2/ref_n)
            t  = abs(delta) / se if se > 0 else 0
            df_n = (g["tts_sd"]**2/n + ref_sd**2/ref_n)**2
            df_d = (g["tts_sd"]**2/n)**2/(n-1) + (ref_sd**2/ref_n)**2/(ref_n-1)
            df   = df_n / df_d if df_d > 0 else n + ref_n - 2
            p    = 2 * sps.t.sf(t, df)
            pooled_sd = np.sqrt((g["tts_sd"]**2 + ref_sd**2) / 2)
            d    = abs(delta) / pooled_sd if pooled_sd > 0 else 0
            sig_str = f"p={p:.4f}" + (" ***" if p < 0.001 else " **" if p < 0.01
                                       else " *" if p < 0.05 else " ns")
            d_str = f"{d:.2f}"
        elif n == 1:
            sig_str = "n=1 (parcial)"

        lineas_test.append(
            f"{label:<20} {g['tts_media']:>10.1f} {delta:>+14.1f} "
            f"{sig_str:>10} {d_str:>8} {n:>4}"
        )

        pval_tex = sig_str.replace("p=", "$p=").replace(" ***", "$\\,***") \
                          .replace(" ns", "$\\,ns").replace(" **", "$\\,**") \
                          .replace(" *", "$\\,*") if "=" in sig_str else sig_str
        lineas_tex.append(
            f"Algoritmo & {label} & ${g['tts_media']:.1f}\\pm{g['tts_sd']:.1f}$ & "
            f"${g['at_media']:.0f}\\pm{g['at_sd']:.0f}$ & {n} & "
            f"${delta:+.1f}$ & {pval_tex} \\\\"
        )

    # Guardar CSV
    (out_dir / "tabla_comparacion.csv").write_text("\n".join(lineas_csv))

    # Guardar LaTeX
    tex = r"""\begin{table}[htbp]
\centering
\caption{Algorithmic optimization vs manual scenarios.
         $\Delta$ TTS vs Mgmt+Capacity (best manual). Welch's $t$-test, $n$ seeds.}
\label{tab:benchmark_vs_manuales}
\small
\begin{tabular}{llrrrc}
\toprule
\textbf{Type} & \textbf{Method} & \textbf{TTS (days)} & \textbf{Patients} & $\boldsymbol{n}$ \\
\midrule
""" + "\n".join(lineas_tex) + r"""
\bottomrule
\multicolumn{5}{l}{\scriptsize *** $p<0.001$, ** $p<0.01$, * $p<0.05$, ns $p\geq0.05$ (Welch's $t$-test vs Mgmt+Capacity).}
\end{tabular}
\end{table}"""
    (out_dir / "tabla_comparacion.tex").write_text(tex)

    # Guardar tests
    (out_dir / "tests_estadisticos.txt").write_text("\n".join(lineas_test))

    print("\n" + "\n".join(lineas_test))
    print(f"\nCSV:  {out_dir/'tabla_comparacion.csv'}")
    print(f"LaTeX: {out_dir/'tabla_comparacion.tex'}")
    print(f"Tests: {out_dir/'tests_estadisticos.txt'}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--consolidado", required=True,
                    help="Ruta a consolidado_riguroso.json")
    ap.add_argument("--out", default="IFORS/figuras_v2/",
                    help="Directorio de salida para figuras y tablas")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    consolidado = Path(args.consolidado)

    if not consolidado.exists():
        print(f"ERROR: no se encuentra {consolidado}")
        return

    print(f"Cargando {consolidado} ...")
    benchmark = cargar_benchmark(consolidado)

    if not benchmark:
        print("ERROR: No se encontraron corridas válidas en el consolidado.")
        return

    print(f"Módulos encontrados: {list(benchmark.keys())}")
    for mod, g in benchmark.items():
        print(f"  {mod:10s}: n={g['n']}  TTS={g['tts_media']:.1f}±{g['tts_sd']:.1f}"
              f"  At={g['at_media']:.0f}±{g['at_sd']:.0f}")

    print("\nGenerando figuras...")
    figura_scatter(benchmark, out_dir)
    figura_boxplot(benchmark, out_dir)
    tabla_resumen(benchmark, out_dir)
    print("\nListo.")


if __name__ == "__main__":
    main()
