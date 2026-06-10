#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analisis_resultados.py
═══════════════════════════════════════════════════════════════════════
Análisis de los resultados del benchmark riguroso. Produce figuras y
tablas DEFENDIBLES para la presentación IFORS:

  1. Curvas de convergencia costo-vs-evaluaciones (media ± IC95, CRN).
  2. Boxplot del objetivo compuesto final re-evaluado.
  3. Boxplots separados de TTS y total_atenciones (explicabilidad).
  4. Tabla de significancia: Wilcoxon pareado por seed (CRN ⇒ pareo válido).
  5. Evaluaciones hasta umbral de calidad (eficiencia).

Lee consolidado_riguroso.json producido por benchmark_riguroso.py.

Uso:
    python analisis_resultados.py \
        --consolidado ifors_resultados/consolidado_riguroso.json \
        --baseline 270 --out figuras/
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy import stats as sps
    _HAY_SCIPY = True
except ImportError:
    _HAY_SCIPY = False

# Mapeo de label legible para cada módulo
LABELS = {
    "M4":  "M1 SMAC-GP+EI",
    "M7":  "M2 SMAC-SK(EI)",
    "M8":  "M3 SK-Adapt.",
    "M10": "M4 SK-KGCP",
    "M11": "M5 ASTRO-DF",
    "M13": "M6 SPSA",
    "RS":  "RS",
}


def cargar(consolidado: Path) -> list[dict]:
    return json.loads(consolidado.read_text())


def _agrupar_por_metodo(datos: list[dict]) -> dict[str, list[dict]]:
    """Agrupa por 'modulo' (benchmark_riguroso) o 'metodo' (harness_experimentos)."""
    g = {}
    for r in datos:
        key = r.get("modulo") or r.get("metodo") or "?"
        g.setdefault(key, []).append(r)
    return g


def _label(m: str) -> str:
    return LABELS.get(m, m)


def _nan(v):
    try:
        return float("nan") if v != v else float(v)
    except Exception:
        return float("nan")


def _kpi(r: dict, key: str) -> float:
    return _nan(r.get("kpis_incumbente", {}).get(key, float("nan")))


# ───────────────────────────────────────────────────────────────────────
# 1. Curvas de convergencia
# ───────────────────────────────────────────────────────────────────────

def curva_convergencia(grupos: dict, b_sim: int, out: Path,
                       grid_step: int = 1) -> None:
    grid = np.arange(1, b_sim + 1, grid_step)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for metodo, corridas in sorted(grupos.items()):
        curvas = []
        for r in corridas:
            traza = r.get("conv_eval") or r.get("traza", [])
            if not traza:
                continue
            ev   = np.array([t[0] for t in traza])
            best = np.array([t[2] if len(t) > 2 else t[1] for t in traza])
            y = np.empty_like(grid, dtype=float)
            for i, g in enumerate(grid):
                idx = np.searchsorted(ev, g, side="right") - 1
                y[i] = best[idx] if idx >= 0 else np.nan
            curvas.append(y)
        if not curvas:
            continue
        M     = np.vstack(curvas)
        media = np.nanmean(M, axis=0)
        n     = np.sum(~np.isnan(M), axis=0)
        sd    = np.nanstd(M, axis=0, ddof=1)
        ic    = 1.96 * sd / np.sqrt(np.maximum(n, 1))
        ax.plot(grid, media, label=f"{_label(metodo)} (n={len(curvas)})", lw=2)
        ax.fill_between(grid, media - ic, media + ic, alpha=0.18)

    ax.set_xlabel("Evaluaciones del simulador (acumuladas)")
    ax.set_ylabel("Mejor objetivo incumbente")
    ax.set_title("Convergencia a presupuesto igualado\n(media ± IC95 sobre macro-réplicas, CRN)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "convergencia_bandas.png", dpi=150)
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────────
# 2. Boxplot del objetivo compuesto final re-evaluado
# ───────────────────────────────────────────────────────────────────────

def boxplot_objetivo(grupos: dict, baseline: float, out: Path) -> None:
    metodos = sorted(grupos.keys())
    data    = [[_nan(r["reeval"]["media"]) for r in grupos[m]
                if "reeval" in r] for m in metodos]
    labels  = [_label(m) for m in metodos]

    fig, ax = plt.subplots(figsize=(9, 5))
    try:
        ax.boxplot(data, tick_labels=labels, showmeans=True)
    except TypeError:
        ax.boxplot(data, labels=labels, showmeans=True)
    ax.axhline(baseline, ls="--", color="red", alpha=0.7,
               label=f"Baseline TTS ({baseline:.0f} d)")
    ax.set_ylabel("Objetivo compuesto re-evaluado")
    ax.set_title("Distribución del objetivo final por método\n"
                 "(incumbente re-evaluado con réplicas frescas)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "boxplot_objetivo.png", dpi=150)
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────────
# 3. Boxplots separados: TTS y Total Atenciones (explicabilidad)
# ───────────────────────────────────────────────────────────────────────

def boxplot_kpis_separados(grupos: dict, baseline_tts: float, out: Path) -> None:
    """
    Panel izquierdo: TTS total (días) por método.
    Panel derecho:   Total atenciones por método.
    Permite explicar el trade-off del objetivo compuesto.
    """
    metodos = sorted(grupos.keys())
    labels  = [_label(m) for m in metodos]

    tts_data = []
    at_data  = []
    for m in metodos:
        tts_vals = [_kpi(r, "tts_media") for r in grupos[m]
                    if "kpis_incumbente" in r]
        at_vals  = [_kpi(r, "at_media")  for r in grupos[m]
                    if "kpis_incumbente" in r]
        tts_data.append([v for v in tts_vals if v == v])   # quitar nan
        at_data.append( [v for v in at_vals  if v == v])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # — TTS —
    try:
        ax1.boxplot(tts_data, tick_labels=labels, showmeans=True)
    except TypeError:
        ax1.boxplot(tts_data, labels=labels, showmeans=True)
    ax1.axhline(baseline_tts, ls="--", color="red", alpha=0.7,
                label=f"Baseline ({baseline_tts:.0f} d)")
    ax1.set_ylabel("TTS total (días)")
    ax1.set_title("Tiempo de tratamiento total\n(TTS — menor es mejor)")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3, axis="y")
    ax1.tick_params(axis="x", rotation=30)

    # — Atenciones —
    try:
        ax2.boxplot(at_data, tick_labels=labels, showmeans=True)
    except TypeError:
        ax2.boxplot(at_data, labels=labels, showmeans=True)
    ax2.set_ylabel("Total atenciones (pacientes)")
    ax2.set_title("Total atenciones realizadas\n(mayor es mejor)")
    ax2.grid(alpha=0.3, axis="y")
    ax2.tick_params(axis="x", rotation=30)

    fig.suptitle("KPIs individuales del incumbente por método\n"
                 "(re-evaluado con réplicas frescas)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "boxplot_kpis_separados.png", dpi=150)
    plt.close(fig)

    # Figura adicional: dispersión TTS vs Atenciones (trade-off)
    fig2, ax3 = plt.subplots(figsize=(8, 6))
    for m in metodos:
        tts_v = [_kpi(r, "tts_media") for r in grupos[m] if "kpis_incumbente" in r]
        at_v  = [_kpi(r, "at_media")  for r in grupos[m] if "kpis_incumbente" in r]
        tts_v = [v for v in tts_v if v == v]
        at_v  = [v for v in at_v  if v == v]
        if tts_v and at_v:
            ax3.scatter(at_v, tts_v, label=_label(m), s=50, alpha=0.7)
            ax3.annotate(_label(m),
                         (float(np.mean(at_v)), float(np.mean(tts_v))),
                         fontsize=8, ha="center", va="bottom")
    ax3.axhline(baseline_tts, ls="--", color="red", alpha=0.5,
                label=f"Baseline TTS ({baseline_tts:.0f} d)")
    ax3.set_xlabel("Total atenciones (mayor → mejor)")
    ax3.set_ylabel("TTS total — días (menor → mejor)")
    ax3.set_title("Trade-off: TTS vs Atenciones por método\n"
                  "(cada punto = 1 incumbente re-evaluado)")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(out / "scatter_tts_vs_atenciones.png", dpi=150)
    plt.close(fig2)


# ───────────────────────────────────────────────────────────────────────
# 4. Tabla de significancia (Wilcoxon pareado, CRN)
# ───────────────────────────────────────────────────────────────────────

def tabla_significancia(grupos: dict, out: Path) -> str:
    metodos  = sorted(grupos.keys())
    por_seed = {
        m: {r["macro_seed"]: _nan(r["reeval"]["media"])
            for r in grupos[m] if "reeval" in r}
        for m in metodos
    }
    lineas = ["# Significancia estadística (Wilcoxon pareado por seed)\n"]
    lineas.append(f"{'Comparación':<28}{'Δ medio':>12}{'p-valor':>12}{'Significativo':>16}")
    lineas.append("-" * 70)
    for i, a in enumerate(metodos):
        for b in metodos[i+1:]:
            seeds = sorted(set(por_seed[a]) & set(por_seed[b]))
            xa = np.array([por_seed[a][s] for s in seeds])
            xb = np.array([por_seed[b][s] for s in seeds])
            dif = float(np.mean(xa - xb))
            if _HAY_SCIPY and len(seeds) >= 6:
                try:
                    _, p = sps.wilcoxon(xa, xb)
                except ValueError:
                    p = float("nan")
            else:
                p = float("nan")
            sig = "sí (p<0.05)" if (p == p and p < 0.05) else ("no" if p == p else "n/d")
            lineas.append(f"{_label(a) + ' vs ' + _label(b):<28}{dif:>12.2f}{p:>12.4f}{sig:>16}")
    txt = "\n".join(lineas)
    (out / "significancia.txt").write_text(txt)
    return txt


# ───────────────────────────────────────────────────────────────────────
# 5. Evaluaciones hasta umbral de calidad
# ───────────────────────────────────────────────────────────────────────

def evaluaciones_hasta_umbral(grupos: dict, baseline: float, out: Path,
                               pct_objetivo: float = 0.15) -> str:
    umbral = baseline * (1 - pct_objetivo)
    lineas = [f"# Evaluaciones hasta alcanzar {umbral:.1f} d "
              f"(−{pct_objetivo:.0%} vs baseline {baseline:.0f})\n"]
    lineas.append(f"{'Método':<16}{'media eval':>12}{'alcanzaron':>14}")
    lineas.append("-" * 44)
    for metodo, corridas in sorted(grupos.items()):
        evs = []
        for r in corridas:
            traza = r.get("conv_eval") or r.get("traza", [])
            alcanzado = next(
                (t[0] for t in traza if (t[2] if len(t) > 2 else t[1]) <= umbral),
                None
            )
            if alcanzado is not None:
                evs.append(alcanzado)
        if evs:
            lineas.append(f"{_label(metodo):<16}{np.mean(evs):>12.1f}"
                          f"{f'{len(evs)}/{len(corridas)}':>14}")
        else:
            lineas.append(f"{_label(metodo):<16}{'—':>12}{f'0/{len(corridas)}':>14}")
    txt = "\n".join(lineas)
    (out / "eval_hasta_umbral.txt").write_text(txt)
    return txt


# ───────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Análisis post-benchmark: figuras y tablas para IFORS.")
    p.add_argument("--consolidado", required=True,
                   help="Path al consolidado_riguroso.json")
    p.add_argument("--baseline", type=float, default=270.0,
                   help="TTS baseline (días) para línea de referencia.")
    p.add_argument("--b_sim", type=int, default=150,
                   help="Presupuesto máximo de evaluaciones (eje x convergencia).")
    p.add_argument("--out", default="figuras",
                   help="Directorio de salida para figuras y tablas.")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    datos  = cargar(Path(args.consolidado))
    grupos = _agrupar_por_metodo(datos)

    print(f"Métodos encontrados: {sorted(grupos.keys())}")
    print(f"Seeds por método: { {m: len(grupos[m]) for m in grupos} }")

    curva_convergencia(grupos, args.b_sim, out)
    print("✓ convergencia_bandas.png")

    boxplot_objetivo(grupos, args.baseline, out)
    print("✓ boxplot_objetivo.png")

    boxplot_kpis_separados(grupos, args.baseline, out)
    print("✓ boxplot_kpis_separados.png  |  scatter_tts_vs_atenciones.png")

    print("\n" + tabla_significancia(grupos, out))
    print("✓ significancia.txt")

    print("\n" + evaluaciones_hasta_umbral(grupos, args.baseline, out))
    print("✓ eval_hasta_umbral.txt")

    print(f"\nTodas las figuras y tablas en: {out}/")


if __name__ == "__main__":
    main()
