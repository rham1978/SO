#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analisis_resultados.py
═══════════════════════════════════════════════════════════════════════
Análisis de los resultados del harness. Produce las figuras y los
estadísticos DEFENDIBLES que reemplazan la tabla de "mejor costo" del
benchmark original:

  1. Curvas de convergencia costo-vs-evaluaciones, media ± IC95 sobre
     macro-réplicas, todos los métodos en el mismo eje a presupuesto común.
  2. Boxplots del costo final RE-EVALUADO (valor esperado real, no ruidoso).
  3. Tabla de significancia: ¿las diferencias entre métodos son reales?
     (test de Mann-Whitney pareado por seed, dado que con CRN las seeds
     están pareadas entre métodos).
  4. "Evaluaciones hasta umbral": cuántas evaluaciones necesita cada método
     para alcanzar -X% sobre baseline (métrica de eficiencia).

Lee el consolidado.json producido por harness_experimentos.py.

Uso:
    python analisis_resultados.py --consolidado resultados/consolidado.json \
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


def cargar(consolidado: Path) -> list[dict]:
    return json.loads(consolidado.read_text())


def _agrupar_por_metodo(datos: list[dict]) -> dict[str, list[dict]]:
    g = {}
    for r in datos:
        g.setdefault(r["metodo"], []).append(r)
    return g


# ───────────────────────────────────────────────────────────────────────
# 1. Curvas de convergencia con banda de confianza
# ───────────────────────────────────────────────────────────────────────

def curva_convergencia(grupos: dict, b_sim: int, out: Path,
                       grid_step: int = 1) -> None:
    """
    Interpola la traza de cada corrida a una grilla común de evaluaciones
    [1..b_sim] (escalera: mejor-hasta-ahora), promedia sobre seeds y dibuja
    media ± IC95.
    """
    grid = np.arange(1, b_sim + 1, grid_step)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for metodo, corridas in sorted(grupos.items()):
        curvas = []
        for r in corridas:
            traza = r["traza"]                       # [(n_eval, costo, mejor)]
            if not traza:
                continue
            ev   = np.array([t[0] for t in traza])
            best = np.array([t[2] for t in traza])    # mejor-hasta-ahora
            # Escalera: para cada punto de la grilla, el mejor alcanzado con <= ese nº de eval
            y = np.empty_like(grid, dtype=float)
            for i, g in enumerate(grid):
                idx = np.searchsorted(ev, g, side="right") - 1
                y[i] = best[idx] if idx >= 0 else np.nan
            curvas.append(y)
        if not curvas:
            continue
        M = np.vstack(curvas)
        media = np.nanmean(M, axis=0)
        n     = np.sum(~np.isnan(M), axis=0)
        sd    = np.nanstd(M, axis=0, ddof=1)
        ic    = 1.96 * sd / np.sqrt(np.maximum(n, 1))

        ax.plot(grid, media, label=f"{metodo} (n={len(curvas)})", lw=2)
        ax.fill_between(grid, media - ic, media + ic, alpha=0.18)

    ax.set_xlabel("Evaluaciones del simulador (acumuladas)")
    ax.set_ylabel("Mejor costo incumbente — TTS total (días)")
    ax.set_title("Convergencia a presupuesto igualado\n(media ± IC95 sobre macro-réplicas, CRN)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "convergencia_bandas.png", dpi=150)
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────────
# 2. Boxplots del costo final re-evaluado
# ───────────────────────────────────────────────────────────────────────

def boxplot_final(grupos: dict, baseline: float, out: Path) -> None:
    metodos = sorted(grupos.keys())
    data = [[r["reeval"]["media"] for r in grupos[m]] for m in metodos]

    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        ax.boxplot(data, tick_labels=metodos, showmeans=True)   # mpl ≥ 3.9
    except TypeError:
        ax.boxplot(data, labels=metodos, showmeans=True)        # mpl < 3.9
    ax.axhline(baseline, ls="--", color="red", alpha=0.7,
               label=f"Línea base ({baseline:.0f} d)")
    ax.set_ylabel("Costo final re-evaluado — TTS total (días)")
    ax.set_title("Distribución del costo final por método\n(incumbente re-evaluado con réplicas frescas)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "boxplot_final.png", dpi=150)
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────────
# 3. Significancia estadística pareada por seed (CRN ⇒ seeds pareadas)
# ───────────────────────────────────────────────────────────────────────

def tabla_significancia(grupos: dict, out: Path) -> str:
    """
    Para cada par de métodos, test de Wilcoxon pareado sobre el costo final
    re-evaluado, emparejando por macro_seed (válido porque CRN garantiza que
    la seed s usó los mismos números aleatorios en ambos métodos).
    """
    metodos = sorted(grupos.keys())
    # Mapa metodo → {seed: costo_reeval}
    por_seed = {
        m: {r["macro_seed"]: r["reeval"]["media"] for r in grupos[m]}
        for m in metodos
    }
    lineas = ["# Significancia estadística (Wilcoxon pareado por seed)\n"]
    lineas.append(f"{'Comparación':<28}{'Δ medio (d)':>14}{'p-valor':>12}{'¿Significativo?':>18}")
    lineas.append("-" * 72)
    for i, a in enumerate(metodos):
        for b in metodos[i+1:]:
            seeds_comunes = sorted(set(por_seed[a]) & set(por_seed[b]))
            xa = np.array([por_seed[a][s] for s in seeds_comunes])
            xb = np.array([por_seed[b][s] for s in seeds_comunes])
            dif = float(np.mean(xa - xb))
            if _HAY_SCIPY and len(seeds_comunes) >= 6:
                try:
                    _, p = sps.wilcoxon(xa, xb)
                except ValueError:
                    p = float("nan")
            else:
                p = float("nan")
            sig = "sí" if (p == p and p < 0.05) else ("no" if p == p else "n/d")
            lineas.append(f"{a + ' vs ' + b:<28}{dif:>14.2f}{p:>12.4f}{sig:>18}")
    txt = "\n".join(lineas)
    (out / "significancia.txt").write_text(txt)
    return txt


# ───────────────────────────────────────────────────────────────────────
# 4. Evaluaciones hasta umbral de calidad
# ───────────────────────────────────────────────────────────────────────

def evaluaciones_hasta_umbral(grupos: dict, baseline: float, out: Path,
                              pct_objetivo: float = 0.15) -> str:
    """
    Para cada método, nº de evaluaciones para alcanzar (1-pct_objetivo)*baseline,
    promediado sobre seeds (las que lo alcanzaron). Métrica de eficiencia.
    """
    umbral = baseline * (1 - pct_objetivo)
    lineas = [f"# Evaluaciones hasta alcanzar {umbral:.1f} d (−{pct_objetivo:.0%} vs baseline {baseline:.0f})\n"]
    lineas.append(f"{'Método':<12}{'media eval':>12}{'alcanzaron':>14}")
    lineas.append("-" * 40)
    for metodo, corridas in sorted(grupos.items()):
        evs = []
        for r in corridas:
            alcanzado = next((n for (n, _, mejor) in r["traza"] if mejor <= umbral), None)
            if alcanzado is not None:
                evs.append(alcanzado)
        if evs:
            lineas.append(f"{metodo:<12}{np.mean(evs):>12.1f}{f'{len(evs)}/{len(corridas)}':>14}")
        else:
            lineas.append(f"{metodo:<12}{'—':>12}{f'0/{len(corridas)}':>14}")
    txt = "\n".join(lineas)
    (out / "eval_hasta_umbral.txt").write_text(txt)
    return txt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--consolidado", required=True)
    p.add_argument("--baseline", type=float, default=270.0)
    p.add_argument("--b_sim", type=int, default=150)
    p.add_argument("--out", default="figuras")
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    datos  = cargar(Path(args.consolidado))
    grupos = _agrupar_por_metodo(datos)

    curva_convergencia(grupos, args.b_sim, out)
    boxplot_final(grupos, args.baseline, out)
    print(tabla_significancia(grupos, out))
    print()
    print(evaluaciones_hasta_umbral(grupos, args.baseline, out))
    print(f"\nFiguras y tablas en: {out}/")


if __name__ == "__main__":
    main()
