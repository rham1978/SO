#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
caracterizacion_heterocedasticidad.py
═══════════════════════════════════════════════════════════════════════
Experimento que ANCLA la narrativa de la tesis y de IFORS: demostrar
empíricamente que la varianza del evaluador DES depende del punto del
espacio de decisión (heterocedasticidad estructural), y caracterizar
CÓMO depende (típicamente: se dispara cerca de la saturación del sistema).

Diseño:
  - Selecciona K puntos que cubren distintos regímenes operativos, desde
    baja utilización (mucha capacidad) hasta alta utilización (cerca de
    saturación). NO solo alrededor del óptimo.
  - En cada punto corre R réplicas independientes del simulador.
  - Estima σ̂²(x) y la relaciona con una medida de régimen (utilización de
    especialistas / agentes / matronas, que el simulador ya reporta).
  - Prueba formal de heterocedasticidad (Levene) sobre los grupos.

Produce la figura: σ̂(x) vs utilización del sistema → la evidencia de que
la varianza no es constante.

Uso:
    python caracterizacion_heterocedasticidad.py --k_puntos 18 --r 50 \
        --n_cores 9 --out heterocedasticidad/
"""

from __future__ import annotations
import argparse
import concurrent.futures
import dataclasses
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


def _puntos_por_regimen(k: int) -> list[np.ndarray]:
    """
    Genera k puntos en [0,1]^12 ordenados de 'mucha capacidad' (baja
    utilización) a 'poca capacidad' (alta utilización / saturación).

    Estrategia: barrer un factor de capacidad global t∈[0,1]. Para las
    variables de capacidad (cupos, horas, agentes, matronas) t alto = más
    capacidad; para las de pérdida (bloqueo, no contactabilidad, vacías)
    t alto = menos pérdida. Así t≈0 satura el sistema y t≈1 lo holgura.
    Esto produce un gradiente de regímenes operativos, que es justo lo que
    necesitamos para ver cómo cambia la varianza.
    """
    from modulo_comparativa_caja_negra import PARAM_NAMES
    CAPACIDAD = {
        "horas_especialista_1ra", "horas_control_post", "cupos_laboratorio_ugd",
        "cupos_ecografia_matrona", "cupos_ecografia_ugd",
        "num_matronas", "num_agentes_ugd",
    }
    PERDIDA = {
        "pct_bloqueo_1ra", "pct_consultas_vacias",
        "pct_no_contactabilidad", "pct_bloqueo_post_control",
    }
    # dias_publicacion: neutro, fijar a la mitad
    puntos = []
    for t in np.linspace(0.05, 0.95, k):
        x = np.full(12, 0.5)
        for i, nombre in enumerate(PARAM_NAMES):
            if nombre in CAPACIDAD:
                x[i] = t              # más t = más capacidad
            elif nombre in PERDIDA:
                x[i] = 1.0 - t        # más t = menos pérdida
            else:
                x[i] = 0.5
        puntos.append(x)
    return puntos


def _worker(args):
    """Corre una réplica en un punto. args = (cfg_dict, seed_offset, claves_util)."""
    from simulador_clinica_baseline import run_once, SimConfig
    cfg_dict, seed_offset = args
    cfg = SimConfig(**cfg_dict)
    res = run_once(seed_offset=seed_offset, cfg=cfg)
    return {
        "tts":            float(res["tts_full_days_mean"]),
        "util_spec_first": float(res.get("spec_util_first_pct", 0.0)),
        "util_spec_post":  float(res.get("spec_util_post_pct", 0.0)),
        "util_agent":      float(res.get("agent_util_first_pct", 0.0)),
        "util_matrona":    float(res.get("matrona_util_post_pct", 0.0)),
    }


def caracterizar(k_puntos: int, r: int, n_cores: int, out: Path) -> None:
    from harness_experimentos import _cfg_desde_vector

    out.mkdir(parents=True, exist_ok=True)
    puntos = _puntos_por_regimen(k_puntos)

    registros = []
    for j, x in enumerate(puntos):
        cfg_dict = _cfg_desde_vector(x)
        tasks = [(cfg_dict, 500_000 + j * r + rr) for rr in range(r)]  # offsets exclusivos
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as ex:
            reps = list(ex.map(_worker, tasks))

        tts = np.array([d["tts"] for d in reps])
        util = float(np.mean([d["util_spec_first"] for d in reps]))
        reg = {
            "punto":    j,
            "util_spec": util,
            "tts_mean": float(tts.mean()),
            "tts_var":  float(tts.var(ddof=1)),
            "tts_sd":   float(tts.std(ddof=1)),
            "tts_raw":  tts.tolist(),
        }
        registros.append(reg)
        print(f"punto {j:2d}/{k_puntos}: util_spec={util:5.1f}%  "
              f"TTS={tts.mean():6.1f}±{tts.std(ddof=1):5.1f} d")

    (out / "heterocedasticidad.json").write_text(
        json.dumps(registros, indent=2, default=str))

    # ── Prueba formal de heterocedasticidad (Levene) ──
    if _HAY_SCIPY:
        grupos = [np.array(r_["tts_raw"]) for r_ in registros]
        W, p = sps.levene(*grupos, center="median")
        veredicto = ("RECHAZA homocedasticidad (varianza NO constante)"
                     if p < 0.05 else "no rechaza homocedasticidad")
        (out / "levene.txt").write_text(
            f"Test de Levene (center=median)\nW = {W:.4f}\np = {p:.3e}\n→ {veredicto}\n")
        print(f"\nLevene: W={W:.3f}, p={p:.3e} → {veredicto}")

    # ── Figura: σ̂ vs utilización ──
    util = np.array([r_["util_spec"] for r_ in registros])
    sd   = np.array([r_["tts_sd"]    for r_ in registros])
    mean = np.array([r_["tts_mean"]  for r_ in registros])
    orden = np.argsort(util)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(util[orden], sd[orden], "o-", color="#C0392B", lw=2)
    ax1.set_xlabel("Utilización de especialistas (%)")
    ax1.set_ylabel("Desviación estándar del TTS (días)")
    ax1.set_title("Heterocedasticidad estructural:\nla varianza crece con la utilización")
    ax1.grid(alpha=0.3)

    ax2.errorbar(util[orden], mean[orden], yerr=sd[orden], fmt="o-",
                 color="#2E5A88", lw=2, capsize=3)
    ax2.set_xlabel("Utilización de especialistas (%)")
    ax2.set_ylabel("TTS total (días)")
    ax2.set_title("Costo medio ± σ por régimen operativo")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "heterocedasticidad.png", dpi=150)
    plt.close(fig)
    print(f"\nFigura y datos en: {out}/")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k_puntos", type=int, default=18)
    p.add_argument("--r", type=int, default=50)
    p.add_argument("--n_cores", type=int, default=9)
    p.add_argument("--out", default="heterocedasticidad")
    args = p.parse_args()
    caracterizar(args.k_puntos, args.r, args.n_cores, Path(args.out))


if __name__ == "__main__":
    main()
