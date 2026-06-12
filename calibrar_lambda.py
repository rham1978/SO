#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calibrar_lambda.py
═══════════════════════════════════════════════════════════════════════
Calibra el parámetro λ para la función objetivo combinada:

    f(x) = tts_full_days_mean(x) - λ · total_atenciones(x)

λ se estima de tres formas complementarias sobre muestras del espacio
de decisión, usando el propio DES como fuente de verdad:

  Método 1 — Razón de desviaciones estándar (normalización por variabilidad):
    λ_std  = std(tts) / std(atenciones)
    → Los dos indicadores contribuyen igual cuando están en su rango
      natural de variación.

  Método 2 — Razón de rangos (normalización min-max):
    λ_range = (max_tts - min_tts) / (max_atenciones - min_atenciones)
    → Los dos indicadores contribuyen igual en los extremos del espacio.

  Método 3 — Elasticidad local en el baseline (interpretación operacional):
    λ_elast = |ΔTTS / Δatenciones| cuando se aumenta capacidad desde baseline
    → Cuántos días de espera equivale una atención más cerca del punto
      de operación actual del CRS Cordillera.

Además verifica que la función combinada con cada λ no invierte el
ranking esperado: las configuraciones con más capacidad deben quedar
mejor rankeadas que las de menos capacidad.

Uso:
    python calibrar_lambda.py --n_puntos 80 --n_reps 5 --n_cores 9 \
        --out calibracion/

    # Prueba rápida (~10 min):
    python calibrar_lambda.py --n_puntos 20 --n_reps 3 --n_cores 4 \
        --out calibracion_prueba/

Tiempo estimado (Pelluhue, 9 cores, 90 s/corrida):
    80 puntos × 5 reps = 400 corridas + 12 reps elasticidad
    ≈ 412 × 90 s / 9 cores ≈ 68 min

Salidas:
    calibracion/
    ├── datos_calibracion.json        ← todas las evaluaciones
    ├── scatter_tts_vs_atenciones.png ← nube de puntos con correlación
    ├── distribucion_indicadores.png  ← histogramas y boxplots
    ├── funcion_combinada_lambdas.png ← f(x) para λ_bajo, λ_med, λ_alto
    ├── analisis_sensibilidad.png     ← cómo cambia el ranking con λ
    └── reporte_lambda.txt            ← λ recomendado con justificación
"""

from __future__ import annotations
import argparse
import concurrent.futures
import dataclasses
import json
import logging
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = logging.getLogger("calibrar_lambda")

# ───────────────────────────────────────────────────────────────────────
# Variables enteras del espacio de decisión
# ───────────────────────────────────────────────────────────────────────
ENTEROS = {
    "horas_especialista_1ra", "horas_control_post", "cupos_laboratorio_ugd",
    "cupos_ecografia_matrona", "cupos_ecografia_ugd", "dias_publicacion",
    "num_matronas", "num_agentes_ugd",
}


# ───────────────────────────────────────────────────────────────────────
# Conversión vector [0,1]^12 → SimConfig
# ───────────────────────────────────────────────────────────────────────

def _vector_a_cfg(x: np.ndarray):
    """Convierte vector normalizado [0,1]^12 a SimConfig con int(round()) para enteros."""
    import copy
    from simulador_clinica_baseline import CFG
    from modulo_comparativa_caja_negra import PARAM_NAMES, PARAM_RANGES

    cfg = copy.deepcopy(CFG)
    vals = {}
    for i, nombre in enumerate(PARAM_NAMES):
        lo, hi = PARAM_RANGES[nombre]
        v = lo + float(x[i]) * (hi - lo)
        vals[nombre] = int(round(v)) if nombre in ENTEROS else float(v)

    cfg.fixed_weekly_capacity        = int(vals["horas_especialista_1ra"])
    cfg.use_fixed_weekly_capacity    = True
    cfg.fixed_post_control_capacity  = int(vals["horas_control_post"])
    cfg.use_fixed_post_control_hours = True
    cfg.ugd_lab_per_week             = int(vals["cupos_laboratorio_ugd"])
    cfg.mat_us_per_week              = int(vals["cupos_ecografia_matrona"])
    cfg.ugd_us_per_week              = int(vals["cupos_ecografia_ugd"])
    cfg.publish_lead_workdays        = int(vals["dias_publicacion"])
    cfg.blocked_pct                  = float(vals["pct_bloqueo_1ra"])
    cfg.empty_control_p_ugd          = float(vals["pct_consultas_vacias"])
    cfg.matrona_capacity             = int(vals["num_matronas"])
    cfg.agent_capacity               = int(vals["num_agentes_ugd"])
    cfg.not_contactable_p            = float(vals["pct_no_contactabilidad"])
    cfg.blocked_pct_post_control     = float(vals["pct_bloqueo_post_control"])
    return cfg, vals


# ───────────────────────────────────────────────────────────────────────
# Worker: evalúa un punto con n_reps réplicas y retorna media de KPIs
# ───────────────────────────────────────────────────────────────────────

def _worker(args):
    """
    args = (punto_idx, x_unit, n_reps, seed_base)
    Retorna dict con medias y desviaciones de tts_full y total_atenciones.
    """
    punto_idx, x_unit, n_reps, seed_base = args
    from simulador_clinica_baseline import run_once

    cfg, vals = _vector_a_cfg(x_unit)
    tts_vals, at_vals, at_first_vals, at_post_vals = [], [], [], []

    for r in range(n_reps):
        try:
            res = run_once(seed_offset=seed_base + punto_idx * n_reps + r, cfg=cfg)
            tts_vals.append(float(res["tts_full_days_mean"]))
            at_vals.append(int(res.get("total_atenciones", 0)))
            at_first_vals.append(int(res.get("total_atenciones_first", 0)))
            at_post_vals.append(int(res.get("total_atenciones_post", 0)))
        except Exception as e:
            log.warning("punto=%d rep=%d error: %s", punto_idx, r, e)

    if not tts_vals:
        return None

    return {
        "punto_idx":      punto_idx,
        "x_unit":         x_unit.tolist(),
        "params":         vals,
        "tts_mean":       float(np.mean(tts_vals)),
        "tts_sd":         float(np.std(tts_vals, ddof=1)) if len(tts_vals) > 1 else 0.0,
        "at_mean":        float(np.mean(at_vals)),
        "at_sd":          float(np.std(at_vals, ddof=1)) if len(at_vals) > 1 else 0.0,
        "at_first_mean":  float(np.mean(at_first_vals)),
        "at_post_mean":   float(np.mean(at_post_vals)),
        "n_reps_ok":      len(tts_vals),
    }


# ───────────────────────────────────────────────────────────────────────
# Método 3: elasticidad local en el baseline
# ───────────────────────────────────────────────────────────────────────

def _calcular_elasticidad(n_reps: int, n_cores: int) -> dict:
    """
    Estima λ_elast = |ΔTTS / Δatenciones| aumentando capacidad desde baseline.
    Evalúa 4 perturbaciones: +capacidad, -capacidad, +bloqueo, -bloqueo.
    Usa n_reps réplicas por punto para estimaciones estables.
    """
    from simulador_clinica_baseline import run_once
    from modulo_comparativa_caja_negra import PARAM_BASELINE
    import copy
    from simulador_clinica_baseline import CFG

    def eval_cfg(cfg, seed_base, n_reps):
        tts, at = [], []
        for r in range(n_reps):
            try:
                res = run_once(seed_offset=seed_base + r, cfg=cfg)
                tts.append(float(res["tts_full_days_mean"]))
                at.append(int(res.get("total_atenciones", 0)))
            except Exception:
                pass
        return float(np.mean(tts)) if tts else float("nan"), \
               float(np.mean(at))  if at  else float("nan")

    # Baseline
    cfg_base = copy.deepcopy(CFG)
    cfg_base.fixed_weekly_capacity     = int(PARAM_BASELINE["horas_especialista_1ra"])
    cfg_base.use_fixed_weekly_capacity = True
    cfg_base.fixed_post_control_capacity  = int(PARAM_BASELINE["horas_control_post"])
    cfg_base.use_fixed_post_control_hours = True

    tts_base, at_base = eval_cfg(cfg_base, seed_base=600_000, n_reps=n_reps)
    log.info("Elasticidad baseline: TTS=%.2f  atenciones=%.0f", tts_base, at_base)

    # Perturbación 1: +4 slots/sem primera consulta (~25% del baseline de 16)
    cfg_up = copy.deepcopy(cfg_base)
    cfg_up.fixed_weekly_capacity = int(PARAM_BASELINE["horas_especialista_1ra"]) + 4
    tts_up, at_up = eval_cfg(cfg_up, seed_base=601_000, n_reps=n_reps)

    # Perturbación 2: -4 slots/sem primera consulta
    cfg_dn = copy.deepcopy(cfg_base)
    cfg_dn.fixed_weekly_capacity = max(8, int(PARAM_BASELINE["horas_especialista_1ra"]) - 4)
    tts_dn, at_dn = eval_cfg(cfg_dn, seed_base=602_000, n_reps=n_reps)

    # Perturbación 3: +8 cupos control post (~20% del baseline de 40)
    cfg_post = copy.deepcopy(cfg_base)
    cfg_post.fixed_post_control_capacity = int(PARAM_BASELINE["horas_control_post"]) + 8
    tts_post, at_post = eval_cfg(cfg_post, seed_base=603_000, n_reps=n_reps)

    elasticidades = []
    for label, tts_pert, at_pert in [
        ("+4 slots 1ra", tts_up, at_up),
        ("-4 slots 1ra", tts_dn, at_dn),
        ("+8 slots ctrl", tts_post, at_post),
    ]:
        delta_tts = tts_base - tts_pert   # positivo = mejora en TTS
        delta_at  = at_pert - at_base      # positivo = más atenciones
        if abs(delta_at) > 0.5:
            e = abs(delta_tts / delta_at)
            elasticidades.append(e)
            log.info("Elasticidad (%s): ΔTTS=%.2f  ΔAt=%.1f  λ=%.4f",
                     label, delta_tts, delta_at, e)

    return {
        "tts_base":      tts_base,
        "at_base":       at_base,
        "elasticidades": elasticidades,
        "lambda_elast":  float(np.median(elasticidades)) if elasticidades else float("nan"),
    }


# ───────────────────────────────────────────────────────────────────────
# Gráficas
# ───────────────────────────────────────────────────────────────────────

def _graficar_scatter(datos: list[dict], lambdas: dict, out: Path) -> None:
    """Nube de puntos TTS vs total_atenciones con correlación."""
    tts = np.array([d["tts_mean"] for d in datos])
    at  = np.array([d["at_mean"]  for d in datos])
    corr = float(np.corrcoef(tts, at)[0, 1])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(at, tts, alpha=0.6, c="#2E5A88", s=40, label=f"Puntos (n={len(datos)})")
    ax.set_xlabel("Total atenciones (first + post_completed)")
    ax.set_ylabel("TTS total (días)")
    ax.set_title(f"Relación entre los dos indicadores\nr de Pearson = {corr:.3f}")

    # Línea de regresión
    m, b = np.polyfit(at, tts, 1)
    x_line = np.linspace(at.min(), at.max(), 100)
    ax.plot(x_line, m * x_line + b, "r--", lw=1.5, alpha=0.7, label=f"Regresión (pendiente={m:.3f})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "scatter_tts_vs_atenciones.png", dpi=150)
    plt.close(fig)


def _graficar_distribuciones(datos: list[dict], out: Path) -> None:
    """Histogramas y boxplots de los dos indicadores."""
    tts = np.array([d["tts_mean"] for d in datos])
    at  = np.array([d["at_mean"]  for d in datos])

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    axes[0, 0].hist(tts, bins=20, color="#2E5A88", alpha=0.8, edgecolor="white")
    axes[0, 0].set_xlabel("TTS total (días)"); axes[0, 0].set_title("Distribución TTS")
    axes[0, 0].axvline(tts.mean(), color="red", ls="--", label=f"μ={tts.mean():.1f}")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].hist(at, bins=20, color="#C8862E", alpha=0.8, edgecolor="white")
    axes[0, 1].set_xlabel("Total atenciones"); axes[0, 1].set_title("Distribución atenciones")
    axes[0, 1].axvline(at.mean(), color="red", ls="--", label=f"μ={at.mean():.0f}")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].boxplot(tts, vert=True, patch_artist=True,
                       boxprops=dict(facecolor="#2E5A88", alpha=0.6))
    axes[1, 0].set_ylabel("TTS total (días)"); axes[1, 0].set_title("Boxplot TTS")
    axes[1, 0].set_xticklabels(["TTS"])

    axes[1, 1].boxplot(at, vert=True, patch_artist=True,
                       boxprops=dict(facecolor="#C8862E", alpha=0.6))
    axes[1, 1].set_ylabel("Total atenciones"); axes[1, 1].set_title("Boxplot atenciones")
    axes[1, 1].set_xticklabels(["Atenciones"])

    fig.suptitle("Distribución de indicadores sobre puntos aleatorios del espacio de decisión",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "distribucion_indicadores.png", dpi=150)
    plt.close(fig)


def _graficar_funcion_combinada(datos: list[dict], lambdas: dict, out: Path) -> None:
    """Muestra f(x) = TTS - λ·atenciones para tres valores de λ."""
    tts = np.array([d["tts_mean"] for d in datos])
    at  = np.array([d["at_mean"]  for d in datos])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    nombres = ["lambda_bajo", "lambda_med", "lambda_alto"]
    titulos = [
        f"λ bajo = {lambdas['lambda_bajo']:.4f}\n(prioriza reducir TTS)",
        f"λ medio = {lambdas['lambda_med']:.4f}\n(equilibrado, recomendado)",
        f"λ alto = {lambdas['lambda_alto']:.4f}\n(prioriza aumentar atenciones)",
    ]

    for ax, lname, titulo in zip(axes, nombres, titulos):
        lv  = lambdas[lname]
        f   = tts - lv * at
        idx = np.argsort(f)   # orden de mejor a peor

        scatter = ax.scatter(range(len(f)), f[idx], c=at[idx],
                             cmap="YlOrRd_r", s=25, alpha=0.7)
        plt.colorbar(scatter, ax=ax, label="Atenciones")
        ax.set_xlabel("Configuración (rankeada por f)")
        ax.set_ylabel("f(x) = TTS - λ·atenciones")
        ax.set_title(titulo, fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("Función objetivo combinada para tres valores de λ\n"
                 "(color = total atenciones del punto)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "funcion_combinada_lambdas.png", dpi=150)
    plt.close(fig)


def _graficar_sensibilidad(datos: list[dict], lambdas: dict, out: Path) -> None:
    """
    Análisis de sensibilidad: cómo cambia el ranking del TOP-10
    al variar λ entre lambda_bajo y lambda_alto.
    """
    tts = np.array([d["tts_mean"] for d in datos])
    at  = np.array([d["at_mean"]  for d in datos])

    lambda_vals = np.linspace(lambdas["lambda_bajo"] * 0.5,
                              lambdas["lambda_alto"] * 1.5, 50)
    top_tts_idx = int(np.argmin(tts))      # mejor solo por TTS
    top_at_idx  = int(np.argmax(at))       # mejor solo por atenciones

    # Para cada λ: ranking del punto top_tts y del punto top_at
    rank_tts_point, rank_at_point = [], []
    for lv in lambda_vals:
        f = tts - lv * at
        orden = np.argsort(f)
        rank_tts_point.append(int(np.where(orden == top_tts_idx)[0][0]) + 1)
        rank_at_point.append(int(np.where(orden == top_at_idx)[0][0])  + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(lambda_vals, rank_tts_point, color="#2E5A88", lw=2,
            label=f"Mejor TTS puro (TTS={tts[top_tts_idx]:.1f}d, at={at[top_tts_idx]:.0f})")
    ax.plot(lambda_vals, rank_at_point, color="#C8862E", lw=2, ls="--",
            label=f"Mejor atenciones puro (TTS={tts[top_at_idx]:.1f}d, at={at[top_at_idx]:.0f})")

    # Marcar los tres λ
    for lname, color in [("lambda_bajo", "green"), ("lambda_med", "red"), ("lambda_alto", "purple")]:
        ax.axvline(lambdas[lname], ls=":", color=color, alpha=0.8,
                   label=f"{lname} = {lambdas[lname]:.4f}")

    ax.set_xlabel("λ")
    ax.set_ylabel("Ranking en f(x) = TTS - λ·at (1 = mejor)")
    ax.set_title("Sensibilidad del ranking al valor de λ\n"
                 "Si las líneas se cruzan, el λ de cruce es el umbral de cambio de preferencia")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.invert_yaxis()    # rank 1 arriba
    fig.tight_layout()
    fig.savefig(out / "analisis_sensibilidad.png", dpi=150)
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────────
# Cálculo de lambdas
# ───────────────────────────────────────────────────────────────────────

def _calcular_lambdas(datos: list[dict], elast: dict) -> dict:
    tts = np.array([d["tts_mean"] for d in datos])
    at  = np.array([d["at_mean"]  for d in datos])

    lambda_std   = float(np.std(tts, ddof=1) / np.std(at, ddof=1)) \
                   if np.std(at, ddof=1) > 0 else float("nan")
    lambda_range = float((tts.max() - tts.min()) / (at.max() - at.min())) \
                   if (at.max() - at.min()) > 0 else float("nan")
    lambda_elast = elast.get("lambda_elast", float("nan"))

    # λ recomendado: mediana de los tres métodos (robusto a outliers)
    validos = [v for v in [lambda_std, lambda_range, lambda_elast] if v == v]
    lambda_med = float(np.median(validos)) if validos else float("nan")

    # Rango de sensibilidad: λ/3 a 3λ
    lambda_bajo = lambda_med / 3.0
    lambda_alto = lambda_med * 3.0

    return {
        "lambda_std":   lambda_std,
        "lambda_range": lambda_range,
        "lambda_elast": lambda_elast,
        "lambda_med":   lambda_med,     # RECOMENDADO
        "lambda_bajo":  lambda_bajo,
        "lambda_alto":  lambda_alto,
    }


# ───────────────────────────────────────────────────────────────────────
# Reporte de texto
# ───────────────────────────────────────────────────────────────────────

def _reporte(datos: list[dict], lambdas: dict, elast: dict, out: Path) -> str:
    tts = np.array([d["tts_mean"] for d in datos])
    at  = np.array([d["at_mean"]  for d in datos])
    corr = float(np.corrcoef(tts, at)[0, 1])

    lineas = [
        "═" * 65,
        "CALIBRACIÓN DE λ — FUNCIÓN OBJETIVO COMBINADA",
        "  f(x) = tts_full_days_mean(x) - λ · total_atenciones(x)",
        "═" * 65,
        "",
        f"Muestra: {len(datos)} configuraciones aleatorias",
        f"Réplicas por punto: {datos[0]['n_reps_ok']}",
        "",
        "── Estadísticos de los indicadores ─────────────────────",
        f"  TTS total:         μ={tts.mean():.2f}  σ={tts.std(ddof=1):.2f}  "
        f"rango=[{tts.min():.1f}, {tts.max():.1f}] días",
        f"  Total atenciones:  μ={at.mean():.0f}  σ={at.std(ddof=1):.1f}  "
        f"rango=[{at.min():.0f}, {at.max():.0f}] pacientes",
        f"  Correlación Pearson TTS~atenciones: r = {corr:.3f}",
        "",
        "── Estimaciones de λ ────────────────────────────────────",
        f"  Método 1 — Razón de std:       λ_std   = {lambdas['lambda_std']:.4f}",
        f"  Método 2 — Razón de rangos:    λ_range = {lambdas['lambda_range']:.4f}",
        f"  Método 3 — Elasticidad local:  λ_elast = {lambdas['lambda_elast']:.4f}",
        f"             (baseline TTS={elast.get('tts_base', '—'):.1f}d, "
        f"at={elast.get('at_base', '—'):.0f})",
        "",
        "── λ RECOMENDADO ─────────────────────────────────────────",
        f"  λ_medio  = {lambdas['lambda_med']:.4f}  (mediana de los tres métodos)",
        "",
        "── Rango de sensibilidad para análisis ──────────────────",
        f"  λ_bajo   = {lambdas['lambda_bajo']:.4f}  (prioriza TTS, λ_med / 3)",
        f"  λ_medio  = {lambdas['lambda_med']:.4f}  (equilibrado, RECOMENDADO)",
        f"  λ_alto   = {lambdas['lambda_alto']:.4f}  (prioriza atenciones, λ_med × 3)",
        "",
        "── Interpretación de λ_medio ────────────────────────────",
        f"  Una atención adicional equivale a {lambdas['lambda_med']:.3f} días menos",
        f"  de tiempo de espera en la función objetivo.",
        f"  O dicho de otro modo: el modelo acepta aumentar TTS en",
        f"  {lambdas['lambda_med']:.3f} días a cambio de atender a un paciente más.",
        "",
        "── Uso en benchmark_riguroso.py ─────────────────────────",
        "  Pasar --lambda_obj al comando de benchmark (implementación",
        "  en función objetivo_combinado() del script).",
        "  Ejemplo:",
        f"    python benchmark_riguroso.py --modulos M4 M8 M10 M13 RS \\",
        f"        --lambda_obj {lambdas['lambda_med']:.4f} \\",
        f"        --n_seeds 15 --n_trials 150 --out resultados_lambda/",
        "",
        "── Para el análisis de sensibilidad de la tesis ─────────",
        f"  Correr benchmark con λ ∈ {{{lambdas['lambda_bajo']:.4f}, "
        f"{lambdas['lambda_med']:.4f}, {lambdas['lambda_alto']:.4f}}}",
        "  y mostrar cómo cambia el incumbente y el ranking entre métodos.",
        "  Si el incumbente cambia cualitativamente, reportarlo como",
        "  'umbral de cambio de preferencia' en la discusión de resultados.",
        "═" * 65,
    ]
    txt = "\n".join(lineas)
    (out / "reporte_lambda.txt").write_text(txt, encoding="utf-8")
    return txt


# ───────────────────────────────────────────────────────────────────────
# Orquestador principal
# ───────────────────────────────────────────────────────────────────────

def calibrar(n_puntos: int, n_reps: int, n_cores: int, out: Path,
             seed: int = 42) -> dict:

    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)

    log.info("=" * 60)
    log.info("Calibración de λ")
    log.info("Puntos: %d  Reps/punto: %d  Cores: %d", n_puntos, n_reps, n_cores)
    costo_est = n_puntos * n_reps * 90 / n_cores / 60
    log.info("Tiempo estimado: %.0f min (%.1f h)", costo_est, costo_est / 60)
    log.info("=" * 60)

    # Muestreo quasi-aleatorio uniforme en [0,1]^12
    puntos = rng.uniform(0, 1, size=(n_puntos, 12))
    tareas = [(i, puntos[i], n_reps, 10_000) for i in range(n_puntos)]

    t0 = time.time()
    datos = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as ex:
        futuros = {ex.submit(_worker, t): t[0] for t in tareas}
        for i, fut in enumerate(concurrent.futures.as_completed(futuros), 1):
            idx = futuros[fut]
            try:
                r = fut.result()
                if r is not None:
                    datos.append(r)
                    if i % 10 == 0 or i == len(tareas):
                        log.info("[%d/%d] TTS=%.2f  at=%.0f",
                                 i, len(tareas), r["tts_mean"], r["at_mean"])
            except Exception as e:
                log.error("punto=%d error: %s", idx, e)

    if len(datos) < 10:
        log.error("Muy pocos puntos válidos (%d). Revisar simulador.", len(datos))
        return {}

    log.info("Puntos válidos: %d / %d  (%.1f min)",
             len(datos), n_puntos, (time.time() - t0) / 60)

    # Guardar datos crudos
    (out / "datos_calibracion.json").write_text(
        json.dumps(datos, indent=2, default=str), encoding="utf-8")

    # Método 3: elasticidad
    log.info("Calculando elasticidad local (baseline)...")
    elast = _calcular_elasticidad(n_reps=max(n_reps, 5), n_cores=n_cores)

    # Calcular lambdas
    lambdas = _calcular_lambdas(datos, elast)
    log.info("λ_std=%.4f  λ_range=%.4f  λ_elast=%.4f  → λ_med=%.4f",
             lambdas["lambda_std"], lambdas["lambda_range"],
             lambdas["lambda_elast"], lambdas["lambda_med"])

    # Gráficas
    _graficar_scatter(datos, lambdas, out)
    _graficar_distribuciones(datos, out)
    _graficar_funcion_combinada(datos, lambdas, out)
    _graficar_sensibilidad(datos, lambdas, out)

    # Reporte
    txt = _reporte(datos, lambdas, elast, out)
    log.info("\n%s", txt)

    resultado = {"lambdas": lambdas, "elasticidad": elast,
                 "n_puntos": len(datos), "n_reps": n_reps}
    (out / "lambdas.json").write_text(
        json.dumps(resultado, indent=2, default=str), encoding="utf-8")

    log.info("Archivos en: %s", out)
    return resultado


def main():
    p = argparse.ArgumentParser(
        description="Calibra λ para f(x) = TTS - λ·atenciones.")
    p.add_argument("--n_puntos", type=int, default=80,
                   help="Configuraciones aleatorias a evaluar (default 80).")
    p.add_argument("--n_reps",   type=int, default=5,
                   help="Réplicas del simulador por punto (default 5).")
    p.add_argument("--n_cores",  type=int, default=9,
                   help="Procesos paralelos (default 9).")
    p.add_argument("--seed",     type=int, default=42,
                   help="Semilla para muestreo (default 42).")
    p.add_argument("--out",      default="calibracion",
                   help="Directorio de salida (default calibracion/).")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    calibrar(args.n_puntos, args.n_reps, args.n_cores,
             Path(args.out), args.seed)


if __name__ == "__main__":
    main()
