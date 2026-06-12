#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
modulo_sa_alrefaei.py
═══════════════════════════════════════════════════════════════════════
Simulated Annealing con temperatura constante.
Basado en: Alrefaei & Andradóttir (1999) — Management Science 45(5).

Característica clave: acumulación de réplicas en zonas prometedoras.
Cada vez que se visita una solución, sus réplicas se acumulan en
A[x]/C[x], dando estimaciones más precisas a las soluciones más
visitadas.  Conectado conceptualmente con OpQuest.

Adaptación al espacio continuo/entero del simulador clínico:
  - Vecindario global: un parámetro se re-muestrea uniformemente
  - Acumulación por clave discreta (parámetros redondeados)
  - Criterio de aceptación para minimización

Uso desde benchmark_riguroso:
    from modulo_sa_alrefaei import sa_alrefaei_runner
    resultado = sa_alrefaei_runner(n_trials=150, seed=0, pesos_kpi={...})
"""

from __future__ import annotations
import dataclasses
import logging
import time

import numpy as np

log = logging.getLogger("modulo_sa_alrefaei")

_ENTEROS = {
    "horas_especialista_1ra", "horas_control_post", "cupos_laboratorio_ugd",
    "cupos_ecografia_matrona", "cupos_ecografia_ugd", "dias_publicacion",
    "num_matronas", "num_agentes_ugd",
}


# ───────────────────────────────────────────────────────────────────────
# Helpers internos
# ───────────────────────────────────────────────────────────────────────

def _vals_a_cfg(vals: dict):
    """Convierte dict de parámetros a SimConfig con benchmark_mode=True."""
    from simulador_clinica_baseline import CFG
    cfg = dataclasses.replace(CFG)
    cfg.fixed_weekly_capacity        = int(round(vals["horas_especialista_1ra"]))
    cfg.use_fixed_weekly_capacity    = True
    cfg.fixed_post_control_capacity  = int(round(vals["horas_control_post"]))
    cfg.use_fixed_post_control_hours = True
    cfg.ugd_lab_per_week             = int(round(vals["cupos_laboratorio_ugd"]))
    cfg.mat_us_per_week              = int(round(vals["cupos_ecografia_matrona"]))
    cfg.ugd_us_per_week              = int(round(vals["cupos_ecografia_ugd"]))
    cfg.publish_lead_workdays        = int(round(vals["dias_publicacion"]))
    cfg.blocked_pct                  = float(vals["pct_bloqueo_1ra"])
    cfg.empty_control_p_ugd          = float(vals["pct_consultas_vacias"])
    cfg.matrona_capacity             = int(round(vals["num_matronas"]))
    cfg.agent_capacity               = int(round(vals["num_agentes_ugd"]))
    cfg.not_contactable_p            = float(vals["pct_no_contactabilidad"])
    cfg.blocked_pct_post_control     = float(vals["pct_bloqueo_post_control"])
    cfg.benchmark_mode               = True
    return cfg


def _calc_objetivo(res: dict, objetivo: str, pesos_kpi: dict | None) -> float:
    if objetivo == "compuesto" and pesos_kpi:
        return sum(float(pesos_kpi.get(k, 0.0)) * float(res.get(k, 0.0))
                   for k in pesos_kpi)
    return float(res.get(objetivo, 1e9))


def _simular_L(vals: dict, L: int, seed_base: int,
               objetivo: str, pesos_kpi: dict | None) -> float:
    """Evalúa una configuración con L réplicas y retorna la media."""
    from simulador_clinica_baseline import run_once
    cfg = _vals_a_cfg(vals)
    resultados = []
    for r in range(L):
        try:
            res = run_once(seed_offset=seed_base + r, cfg=cfg)
            resultados.append(_calc_objetivo(res, objetivo, pesos_kpi))
        except Exception as e:
            log.warning("SA eval r=%d falló: %s", r, e)
    return float(np.mean(resultados)) if resultados else float("nan")


def _cfg_key(vals: dict, param_names: list) -> tuple:
    """Clave discreta: enteros exactos, continuos redondeados a 3 dec."""
    return tuple(
        int(vals[n]) if n in _ENTEROS else round(float(vals[n]), 3)
        for n in param_names
    )


def _punto_central(param_names: list, param_ranges: dict) -> dict:
    """Punto inicial = centro del espacio de parámetros."""
    vals = {}
    for n in param_names:
        lo, hi = param_ranges[n]
        v = (lo + hi) / 2.0
        vals[n] = int(round(v)) if n in _ENTEROS else float(v)
    return vals


def _generar_vecino(x_vals: dict, param_names: list, param_ranges: dict,
                    rng: np.random.Generator) -> dict:
    """
    Vecindario global: re-muestrea UN parámetro uniformemente en su rango.
    Preserva el espíritu del vecindario global de Alrefaei & Andradóttir.
    """
    z_vals = x_vals.copy()
    n = param_names[int(rng.integers(0, len(param_names)))]
    lo, hi = param_ranges[n]
    if n in _ENTEROS:
        z_vals[n] = int(rng.integers(lo, hi + 1))
    else:
        z_vals[n] = float(lo + rng.random() * (hi - lo))
    return z_vals


# ───────────────────────────────────────────────────────────────────────
# Runner principal
# ───────────────────────────────────────────────────────────────────────

def sa_alrefaei_runner(n_trials: int, seed: int, L: int = 3,
                       temp: float = 1.0,
                       pesos_kpi: dict | None = None) -> dict:
    """
    SA temperatura constante — Alrefaei & Andradóttir (1999).

    Parámetros
    ----------
    n_trials   : presupuesto total de evaluaciones del simulador
    seed       : semilla macro-réplica
    L          : réplicas por visita (default 3, igual que SMAC)
    temp       : temperatura constante (default 1.0)
    pesos_kpi  : dict para objetivo compuesto; None → solo TTS

    Retorna dict compatible con benchmark_riguroso._correr_una.
    """
    import modulo_comparativa_caja_negra as comp

    PARAM_NAMES = comp.PARAM_NAMES
    PARAM_RANGES = comp.PARAM_RANGES
    objetivo    = "compuesto" if pesos_kpi else "tts_full_days_mean"
    max_iter    = max(5, n_trials // (2 * L))  # evalúa x + z cada iter
    rng         = np.random.default_rng(seed=seed)
    seed_sim    = seed * 200_000
    t0          = time.time()

    log.info("SA Alrefaei-Andradóttir: max_iter=%d  L=%d  T=%.1f  objetivo=%s",
             max_iter, L, temp, objetivo)

    # ── Acumuladores: A[key]=suma_ponderada, C[key]=n_reps ───────────────
    A: dict[tuple, float] = {}
    C: dict[tuple, int]   = {}
    vals_map: dict[tuple, dict] = {}   # key → parámetros originales

    # ── Punto inicial ────────────────────────────────────────────────────
    x_vals = _punto_central(PARAM_NAMES, PARAM_RANGES)

    historia   = []
    conv_eval  = []
    n_eval     = 0
    mejor_val  = float("inf")
    mejor_key  = None

    for k in range(max_iter):

        z_vals = _generar_vecino(x_vals, PARAM_NAMES, PARAM_RANGES, rng)

        # ── Evaluar x y z, acumular ──────────────────────────────────────
        for sol_vals in [x_vals, z_vals]:
            key = _cfg_key(sol_vals, PARAM_NAMES)
            f   = _simular_L(sol_vals, L, seed_sim, objetivo, pesos_kpi)
            seed_sim += L
            n_eval   += L

            if key in A:
                A[key] += f * L
                C[key] += L
            else:
                A[key]       = f * L
                C[key]       = L
                vals_map[key] = sol_vals.copy()

        # ── Criterio de aceptación (minimización) ────────────────────────
        x_key   = _cfg_key(x_vals, PARAM_NAMES)
        z_key   = _cfg_key(z_vals, PARAM_NAMES)
        f_hat_x = A[x_key] / C[x_key]
        f_hat_z = A[z_key] / C[z_key]

        # prob = exp(-max(deterioro, 0) / T); acepta siempre mejora
        diff = max(f_hat_z - f_hat_x, 0.0)
        if rng.random() < np.exp(-diff / temp):
            x_vals = z_vals

        # ── Mejor solución acumulada ─────────────────────────────────────
        mejor_key = min(A, key=lambda kk: A[kk] / C[kk])
        mejor_val = A[mejor_key] / C[mejor_key]

        historia.append({
            "iter":              k,
            "costo":             round(f_hat_z, 4),
            "mejor_hasta_ahora": round(mejor_val, 4),
            "t_seg":             round(time.time() - t0, 2),
        })
        conv_eval.append((n_eval, round(mejor_val, 4)))

        log.info("SA iter=%3d  f(z)=%.2f  mejor=%.2f  n_eval=%d",
                 k, f_hat_z, mejor_val, n_eval)

    # ── Resultado final ──────────────────────────────────────────────────
    incumbente_vals = vals_map.get(mejor_key, x_vals)

    return {
        "modulo":            "SA",
        "algoritmo":         "SA Alrefaei-Andradottir (1999)",
        "costo_incumbente":  round(mejor_val, 4),
        "tiempo_seg":        round(time.time() - t0, 2),
        "n_evaluaciones":    n_eval,
        "incumbente":        incumbente_vals,
        "historia_costos":   historia,
        "conv_eval":         conv_eval,
        "conv_time":         [(h["t_seg"], h["mejor_hasta_ahora"]) for h in historia],
        "temp":              temp,
        "L":                 L,
        "max_iter":          max_iter,
    }
