"""
===============================================================================
MÓDULO COMPARATIVA — Benchmarking de Algoritmos de Optimización
===============================================================================
Compara los siguientes módulos usando simulador_clinica_baseline.py como
caja negra (black-box), midiendo:
  · Curva de convergencia (mejor costo vs. evaluaciones acumuladas)
  · Curva de convergencia vs. tiempo de cómputo
  · Tiempo total y mejor valor alcanzado por módulo
  · Comparativa de configuraciones óptimas encontradas

Módulos incluidos:
  M4   — SMAC BlackBox (GP + EI)
  M7   — SMAC + Stochastic Kriging (EI)
  M8   — SMAC + SK Adaptativo
  M9   — SMAC + SK REVI
  M10  — SMAC + SK KGCP
  M11  — ASTRO-DF (Trust-Region lineal)
  M12  — STRONG   (Trust-Region cuadrático)
  M13  — SPSA     (gradiente por perturbación simultánea)
  M14  — ALOE     (Armijo relajado + diferencias finitas)

Uso rápido (presupuesto reducido para demostración):
    python modulo_comparativa_caja_negra.py --modulos M4 M13 M14 --n_trials 20 --max_iter 30

Uso completo:
    python modulo_comparativa_caja_negra.py --todos --n_trials 50 --max_iter 100

Re-graficar desde resultados guardados:
    python modulo_comparativa_caja_negra.py --solo_graficas resultados_comparativa.json
===============================================================================
"""

import sys
import os
import json
import time
import argparse
import logging
import numpy as np
from dataclasses import asdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("comparativa")

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

# ─────────────────────────────────────────────────────────────────
# Valor baseline (referencia visual en gráficas)
# ─────────────────────────────────────────────────────────────────
BASELINE_COSTO = 279.78   # tts_full_days_mean con CFG original

# ─────────────────────────────────────────────────────────────────
# Paleta y etiquetas por módulo
# ─────────────────────────────────────────────────────────────────
MODULO_META = {
    "M4":   {"label": "M1 SMAC-BB (GP)",    "color": "#1f77b4", "ls": "-",  "marker": "o"},
    "M7":   {"label": "M2 SMAC-SK (EI)",    "color": "#ff7f0e", "ls": "-",  "marker": "s"},
    "M8":   {"label": "M3 SK-Adapt.",       "color": "#2ca02c", "ls": "--", "marker": "D"},
    "M9":   {"label": "M9 SK-REVI",         "color": "#d62728", "ls": "--", "marker": "v"},
    "M10":  {"label": "M4 SK-KGCP",         "color": "#9467bd", "ls": "-.", "marker": "^"},
    "M11":  {"label": "M5 ASTRO-DF",        "color": "#8c564b", "ls": "-.", "marker": "p"},
    "M12":  {"label": "M12 STRONG",         "color": "#e377c2", "ls": ":",  "marker": "h"},
    "M13":  {"label": "M6 SPSA",            "color": "#7f7f7f", "ls": ":",  "marker": "x"},
    "M14":  {"label": "M14 ALOE",           "color": "#bcbd22", "ls": (0, (3,1,1,1)), "marker": "*"},
}

# Nombres de parámetros del incumbente
PARAM_NAMES = [
    "horas_especialista_1ra",
    "horas_control_post",
    "cupos_laboratorio_ugd",
    "cupos_ecografia_matrona",
    "cupos_ecografia_ugd",
    "dias_publicacion",
    "pct_bloqueo_1ra",
    "pct_consultas_vacias",
    "num_matronas",
    "num_agentes_ugd",
    "pct_no_contactabilidad",
    "pct_bloqueo_post_control",
]
# Rangos para normalización
PARAM_RANGES = {
    "horas_especialista_1ra":   (8,    30),
    "horas_control_post":       (20,   70),
    "cupos_laboratorio_ugd":    (20,  100),
    "cupos_ecografia_matrona":  (10,   50),
    "cupos_ecografia_ugd":      (10,   50),
    "dias_publicacion":         (1,    10),
    "pct_bloqueo_1ra":          (0.05, 0.50),
    "pct_consultas_vacias":     (0.05, 0.50),
    "num_matronas":             (1,    4),
    "num_agentes_ugd":          (1,    4),
    "pct_no_contactabilidad":   (0.05, 0.50),
    "pct_bloqueo_post_control": (0.05, 0.50),
}
PARAM_BASELINE = {
    "horas_especialista_1ra":   16,
    "horas_control_post":       40,
    "cupos_laboratorio_ugd":    54,
    "cupos_ecografia_matrona":  25,
    "cupos_ecografia_ugd":      25,
    "dias_publicacion":         5,
    "pct_bloqueo_1ra":          0.32,
    "pct_consultas_vacias":     0.30,
    "num_matronas":             1,
    "num_agentes_ugd":          1,
    "pct_no_contactabilidad":   0.15,
    "pct_bloqueo_post_control": 0.34,
}
PARAM_LABELS = [
    "Slots 1ra/sem",
    "Pac control/sem",
    "Cupos lab UGD",
    "Cupos eco mat.",
    "Cupos eco UGD",
    "Días publicac.",
    "% Bloq. 1ra",
    "% Cons. vacías",
    "Nº Matronas",
    "Nº Agentes UGD",
    "% No contactab.",
    "% Bloq. post",
]


def _incumbente_a_dict(obj) -> dict:
    """Convierte incumbente a dict, sea dataclass, Configuration de SMAC, o dict."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "get_dictionary"):
        return obj.get_dictionary()
    try:
        return asdict(obj)
    except TypeError:
        pass
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {}


# ═════════════════════════════════════════════════════════════════
# Helpers para extraer curvas de convergencia
# ═════════════════════════════════════════════════════════════════

def _convergencia_smac(historia_costos: list, n_corridas_eval: int = 2):
    """
    Extrae (eval_acumulada, mejor_hasta_ahora) para módulos SMAC (M4-M10).

    Orden cronológico: por t_seg real si el tracker lo guardó (M4-M10 sí lo hacen),
    por config_id como fallback si no hay timestamps.

    Incumbente acumulado: usa mejor_hasta_ahora pre-calculado por cada módulo
    si existe (más preciso que recalcularlo); cae a 'costo' si no.

    n_reps por entrada: usa el campo n_reps real si existe (M8/M9 adaptativo),
    usa n_corridas_eval como fallback (M4/M7 réplicas fijas).
    """
    tiene_tiempo = any(e.get("t_seg") is not None for e in historia_costos)
    if tiene_tiempo:
        entries = sorted(historia_costos,
                         key=lambda e: float(e.get("t_seg", 0.0)))
    else:
        entries = sorted(historia_costos,
                         key=lambda e: e.get("config_id", 0))

    mejores = []
    mejor   = float("inf")
    n_eval  = 0
    for e in entries:
        val    = e.get("mejor_hasta_ahora",
                 e.get("costo", float("inf")))
        n_eval += e.get("n_reps", n_corridas_eval)
        mejor   = min(mejor, val)
        mejores.append((n_eval, mejor))
    return mejores


def _convergencia_iterativa(historia_costos: list, evals_por_iter: int = None):
    """
    Para módulos iterativos (M11-M14): extrae (eval_acumulada, mejor_hasta_ahora).
    Si evals_por_iter es None, usa el campo n_reps de cada entrada si existe.
    Si tampoco existe n_reps, cuenta 1 evaluación por entrada.
    """
    mejores = []
    mejor   = float("inf")
    n_eval  = 0
    for e in historia_costos:
        # Obtener n_reps de la entrada o usar el parámetro fijo
        if evals_por_iter is not None:
            n = evals_por_iter
        else:
            n = e.get("n_reps", 1)

        n_eval += n
        # Usar campo que indique el mejor hasta ahora
        val = e.get("mejor_hasta_ahora",
              e.get("f_incumbente",
              e.get("costo", float("inf"))))
        if val < mejor:
            mejor = val
        mejores.append((n_eval, mejor))
    return mejores


def _tiempos_convergencia(historia_costos: list):
    """Extrae (t_seg, mejor_hasta_ahora) para curva vs. tiempo."""
    mejores = []
    mejor   = float("inf")
    for e in historia_costos:
        t = e.get("t_seg", 0.0)
        val = e.get("mejor_hasta_ahora",
              e.get("f_incumbente",
              e.get("costo", float("inf"))))
        if val < mejor:
            mejor = val
        mejores.append((t, mejor))
    return mejores


# ═════════════════════════════════════════════════════════════════
# Runners para cada módulo
# ═════════════════════════════════════════════════════════════════

def run_m4_blackbox(n_trials: int = 30, n_corridas: int = 2, seed: int = 42,
                    pesos_kpi: dict = None) -> dict:
    """M4: SMAC BlackBox (Gaussian Process + EI)."""
    log.info("▶ M4 — SMAC BlackBox (GP+EI)  [n_trials=%d, n_corridas=%d]",
             n_trials, n_corridas)
    import modulo4_smac_v2 as m4
    res = m4.optimizar(
        tipo             = "blackbox",
        n_trials         = n_trials,
        n_corridas_eval  = n_corridas,
        seed             = seed,
        objetivo         = "compuesto" if pesos_kpi else "tts_full_days_mean",
        pesos_kpi        = pesos_kpi,
        guardar_json     = "resultado_comparativa_m4.json",
        output_dir       = "smac_comp_m4",
    )
    conv_eval = _convergencia_smac(res.historia_costos, n_corridas)
    conv_time = _tiempos_smac(res.historia_costos, res.tiempo_seg)
    return {
        "modulo":           "M4",
        "algoritmo":        "SMAC BlackBox (GP+EI)",
        "costo_incumbente": res.costo_incumbente,
        "tiempo_seg":       res.tiempo_seg,
        "n_evaluaciones":   len(res.historia_costos) * n_corridas,
        "incumbente":       _incumbente_a_dict(res.incumbente),
        "conv_eval":        conv_eval,
        "conv_time":        conv_time,
    }


def run_m7_sk(n_trials: int = 30, n_corridas: int = 2, seed: int = 42,
              pesos_kpi: dict = None) -> dict:
    """M7: SMAC + Stochastic Kriging (EI)."""
    log.info("▶ M7 — SMAC + SK (EI)  [n_trials=%d, n_corridas=%d]", n_trials, n_corridas)
    import modulo7_smac_sk as m7
    res = m7.optimizar(
        tipo             = "blackbox_sk",
        n_trials         = n_trials,
        n_corridas_eval  = n_corridas,
        seed             = seed,
        objetivo         = "compuesto" if pesos_kpi else "tts_full_days_mean",
        pesos_kpi        = pesos_kpi,
        guardar_json     = "resultado_comparativa_m7.json",
        output_dir       = "smac_comp_m7",
    )
    conv_eval = _convergencia_smac(res.historia_costos, n_corridas)
    conv_time = _tiempos_smac(res.historia_costos, res.tiempo_seg)
    return {
        "modulo":           "M7",
        "algoritmo":        "SMAC + SK (EI)",
        "costo_incumbente": res.costo_incumbente,
        "tiempo_seg":       res.tiempo_seg,
        "n_evaluaciones":   len(res.historia_costos) * n_corridas,
        "incumbente":       _incumbente_a_dict(res.incumbente),
        "conv_eval":        conv_eval,
        "conv_time":        conv_time,
    }


def run_m8_sk_adaptativo(n_trials: int = 30, n_corridas: int = 2, seed: int = 42,
                         pesos_kpi: dict = None) -> dict:
    """M8: SMAC + SK con replicación adaptativa."""
    log.info("▶ M8 — SK Adaptativo  [n_trials=%d]", n_trials)
    import modulo8_sk_adaptativo_paralelizado as m8
    res = m8.optimizar(
        tipo             = "blackbox_sk",
        n_trials         = n_trials,
        n_corridas_eval  = n_corridas,
        seed             = seed,
        adaptativo       = True,
        n_min_reps       = 2,
        n_max_reps       = 6,
        n_warmup         = min(10, n_trials // 3),
        objetivo         = "compuesto" if pesos_kpi else "tts_full_days_mean",
        pesos_kpi        = pesos_kpi,
        guardar_json     = "resultado_comparativa_m8.json",
        output_dir       = "smac_comp_m8",
    )
    conv_eval = _convergencia_smac(res.historia_costos, n_corridas)
    conv_time = _tiempos_smac(res.historia_costos, res.tiempo_seg)
    return {
        "modulo":           "M8",
        "algoritmo":        "SK Adaptativo",
        "costo_incumbente": res.costo_incumbente,
        "tiempo_seg":       res.tiempo_seg,
        "n_evaluaciones":   sum(
            e.get("n_reps", n_corridas) for e in res.historia_costos
        ),
        "incumbente":       _incumbente_a_dict(res.incumbente),
        "conv_eval":        conv_eval,
        "conv_time":        conv_time,
    }


def run_m9_revi(n_trials: int = 30, n_corridas: int = 2, seed: int = 42,
                pesos_kpi: dict = None) -> dict:
    """M9: SMAC + SK con REVI."""
    log.info("▶ M9 — SK-REVI  [n_trials=%d]", n_trials)
    import modulo9_sk_revi as m9
    res = m9.optimizar(
        tipo             = "blackbox_sk",
        n_trials         = n_trials,
        n_corridas_eval  = n_corridas,
        seed             = seed,
        revi             = True,
        n_min_reps       = 2,
        n_max_reps       = 6,
        n_warmup         = min(10, n_trials // 3),
        objetivo         = "compuesto" if pesos_kpi else "tts_full_days_mean",
        pesos_kpi        = pesos_kpi,
        guardar_json     = "resultado_comparativa_m9.json",
        output_dir       = "smac_comp_m9",
    )
    conv_eval = _convergencia_smac(res.historia_costos, n_corridas)
    conv_time = _tiempos_smac(res.historia_costos, res.tiempo_seg)
    return {
        "modulo":           "M9",
        "algoritmo":        "SK-REVI",
        "costo_incumbente": res.costo_incumbente,
        "tiempo_seg":       res.tiempo_seg,
        "n_evaluaciones":   sum(
            e.get("n_reps", n_corridas) for e in res.historia_costos
        ),
        "incumbente":       _incumbente_a_dict(res.incumbente),
        "conv_eval":        conv_eval,
        "conv_time":        conv_time,
    }


def run_m10_kgcp(n_trials: int = 30, n_corridas: int = 2, seed: int = 42,
                 pesos_kpi: dict = None) -> dict:
    """M10: SMAC + SK con KGCP."""
    log.info("▶ M10 — SK-KGCP  [n_trials=%d]", n_trials)
    import modulo10_sk_kgcp as m10
    res = m10.optimizar(
        tipo             = "blackbox_kgcp",
        n_trials         = n_trials,
        n_corridas_eval  = n_corridas,
        seed             = seed,
        n_kgcp_mc        = 32,
        n_kgcp_cand      = 200,
        objetivo         = "compuesto" if pesos_kpi else "tts_full_days_mean",
        pesos_kpi        = pesos_kpi,
        guardar_json     = "resultado_comparativa_m10.json",
        output_dir       = "smac_comp_m10",
    )
    conv_eval = _convergencia_smac(res.historia_costos, n_corridas)
    conv_time = _tiempos_smac(res.historia_costos, res.tiempo_seg)
    return {
        "modulo":           "M10",
        "algoritmo":        "SK-KGCP",
        "costo_incumbente": res.costo_incumbente,
        "tiempo_seg":       res.tiempo_seg,
        "n_evaluaciones":   sum(
            e.get("n_reps", n_corridas) for e in res.historia_costos
        ),
        "incumbente":       _incumbente_a_dict(res.incumbente),
        "conv_eval":        conv_eval,
        "conv_time":        conv_time,
    }


def run_m11_astrodf(max_iter: int = 50, n_workers: int = 0, seed: int = 42,
                    pesos_kpi: dict = None) -> dict:
    """M11: ASTRO-DF (trust-region lineal con muestreo adaptativo)."""
    log.info("▶ M11 — ASTRO-DF  [max_iter=%d]", max_iter)
    import modulo_11_astrodf as m11
    res = m11.optimizar_astro_df(
        seed             = seed,
        max_iter         = max_iter,
        n_workers        = n_workers,
        objetivo         = "compuesto" if pesos_kpi else "tts_full_days_mean",
        pesos_kpi        = pesos_kpi,
        guardar_json     = "resultado_comparativa_m11.json",
    )
    conv_eval = _convergencia_iterativa(res.historia_costos)
    conv_time = _tiempos_convergencia(res.historia_costos)
    return {
        "modulo":           "M11",
        "algoritmo":        "ASTRO-DF",
        "costo_incumbente": res.costo_incumbente,
        "tiempo_seg":       res.tiempo_seg,
        "n_evaluaciones":   sum(e.get("n_reps", 1) for e in res.historia_costos),
        "incumbente":       _incumbente_a_dict(res.incumbente),
        "conv_eval":        conv_eval,
        "conv_time":        conv_time,
    }


def run_m12_strong(max_iter: int = 50, n_workers: int = 0, seed: int = 42,
                   pesos_kpi: dict = None) -> dict:
    """M12: STRONG (trust-region cuadrático)."""
    log.info("▶ M12 — STRONG  [max_iter=%d]", max_iter)
    import modulo12_strong as m12
    res = m12.optimizar_strong(
        seed             = seed,
        max_iter         = max_iter,
        n_workers        = n_workers,
        objetivo         = "compuesto" if pesos_kpi else "tts_full_days_mean",
        pesos_kpi        = pesos_kpi,
        guardar_json     = "resultado_comparativa_m12.json",
    )
    conv_eval = _convergencia_iterativa(res.historia_costos)
    conv_time = _tiempos_convergencia(res.historia_costos)
    return {
        "modulo":           "M12",
        "algoritmo":        "STRONG",
        "costo_incumbente": res.costo_incumbente,
        "tiempo_seg":       res.tiempo_seg,
        "n_evaluaciones":   sum(e.get("n_reps", 1) for e in res.historia_costos),
        "incumbente":       _incumbente_a_dict(res.incumbente),
        "conv_eval":        conv_eval,
        "conv_time":        conv_time,
    }


def run_m13_spsa(max_iter: int = 50, n_reps: int = 5, n_workers: int = 0, seed: int = 42,
                 pesos_kpi: dict = None) -> dict:
    """M13: SPSA (gradiente por perturbación simultánea)."""
    log.info("▶ M13 — SPSA  [max_iter=%d, n_reps=%d]", max_iter, n_reps)
    import modulo13_spsa as m13
    res = m13.optimizar_spsa(
        seed             = seed,
        max_iter         = max_iter,
        n_reps           = n_reps,
        n_workers        = n_workers,
        objetivo         = "compuesto" if pesos_kpi else "tts_full_days_mean",
        pesos_kpi        = pesos_kpi,
        guardar_json     = "resultado_comparativa_m13.json",
    )
    conv_eval = _convergencia_iterativa(res.historia_costos, evals_por_iter=2*n_reps)
    conv_time = _tiempos_convergencia(res.historia_costos)
    return {
        "modulo":           "M13",
        "algoritmo":        "SPSA",
        "costo_incumbente": res.costo_incumbente,
        "tiempo_seg":       res.tiempo_seg,
        "n_evaluaciones":   len(res.historia_costos) * 2 * n_reps,
        "incumbente":       _incumbente_a_dict(res.incumbente),
        "conv_eval":        conv_eval,
        "conv_time":        conv_time,
    }


def run_m14_aloe(max_iter: int = 50, r: int = 5, n_workers: int = 0, seed: int = 42,
                 pesos_kpi: dict = None) -> dict:
    """M14: ALOE (Armijo relajado + diferencias finitas)."""
    log.info("▶ M14 — ALOE  [max_iter=%d, r=%d]", max_iter, r)
    import modulo14_aloe as m14
    res = m14.optimizar_aloe(
        seed             = seed,
        max_iter         = max_iter,
        r                = r,
        n_workers        = n_workers,
        objetivo         = "compuesto" if pesos_kpi else "tts_full_days_mean",
        pesos_kpi        = pesos_kpi,
        guardar_json     = "resultado_comparativa_m14.json",
    )
    # ALOE: por iteración → gradiente (2*d evals con n_grad=2 fijo) + 2 evals Armijo
    # n_grad=2 es el valor corregido en modulo14_aloe.py (bug original usaba r//4)
    _d = 12
    evals_grad = 2 * _d * 2          # 24 sims para gradiente (n_grad=2 fijo)
    evals_arm  = 2 * r
    evals_iter = evals_grad + evals_arm
    conv_eval  = _convergencia_iterativa(res.historia_costos, evals_por_iter=evals_iter)
    conv_time  = _tiempos_convergencia(res.historia_costos)
    return {
        "modulo":           "M14",
        "algoritmo":        "ALOE",
        "costo_incumbente": res.costo_incumbente,
        "tiempo_seg":       res.tiempo_seg,
        "n_evaluaciones":   len(res.historia_costos) * evals_iter,
        "incumbente":       _incumbente_a_dict(res.incumbente),
        "conv_eval":        conv_eval,
        "conv_time":        conv_time,
    }


def _tiempos_smac(historia_costos: list, tiempo_total: float) -> list:
    """
    Extrae (t_seg, mejor_hasta_ahora) para curva vs. tiempo de módulos SMAC.
    Lee t_seg real del tracker si existe en las entradas (M4-M10 lo guardan);
    usa interpolación lineal solo como fallback si ninguna entrada tiene t_seg.
    """
    entries = sorted(historia_costos, key=lambda e: e.get("config_id", 0))
    tiene_tiempo = any(e.get("t_seg") is not None for e in entries)
    mejores = []
    mejor   = float("inf")
    n       = len(entries)

    for i, e in enumerate(entries):
        costo = e.get("mejor_hasta_ahora",
                e.get("costo", float("inf")))
        if tiene_tiempo:
            t = float(e.get("t_seg", 0.0))
        else:
            # Fallback: interpolación lineal con tiempo total
            t = (i + 1) * tiempo_total / max(n, 1)
        if costo < mejor:
            mejor = costo
        mejores.append((t, mejor))
    return mejores


# ═════════════════════════════════════════════════════════════════
# Mapa de runners
# ═════════════════════════════════════════════════════════════════

_RUNNERS = {
    "M4":  run_m4_blackbox,
    "M7":  run_m7_sk,
    "M8":  run_m8_sk_adaptativo,
    "M9":  run_m9_revi,
    "M10": run_m10_kgcp,
    "M11": run_m11_astrodf,
    "M12": run_m12_strong,
    "M13": run_m13_spsa,
    "M14": run_m14_aloe,
}


# ═════════════════════════════════════════════════════════════════
# Ejecutar comparativa
# ═════════════════════════════════════════════════════════════════

def ejecutar_comparativa(
    modulos:    list[str] = None,
    n_trials:   int  = 30,
    max_iter:   int  = 50,
    n_corridas: int  = 2,
    n_reps:     int  = 5,
    n_workers:  int  = 0,
    seed:       int  = 42,
    guardar_json: str = "resultados_comparativa.json",
    pesos_kpi:  dict = None,
) -> dict:
    """
    Corre la comparativa para los módulos indicados.
    Retorna dict {modulo: resultado}.
    """
    modulos = modulos or list(_RUNNERS.keys())
    resultados = {}

    t_total_inicio = time.time()
    for m in modulos:
        if m not in _RUNNERS:
            log.warning("Módulo '%s' no reconocido. Opciones: %s", m, list(_RUNNERS.keys()))
            continue
        log.info("\n%s\nEjecutando %s\n%s", "="*60, m, "="*60)
        try:
            kwargs = {}
            if m in ("M4", "M7", "M8", "M9", "M10"):
                kwargs = {"n_trials": n_trials, "n_corridas": n_corridas, "seed": seed,
                          "pesos_kpi": pesos_kpi}
            elif m in ("M11", "M12"):
                kwargs = {"max_iter": max_iter, "n_workers": n_workers, "seed": seed,
                          "pesos_kpi": pesos_kpi}
            elif m == "M13":
                kwargs = {"max_iter": max_iter, "n_reps": n_reps,
                          "n_workers": n_workers, "seed": seed,
                          "pesos_kpi": pesos_kpi}
            elif m == "M14":
                kwargs = {"max_iter": max_iter, "r": n_reps,
                          "n_workers": n_workers, "seed": seed,
                          "pesos_kpi": pesos_kpi}

            res = _RUNNERS[m](**kwargs)
            resultados[m] = res
            log.info("✓ %s completado: costo=%.3f  tiempo=%.0fs",
                     m, res["costo_incumbente"], res["tiempo_seg"])
        except Exception as ex:
            log.error("✗ Error en %s: %s", m, ex, exc_info=True)

    t_total = time.time() - t_total_inicio
    log.info("\n%s\nComparativa completada en %.0f s\n%s", "="*60, t_total, "="*60)

    # Guardar resultados
    _guardar_resultados(resultados, guardar_json)
    return resultados


def _guardar_resultados(resultados: dict, path: str):
    """Serializa resultados a JSON (conv_eval/conv_time como listas)."""
    serial = {}
    for m, r in resultados.items():
        serial[m] = {
            "modulo":           r["modulo"],
            "algoritmo":        r["algoritmo"],
            "costo_incumbente": r["costo_incumbente"],
            "tiempo_seg":       r["tiempo_seg"],
            "n_evaluaciones":   r["n_evaluaciones"],
            "incumbente":       r["incumbente"],
            "conv_eval":        r["conv_eval"],
            "conv_time":        r["conv_time"],
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serial, f, ensure_ascii=False, indent=2)
    log.info("Resultados guardados en '%s'", path)


def cargar_resultados(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


# ═════════════════════════════════════════════════════════════════
# Gráficas de comparación
# ═════════════════════════════════════════════════════════════════

def graficar_comparativa(
    resultados:  dict,
    guardar_png: str  = "comparativa_modulos.png",
    mostrar:     bool = False,
):
    """
    Genera figura multi-panel con 4 gráficas de comparación:
      (1) Convergencia vs. evaluaciones acumuladas
      (2) Convergencia vs. tiempo de cómputo
      (3) Barra: mejor costo por módulo
      (4) Barra: tiempo total por módulo
    """
    modulos_presentes = [m for m in MODULO_META if m in resultados]
    if not modulos_presentes:
        log.error("No hay resultados para graficar.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Comparativa de Módulos de Optimización — Caja Negra\n"
        "Objetivo: tts_full_days_mean (días) — Simulador Clínica DES",
        fontsize=14, fontweight="bold",
    )

    ax_eval, ax_time, ax_cost, ax_ttime = (
        axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    )

    # ── Panel 1: Convergencia vs. evaluaciones ──────────────────
    for m in modulos_presentes:
        r    = resultados[m]
        meta = MODULO_META[m]
        cv   = r["conv_eval"]
        if not cv:
            continue
        xs = [p[0] for p in cv]
        ys = [p[1] for p in cv]
        ax_eval.plot(xs, ys,
                     label  = meta["label"],
                     color  = meta["color"],
                     ls     = meta["ls"],
                     marker = meta["marker"],
                     markevery = max(1, len(xs)//10),
                     markersize = 5, linewidth=1.8)

    ax_eval.axhline(BASELINE_COSTO, color="black", ls="--", lw=1.2, alpha=0.6,
                    label=f"Baseline ({BASELINE_COSTO:.1f} días)")
    ax_eval.set_xlabel("Evaluaciones del simulador (acumuladas)", fontsize=10)
    ax_eval.set_ylabel("Mejor TTS total (días)", fontsize=10)
    ax_eval.set_title("Convergencia vs. Evaluaciones", fontsize=11, fontweight="bold")
    ax_eval.legend(fontsize=7.5, loc="upper right", ncol=1)
    ax_eval.grid(True, alpha=0.3)
    ax_eval.set_ylim(bottom=max(0, min(r["costo_incumbente"]
                     for r in resultados.values()) * 0.90))

    # ── Panel 2: Convergencia vs. tiempo ─────────────────────────
    for m in modulos_presentes:
        r    = resultados[m]
        meta = MODULO_META[m]
        cv   = r["conv_time"]
        if not cv:
            continue
        xs = [p[0] for p in cv]
        ys = [p[1] for p in cv]
        ax_time.plot(xs, ys,
                     label  = meta["label"],
                     color  = meta["color"],
                     ls     = meta["ls"],
                     marker = meta["marker"],
                     markevery = max(1, len(xs)//10),
                     markersize = 5, linewidth=1.8)

    ax_time.axhline(BASELINE_COSTO, color="black", ls="--", lw=1.2, alpha=0.6,
                    label=f"Baseline ({BASELINE_COSTO:.1f} días)")
    ax_time.set_xlabel("Tiempo de cómputo (segundos)", fontsize=10)
    ax_time.set_ylabel("Mejor TTS total (días)", fontsize=10)
    ax_time.set_title("Convergencia vs. Tiempo", fontsize=11, fontweight="bold")
    ax_time.legend(fontsize=7.5, loc="upper right", ncol=1)
    ax_time.grid(True, alpha=0.3)
    ax_time.set_ylim(bottom=max(0, min(r["costo_incumbente"]
                     for r in resultados.values()) * 0.90))

    # ── Panel 3: Mejor costo por módulo ──────────────────────────
    labels_c = [MODULO_META[m]["label"] for m in modulos_presentes]
    costos   = [resultados[m]["costo_incumbente"] for m in modulos_presentes]
    colors_c = [MODULO_META[m]["color"] for m in modulos_presentes]

    bars = ax_cost.barh(labels_c, costos, color=colors_c, edgecolor="white", height=0.6)
    ax_cost.axvline(BASELINE_COSTO, color="black", ls="--", lw=1.5, alpha=0.7,
                    label=f"Baseline ({BASELINE_COSTO:.1f} días)")
    for bar, val in zip(bars, costos):
        ax_cost.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                     f"{val:.1f}", va="center", ha="left", fontsize=8)
    ax_cost.set_xlabel("TTS total incumbente (días)", fontsize=10)
    ax_cost.set_title("Mejor Costo por Módulo", fontsize=11, fontweight="bold")
    ax_cost.legend(fontsize=8)
    ax_cost.grid(True, axis="x", alpha=0.3)

    # ── Panel 4: Tiempo total por módulo ─────────────────────────
    tiempos  = [resultados[m]["tiempo_seg"] for m in modulos_presentes]
    bars_t   = ax_ttime.barh(labels_c, tiempos, color=colors_c, edgecolor="white", height=0.6)
    for bar, val in zip(bars_t, tiempos):
        ax_ttime.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                      f"{val:.0f}s", va="center", ha="left", fontsize=8)
    ax_ttime.set_xlabel("Tiempo de cómputo (segundos)", fontsize=10)
    ax_ttime.set_title("Tiempo Total por Módulo", fontsize=11, fontweight="bold")
    ax_ttime.grid(True, axis="x", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(guardar_png, dpi=150, bbox_inches="tight")
    log.info("Gráfica guardada en '%s'", guardar_png)
    if mostrar:
        plt.show()
    plt.close(fig)


def graficar_configuraciones(
    resultados:  dict,
    guardar_png: str  = "comparativa_configuraciones.png",
    mostrar:     bool = False,
):
    """
    Genera dos paneles con la comparativa de configuraciones óptimas:
      (1) Heatmap de parámetros normalizados [0,1] por módulo
      (2) Gráfico de barras agrupadas para cada parámetro
    """
    modulos_presentes = [m for m in MODULO_META if m in resultados]
    if not modulos_presentes:
        return

    n_m = len(modulos_presentes)
    n_p = len(PARAM_NAMES)

    # Construir matriz normalizada: filas=módulos, cols=parámetros
    mat = np.zeros((n_m, n_p))
    for i, m in enumerate(modulos_presentes):
        inc = resultados[m].get("incumbente", {})
        for j, p in enumerate(PARAM_NAMES):
            lo, hi = PARAM_RANGES[p]
            val = float(inc.get(p, PARAM_BASELINE[p]))
            mat[i, j] = (val - lo) / (hi - lo)

    # Fila baseline
    baseline_norm = np.array([
        (PARAM_BASELINE[p] - PARAM_RANGES[p][0]) /
        (PARAM_RANGES[p][1] - PARAM_RANGES[p][0])
        for p in PARAM_NAMES
    ])

    fig, axes = plt.subplots(1, 2, figsize=(18, max(6, n_m * 0.7 + 2)))
    fig.suptitle(
        "Comparativa de Configuraciones Óptimas por Módulo\n"
        "(valores normalizados al rango [lo, hi] de cada parámetro)",
        fontsize=13, fontweight="bold",
    )

    ax_heat, ax_bar = axes

    # ── Heatmap ──────────────────────────────────────────────────
    im = ax_heat.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax_heat.set_xticks(range(n_p))
    ax_heat.set_xticklabels(PARAM_LABELS, rotation=45, ha="right", fontsize=8)
    ax_heat.set_yticks(range(n_m))
    ax_heat.set_yticklabels([MODULO_META[m]["label"] for m in modulos_presentes], fontsize=9)
    ax_heat.set_title("Heatmap: Parámetros Óptimos Normalizados", fontsize=11, fontweight="bold")

    for i in range(n_m):
        for j in range(n_p):
            ax_heat.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                         fontsize=6.5, color="black")

    plt.colorbar(im, ax=ax_heat, shrink=0.8, label="Valor norm. [0=mín, 1=máx]")

    # ── Barras agrupadas ──────────────────────────────────────────
    x      = np.arange(n_p)
    width  = 0.8 / (n_m + 1)
    offset = -(n_m / 2) * width

    for i, m in enumerate(modulos_presentes):
        meta = MODULO_META[m]
        ax_bar.bar(x + offset + i * width, mat[i],
                   width   = width * 0.9,
                   color   = meta["color"],
                   label   = meta["label"],
                   edgecolor="white", linewidth=0.5)

    # Línea baseline
    ax_bar.step(x + offset + (n_m / 2) * width,
                baseline_norm, where="mid",
                color="black", ls="--", lw=1.5, label="Baseline")

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(PARAM_LABELS, rotation=45, ha="right", fontsize=8)
    ax_bar.set_ylabel("Valor normalizado [0,1]", fontsize=10)
    ax_bar.set_title("Parámetros Óptimos por Módulo", fontsize=11, fontweight="bold")
    ax_bar.legend(fontsize=7, loc="upper right", ncol=2)
    ax_bar.grid(True, axis="y", alpha=0.3)
    ax_bar.set_ylim(0, 1.05)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(guardar_png, dpi=150, bbox_inches="tight")
    log.info("Gráfica de configuraciones guardada en '%s'", guardar_png)
    if mostrar:
        plt.show()
    plt.close(fig)


def graficar_tabla_resumen(
    resultados:  dict,
    guardar_png: str  = "comparativa_tabla_resumen.png",
    mostrar:     bool = False,
):
    """
    Tabla resumen con costo incumbente, mejora vs baseline,
    tiempo y nº evaluaciones.
    """
    modulos_presentes = [m for m in MODULO_META if m in resultados]
    if not modulos_presentes:
        return

    cols = ["Módulo", "Algoritmo", "Costo\n(días)", "Mejora\nvs baseline (%)",
            "Tiempo\n(seg)", "Evaluaciones\nsimulador"]
    rows = []
    for m in modulos_presentes:
        r       = resultados[m]
        costo   = r["costo_incumbente"]
        mejora  = (BASELINE_COSTO - costo) / BASELINE_COSTO * 100
        rows.append([
            m,
            r["algoritmo"],
            f"{costo:.2f}",
            f"{mejora:+.1f}%",
            f"{r['tiempo_seg']:.0f}",
            f"{r['n_evaluaciones']:,}",
        ])

    # Ordenar por costo (menor primero)
    rows.sort(key=lambda x: float(x[2]))

    fig, ax = plt.subplots(figsize=(14, max(3, len(rows) * 0.55 + 1.5)))
    fig.suptitle("Tabla Resumen — Comparativa de Módulos de Optimización",
                 fontsize=13, fontweight="bold")
    ax.axis("off")

    tabla = ax.table(
        cellText = rows,
        colLabels = cols,
        cellLoc  = "center",
        loc      = "center",
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(9)
    tabla.scale(1.2, 1.7)

    # Colorear filas según mejora
    for i, row in enumerate(rows):
        mejora_val = float(row[3].replace("%", "").replace("+", ""))
        color = "#d4f4d4" if mejora_val > 5 else \
                "#fff3cd" if mejora_val > 0 else \
                "#f8d7da"
        for j in range(len(cols)):
            tabla[i + 1, j].set_facecolor(color)

    # Header
    for j in range(len(cols)):
        tabla[0, j].set_facecolor("#4a4a8a")
        tabla[0, j].set_text_props(color="white", fontweight="bold")

    plt.tight_layout()
    fig.savefig(guardar_png, dpi=150, bbox_inches="tight")
    log.info("Tabla resumen guardada en '%s'", guardar_png)
    if mostrar:
        plt.show()
    plt.close(fig)


def graficar_todo(
    resultados:  dict,
    prefijo:     str  = "comparativa",
    mostrar:     bool = False,
):
    """Genera los 3 conjuntos de gráficas."""
    graficar_comparativa(
        resultados, f"{prefijo}_convergencia.png", mostrar)
    graficar_configuraciones(
        resultados, f"{prefijo}_configuraciones.png", mostrar)
    graficar_tabla_resumen(
        resultados, f"{prefijo}_tabla_resumen.png", mostrar)


# ═════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════

def _build_parser():
    p = argparse.ArgumentParser(
        description="Comparativa de módulos de optimización sobre simulador caja negra"
    )
    p.add_argument("--modulos", nargs="+",
                   choices=list(_RUNNERS.keys()) + ["todos"],
                   default=["M4", "M13", "M14"],
                   help="Módulos a comparar (default: M4 M13 M14)")
    p.add_argument("--todos", action="store_true",
                   help="Ejecutar todos los módulos")
    p.add_argument("--n_trials",   type=int,   default=20,
                   help="Trials para módulos SMAC (M4-M10) [default: 20]")
    p.add_argument("--max_iter",   type=int,   default=30,
                   help="Iteraciones para M11-M14 [default: 30]")
    p.add_argument("--n_corridas", type=int,   default=2,
                   help="Corridas por evaluación en SMAC [default: 2]")
    p.add_argument("--n_reps",     type=int,   default=5,
                   help="Réplicas por punto en M13/M14 [default: 5]")
    p.add_argument("--n_workers",  type=int,   default=0,
                   help="Workers paralelos (0=auto) para M11-M14 [default: 0]")
    p.add_argument("--seed",       type=int,   default=42,
                   help="Semilla aleatoria [default: 42]")
    p.add_argument("--json",       default="resultados_comparativa.json",
                   help="Archivo JSON de resultados")
    p.add_argument("--prefijo",    default="comparativa",
                   help="Prefijo de archivos PNG de salida")
    p.add_argument("--solo_graficas", default=None, metavar="JSON",
                   help="Re-graficar desde JSON ya existente (no re-ejecuta)")
    p.add_argument("--lambda_obj", type=float, default=None,
                   help="Valor λ para objetivo compuesto f=tts - λ·total_atenciones "
                        "(obtenido con calibrar_lambda.py). Si se omite usa sólo TTS.")
    return p


def main():
    args = _build_parser().parse_args()

    if args.solo_graficas:
        log.info("Cargando resultados desde '%s'...", args.solo_graficas)
        resultados = cargar_resultados(args.solo_graficas)
        graficar_todo(resultados, prefijo=args.prefijo)
        return

    modulos = list(_RUNNERS.keys()) if args.todos else args.modulos
    if "todos" in modulos:
        modulos = list(_RUNNERS.keys())

    pesos_kpi = None
    if args.lambda_obj is not None:
        pesos_kpi = {"tts_full_days_mean": 1.0, "total_atenciones": -args.lambda_obj}
        log.info("Objetivo compuesto activo: f = tts_full - %.4f · total_atenciones",
                 args.lambda_obj)

    resultados = ejecutar_comparativa(
        modulos     = modulos,
        n_trials    = args.n_trials,
        max_iter    = args.max_iter,
        n_corridas  = args.n_corridas,
        n_reps      = args.n_reps,
        n_workers   = args.n_workers,
        seed        = args.seed,
        guardar_json= args.json,
        pesos_kpi   = pesos_kpi,
    )

    if resultados:
        graficar_todo(resultados, prefijo=args.prefijo)
        log.info("\n✓ Comparativa finalizada. Archivos generados:")
        log.info("  · %s", args.json)
        log.info("  · %s_convergencia.png", args.prefijo)
        log.info("  · %s_configuraciones.png", args.prefijo)
        log.info("  · %s_tabla_resumen.png", args.prefijo)


if __name__ == "__main__":
    main()
