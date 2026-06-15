#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
modulo_smac_rf.py
══════════════════════════════════════════════════════════════════════
SMAC con Random Forest (HPO Facade) — SMAC 2.x API.

Diferencia clave vs M4-M10:
  · M4-M10 usan scikit-optimize (GP como sustituto)
  · SMAC_RF usa el Random Forest de SMAC3 (pyrfr backend)
    via HyperparameterOptimizationFacade → mismo marco teórico
    que Hutter et al. (2011), modelo original de SMAC.

Referencia:
  Hutter, F., Hoos, H.H., Leyton-Brown, K. (2011).
  Sequential Model-Based Optimization for General Algorithm Configuration.
  LION 5, LNCS 6683, pp. 507–523. DOI: 10.1007/978-3-642-25566-3_40
══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
import concurrent.futures
import dataclasses
import logging
import tempfile
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("modulo_smac_rf")

ENTEROS = {
    "horas_especialista_1ra", "horas_control_post", "cupos_laboratorio_ugd",
    "cupos_ecografia_matrona", "cupos_ecografia_ugd", "dias_publicacion",
    "num_matronas", "num_agentes_ugd",
}


def _build_configspace(seed: int = 42):
    """Construye el ConfigSpace de SMAC con los 12 parámetros del simulador."""
    from ConfigSpace import (
        ConfigurationSpace,
        UniformIntegerHyperparameter as UInt,
        UniformFloatHyperparameter   as UFloat,
    )
    cs = ConfigurationSpace(seed=seed)
    cs.add_hyperparameters([
        UInt  ("horas_especialista_1ra",   8,    30),
        UInt  ("horas_control_post",       20,   70),
        UInt  ("cupos_laboratorio_ugd",    20,  100),
        UInt  ("cupos_ecografia_matrona",  10,   50),
        UInt  ("cupos_ecografia_ugd",      10,   50),
        UInt  ("dias_publicacion",         1,    10),
        UFloat("pct_bloqueo_1ra",          0.05, 0.50),
        UFloat("pct_consultas_vacias",     0.05, 0.50),
        UInt  ("num_matronas",             1,    4),
        UInt  ("num_agentes_ugd",          1,    4),
        UFloat("pct_no_contactabilidad",   0.05, 0.50),
        UFloat("pct_bloqueo_post_control", 0.05, 0.50),
    ])
    return cs


def _evaluar_config_worker(args):
    """Worker picklable para ProcessPoolExecutor."""
    seed_offset, cfg_dict, objetivo, pesos_kpi = args
    from simulador_clinica_baseline import run_once, SimConfig
    cfg = SimConfig(**{k: v for k, v in cfg_dict.items()
                       if k in SimConfig.__dataclass_fields__})
    res = run_once(seed_offset=seed_offset, cfg=cfg)
    if objetivo == "compuesto" and pesos_kpi:
        return sum(float(pesos_kpi.get(k, 0.0)) * float(res.get(k, 0.0))
                   for k in pesos_kpi)
    return float(res.get(objetivo, 1e9))


def _vals_a_simcfg(vals: dict):
    """Mapea dict de parámetros → SimConfig."""
    from simulador_clinica_baseline import SimConfig, CFG
    cfg = dataclasses.replace(CFG)
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
    cfg.benchmark_mode               = True
    return cfg


def _evaluar_config(vals: dict, n_corridas: int, seed_base: int,
                    pesos_kpi: dict | None) -> float:
    """Evalúa una configuración con n_corridas réplicas. Retorna media."""
    objetivo = "compuesto" if pesos_kpi else "tts_full_days_mean"
    cfg = _vals_a_simcfg(vals)
    cfg_dict = dataclasses.asdict(cfg)
    resultados = []
    for r in range(n_corridas):
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(_evaluar_config_worker,
                                (seed_base + r, cfg_dict, objetivo, pesos_kpi))
                val = fut.result(timeout=600.0)
            resultados.append(float(val))
        except Exception as e:
            log.warning("SMAC-RF eval error rep=%d: %s", r, e)
    return float(np.mean(resultados)) if resultados else 1e9


class _SMACTarget:
    """Callable que registra historia de evaluaciones para SMAC."""

    def __init__(self, n_corridas: int, seed_base: int,
                 pesos_kpi: dict | None, historia: list, t0: float):
        self.n_corridas = n_corridas
        self.seed_base  = seed_base
        self.pesos_kpi  = pesos_kpi
        self.historia   = historia
        self.t0         = t0
        self.call_count = 0
        self.best       = float("inf")
        self.best_vals  = {}

    def __call__(self, config, seed: int = 0) -> float:
        self.call_count += 1
        vals = {k: (int(v) if k in ENTEROS else float(v))
                for k, v in dict(config).items()}
        sb = self.seed_base + self.call_count * self.n_corridas
        costo = _evaluar_config(vals, self.n_corridas, sb, self.pesos_kpi)
        if costo < self.best:
            self.best      = costo
            self.best_vals = vals.copy()
        self.historia.append({
            "iter":              self.call_count,
            "costo":             round(costo, 4),
            "mejor_hasta_ahora": round(self.best, 4),
            "n_reps":            self.n_corridas,
            "t_seg":             round(time.time() - self.t0, 2),
        })
        log.info("SMAC-RF iter=%3d  f=%.2f  mejor=%.2f  n_eval=%d",
                 self.call_count, costo, self.best,
                 self.call_count * self.n_corridas)
        return costo


def smac_rf_runner(n_trials: int = 50, n_corridas: int = 3,
                   seed: int = 42, pesos_kpi: dict | None = None) -> dict:
    """
    Optimiza con SMAC HyperparameterOptimizationFacade (Random Forest + EI).

    Parámetros
    ----------
    n_trials   : nº de configuraciones evaluadas por SMAC
    n_corridas : réplicas del simulador por configuración
    seed       : semilla global
    pesos_kpi  : dict de pesos para objetivo compuesto (None → TTS)

    Retorna
    -------
    dict con costo_incumbente, incumbente, historia_costos, conv_eval, conv_time
    """
    from smac import HyperparameterOptimizationFacade, Scenario

    t0      = time.time()
    historia: list[dict] = []

    cs     = _build_configspace(seed=seed)
    target = _SMACTarget(n_corridas, seed * 100_000, pesos_kpi, historia, t0)

    with tempfile.TemporaryDirectory() as tmpdir:
        scenario = Scenario(
            configspace = cs,
            n_trials    = n_trials,
            seed        = seed,
            output_directory = Path(tmpdir),
            n_workers   = 1,
        )
        smac = HyperparameterOptimizationFacade(
            scenario,
            target,
            overwrite   = True,
            logging_level = logging.WARNING,
        )
        incumbent = smac.optimize()

    # Extraer incumbente final (SMAC puede haber encontrado uno mejor al final)
    best_vals = {k: (int(v) if k in ENTEROS else float(v))
                 for k, v in dict(incumbent).items()}
    best_cost = target.best
    if best_vals != target.best_vals:
        # Verificar costo del incumbente SMAC
        c = _evaluar_config(best_vals, n_corridas,
                            seed * 100_000 + 999_000, pesos_kpi)
        if c < best_cost:
            best_cost = c
            target.best_vals = best_vals

    # Construir conv_eval y conv_time (acumulado)
    eval_acum = 0
    conv_eval, conv_time = [], []
    for h in historia:
        eval_acum += h["n_reps"]
        conv_eval.append([eval_acum, h["mejor_hasta_ahora"]])
        conv_time.append([h["t_seg"],  h["mejor_hasta_ahora"]])

    return {
        "modulo":           "SMAC_RF",
        "algoritmo":        "SMAC-HPO (Random Forest + EI)",
        "costo_incumbente": round(best_cost, 4),
        "tiempo_seg":       round(time.time() - t0, 2),
        "n_evaluaciones":   target.call_count * n_corridas,
        "incumbente":       target.best_vals,
        "historia_costos":  historia,
        "conv_eval":        conv_eval,
        "conv_time":        conv_time,
        "seed":             seed,
    }
