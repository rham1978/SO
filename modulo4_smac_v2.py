"""
===============================================================================
MÓDULO 4 — OPTIMIZACIÓN SMAC3
===============================================================================
Conectado con: simulador_clinica_baseline.py

Implementa los 6 tipos de SMAC3 según la imagen de referencia:
  1. Black-Box                  (Gaussian Process + EI + Sobol)
  2. Hyperparameter Optimization (Random Forest + EI + Sobol)
  3. Multi-Fidelity             (Random Forest + EI + Sobol + Hyperband)
  4. Algorithm Configuration    (Random Forest + EI + Default + Hyperband)
  5. Random                     (búsqueda aleatoria pura)
  6. Hyperband                  (bandit-based intensification)

Parámetros del modelo que se optimizan (11):
  1.  horas_especialista_1ra   → fixed_weekly_capacity (slots/sem 1ra consulta)
  2.  horas_control_post       → fixed_post_control_capacity (pac/sem control)
  3.  cupos_laboratorio_ugd    → ugd_lab_per_week
  4.  cupos_ecografia_matrona  → mat_us_per_week
  5.  cupos_ecografia_ugd      → ugd_us_per_week
  6.  dias_publicacion         → publish_lead_workdays
  7.  pct_bloqueo_1ra          → blocked_pct
  8.  pct_consultas_vacias     → empty_control_p_ugd
  9.  num_matronas             → matrona capacity
  10. num_agentes_ugd          → agent_capacity
  11. pct_no_contactabilidad   → not_contactable_p

Objetivo por defecto: tts_full_days_mean (RECOMENDADO)
─────────────────────────────────────────────────────
  Mide el tiempo total desde que el paciente ingresa al sistema hasta
  que completa toda su ruta post-consulta (alta o cirugía).

  Incluye:
    · Backlog histórico pre-t=0  (~54 días en baseline)
      Tiempo que el paciente esperaba ANTES de que empiece la simulación.
    · Proceso primera consulta   (~140 días en baseline)
      Desde ingreso hasta atención primera consulta (dentro de la sim).
    · Post-consulta              (~127 días en baseline)
      Desde primera consulta hasta alta/cirugía (control, preq, quir).

  Baseline: tts_full_days_mean = 279.78 días
  SMAC busca configuraciones que reduzcan este valor.

  Por qué es el mejor objetivo:
    · Es el KPI clínico más completo — captura toda la experiencia del paciente
    · Incluye el backlog histórico, que refleja la realidad del sistema
    · Es sensible a mejoras en primera consulta Y en post-consulta
    · Penaliza tanto las esperas iniciales como los cuellos de botella post

Otros objetivos disponibles (a minimizar):
  - wl_total_end               : lista espera total al final del horizonte
  - wl_first_end               : lista espera 1ra consulta al final
  - tts_first_attended_days_mean: TTS primera consulta atendida (días)
  - compuesto                  : suma ponderada normalizada de múltiples KPIs

Uso:
    pip install smac

    # Correr con objetivo por defecto (tts_full_days_mean)
    python modulo4_smac.py --tipo hpo --n_trials 50 --n_corridas 3

    # Con paralelismo (3 workers = 1 por réplica, speedup ~3x)
    python modulo4_smac.py --tipo hpo --n_trials 50 --n_corridas 3 --n_workers 3

    # Correr todos los tipos y comparar
    python modulo4_smac.py --tipo todos --n_trials 30 --n_corridas 2

    # Sin gráficos PNG
    python modulo4_smac.py --tipo hpo --n_trials 50 --no_plot

    # Mostrar espacio de búsqueda y valor baseline del objetivo
    python modulo4_smac.py --mostrar_espacio

    # Evaluar incumbente encontrado con más corridas
    python modulo4_smac.py --evaluar_incumbente resultado_smac_hpo.json

O importar:
    from modulo4_smac import optimizar, comparar_todos, TIPOS_SMAC
===============================================================================
"""

import logging
import json
import time
import sys
import os
import copy
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

# ── Paralelismo (Nivel 1: corridas dentro de cada evaluación) ──
import concurrent.futures
import multiprocessing
import dataclasses

# ── Visualización ──────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")          # sin GUI — guarda PNG directamente
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("smac_opt")


# ──────────────────────────────────────────────────────────────────────────────
# Descripción de los 6 tipos de SMAC (según imagen de referencia)
# ──────────────────────────────────────────────────────────────────────────────
TIPOS_SMAC = {
    "blackbox": {
        "nombre":        "Black-Box",
        "modelo":        "Gaussian Process",
        "adquisicion":   "Expected Improvement (EI)",
        "disenio_inicial":"Sobol",
        "intensificador":"Default",
        "prob_aleatorio": 0.085,
        "multi_fidelity": False,
        "descripcion":   "Optimización caja negra con GP. Ideal para pocos parámetros.",
    },
    "hpo": {
        "nombre":        "Hyperparameter Optimization",
        "modelo":        "Random Forest",
        "adquisicion":   "Expected Improvement (EI)",
        "disenio_inicial":"Sobol",
        "intensificador":"Default",
        "prob_aleatorio": 0.20,
        "multi_fidelity": False,
        "descripcion":   "HPO con RF. Robusto para espacios grandes. Recomendado.",
    },
    "multifidelity": {
        "nombre":        "Multi-Fidelity",
        "modelo":        "Random Forest",
        "adquisicion":   "Expected Improvement (EI)",
        "disenio_inicial":"Sobol",
        "intensificador":"Hyperband",
        "prob_aleatorio": 0.20,
        "multi_fidelity": True,
        "descripcion":   "Multi-fidelidad con Hyperband. Usa menos corridas para descartar configs malas.",
    },
    "algconfig": {
        "nombre":        "Algorithm Configuration",
        "modelo":        "Random Forest",
        "adquisicion":   "Expected Improvement (EI)",
        "disenio_inicial":"Default",
        "intensificador":"Hyperband",
        "prob_aleatorio": 0.50,
        "multi_fidelity": False,
        "descripcion":   "Configuración de algoritmos. Alta exploración aleatoria (50%).",
    },
    "random": {
        "nombre":        "Random",
        "modelo":        "No usado",
        "adquisicion":   "No usado",
        "disenio_inicial":"Default",
        "intensificador":"Default",
        "prob_aleatorio": 1.0,
        "multi_fidelity": False,
        "descripcion":   "Búsqueda aleatoria pura. Útil como línea base de comparación.",
    },
    "hyperband": {
        "nombre":        "Hyperband",
        "modelo":        "No usado",
        "adquisicion":   "No usado",
        "disenio_inicial":"Default",
        "intensificador":"Hyperband",
        "prob_aleatorio": 1.0,
        "multi_fidelity": True,
        "descripcion":   "Bandit-based multi-fidelidad puro. Rápido para descartar configs.",
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Conexión con el modelo baseline
# ──────────────────────────────────────────────────────────────────────────────
_baseline_cache = None

def _importar_baseline():
    global _baseline_cache
    if _baseline_cache is not None:
        return _baseline_cache

    directorio = os.path.dirname(os.path.abspath(__file__))
    if directorio not in sys.path:
        sys.path.insert(0, directorio)

    try:
        from simulador_clinica_baseline import ClinicModelAdjusted, CFG, run_once
        import simpy, random as _random
        _baseline_cache = (ClinicModelAdjusted, CFG, run_once, simpy, _random)
        log.info("Baseline importado: simulador_clinica_baseline.py")
        return _baseline_cache
    except ImportError as e:
        log.error("No se pudo importar simulador_clinica_baseline.py: %s", e)
        raise


def _aplicar_config_a_cfg(config, cfg_base):
    """
    Aplica los parámetros de una ConfigSpace.Configuration al SimConfig del baseline.
    Retorna una copia modificada del CFG.
    """
    cfg = copy.deepcopy(cfg_base)

    # 1. Horas especialista 1ra consulta → slots/sem
    if "horas_especialista_1ra" in config:
        cfg.fixed_weekly_capacity     = int(config["horas_especialista_1ra"])
        cfg.use_fixed_weekly_capacity = True

    # 2. Pac/sem control post → fixed_post_control_capacity
    if "horas_control_post" in config:
        cfg.fixed_post_control_capacity  = int(config["horas_control_post"])
        cfg.use_fixed_post_control_hours = True

    # 3. Cupos laboratorio UGD
    if "cupos_laboratorio_ugd" in config:
        cfg.ugd_lab_per_week = int(config["cupos_laboratorio_ugd"])

    # 4. Cupos ecografía matrona
    if "cupos_ecografia_matrona" in config:
        cfg.mat_us_per_week = int(config["cupos_ecografia_matrona"])

    # 5. Cupos ecografía UGD
    if "cupos_ecografia_ugd" in config:
        cfg.ugd_us_per_week = int(config["cupos_ecografia_ugd"])

    # 6. Días de publicación anticipada
    if "dias_publicacion" in config:
        cfg.publish_lead_workdays = int(config["dias_publicacion"])

    # 7. % Bloqueo primera consulta
    if "pct_bloqueo_1ra" in config:
        cfg.blocked_pct = float(config["pct_bloqueo_1ra"])

    # 8. % Consultas vacías UGD control
    if "pct_consultas_vacias" in config:
        cfg.empty_control_p_ugd = float(config["pct_consultas_vacias"])

    # 9. Número de matronas (capacidad recurso)
    if "num_matronas" in config:
        cfg.matrona_capacity = int(config["num_matronas"])

    # 10. Número de agentes UGD
    if "num_agentes_ugd" in config:
        cfg.agent_capacity = int(config["num_agentes_ugd"])

    # 11. % No contactabilidad
    if "pct_no_contactabilidad" in config:
        cfg.not_contactable_p = float(config["pct_no_contactabilidad"])

    # 12. % Bloqueo post-control
    if "pct_bloqueo_post_control" in config:
        cfg.blocked_pct_post_control = float(config["pct_bloqueo_post_control"])

    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Worker pickleable para ProcessPoolExecutor
# ──────────────────────────────────────────────────────────────────────────────
def _worker_run_once(args: tuple) -> float:
    """
    Worker de nivel módulo — requerido para multiprocessing (pickleable).
    args = (seed_offset, cfg_dict, objetivo, pesos_kpi)

    Importa el simulador dentro del worker para compatibilidad con
    el método 'spawn' de multiprocessing (Windows / macOS).
    Retorna el valor escalar del KPI para una réplica.
    """
    seed_offset, cfg_dict, objetivo, pesos_kpi = args
    try:
        import sys, os
        for d in [os.path.dirname(os.path.abspath(__file__)), os.getcwd()]:
            if d not in sys.path:
                sys.path.insert(0, d)
        from simulador_clinica_baseline import run_once, SimConfig
        cfg = SimConfig(**cfg_dict)
        res = run_once(seed_offset=seed_offset, cfg=cfg)
        if objetivo == "compuesto" and pesos_kpi:
            return sum(float(pesos_kpi.get(k, 0.0)) * float(res.get(k, 0.0))
                       for k in pesos_kpi)
        return float(res.get(objetivo, 1e9))
    except Exception as e:
        import logging
        logging.getLogger("smac_opt").warning("Worker error seed=%d: %s", seed_offset, e)
        return 1e9


def evaluar_configuracion(
    config,
    seed:       int   = 0,
    n_corridas: int   = 3,
    seed_base:  int   = 202,
    objetivo:   str   = "tts_full_days_mean",
    pesos_kpi:  dict  = None,
    budget:     float = None,
    n_workers:  int   = 0,
) -> float:
    """
    Función objetivo compartida por todos los tipos de SMAC — versión paralelizada.
    Las n_corridas de run_once se ejecutan en paralelo via ProcessPoolExecutor.

    n_workers: procesos paralelos (0 = automático = min(n, nCPUs)).
    """
    _, CFG_base, _, _, _ = _importar_baseline()
    cfg = _aplicar_config_a_cfg(config, CFG_base)
    cfg_dict = dataclasses.asdict(cfg)

    n = max(1, int(round(budget))) if budget is not None else n_corridas

    workers = min(n_workers if n_workers > 0 else n,
                  n,
                  multiprocessing.cpu_count())

    tasks = [
        (seed_base + seed + r, cfg_dict, objetivo, pesos_kpi or {})
        for r in range(n)
    ]

    if workers <= 1 or n == 1:
        valores = [_worker_run_once(t) for t in tasks]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
            valores = list(ex.map(_worker_run_once, tasks))

    costo = float(np.mean(valores))
    log.info("  Config eval → %s=%.2f (n=%d, workers=%d, seed=%d)",
             objetivo, costo, n, workers if n > 1 else 1, seed)
    return costo


# ──────────────────────────────────────────────────────────────────────────────
# Espacio de configuración (11 parámetros)
# ──────────────────────────────────────────────────────────────────────────────
def crear_espacio_configuracion(seed: int = 42):
    """
    Crea el ConfigurationSpace con los 11 parámetros del modelo clínico.
    Rangos calibrados respecto al baseline documentado.
    """
    try:
        from ConfigSpace import ConfigurationSpace
        from ConfigSpace.hyperparameters import (
            UniformIntegerHyperparameter as IntHP,
            UniformFloatHyperparameter   as FloatHP,
        )
    except ImportError:
        raise ImportError("Instala ConfigSpace: pip install smac")

    cs = ConfigurationSpace(seed=seed)
    cs.add([
        # 1. Slots/sem primera consulta (baseline=16)
        IntHP("horas_especialista_1ra",  lower=8,   upper=30,  default_value=16),
        # 2. Pac/sem control post (baseline=40)
        IntHP("horas_control_post",      lower=20,  upper=70,  default_value=40),
        # 3. Cupos lab UGD/sem (baseline=54)
        IntHP("cupos_laboratorio_ugd",   lower=20,  upper=100, default_value=54),
        # 4. Cupos eco matrona/sem (baseline=25)
        IntHP("cupos_ecografia_matrona", lower=10,  upper=50,  default_value=25),
        # 5. Cupos eco UGD/sem (baseline=25)
        IntHP("cupos_ecografia_ugd",     lower=10,  upper=50,  default_value=25),
        # 6. Días publicación anticipada (baseline=5)
        IntHP("dias_publicacion",        lower=1,   upper=10,  default_value=5),
        # 7. % Bloqueo 1ra (baseline=0.32)
        FloatHP("pct_bloqueo_1ra",       lower=0.05, upper=0.50, default_value=0.32),
        # 8. % Consultas vacías UGD (baseline=0.30)
        FloatHP("pct_consultas_vacias",  lower=0.05, upper=0.50, default_value=0.30),
        # 9. Número matronas (baseline=1)
        IntHP("num_matronas",            lower=1,   upper=4,   default_value=1),
        # 10. Número agentes UGD (baseline=1)
        IntHP("num_agentes_ugd",         lower=1,   upper=4,   default_value=1),
        # 11. % No contactabilidad (baseline=0.30)
        FloatHP("pct_no_contactabilidad",lower=0.05, upper=0.50, default_value=0.30),
        FloatHP("pct_bloqueo_post_control", lower=0.05, upper=0.50, default_value=0.34),
        
    ])
    return cs


# ──────────────────────────────────────────────────────────────────────────────
# Visualización: convergencia + tiempo de proceso
# ──────────────────────────────────────────────────────────────────────────────
def graficar(
    resultado,
    guardar_png: str  = None,
    mostrar:     bool = False,
) -> str:
    """
    Genera figura de 2 paneles (convergencia + tiempo) a partir de un ResultadoSMAC.
    Acepta dataclass o dict (compatible con JSON cargado).
    Retorna la ruta del PNG guardado.
    """
    r = resultado.__dict__ if hasattr(resultado, "__dict__") else dict(resultado)

    historia      = r.get("historia_costos", [])
    tipo          = r.get("tipo", "")
    objetivo      = r.get("objetivo", "objetivo")
    t_total       = float(r.get("tiempo_seg", 0.0))
    costo_final   = float(r.get("costo_incumbente", float("inf")))
    n_corridas    = int(r.get("n_corridas_eval", 1))
    nombre_tipo   = r.get("nombre_tipo", tipo)

    if not historia:
        log.warning("graficar(): historia_costos vacía.")
        return ""

    n           = len(historia)
    costos      = [h.get("costo", float("nan"))              for h in historia]
    incumbentes = [h.get("mejor_hasta_ahora", float("nan"))  for h in historia]
    t_secs      = [h.get("t_seg", None)                      for h in historia]

    tiene_tiempo = any(v is not None for v in t_secs)
    if not tiene_tiempo and t_total > 0:
        t_secs = [round(t_total * (i + 1) / n, 1) for i in range(n)]
    t_secs  = [v if v is not None else 0.0 for v in t_secs]
    t_delta = [t_secs[0]] + [max(0.0, t_secs[i] - t_secs[i-1]) for i in range(1, n)]
    trials  = list(range(1, n + 1))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 8),
        gridspec_kw={"height_ratios": [3, 2]},
    )
    fig.suptitle(
        f"SMAC3 — {nombre_tipo}\n"
        f"Objetivo: {objetivo}  ·  {n} evaluaciones  ·  "
        f"{n_corridas} réplicas/config  ·  {t_total:.0f}s totales",
        fontsize=11, fontweight="bold", y=0.98,
    )

    # Panel 1: Convergencia
    validos = [(i, c) for i, c in zip(trials, costos) if not np.isnan(c) and c < 1e8]
    if validos:
        xi, yi = zip(*validos)
        ax1.scatter(xi, yi, s=18, color="#aaaaaa", alpha=0.55, zorder=2,
                    label="Evaluaciones")
    inc_v = [(i, v) for i, v in zip(trials, incumbentes) if not np.isnan(v)]
    if inc_v:
        xi, yi = zip(*inc_v)
        ax1.step(xi, yi, where="post", color="#1a6db5", linewidth=2.2,
                 zorder=3, label="Mejor incumbente")
        ax1.fill_between(xi, yi, step="post", alpha=0.10, color="#1a6db5")
    if not np.isinf(costo_final):
        ax1.axhline(costo_final, color="#c0392b", linewidth=1.2,
                    linestyle="--", alpha=0.75,
                    label=f"Mejor final = {costo_final:.2f}")
    n_init = max(1, round(n * 0.10))
    ax1.axvspan(0.5, n_init + 0.5, alpha=0.07, color="#e67e22",
                label=f"Diseño inicial (~{n_init} configs)")
    ax1.axvline(n_init + 0.5, color="#e67e22", linewidth=0.8,
                linestyle=":", alpha=0.6)
    ax1.set_xlabel("N° evaluación", fontsize=10)
    ax1.set_ylabel(objetivo, fontsize=10)
    ax1.set_title("Convergencia", fontsize=10, pad=4)
    ax1.legend(fontsize=8.5, framealpha=0.9)
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=12))
    ax1.set_xlim(0.5, n + 0.5)

    # Panel 2: Tiempo de proceso
    ax2.bar(trials, t_delta, color="#b0c4de", edgecolor="none",
            alpha=0.7, label="Tiempo por evaluación (s)", zorder=2)
    ax2r = ax2.twinx()
    ax2r.plot(trials, t_secs, color="#1a6db5", linewidth=1.8,
              alpha=0.85, label="Tiempo acumulado (s)", zorder=3)
    ax2r.fill_between(trials, t_secs, alpha=0.08, color="#1a6db5")
    ax2r.set_ylabel("Tiempo acumulado (s)", fontsize=9, color="#1a6db5")
    ax2r.tick_params(axis="y", labelcolor="#1a6db5")
    t_medio = t_total / n if n > 0 else 0
    ax2.axhline(t_medio, color="#e74c3c", linewidth=1.2, linestyle="--",
                alpha=0.75, label=f"Promedio = {t_medio:.1f}s/eval")
    ax2r.axhline(t_total, color="#1a6db5", linewidth=0.8,
                 linestyle=":", alpha=0.5)
    ax2r.text(n * 0.98, t_total * 1.01, f"{t_total:.0f}s",
              ha="right", va="bottom", fontsize=8, color="#1a6db5")
    ax2.set_xlabel("N° evaluación", fontsize=10)
    ax2.set_ylabel("Tiempo incremental (s)", fontsize=9)
    ax2.set_title("Tiempo de proceso por evaluación", fontsize=10, pad=4)
    ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=12))
    ax2.set_xlim(0.5, n + 0.5)
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8.5,
               framealpha=0.9, loc="upper left")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if guardar_png is None:
        guardar_png = f"convergencia_{tipo}_{objetivo}.png"
    try:
        fig.savefig(guardar_png, dpi=150, bbox_inches="tight")
        log.info("Gráfico guardado: %s", guardar_png)
    except Exception as e:
        log.warning("No pudo guardar %s: %s", guardar_png, e)
    if mostrar:
        plt.show()
    plt.close(fig)
    return guardar_png


def graficar_comparacion(
    resultados:  dict,
    guardar_png: str  = "convergencia_comparacion_todos.png",
    mostrar:     bool = False,
) -> str:
    """
    Gráfico de convergencia y tiempo superpuesto para comparar múltiples tipos SMAC.
    resultados: {tipo: ResultadoSMAC_o_dict, ...}  (hasta 6 tipos)
    """
    colores = ["#1a6db5", "#c0392b", "#27ae60", "#8e44ad", "#e67e22", "#7f8c8d"]
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                    gridspec_kw={"height_ratios": [3, 2]})
    objetivo = ""
    for idx, (nombre, res) in enumerate(resultados.items()):
        r          = res.__dict__ if hasattr(res, "__dict__") else dict(res)
        historia   = r.get("historia_costos", [])
        objetivo   = r.get("objetivo", objetivo)
        t_total    = float(r.get("tiempo_seg", 0.0))
        costo_fin  = float(r.get("costo_incumbente", float("inf")))
        color      = colores[idx % len(colores)]
        n          = len(historia)
        if not historia:
            continue
        incumbentes = [h.get("mejor_hasta_ahora", float("nan")) for h in historia]
        t_secs      = [h.get("t_seg", None) for h in historia]
        if not any(v is not None for v in t_secs) and t_total > 0:
            t_secs = [round(t_total * (i + 1) / n, 1) for i in range(n)]
        t_secs = [v if v is not None else 0.0 for v in t_secs]
        trials = list(range(1, n + 1))
        inc_v  = [(i, v) for i, v in zip(trials, incumbentes) if not np.isnan(v)]
        if inc_v:
            xi, yi = zip(*inc_v)
            ax1.step(xi, yi, where="post", color=color, linewidth=2.0,
                     label=f"{nombre}  (final={costo_fin:.2f})", zorder=3)
        ax2.plot(trials, t_secs, color=color, linewidth=1.8,
                 label=f"{nombre}  ({t_total:.0f}s)")

    ax1.set_xlabel("N° evaluación", fontsize=10)
    ax1.set_ylabel(objetivo, fontsize=10)
    ax1.set_title("Convergencia comparativa", fontsize=10, pad=4)
    ax1.legend(fontsize=8.5, framealpha=0.9)
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=12))
    ax2.set_xlabel("N° evaluación", fontsize=10)
    ax2.set_ylabel("Tiempo acumulado (s)", fontsize=10)
    ax2.set_title("Tiempo acumulado comparativo", fontsize=10, pad=4)
    ax2.legend(fontsize=8.5, framealpha=0.9)
    ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=12))
    fig.suptitle(f"Comparación SMAC3  ·  objetivo: {objetivo}",
                 fontsize=11, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    try:
        fig.savefig(guardar_png, dpi=150, bbox_inches="tight")
        log.info("Gráfico comparativo guardado: %s", guardar_png)
    except Exception as e:
        log.warning("No pudo guardar %s: %s", guardar_png, e)
    if mostrar:
        plt.show()
    plt.close(fig)
    return guardar_png


# ──────────────────────────────────────────────────────────────────────────────
# Dataclass resultado
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class ResultadoSMAC:
    tipo:             str   = ""
    nombre_tipo:      str   = ""
    objetivo:         str   = ""
    n_trials:         int   = 0
    n_corridas_eval:  int   = 0
    seed:             int   = 42
    incumbente:       dict  = field(default_factory=dict)
    costo_incumbente: float = 0.0
    n_evaluaciones:   int   = 0
    tiempo_seg:       float = 0.0
    output_dir:       str   = ""
    historia_costos:  list  = field(default_factory=list)
    convergido:       bool  = False
    descripcion:      str   = ""


# ──────────────────────────────────────────────────────────────────────────────
# Builders de cada tipo de SMAC
# ──────────────────────────────────────────────────────────────────────────────
def _scenario_base(cs, n_trials, output_dir, seed, min_b=None, max_b=None):
    from smac import Scenario
    kwargs = dict(
        configspace      = cs,
        n_trials         = n_trials,
        output_directory = Path(output_dir),
        seed             = seed,
        name             = Path(output_dir).name,
    )
    if min_b is not None: kwargs["min_budget"] = min_b
    if max_b is not None: kwargs["max_budget"] = max_b
    return Scenario(**kwargs)


def _build_blackbox(scenario, fn_obj):
    """
    Black-Box: Gaussian Process + EI + Sobol + Default intensifier
    prob_random = 8.5%
    """
    from smac import BlackBoxFacade
    from smac.initial_design import SobolInitialDesign
    from smac.acquisition.function import EI
    from smac.random_design import ProbabilityRandomDesign

    return BlackBoxFacade(
        scenario          = scenario,
        target_function   = fn_obj,
        initial_design    = SobolInitialDesign(scenario),
        acquisition_function = EI(),
        random_design     = ProbabilityRandomDesign(probability=0.085),
        overwrite         = True,
    )


def _build_hpo(scenario, fn_obj):
    """
    HPO: Random Forest + EI + Sobol + Default intensifier
    prob_random = 20%
    """
    from smac import HyperparameterOptimizationFacade
    from smac.initial_design import SobolInitialDesign
    from smac.acquisition.function import EI
    from smac.random_design import ProbabilityRandomDesign

    return HyperparameterOptimizationFacade(
        scenario          = scenario,
        target_function   = fn_obj,
        initial_design    = SobolInitialDesign(scenario),
        acquisition_function = EI(),
        random_design     = ProbabilityRandomDesign(probability=0.20),
        overwrite         = True,
    )


def _build_multifidelity(scenario, fn_obj):
    """
    Multi-Fidelity: Random Forest + EI + Sobol + Hyperband intensifier
    prob_random = 20%
    """
    from smac import MultiFidelityFacade
    from smac.initial_design import SobolInitialDesign
    from smac.acquisition.function import EI
    from smac.random_design import ProbabilityRandomDesign
    from smac.intensifier import Hyperband

    return MultiFidelityFacade(
        scenario          = scenario,
        target_function   = fn_obj,
        initial_design    = SobolInitialDesign(scenario),
        acquisition_function = EI(),
        intensifier       = Hyperband(scenario, incumbent_selection="highest_budget"),
        random_design     = ProbabilityRandomDesign(probability=0.20),
        overwrite         = True,
    )


def _build_algconfig(scenario, fn_obj):
    """
    Algorithm Configuration: Random Forest + EI + Default design + Hyperband
    prob_random = 50%
    """
    from smac.facade.algorithm_configuration_facade import AlgorithmConfigurationFacade
    from smac.initial_design import DefaultInitialDesign
    from smac.acquisition.function import EI
    from smac.random_design import ProbabilityRandomDesign
    from smac.intensifier import Hyperband

    return AlgorithmConfigurationFacade(
        scenario          = scenario,
        target_function   = fn_obj,
        initial_design    = DefaultInitialDesign(scenario),
        acquisition_function = EI(),
        intensifier       = Hyperband(scenario, incumbent_selection="highest_budget"),
        random_design     = ProbabilityRandomDesign(probability=0.50),
        overwrite         = True,
    )


def _build_random(scenario, fn_obj):
    """
    Random: búsqueda aleatoria pura
    """
    from smac import RandomFacade
    from smac.initial_design import RandomInitialDesign

    return RandomFacade(
        scenario       = scenario,
        target_function= fn_obj,
        initial_design = RandomInitialDesign(scenario),
        overwrite      = True,
    )


def _build_hyperband(scenario, fn_obj):
    """
    Hyperband: bandit-based multi-fidelidad puro
    """
    from smac.facade.hyperband_facade import HyperbandFacade
    from smac.initial_design import DefaultInitialDesign
    from smac.intensifier import Hyperband

    return HyperbandFacade(
        scenario        = scenario,
        target_function = fn_obj,
        initial_design  = DefaultInitialDesign(scenario),
        intensifier     = Hyperband(scenario, incumbent_selection="highest_budget"),
        overwrite       = True,
    )


_BUILDERS = {
    "blackbox":     _build_blackbox,
    "hpo":          _build_hpo,
    "multifidelity":_build_multifidelity,
    "algconfig":    _build_algconfig,
    "random":       _build_random,
    "hyperband":    _build_hyperband,
}


# ──────────────────────────────────────────────────────────────────────────────
# Función principal: optimizar con un tipo de SMAC
# ──────────────────────────────────────────────────────────────────────────────
def optimizar(
    tipo:            str   = "hpo",
    n_trials:        int   = 50,
    n_corridas_eval: int   = 3,
    seed:            int   = 42,
    seed_base_modelo:int   = 202,
    objetivo:        str   = "tts_full_days_mean",
    pesos_kpi:       dict  = None,
    output_dir:      str   = None,
    guardar_json:    str   = None,
    min_budget:      float = None,
    max_budget:      float = None,
    n_workers:       int   = 0,
) -> ResultadoSMAC:
    """
    Ejecuta la optimización con el tipo de SMAC especificado.

    n_workers: procesos paralelos por evaluación (0 = auto = min(n_corridas, nCPUs)).
    """
    if tipo not in _BUILDERS:
        raise ValueError(f"Tipo '{tipo}' no válido. Opciones: {list(_BUILDERS.keys())}")

    info = TIPOS_SMAC[tipo]
    output_dir  = output_dir  or f"smac_output_{tipo}"
    guardar_json= guardar_json or f"resultado_smac_{tipo}.json"

    log.info("="*65)
    log.info("SMAC3 — %s", info["nombre"])
    log.info("Objetivo  : %s", objetivo)
    log.info("n_trials  : %d  |  n_corridas_eval: %d  |  seed: %d",
             n_trials, n_corridas_eval, seed)
    log.info("="*65)

    cs = crear_espacio_configuracion(seed=seed)

    _pesos = pesos_kpi or {}
    _tracker = {"t0_opt": None, "calls": []}

    def fn_obj(config, seed: int = 0, budget: float = None) -> float:
        if _tracker["t0_opt"] is None:
            _tracker["t0_opt"] = time.time()
        costo = evaluar_configuracion(
            config      = config,
            seed        = seed,
            n_corridas  = n_corridas_eval,
            seed_base   = seed_base_modelo,
            objetivo    = objetivo,
            pesos_kpi   = _pesos,
            budget      = budget,
            n_workers   = n_workers,
        )
        _tracker["calls"].append({
            "t_seg": round(time.time() - _tracker["t0_opt"], 2),
            "costo": round(float(costo), 4),
        })
        return costo

    es_multifidelidad = info["multi_fidelity"]
    if es_multifidelidad:
        min_b = min_budget if min_budget is not None else 1
        max_b = max_budget if max_budget is not None else n_corridas_eval
    else:
        min_b = max_b = None

    scenario = _scenario_base(cs, n_trials, output_dir, seed, min_b, max_b)
    smac = _BUILDERS[tipo](scenario, fn_obj)

    t0 = time.time()
    log.info("Iniciando optimización...")
    incumbente_config = smac.optimize()
    t_total = time.time() - t0

    mejor_config = dict(incumbente_config)
    try:
        mejor_costo = float(smac.runhistory.get_cost(incumbente_config))
    except Exception:
        mejor_costo = float('inf')

    # Historia enriquecida con timestamps y mejor incumbente acumulado
    tracker_calls = _tracker.get("calls", [])
    historia = []
    for idx, (trial_key, trial_val) in enumerate(smac.runhistory.items()):
        try:
            entrada = {
                "config_id": trial_key.config_id,
                "seed":      trial_key.seed,
                "budget":    trial_key.budget,
                "costo":     round(float(trial_val.cost), 4),
            }
            if idx < len(tracker_calls):
                entrada["t_seg"] = tracker_calls[idx]["t_seg"]
            historia.append(entrada)
        except Exception:
            pass

    mejor_hasta = float("inf")
    for entrada in historia:
        mejor_hasta = min(mejor_hasta, entrada["costo"])
        entrada["mejor_hasta_ahora"] = round(mejor_hasta, 4)

    log.info("─"*65)
    log.info("Completado en %.0fs | Mejor %s = %.3f", t_total, objetivo, mejor_costo)
    log.info("Mejor configuración:")
    for k, v in mejor_config.items():
        log.info("  %-30s = %s", k, v)

    resultado = ResultadoSMAC(
        tipo             = tipo,
        nombre_tipo      = info["nombre"],
        objetivo         = objetivo,
        n_trials         = n_trials,
        n_corridas_eval  = n_corridas_eval,
        seed             = seed,
        incumbente       = {k: (int(v) if isinstance(v, (int, np.integer))
                               else round(float(v), 6)) for k, v in mejor_config.items()},
        costo_incumbente = round(mejor_costo, 4),
        n_evaluaciones   = len(historia),
        tiempo_seg       = round(t_total, 2),
        output_dir       = output_dir,
        historia_costos  = historia,
        convergido       = True,
        descripcion      = info["descripcion"],
    )

    out = asdict(resultado)
    with open(guardar_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log.info("Resultado guardado en '%s'", guardar_json)

    if not getattr(optimizar, "_no_plot", False):
        png_path = guardar_json.replace(".json", ".png")
        graficar(resultado, guardar_png=png_path)

    return resultado


# ──────────────────────────────────────────────────────────────────────────────
# Comparación de todos los tipos
# ──────────────────────────────────────────────────────────────────────────────
def comparar_todos(
    n_trials:        int   = 30,
    n_corridas_eval: int   = 2,
    seed:            int   = 42,
    seed_base_modelo:int   = 202,
    objetivo:        str   = "tts_full_days_mean",
    pesos_kpi:       dict  = None,
    tipos:           list  = None,
    guardar_json:    str   = "resultado_smac_comparacion.json",
    n_workers:       int   = 0,
) -> dict:
    """
    Corre todos (o un subconjunto) de los 6 tipos de SMAC y los compara.

    Parámetros
    ----------
    tipos : lista de tipos a correr (None = todos los 6)
    Resto : igual que optimizar()

    Retorna dict con resultados de cada tipo y tabla comparativa
    """
    tipos_a_correr = tipos or list(_BUILDERS.keys())
    log.info("Comparando %d tipos de SMAC: %s", len(tipos_a_correr), tipos_a_correr)

    resultados = {}
    for tipo in tipos_a_correr:
        log.info("\n%s\n[%s]\n%s", "="*65, tipo.upper(), "="*65)
        try:
            r = optimizar(
                tipo             = tipo,
                n_trials         = n_trials,
                n_corridas_eval  = n_corridas_eval,
                seed             = seed,
                seed_base_modelo = seed_base_modelo,
                objetivo         = objetivo,
                pesos_kpi        = pesos_kpi,
                output_dir       = f"smac_output_{tipo}",
                guardar_json     = f"resultado_smac_{tipo}.json",
                n_workers        = n_workers,
            )
            resultados[tipo] = asdict(r)
        except Exception as e:
            log.error("[%s] Error: %s", tipo, e)
            resultados[tipo] = {"error": str(e)}

    # Tabla comparativa
    print("\n" + "="*80)
    print("COMPARACIÓN DE TODOS LOS TIPOS SMAC")
    print(f"Objetivo: {objetivo}  |  n_trials: {n_trials}  |  n_corridas_eval: {n_corridas_eval}")
    print("="*80)
    print(f"  {'Tipo':<20} {'Nombre':<30} {'Mejor costo':>12} {'N eval':>8} {'Tiempo':>8}")
    print("  " + "─"*76)

    ordenados = sorted(
        [(t, r) for t, r in resultados.items() if "error" not in r],
        key=lambda x: x[1].get("costo_incumbente", 1e9)
    )
    for tipo, r in ordenados:
        print(f"  {tipo:<20} {r.get('nombre_tipo',''):<30} "
              f"{r.get('costo_incumbente',0):>12.2f} "
              f"{r.get('n_evaluaciones',0):>8d} "
              f"{r.get('tiempo_seg',0):>7.0f}s")

    for tipo, r in resultados.items():
        if "error" in r:
            print(f"  {tipo:<20} ERROR: {r['error']}")

    print("\n  Mejor incumbente global:")
    if ordenados:
        mejor_tipo, mejor_r = ordenados[0]
        print(f"  Tipo: {mejor_tipo} — {mejor_r.get('nombre_tipo','')}")
        print(f"  Costo: {mejor_r.get('costo_incumbente', 0):.2f}")
        print("  Configuración:")
        for k, v in mejor_r.get("incumbente", {}).items():
            baseline_val = _get_baseline_val(k)
            diff = f"  (baseline={baseline_val})" if baseline_val else ""
            print(f"    {k:<35} = {v}{diff}")

    comparacion = {
        "objetivo":         objetivo,
        "n_trials":         n_trials,
        "n_corridas_eval":  n_corridas_eval,
        "seed":             seed,
        "resultados":       resultados,
        "ranking":          [t for t, _ in ordenados],
    }
    with open(guardar_json, "w", encoding="utf-8") as f:
        json.dump(comparacion, f, ensure_ascii=False, indent=2)
    log.info("Comparación guardada en '%s'", guardar_json)

    if not getattr(optimizar, "_no_plot", False):
        graficar_comparacion(
            {t: r for t, r in resultados.items() if "error" not in r},
            guardar_png="convergencia_comparacion_todos.png",
        )

    return comparacion


def _get_baseline_val(param: str) -> str:
    """Retorna el valor baseline de un parámetro para mostrar en la comparación."""
    baseline = {
        "horas_especialista_1ra":  "16",
        "horas_control_post":      "40",
        "cupos_laboratorio_ugd":   "54",
        "cupos_ecografia_matrona": "25",
        "cupos_ecografia_ugd":     "25",
        "dias_publicacion":        "5",
        "pct_bloqueo_1ra":         "0.32",
        "pct_consultas_vacias":    "0.30",
        "num_matronas":            "1",
        "num_agentes_ugd":         "1",
        "pct_no_contactabilidad":  "0.30",
        "pct_bloqueo_post_control": "0.34",
    }
    return baseline.get(param, "")


# ──────────────────────────────────────────────────────────────────────────────
# Evaluar incumbente en el modelo completo (validación post-optimización)
# ──────────────────────────────────────────────────────────────────────────────
def evaluar_incumbente(
    incumbente:      dict,
    n_corridas:      int   = 10,
    seed_base:       int   = 300,
    objetivo:        str   = "tts_full_days_mean",
    guardar_json:    str   = "evaluacion_incumbente.json",
    n_workers:       int   = 0,
) -> dict:
    """
    Evalúa el mejor incumbente con más corridas — versión paralelizada.
    """
    _, CFG_base, _, _, _ = _importar_baseline()

    class _FakeConfig(dict):
        def __getitem__(self, k): return super().__getitem__(k)
        def __contains__(self, k): return super().__contains__(k)

    cfg = _aplicar_config_a_cfg(_FakeConfig(incumbente), CFG_base)
    cfg_dict = dataclasses.asdict(cfg)
    log.info("Evaluando incumbente con %d corridas...", n_corridas)

    workers = min(n_workers if n_workers > 0 else n_corridas,
                  n_corridas,
                  multiprocessing.cpu_count())

    tasks = [
        (seed_base + r, cfg_dict, objetivo, {})
        for r in range(n_corridas)
    ]

    if workers <= 1:
        valores = [_worker_run_once(t) for t in tasks]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
            valores = list(ex.map(_worker_run_once, tasks))

    for r, v in enumerate(valores):
        log.info("  Corrida %d/%d: %s=%.2f", r+1, n_corridas, objetivo, v)

    mu  = float(np.mean(valores))
    sd  = float(np.std(valores, ddof=1)) if len(valores) > 1 else 0.0
    try:
        from scipy.stats import t
        tc = float(t.ppf(0.975, len(valores)-1))
    except Exception:
        tc = 1.96
    se  = sd / np.sqrt(len(valores))

    resultado = {
        "incumbente":        incumbente,
        "objetivo":          objetivo,
        "n_corridas":        len(valores),
        "valores":           [round(v, 4) for v in valores],
        "media":             round(mu, 4),
        "sd":                round(sd, 4),
        "ic_95_bajo":        round(mu - tc*se, 4),
        "ic_95_alto":        round(mu + tc*se, 4),
    }

    with open(guardar_json, "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)

    log.info("─"*50)
    log.info("Resultado evaluación incumbente:")
    log.info("  %s = %.2f ± %.2f  IC95=[%.2f, %.2f]",
             objetivo, mu, sd, mu-tc*se, mu+tc*se)
    log.info("Guardado en '%s'", guardar_json)

    return resultado


# ──────────────────────────────────────────────────────────────────────────────
# Mostrar espacio de búsqueda
# ──────────────────────────────────────────────────────────────────────────────
def mostrar_espacio():
    cs = crear_espacio_configuracion()

    print("\n" + "="*80)
    print("ESPACIO DE CONFIGURACIÓN — 11 PARÁMETROS DEL MODELO CLÍNICO")
    print("="*80)
    print(f"\n  {'Parámetro':<35} {'Tipo':<8} {'Min':>6} {'Max':>6} {'Default':>8}  Descripción")
    print("  " + "─"*100)
    descs = {
        "horas_especialista_1ra":   "Slots/sem primera consulta (baseline=16)",
        "horas_control_post":       "Pac/sem control post especialista (baseline=40)",
        "cupos_laboratorio_ugd":    "Cupos lab UGD por semana (baseline=54)",
        "cupos_ecografia_matrona":  "Cupos eco matrona por semana (baseline=25)",
        "cupos_ecografia_ugd":      "Cupos eco UGD por semana (baseline=25)",
        "dias_publicacion":         "Días hábiles anticipación agenda (baseline=5)",
        "pct_bloqueo_1ra":          "Fracción slots bloqueados 1ra consulta (baseline=0.32)",
        "pct_consultas_vacias":     "Fracción consultas vacías UGD control (baseline=0.30)",
        "num_matronas":             "Capacidad matrona — N profesionales (baseline=1)",
        "num_agentes_ugd":          "Capacidad agente UGD — N agentes (baseline=1)",
        "pct_no_contactabilidad":   "Fracción no contactabilidad (baseline=0.30)",
    }
    for hp in cs.get_hyperparameters():
        tipo = "Int" if isinstance(getattr(hp, 'lower', 0), int) else "Float"
        low  = getattr(hp, 'lower',        '?')
        high = getattr(hp, 'upper',        '?')
        defv = getattr(hp, 'default_value','?')
        desc = descs.get(hp.name, "")
        print(f"  {hp.name:<33} {tipo:<8} {str(low):>6} {str(high):>6} {str(defv):>8}  {desc}")

    print(f"\n  Total: {len(cs.get_hyperparameters())} hiperparámetros")

    print("\n" + "="*80)
    print("OBJETIVO DE OPTIMIZACIÓN — tts_full_days_mean (MINIMIZAR)")
    print("="*80)
    print(f"""
  Definición:
    env.now - patient.enqueued_at  para pacientes que COMPLETAN toda la ruta post.

  Descomposición del valor BASELINE (279.78 días):
    Backlog histórico pre-t=0    :  54.33 días  (19%)
      → Tiempo que los pacientes ya llevaban esperando antes de t=0
    Proceso primera consulta     : 139.53 días  (50%)
      → Desde ingreso hasta atención primera consulta (dentro de la simulación)
    Post-consulta                : 126.98 días  (45%)
      → Desde primera consulta hasta alta/cirugía
    ──────────────────────────────────────────────
    Nota: la suma supera 279.78d porque son poblaciones distintas.
    tts_full solo cuenta pacientes que COMPLETARON post dentro del horizonte.

  SMAC busca configuraciones que reduzcan los 279.78 días baseline.
  Un mejor resultado → menos días promedio de espera total por paciente.
""")

    print("="*80)
    print("TIPOS DE SMAC DISPONIBLES")
    print("="*80)
    print(f"\n  {'Tipo':<20} {'Nombre':<32} {'Modelo':<22} {'Multi-fid':<10} {'Prob.aleat'}")
    print("  " + "─"*90)
    for tipo, info in TIPOS_SMAC.items():
        print(f"  {tipo:<20} {info['nombre']:<32} {info['modelo']:<22} "
              f"{'Sí' if info['multi_fidelity'] else 'No':<10} "
              f"{info['prob_aleatorio']*100:.0f}%")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
# ── Guard obligatorio para multiprocessing en Windows y macOS ─────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Módulo 4: Optimización SMAC3 — 6 tipos — conectado con modelo baseline"
    )
    parser.add_argument(
        "--tipo", type=str, default="hpo",
        choices=list(_BUILDERS.keys()) + ["todos"],
        help="Tipo de SMAC a usar (default: hpo). 'todos' corre los 6 y compara."
    )
    parser.add_argument(
        "--n_trials", type=int, default=50,
        help="Evaluaciones totales del modelo (default: 50)"
    )
    parser.add_argument(
        "--n_corridas", type=int, default=3,
        help="Corridas del modelo por evaluación (default: 3)"
    )
    parser.add_argument(
        "--objetivo", type=str, default="tts_full_days_mean",
        choices=["wl_total_end", "wl_first_end",
                 "tts_first_attended_days_mean", "tts_full_days_mean"],
        help="KPI a minimizar (default: tts_full_days_mean)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Semilla SMAC (default: 42)"
    )
    parser.add_argument(
        "--seed_modelo", type=int, default=202,
        help="Semilla base modelo de simulación (default: 202)"
    )
    parser.add_argument(
        "--min_budget", type=float, default=None,
        help="Presupuesto mínimo multi-fidelidad (default: 1)"
    )
    parser.add_argument(
        "--max_budget", type=float, default=None,
        help="Presupuesto máximo multi-fidelidad (default: n_corridas)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directorio salida SMAC"
    )
    parser.add_argument(
        "--guardar_json", type=str, default=None,
        help="Archivo JSON resultado"
    )
    parser.add_argument(
        "--n_workers", type=int, default=0,
        help="Procesos paralelos por evaluación (0=auto, default: 0)"
    )
    parser.add_argument(
        "--no_plot", action="store_true",
        help="No generar PNG de convergencia/tiempo"
    )
    parser.add_argument(
        "--mostrar_espacio", action="store_true",
        help="Muestra el espacio de búsqueda y sale"
    )
    parser.add_argument(
        "--listar", action="store_true",
        help="Lista los tipos disponibles y sale"
    )
    parser.add_argument(
        "--evaluar_incumbente", type=str, default=None,
        help="Ruta a JSON de resultado SMAC para evaluar el incumbente con más corridas"
    )
    args = parser.parse_args()

    optimizar._no_plot = args.no_plot

    if args.listar:
        print("\nTipos de SMAC disponibles:")
        for tipo, info in TIPOS_SMAC.items():
            print(f"\n  [{tipo}] {info['nombre']}")
            print(f"    Modelo        : {info['modelo']}")
            print(f"    Adquisición   : {info['adquisicion']}")
            print(f"    Diseño inicial: {info['disenio_inicial']}")
            print(f"    Intensificador: {info['intensificador']}")
            print(f"    Prob. aleat.  : {info['prob_aleatorio']*100:.0f}%")
            print(f"    Multi-fid.    : {'Sí' if info['multi_fidelity'] else 'No'}")
            print(f"    Descripción   : {info['descripcion']}")
        sys.exit(0)

    if args.mostrar_espacio:
        mostrar_espacio()
        sys.exit(0)

    if args.evaluar_incumbente:
        with open(args.evaluar_incumbente) as f:
            data = json.load(f)
        evaluar_incumbente(
            incumbente   = data.get("incumbente", {}),
            n_corridas   = max(10, args.n_corridas * 3),
            objetivo     = data.get("objetivo", args.objetivo),
            guardar_json = "evaluacion_incumbente.json",
            n_workers    = args.n_workers,
        )
        sys.exit(0)

    if args.tipo == "todos":
        comparar_todos(
            n_trials         = args.n_trials,
            n_corridas_eval  = args.n_corridas,
            seed             = args.seed,
            seed_base_modelo = args.seed_modelo,
            objetivo         = args.objetivo,
            guardar_json     = args.guardar_json or "resultado_smac_comparacion.json",
            n_workers        = args.n_workers,
        )
    else:
        optimizar(
            tipo             = args.tipo,
            n_trials         = args.n_trials,
            n_corridas_eval  = args.n_corridas,
            seed             = args.seed,
            seed_base_modelo = args.seed_modelo,
            objetivo         = args.objetivo,
            output_dir       = args.output_dir,
            guardar_json     = args.guardar_json,
            min_budget       = args.min_budget,
            max_budget       = args.max_budget,
            n_workers        = args.n_workers,
        )
