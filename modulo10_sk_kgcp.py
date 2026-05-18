"""
===============================================================================
MÓDULO 10 — SK con KGCP (Knowledge Gradient with Correlated Priors)
===============================================================================
Extiende el Módulo 9 (SK + REVI) reemplazando la función de adquisición EI
por KGCP, que planifica un paso adelante y propaga información globalmente.

Diferencia fundamental vs EI (módulos 7–9):
  EI (greedy):  EI(x) = E[max(μ* - f(x), 0)]
                Solo mira la mejora directa en x propuesto.

  KGCP (lookahead): KG(x) = E_Z[max_{x'} μ_n+1(x'|obs en x)] - max μ_n
                Mide cuánto mejoraría el INCUMBENTE GLOBAL si evaluamos x
                y actualizamos el modelo. La evaluación en x informa μ(x')
                para TODOS los x' mediante el kernel correlacionado del SK.

Formulación (Frazier, Powell & Dayanik 2009, discretización MC):
  Para cada candidato x:
    σ_kg(x, x') = K_n(x, x') / sqrt(K_n(x,x) + σ²_sim(x)/n_reps)
    KG(x) ≈ mean_Z[ max_{x'∈X_cand} (μ_n(x') + σ_kg(x,x')·Z) ] - max μ_n

  donde K_n es el kernel posterior del SK (ya heteroscedástico con diag(V)).
  Z ~ N(0,1) se muestrea con n_kgcp_mc muestras Monte Carlo.
  X_cand es un conjunto de n_kgcp_cand candidatos aleatorios + puntos ya evaluados.

Por qué KGCP supera a EI en simuladores ruidosos:
  · EI sobreexplota zonas con alta varianza del modelo (falsos positivos).
  · KGCP descuenta la incertidumbre: si evaluar x no va a informar el
    incumbente global (por baja correlación), KG(x) ≈ 0 → no se evalúa.
  · Es óptimo en horizonte de 1 paso bajo el prior GP, incluso con ruido.

Indicador fijo : tts_full_days_mean
Replicación    : compatible con fijo, adaptativo (M8) y REVI (M9)
Tipos nuevos   : 'hpo_kgcp', 'blackbox_kgcp'
Gráficos       : convergencia + tiempo + réplicas + σ² (igual que M9)
                 + panel extra de KG vs EI por evaluación

Uso:
  python modulo10_sk_kgcp.py --tipo hpo_kgcp --n_trials 100
  python modulo10_sk_kgcp.py --tipo hpo_kgcp --n_trials 100 --n_workers 4
  python modulo10_sk_kgcp.py --tipo hpo_kgcp --n_trials 100 \\
      --n_kgcp_mc 64 --n_kgcp_cand 500
  python modulo10_sk_kgcp.py --tipo hpo_kgcp --n_trials 100 --revi
  python modulo10_sk_kgcp.py --evaluar_incumbente resultado_m10.json
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
from scipy.optimize import minimize

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
log = logging.getLogger("smac_sk")


# ──────────────────────────────────────────────────────────────
# Importar simulador (igual que módulo 4)
# ──────────────────────────────────────────────────────────────
_baseline_cache = None

def _importar_baseline():
    global _baseline_cache
    if _baseline_cache is not None:
        return _baseline_cache
    for d in [os.path.dirname(os.path.abspath(__file__)), os.getcwd()]:
        if d not in sys.path:
            sys.path.insert(0, d)
    try:
        from simulador_clinica_baseline import ClinicModelAdjusted, CFG, run_once
        import simpy, random as _random
        _baseline_cache = (ClinicModelAdjusted, CFG, run_once, simpy, _random)
        log.info("✓ Simulador importado: simulador_clinica_baseline.py")
        return _baseline_cache
    except ImportError as e:
        log.error("No se pudo importar simulador_clinica_baseline.py: %s", e)
        raise


# ──────────────────────────────────────────────────────────────
# Aplicar config a SimConfig — IDÉNTICO al módulo 4
# ──────────────────────────────────────────────────────────────
def _aplicar_config_a_cfg(config, cfg_base):
    """
    Aplica parámetros de ConfigSpace.Configuration al SimConfig.
    Rangos y nombres idénticos al módulo 4.
    """
    cfg = copy.deepcopy(cfg_base)

    # 1. Slots/sem primera consulta
    if "horas_especialista_1ra" in config:
        cfg.fixed_weekly_capacity     = int(config["horas_especialista_1ra"])
        cfg.use_fixed_weekly_capacity = True

    # 2. Pac/sem control post
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

    # 6. Días publicación anticipada
    if "dias_publicacion" in config:
        cfg.publish_lead_workdays = int(config["dias_publicacion"])

    # 7. % Bloqueo 1ra consulta
    if "pct_bloqueo_1ra" in config:
        cfg.blocked_pct = float(config["pct_bloqueo_1ra"])

    # 8. % Consultas vacías UGD control
    if "pct_consultas_vacias" in config:
        cfg.empty_control_p_ugd = float(config["pct_consultas_vacias"])

    # 9. Número matronas
    if "num_matronas" in config:
        cfg.matrona_capacity = int(config["num_matronas"])

    # 10. Número agentes UGD
    if "num_agentes_ugd" in config:
        cfg.agent_capacity = int(config["num_agentes_ugd"])

    # 11. % No contactabilidad
    if "pct_no_contactabilidad" in config:
        cfg.not_contactable_p = float(config["pct_no_contactabilidad"])

    # 12. % Bloqueo post-control
    if "pct_bloqueo_post_control" in config:
        cfg.blocked_pct_post_control = float(config["pct_bloqueo_post_control"])

    return cfg


# ──────────────────────────────────────────────────────────────
# Espacio de configuración — IDÉNTICO al módulo 4
# ──────────────────────────────────────────────────────────────
def crear_espacio_configuracion(seed: int = 42):
    """
    ConfigSpace con 12 parámetros y rangos exactos del módulo 4
    y SimConfig del simulador.
    """
    try:
        from ConfigSpace import ConfigurationSpace
        from ConfigSpace.hyperparameters import (
            UniformIntegerHyperparameter as IntHP,
            UniformFloatHyperparameter   as FloatHP,
        )
    except ImportError:
        raise ImportError("Instala con: pip install smac")

    cs = ConfigurationSpace(seed=seed)
    cs.add([
        # 1. Slots/sem primera consulta (SimConfig: fixed_weekly_capacity, baseline=16)
        IntHP("horas_especialista_1ra",     lower=8,    upper=30,  default_value=16),
        # 2. Pac/sem control post (SimConfig: fixed_post_control_capacity, baseline=40)
        IntHP("horas_control_post",         lower=20,   upper=70,  default_value=40),
        # 3. Cupos lab UGD/sem (SimConfig: ugd_lab_per_week, baseline=54)
        IntHP("cupos_laboratorio_ugd",      lower=20,   upper=100, default_value=54),
        # 4. Cupos eco matrona/sem (SimConfig: mat_us_per_week, baseline=25)
        IntHP("cupos_ecografia_matrona",    lower=10,   upper=50,  default_value=25),
        # 5. Cupos eco UGD/sem (SimConfig: ugd_us_per_week, baseline=25)
        IntHP("cupos_ecografia_ugd",        lower=10,   upper=50,  default_value=25),
        # 6. Días publicación anticipada (SimConfig: publish_lead_workdays, baseline=5)
        IntHP("dias_publicacion",           lower=1,    upper=10,  default_value=5),
        # 7. % Bloqueo 1ra (SimConfig: blocked_pct, baseline=0.32)
        FloatHP("pct_bloqueo_1ra",          lower=0.05, upper=0.50, default_value=0.32),
        # 8. % Consultas vacías UGD (SimConfig: empty_control_p_ugd, baseline=0.30)
        FloatHP("pct_consultas_vacias",     lower=0.05, upper=0.50, default_value=0.30),
        # 9. Número matronas (SimConfig: matrona_capacity, baseline=1)
        IntHP("num_matronas",               lower=1,    upper=4,   default_value=1),
        # 10. Número agentes UGD (SimConfig: agent_capacity, baseline=1)
        IntHP("num_agentes_ugd",            lower=1,    upper=4,   default_value=1),
        # 11. % No contactabilidad (SimConfig: not_contactable_p, baseline=0.15)
        FloatHP("pct_no_contactabilidad",   lower=0.05, upper=0.50, default_value=0.15),
        # 12. % Bloqueo post-control (SimConfig: blocked_pct_post_control, baseline=0.34)
        FloatHP("pct_bloqueo_post_control", lower=0.05, upper=0.50, default_value=0.34),
    ])
    return cs


# Dimensión del espacio
_DIM = 12
_PARAM_NAMES = [
    "horas_especialista_1ra", "horas_control_post", "cupos_laboratorio_ugd",
    "cupos_ecografia_matrona", "cupos_ecografia_ugd", "dias_publicacion",
    "pct_bloqueo_1ra", "pct_consultas_vacias", "num_matronas",
    "num_agentes_ugd", "pct_no_contactabilidad", "pct_bloqueo_post_control",
]
_BASELINES = {
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


# ──────────────────────────────────────────────────────────────
# Almacén global de varianzas — clave para SK heteroscedástico
# ──────────────────────────────────────────────────────────────
_variance_store: dict = {}   # config_key → varianza_muestral
_n_reps_store:   dict = {}   # config_key → n_réplicas usadas
_N_REPS_GLOBAL:  int  = 3    # se actualiza en evaluar_configuracion_sk


def _config_key(config) -> str:
    """Clave única reproducible para una configuración."""
    return str(sorted({k: round(float(v), 6) for k, v in dict(config).items()}.items()))


# ──────────────────────────────────────────────────────────────
# Política de replicación adaptativa — NUEVO en módulo 8
# ──────────────────────────────────────────────────────────────
class AdaptiveReplicationPolicy:
    """
    Decide cuántas réplicas asignar a cada config según su varianza estimada.

    Durante el warmup (primeras n_warmup evaluaciones) usa n_min réplicas
    para poblar _variance_store con datos reales antes de adaptar.

    Después del warmup, n(x) se calcula como:
      percentil = rank de σ²(x) en la distribución de varianzas conocidas
      n(x) = n_min + round( percentil × (n_max - n_min) )

    La varianza σ²(x) se estima como:
      - Si x ya fue evaluada: valor exacto de _variance_store
      - Si x es nueva: mediana de los k vecinos más cercanos en espacio
        normalizado (k=3), o mediana global si hay < 3 vecinos.
    """

    def __init__(
        self,
        n_min:    int   = 2,
        n_max:    int   = 8,
        n_warmup: int   = 10,
        k_neighbors: int = 3,
    ):
        self.n_min       = int(n_min)
        self.n_max       = int(n_max)
        self.n_warmup    = int(n_warmup)
        self.k_neighbors = int(k_neighbors)
        self._eval_count = 0           # evaluaciones realizadas hasta ahora
        self._X_norm     = []          # vectores normalizados de configs evaluadas
        self._keys       = []          # claves correspondientes (mismo orden)
        self._bounds_lo  = None        # límites para normalización (calculados 1 vez)
        self._bounds_hi  = None

    # ── Normalización del espacio de parámetros ────────────────
    def _normalize(self, config_dict: dict) -> np.ndarray:
        """Convierte un config dict a vector [0,1]^d usando los rangos del espacio."""
        # Rangos hardcoded del espacio de configuración (igual que crear_espacio_configuracion)
        _RANGES = {
            "horas_especialista_1ra":   (8,    30),
            "horas_control_post":       (20,   70),
            "cupos_laboratorio_ugd":    (20,  100),
            "cupos_ecografia_matrona":  (10,   50),
            "cupos_ecografia_ugd":      (10,   50),
            "dias_publicacion":         (1,    10),
            "pct_bloqueo_1ra":          (0.05, 0.50),
            "pct_consultas_vacias":     (0.05, 0.50),
            "num_matronas":             (1,     4),
            "num_agentes_ugd":          (1,     4),
            "pct_no_contactabilidad":   (0.05, 0.50),
            "pct_bloqueo_post_control": (0.05, 0.50),
        }
        keys = sorted(_RANGES.keys())
        vec  = []
        for k in keys:
            lo, hi = _RANGES[k]
            v = float(config_dict.get(k, (lo + hi) / 2))
            vec.append((v - lo) / max(hi - lo, 1e-8))
        return np.array(vec, dtype=float)

    # ── Estimación de σ²(x) para config nueva ─────────────────
    def _estimate_variance(self, config_dict: dict) -> float:
        """
        Estima la varianza simulación de x usando los k vecinos más cercanos
        en espacio normalizado. Si no hay varianzas conocidas, retorna mediana
        global de _variance_store o 1000 (fallback).
        """
        known_vars = list(_variance_store.values())
        if not known_vars:
            return 1000.0

        # Verificar si esta config ya fue evaluada
        key = _config_key(type('_C', (), {'__iter__': lambda s: iter(config_dict.items()),
                                           'items': lambda s: config_dict.items()})())
        if key in _variance_store:
            return _variance_store[key]

        if not self._X_norm:
            return float(np.median(known_vars))

        x = self._normalize(config_dict)
        X = np.array(self._X_norm)
        dists = np.linalg.norm(X - x, axis=1)
        k = min(self.k_neighbors, len(self._keys))
        idx = np.argsort(dists)[:k]
        neighbor_vars = [_variance_store.get(self._keys[i], np.median(known_vars))
                         for i in idx]
        return float(np.mean(neighbor_vars))

    # ── Interfaz principal ─────────────────────────────────────
    def decide_n_reps(self, config) -> int:
        """
        Retorna el número de réplicas para esta config.
        Actualiza el contador interno (se llama una vez por evaluación).
        """
        self._eval_count += 1

        # Fase warmup: n_min fijo
        if self._eval_count <= self.n_warmup:
            log.debug("Warmup %d/%d → n_reps=%d",
                      self._eval_count, self.n_warmup, self.n_min)
            return self.n_min

        # Fase adaptativa
        config_dict = dict(config)
        sigma2 = self._estimate_variance(config_dict)

        known_vars = list(_variance_store.values())
        if len(known_vars) < 2:
            return self.n_min

        # Percentil de σ²(x) en la distribución conocida
        pct = float(np.mean(np.array(known_vars) <= sigma2))  # ∈ [0, 1]

        n = int(self.n_min + round(pct * (self.n_max - self.n_min)))
        n = max(self.n_min, min(self.n_max, n))

        log.debug("Adaptativo eval=%d σ²=%.1f pct=%.2f → n_reps=%d",
                  self._eval_count, sigma2, pct, n)
        return n

    def register(self, config, var: float):
        """
        Registra la varianza observada de una config evaluada.
        Debe llamarse DESPUÉS de cada evaluación para mantener el índice.
        """
        config_dict = dict(config)
        x = self._normalize(config_dict)
        key = _config_key(config)
        self._X_norm.append(x.tolist())
        self._keys.append(key)


# ──────────────────────────────────────────────────────────────
# Función objetivo adaptativa — NUEVO en módulo 8
# ──────────────────────────────────────────────────────────────
def evaluar_configuracion_sk_adaptive(
    config,
    policy:     "AdaptiveReplicationPolicy",
    seed:       int   = 0,
    seed_base:  int   = 202,
    objetivo:   str   = "tts_full_days_mean",
    n_workers:  int   = 0,
) -> float:
    """
    Versión adaptativa de evaluar_configuracion_sk.

    El número de réplicas n(x) lo decide AdaptiveReplicationPolicy
    en función de la varianza σ²(x) estimada para esta config.

    Tras la evaluación actualiza _variance_store y registra en la política.
    """
    global _N_REPS_GLOBAL
    _, CFG_base, _, _, _ = _importar_baseline()
    cfg = _aplicar_config_a_cfg(config, CFG_base)
    cfg_dict = dataclasses.asdict(cfg)

    # Número de réplicas decidido por la política
    n = policy.decide_n_reps(config)
    _N_REPS_GLOBAL = n

    workers = min(n_workers if n_workers > 0 else n,
                  n,
                  multiprocessing.cpu_count())

    tasks = [
        (seed_base + seed + r, cfg_dict, objetivo, {})
        for r in range(n)
    ]

    if workers <= 1 or n == 1:
        valores = [_worker_run_once(t) for t in tasks]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
            valores = list(ex.map(_worker_run_once, tasks))

    media = float(np.mean(valores))
    var   = float(np.var(valores, ddof=1)) if len(valores) > 1 else 1000.0

    # Guardar varianza y registrar en política
    key = _config_key(config)
    _variance_store[key] = var
    _n_reps_store[key]   = len(valores)
    policy.register(config, var)

    log.info("  Adapt eval → %s=%.2f  σ=%.2f  n_reps=%d (eval#%d)",
             objetivo, media, np.sqrt(var), n, policy._eval_count)
    return media


# ──────────────────────────────────────────────────────────────
# Política REVI — NUEVO en módulo 9
# ──────────────────────────────────────────────────────────────
class REVIReplicationPolicy:
    """
    Replicated Expected Value of Information (REVI).

    Asigna réplicas balanceando la varianza del simulador σ²_sim(x)
    contra la incertidumbre del modelo SK σ²_mod(x):

        n*(x) = clip( sqrt( σ²_sim(x) / σ²_mod(x) ), n_min, n_max )

    Derivación:
      La varianza total del estimador de μ(x) con n réplicas es:
        Var_total(x,n) = σ²_mod(x) + σ²_sim(x)/n

      El costo marginal de añadir una réplica es proporcional a 1/n².
      El n óptimo que iguala el costo marginal con la reducción de
      incertidumbre da exactamente sqrt(σ²_sim/σ²_mod).

    Warmup: primeras n_warmup evaluaciones usan n_min para poblar
    _variance_store con σ²_sim reales antes de activar REVI.

    Durante el warmup, σ²_mod se estima como la varianza del modelo
    usando los vecinos más cercanos en espacio normalizado.
    """

    # Rangos del espacio — compartidos con AdaptiveReplicationPolicy
    _RANGES = {
        "horas_especialista_1ra":   (8,    30),
        "horas_control_post":       (20,   70),
        "cupos_laboratorio_ugd":    (20,  100),
        "cupos_ecografia_matrona":  (10,   50),
        "cupos_ecografia_ugd":      (10,   50),
        "dias_publicacion":         (1,    10),
        "pct_bloqueo_1ra":          (0.05, 0.50),
        "pct_consultas_vacias":     (0.05, 0.50),
        "num_matronas":             (1,     4),
        "num_agentes_ugd":          (1,     4),
        "pct_no_contactabilidad":   (0.05, 0.50),
        "pct_bloqueo_post_control": (0.05, 0.50),
    }

    def __init__(
        self,
        n_min:       int   = 2,
        n_max:       int   = 8,
        n_warmup:    int   = 10,
        k_neighbors: int   = 3,
        sigma_mod_floor: float = 1e-3,   # evita div/0 cuando modelo es muy certero
    ):
        self.n_min           = int(n_min)
        self.n_max           = int(n_max)
        self.n_warmup        = int(n_warmup)
        self.k_neighbors     = int(k_neighbors)
        self.sigma_mod_floor = float(sigma_mod_floor)
        self._eval_count     = 0
        self._X_norm         = []
        self._keys           = []
        self._sk_model       = None    # referencia al StochasticKrigingModel
        self._sk_scaler      = None    # normalizador X de SMAC (si disponible)
        # Historial para gráficos
        self.sigma2_sim_hist = []
        self.sigma2_mod_hist = []

    def set_sk_model(self, model):
        """Inyecta referencia al SK entrenado. Llamar en cada iteración de fn_obj."""
        self._sk_model = model

    def _normalize(self, config_dict: dict) -> np.ndarray:
        keys = sorted(self._RANGES.keys())
        vec  = []
        for k in keys:
            lo, hi = self._RANGES[k]
            v = float(config_dict.get(k, (lo + hi) / 2))
            vec.append((v - lo) / max(hi - lo, 1e-8))
        return np.array(vec, dtype=float)

    def _sigma2_sim(self, config_dict: dict) -> float:
        """
        σ²_sim(x): varianza del simulador en x.
        Exacta si x ya fue evaluada; vecinos si no.
        """
        known_vars = list(_variance_store.values())
        if not known_vars:
            return 1000.0

        key = str(sorted({k: round(float(v), 6)
                          for k, v in config_dict.items()}.items()))
        if key in _variance_store:
            return float(_variance_store[key])

        if not self._X_norm:
            return float(np.median(known_vars))

        x = self._normalize(config_dict)
        X = np.array(self._X_norm)
        dists = np.linalg.norm(X - x, axis=1)
        k = min(self.k_neighbors, len(self._keys))
        idx = np.argsort(dists)[:k]
        neighbor_vars = [_variance_store.get(self._keys[i], np.median(known_vars))
                         for i in idx]
        return float(np.mean(neighbor_vars))

    def _sigma2_mod(self, config_dict: dict) -> float:
        """
        σ²_mod(x): incertidumbre del modelo SK en x.
        Usa SK._predict() si el modelo está disponible y entrenado.
        Fallback: mediana de varianzas conocidas del simulador.
        """
        model = self._sk_model
        if model is None or not getattr(model, "_fitted", False):
            # Sin modelo: usar varianza global como proxy conservador
            known_vars = list(_variance_store.values())
            return float(np.median(known_vars)) if known_vars else 1000.0

        try:
            x_norm = self._normalize(config_dict).reshape(1, -1)
            # SK._predict devuelve (mu, var_diag) en escala original
            _, var_out = model._predict(x_norm, covariance_type="diagonal")
            sigma2_mod = float(np.maximum(var_out.ravel()[0], self.sigma_mod_floor))
            return sigma2_mod
        except Exception as e:
            log.debug("REVI σ²_mod fallback: %s", e)
            known_vars = list(_variance_store.values())
            return float(np.median(known_vars)) if known_vars else 1000.0

    def decide_n_reps(self, config) -> int:
        """
        Decide n*(x) según la fórmula REVI.
        Fase warmup: retorna n_min fijo.
        Fase REVI:   n*(x) = clip(sqrt(σ²_sim / σ²_mod), n_min, n_max)
        """
        self._eval_count += 1
        config_dict = dict(config)

        if self._eval_count <= self.n_warmup:
            # Warmup: guardar σ² de referencia para el gráfico
            known = list(_variance_store.values())
            s2_sim = float(np.median(known)) if known else 1000.0
            s2_mod = self._sigma2_mod(config_dict)
            self.sigma2_sim_hist.append(round(s2_sim, 4))
            self.sigma2_mod_hist.append(round(s2_mod, 4))
            log.debug("REVI warmup %d/%d → n_reps=%d",
                      self._eval_count, self.n_warmup, self.n_min)
            return self.n_min

        # Fase REVI
        s2_sim = self._sigma2_sim(config_dict)
        s2_mod = max(self._sigma2_mod(config_dict), self.sigma_mod_floor)

        ratio = s2_sim / s2_mod
        n_opt = int(round(np.sqrt(max(ratio, 0.0))))
        n     = max(self.n_min, min(self.n_max, n_opt))

        self.sigma2_sim_hist.append(round(s2_sim, 4))
        self.sigma2_mod_hist.append(round(s2_mod, 4))

        log.debug("REVI eval=%d σ²_sim=%.1f σ²_mod=%.4f ratio=%.2f → n*=%d → n=%d",
                  self._eval_count, s2_sim, s2_mod, ratio, n_opt, n)
        return n

    def register(self, config, var: float):
        """Registra config evaluada para búsqueda de vecinos."""
        x   = self._normalize(dict(config))
        key = str(sorted({k: round(float(v), 6)
                          for k, v in dict(config).items()}.items()))
        self._X_norm.append(x.tolist())
        self._keys.append(key)


# ──────────────────────────────────────────────────────────────
# Función objetivo REVI — NUEVO en módulo 9
# ──────────────────────────────────────────────────────────────
def evaluar_configuracion_sk_revi(
    config,
    policy:    "REVIReplicationPolicy",
    seed:      int = 0,
    seed_base: int = 202,
    objetivo:  str = "tts_full_days_mean",
    n_workers: int = 0,
) -> float:
    """
    Versión REVI de evaluar_configuracion_sk.
    n*(x) lo decide REVIReplicationPolicy usando σ²_sim y σ²_mod.
    """
    global _N_REPS_GLOBAL
    _, CFG_base, _, _, _ = _importar_baseline()
    cfg      = _aplicar_config_a_cfg(config, CFG_base)
    cfg_dict = dataclasses.asdict(cfg)

    n = policy.decide_n_reps(config)
    _N_REPS_GLOBAL = n

    workers = min(n_workers if n_workers > 0 else n,
                  n, multiprocessing.cpu_count())

    tasks = [(seed_base + seed + r, cfg_dict, objetivo, {}) for r in range(n)]

    if workers <= 1 or n == 1:
        valores = [_worker_run_once(t) for t in tasks]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
            valores = list(ex.map(_worker_run_once, tasks))

    media = float(np.mean(valores))
    var   = float(np.var(valores, ddof=1)) if len(valores) > 1 else 1000.0

    key = _config_key(config)
    _variance_store[key] = var
    _n_reps_store[key]   = len(valores)
    policy.register(config, var)

    log.info("  REVI eval → %s=%.2f  σ_sim=%.2f  n*=%d (eval#%d)",
             objetivo, media, np.sqrt(var), n, policy._eval_count)
    return media
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
        return float(res.get(objetivo, 1e9))
    except Exception as e:
        import logging
        logging.getLogger("smac_sk").warning("Worker error seed=%d: %s", seed_offset, e)
        return 1e9


# ──────────────────────────────────────────────────────────────
# Función objetivo — guarda varianza para SK
# ──────────────────────────────────────────────────────────────
def evaluar_configuracion_sk(
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
    Función objetivo para SMAC+SK — versión paralelizada.

    Las n_corridas de run_once se ejecutan en paralelo via ProcessPoolExecutor.
    La varianza σ²(x) se calcula en el proceso principal tras recibir todos
    los valores, por lo que _variance_store se escribe sin race conditions.

    n_workers: número de procesos paralelos.
               0 = automático (min(n, nCPUs disponibles)).
    """
    global _N_REPS_GLOBAL
    _, CFG_base, _, _, _ = _importar_baseline()
    cfg = _aplicar_config_a_cfg(config, CFG_base)
    cfg_dict = dataclasses.asdict(cfg)

    n = max(1, int(round(budget))) if budget is not None else n_corridas
    _N_REPS_GLOBAL = n

    workers = min(n_workers if n_workers > 0 else n,
                  n,
                  multiprocessing.cpu_count())

    tasks = [
        (seed_base + seed + r, cfg_dict, objetivo, pesos_kpi or {})
        for r in range(n)
    ]

    if workers <= 1 or n == 1:
        # Ruta secuencial — sin overhead de fork para n pequeño
        valores = []
        for t in tasks:
            valores.append(_worker_run_once(t))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
            valores = list(ex.map(_worker_run_once, tasks))

    media = float(np.mean(valores))
    var   = float(np.var(valores, ddof=1)) if len(valores) > 1 else 1000.0

    # Guardar varianza — usada por StochasticKrigingModel._train()
    key = _config_key(config)
    _variance_store[key] = var
    _n_reps_store[key]   = len(valores)

    log.info("  Config eval → %s=%.2f  σ=%.2f  (n=%d, workers=%d)",
             objetivo, media, np.sqrt(var), n, workers if n > 1 else 1)
    return media


# ──────────────────────────────────────────────────────────────
# Stochastic Kriging como AbstractGaussianProcess de SMAC
# ──────────────────────────────────────────────────────────────
try:
    from smac.model.gaussian_process.abstract_gaussian_process import AbstractGaussianProcess
    from smac.model.gaussian_process.kernels import (
        MaternKernel, ConstantKernel, SumKernel, WhiteKernel
    )
    _SMAC_GP_OK = True
except ImportError:
    _SMAC_GP_OK = False


class StochasticKrigingModel(AbstractGaussianProcess if _SMAC_GP_OK else object):
    """
    Stochastic Kriging como surrogate de SMAC.

    Diferencia clave vs GP/RF de SMAC:
      GP SMAC:  K_y = K(X,X) + σ²·I      (homoscedástico)
      SK:       K_y = K(X,X) + diag(V)   (heteroscedástico)
      donde V_i = σ²_muestral(xᵢ) / n_reps_i

    SMAC llama _train(X, Y) en cada iteración.
    El modelo recupera σ²(x) de _variance_store — guardado
    durante evaluar_configuracion_sk().
    """

    def __init__(self, configspace, instance_features=None,
                 pca_components=None, seed: int = 0,
                 nu: float = 2.5, n_restarts: int = 3,
                 normalize_y: bool = True):

        if _SMAC_GP_OK:
            kernel = SumKernel(
                ConstantKernel(2.0, (0.01, 100.0)),
                MaternKernel(1.0, (0.01, 10.0), nu=2.5)
            )
            super().__init__(
                configspace       = configspace,
                kernel            = kernel,
                instance_features = instance_features,
                pca_components    = None,
                seed              = seed,
            )

        self.nu          = nu
        self.n_restarts  = n_restarts
        self.normalize_y = normalize_y
        self._seed       = seed

        # Estado interno SK
        self._X_train   = None
        self._y_train   = None
        self._V_train   = None
        self._ls        = 1.0      # lengthscale global
        self._amp       = 1.0      # amplitud del kernel
        self._K_inv     = None
        self._alpha_vec = None
        self._y_mean    = 0.0
        self._y_std     = 1.0
        self._fitted    = False
        self._var_threshold = 1e-10

    def _get_gaussian_process(self):
        """Requerido por AbstractGaussianProcess — no usamos sklearn GP."""
        return None

    # ── Kernel Matérn 5/2 ──────────────────────────────────────
    def _kern(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        ls   = max(self._ls, 1e-5)
        amp  = max(self._amp, 1e-5)
        diff = X1[:, None, :] - X2[None, :, :]
        r    = np.sqrt(np.sum(diff**2, axis=2)) / ls
        sqrt5r = np.sqrt(5) * r
        return amp * (1 + sqrt5r + sqrt5r**2/3) * np.exp(-sqrt5r)

    # ── Selección de hiperparámetros via grid ──────────────────
    def _fit_hyperparams(self, X: np.ndarray, y: np.ndarray,
                          V: np.ndarray):
        """Grid search sobre lengthscale y amplitud — rápido y estable."""
        n        = len(y)
        best_nll = 1e12
        best_ls  = 1.0
        best_amp = 1.0

        for ls in [0.1, 0.3, 0.5, 1.0, 2.0]:
            for amp in [0.1, 0.5, 1.0, 5.0, 10.0]:
                self._ls  = ls
                self._amp = amp
                K  = self._kern(X, X)
                Ky = K + np.diag(V) + 1e-8 * np.eye(n)
                try:
                    L    = np.linalg.cholesky(Ky)
                    a    = np.linalg.solve(L.T, np.linalg.solve(L, y))
                    nll  = 0.5 * float(
                        y @ a
                        + 2*np.sum(np.log(np.diag(L)))
                        + n*np.log(2*np.pi)
                    )
                    if nll < best_nll:
                        best_nll = nll
                        best_ls  = ls
                        best_amp = amp
                except np.linalg.LinAlgError:
                    pass

        self._ls  = best_ls
        self._amp = best_amp

    # ── _train — interfaz SMAC ──────────────────────────────────
    def _train(self, X: np.ndarray, Y: np.ndarray):
        """
        Ajusta SK.
        X: (n, d) normalizadas por SMAC
        Y: (n, 1) costos (medias del KPI)

        Recupera varianzas de _variance_store para construir diag(V).
        """
        n = X.shape[0]
        y = Y.ravel()

        # Normalizar Y
        if self.normalize_y:
            self._y_mean = float(np.mean(y))
            self._y_std  = max(float(np.std(y)), 1e-8)
            y_n = (y - self._y_mean) / self._y_std
        else:
            y_n = y.copy()

        # Recuperar varianzas de la evaluación
        # SMAC no pasa varianzas; se leen del store global
        keys = list(_variance_store.keys())
        V    = np.zeros(n)
        for i in range(n):
            if i < len(keys):
                raw_var  = _variance_store[keys[i]]
                n_reps_i = _n_reps_store.get(keys[i], _N_REPS_GLOBAL)
                # Varianza del estimador: σ²/n (normalizada si corresponde)
                v_est = raw_var / max(n_reps_i, 1)
                V[i]  = v_est / (self._y_std**2) if self.normalize_y else v_est
            else:
                V[i] = 1.0   # fallback

        V = np.maximum(V, 1e-6)

        self._X_train = X.copy()
        self._y_train = y_n.copy()
        self._V_train = V.copy()

        # Ajustar hiperparámetros
        if n >= 5:
            try:
                self._fit_hyperparams(X, y_n, V)
            except Exception as e:
                log.debug("Error fit_hyperparams: %s", e)

        # K_y^{-1} y alpha
        K  = self._kern(X, X)
        Ky = K + np.diag(V) + 1e-8 * np.eye(n)
        try:
            self._K_inv    = np.linalg.inv(Ky)
            self._alpha_vec = self._K_inv @ y_n
            self._fitted    = True
        except np.linalg.LinAlgError:
            log.warning("SK: matriz singular, usando fallback")
            self._K_inv    = np.eye(n) * 0.01
            self._alpha_vec = y_n * 0.01
            self._fitted    = True

        log.debug("SK entrenado n=%d ls=%.3f amp=%.3f", n, self._ls, self._amp)
        return self

    # ── _predict — interfaz SMAC ────────────────────────────────
    def _predict(self, X: np.ndarray, covariance_type: str = "diagonal"):
        """
        Predicción SK.
        Retorna (μ, σ²) — SMAC los usa para calcular EI.
        σ²(x) es la incertidumbre del modelo (no el ruido simulación).
        """
        if not self._fitted or self._X_train is None:
            n = X.shape[0]
            return np.zeros((n, 1)), np.ones((n, 1))

        K_star = self._kern(X, self._X_train)      # (m, n)
        K_ss   = self._kern(X, X)                   # (m, m)

        mu       = K_star @ self._alpha_vec
        var_diag = np.diag(K_ss) - np.sum(K_star @ self._K_inv * K_star, axis=1)
        var_diag = np.maximum(var_diag, self._var_threshold)

        # Desnormalizar
        mu_out  = mu  * self._y_std + self._y_mean
        var_out = var_diag * self._y_std**2

        if covariance_type == "diagonal":
            return mu_out.reshape(-1, 1), var_out.reshape(-1, 1)
        return mu_out.reshape(-1, 1), np.diag(var_out)


# ──────────────────────────────────────────────────────────────
# KGCP: función de adquisición Knowledge Gradient — NUEVO M10
# ──────────────────────────────────────────────────────────────
try:
    from smac.acquisition.function.abstract_acquisition_function import (
        AbstractAcquisitionFunction,
    )
    _SMAC_ACQ_OK = True
except ImportError:
    _SMAC_ACQ_OK = False


class KGCPAcquisition(AbstractAcquisitionFunction if _SMAC_ACQ_OK else object):
    """
    Knowledge Gradient with Correlated Priors (KGCP).

    Para cada candidato x calcula:
      KG(x) ≈ mean_Z[ max_{x'∈X_cand} (μ(x') + σ_kg(x,x')·Z) ] - max μ

    donde:
      σ_kg(x, x') = K_n(x, x') / sqrt(K_n(x,x) + σ²_sim(x)/n_reps)
      Z ~ N(0,1)  (n_mc muestras Monte Carlo)
      X_cand      = puntos de entrenamiento + muestras aleatorias adicionales

    Parámetros
    ----------
    n_mc   : muestras Monte Carlo para E_Z  (default 64)
    n_cand : candidatos adicionales para max_{x'} (default 500)
    seed   : semilla para reproducibilidad de Z
    """

    def __init__(self, n_mc: int = 64, n_cand: int = 500, seed: int = 0):
        if _SMAC_ACQ_OK:
            super().__init__()
        self.n_mc   = int(n_mc)
        self.n_cand = int(n_cand)
        self._rng   = np.random.RandomState(seed)
        self._model = None          # StochasticKrigingModel (inyectado en update)
        self._eta   = None          # mejor μ actual (incumbente del modelo)
        self._mu_cand  = None       # μ(x') de candidatos   shape (n_cand+n_train,)
        self._X_cand   = None       # X de candidatos        shape (n_cand+n_train, d)
        # Para gráficos: historial de KG medio por evaluación
        self.kg_history = []
        self.ei_history = []

    # ── Interfaz SMAC ──────────────────────────────────────────
    @property
    def name(self) -> str:
        return "KGCP"

    def _compute(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Calcula KG(x) para cada fila de X.
        Llamado internamente por SMAC en cada iteración.
        """
        return self._kg_batch(X)

    def update(self, model, eta: float = None, **kwargs):
        """
        Recibe el modelo SK actualizado tras cada re-entrenamiento.
        Precalcula μ y X_cand para eficiencia.
        """
        self._model = model
        if not getattr(model, "_fitted", False) or model._X_train is None:
            self._eta = eta or 0.0
            return self

        mu_train, _ = model._predict(model._X_train)
        self._eta   = float(np.min(mu_train))   # incumbente modelo (minimización)

        # X_cand = puntos entrenamiento + muestras aleatorias en [0,1]^d
        d = model._X_train.shape[1]
        X_rand = self._rng.uniform(0, 1, size=(self.n_cand, d))
        self._X_cand = np.vstack([model._X_train, X_rand])

        mu_cand, _ = model._predict(self._X_cand)
        self._mu_cand = mu_cand.ravel()

        if eta is not None:
            self._eta = float(eta)
        return self

    # ── Cálculo KG vectorizado ─────────────────────────────────
    def _sigma_kg(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula σ_kg(x, x') para x en X (shape m×d) y x' en X_cand.
        Retorna shape (m, n_cand+n_train).

        σ_kg(x, x') = K_n(x, x') / sqrt(K_n(x,x) + noise(x))
        noise(x) = σ²_sim(x) / n_reps(x), recuperado de _variance_store.
        """
        model = self._model
        if model is None or self._X_cand is None:
            return np.zeros((X.shape[0], 1))

        K_x_cand = model._kern(X, self._X_cand)   # (m, n_cand+n_train)
        K_x_x    = np.diag(model._kern(X, X))     # (m,)

        # Ruido de estimación: σ²_sim/n para cada x
        noise = np.full(X.shape[0], 1.0)
        keys  = list(_variance_store.keys())
        for i in range(min(len(keys), X.shape[0])):
            k = keys[i]
            raw_var = _variance_store.get(k, 1.0)
            n_reps  = max(_n_reps_store.get(k, _N_REPS_GLOBAL), 1)
            # Normalizar igual que en _train
            std2 = max(model._y_std ** 2, 1e-8) if model.normalize_y else 1.0
            noise[i] = (raw_var / n_reps) / std2

        denom = np.sqrt(np.maximum(K_x_x + noise, 1e-10))  # (m,)
        sigma_kg = K_x_cand / denom[:, None]                # (m, n_cand)
        return sigma_kg

    def _kg_batch(self, X: np.ndarray) -> np.ndarray:
        """
        KG(x) ≈ (1/n_mc) Σ_z max_{x'} (μ(x') + σ_kg(x,x')·z) - max μ
        Retorna shape (m, 1) — positivo = bueno para minimización en SMAC.
        """
        if (self._model is None
                or not getattr(self._model, "_fitted", False)
                or self._X_cand is None):
            return np.zeros((X.shape[0], 1))

        sigma_kg  = self._sigma_kg(X)         # (m, n_cand)
        mu_cand   = self._mu_cand             # (n_cand,)
        eta       = self._eta

        # Muestras Z ~ N(0,1) shape (n_mc,)
        Z = self._rng.standard_normal(self.n_mc)

        # KG(x_i) = mean_z[ max_{x'}(μ(x') + σ_kg(x_i,x')·z) ] - eta
        # Broadcasting: sigma_kg (m, n_cand) × Z (n_mc,) → (m, n_cand, n_mc)
        # max sobre x' → (m, n_mc), mean sobre z → (m,)
        kg_vals = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            # (n_cand, n_mc) = mu_cand[:,None] + sigma_kg[i,:,None] * Z[None,:]
            f_cand = mu_cand[:, None] + sigma_kg[i, :, None] * Z[None, :]
            max_per_z = np.max(f_cand, axis=0)    # (n_mc,)
            kg_vals[i] = np.mean(max_per_z) - eta

        # SMAC maximiza la acquisition → negamos (minimización) o dejamos positivo
        # Para minimización de costos, KG > 0 cuando evaluar x es útil.
        # SMAC espera: mayor valor = más prometedor → retornamos KG directo.
        kg_pos = np.maximum(kg_vals, 0.0)

        # Guardar historial para gráficos
        if len(kg_pos) > 0:
            self.kg_history.append(float(np.mean(kg_pos)))

        return kg_pos.reshape(-1, 1)

    def compute_ei_comparable(self, X: np.ndarray) -> np.ndarray:
        """Calcula EI estándar para comparación en gráficos."""
        if (self._model is None
                or not getattr(self._model, "_fitted", False)):
            return np.zeros((X.shape[0], 1))
        mu, var = self._model._predict(X)
        mu  = mu.ravel()
        std = np.sqrt(np.maximum(var.ravel(), 1e-10))
        eta = self._eta
        from scipy.stats import norm
        z   = (eta - mu) / std
        ei  = (eta - mu) * norm.cdf(z) + std * norm.pdf(z)
        ei  = np.maximum(ei, 0.0)
        if len(ei) > 0:
            self.ei_history.append(float(np.mean(ei)))
        return ei.reshape(-1, 1)


# ──────────────────────────────────────────────────────────────
# Builders KGCP — NUEVO M10
# ──────────────────────────────────────────────────────────────
def _build_hpo_kgcp(scenario, fn_obj, cs, seed,
                    n_kgcp_mc: int = 64, n_kgcp_cand: int = 500):
    """
    HPO con KGCP:
      · Surrogate : SK heteroscedástico (igual que módulos 7-9)
      · Acquisition: KGCPAcquisition (reemplaza EI)
      · Initial design: LHS
      · Maximizer : LocalAndSortedRandomSearch
    """
    from smac import HyperparameterOptimizationFacade
    from smac.acquisition.maximizer import LocalAndSortedRandomSearch
    from smac.initial_design import LatinHypercubeInitialDesign
    from smac.random_design import ProbabilityRandomDesign

    sk_model = StochasticKrigingModel(
        configspace=cs, seed=seed, nu=2.5, n_restarts=3, normalize_y=True,
    )
    kgcp_acq = KGCPAcquisition(n_mc=n_kgcp_mc, n_cand=n_kgcp_cand, seed=seed)

    return HyperparameterOptimizationFacade(
        scenario              = scenario,
        target_function       = fn_obj,
        model                 = sk_model,
        acquisition_function  = kgcp_acq,
        acquisition_maximizer = LocalAndSortedRandomSearch(
            configspace=cs, seed=seed, challengers=1000,
        ),
        initial_design        = LatinHypercubeInitialDesign(scenario),
        random_design         = ProbabilityRandomDesign(probability=0.15),
        overwrite             = True,
    )


def _build_blackbox_kgcp(scenario, fn_obj, cs, seed,
                          n_kgcp_mc: int = 64, n_kgcp_cand: int = 500):
    """
    Blackbox con KGCP:
      · Igual que hpo_kgcp pero con SobolInitialDesign y menor prob_random.
    """
    from smac import HyperparameterOptimizationFacade
    from smac.acquisition.maximizer import LocalAndSortedRandomSearch
    from smac.initial_design import SobolInitialDesign
    from smac.random_design import ProbabilityRandomDesign

    sk_model = StochasticKrigingModel(
        configspace=cs, seed=seed, nu=2.5, n_restarts=3, normalize_y=True,
    )
    kgcp_acq = KGCPAcquisition(n_mc=n_kgcp_mc, n_cand=n_kgcp_cand, seed=seed)

    return HyperparameterOptimizationFacade(
        scenario              = scenario,
        target_function       = fn_obj,
        model                 = sk_model,
        acquisition_function  = kgcp_acq,
        acquisition_maximizer = LocalAndSortedRandomSearch(
            configspace=cs, seed=seed, challengers=1000,
        ),
        initial_design        = SobolInitialDesign(scenario),
        random_design         = ProbabilityRandomDesign(probability=0.085),
        overwrite             = True,
    )


# ──────────────────────────────────────────────────────────────
# Builders — 2 tipos SK + mismos 6 del módulo 4
# ──────────────────────────────────────────────────────────────
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


def _build_hpo_sk(scenario, fn_obj, cs, seed):
    """
    HPO con Stochastic Kriging:
      · Surrogate: SK (Matérn 5/2, heteroscedástico)
      · Acquisition: EI
      · Initial design: LHS
      · Acquisition maximizer: LocalAndSortedRandomSearch (1000 candidatos)
    """
    from smac import HyperparameterOptimizationFacade
    from smac.acquisition.function import EI
    from smac.acquisition.maximizer import LocalAndSortedRandomSearch
    from smac.initial_design import LatinHypercubeInitialDesign
    from smac.random_design import ProbabilityRandomDesign

    sk_model = StochasticKrigingModel(
        configspace  = cs,
        seed         = seed,
        nu           = 2.5,
        n_restarts   = 3,
        normalize_y  = True,
    )

    return HyperparameterOptimizationFacade(
        scenario              = scenario,
        target_function       = fn_obj,
        model                 = sk_model,
        acquisition_function  = EI(xi=0.0),
        acquisition_maximizer = LocalAndSortedRandomSearch(
            configspace  = cs,
            seed         = seed,
            challengers  = 1000,
        ),
        initial_design        = LatinHypercubeInitialDesign(scenario),
        random_design         = ProbabilityRandomDesign(probability=0.20),
        overwrite             = True,
    )


def _build_blackbox_sk(scenario, fn_obj, cs, seed):
    """
    Blackbox con Stochastic Kriging:
      · Mismo que hpo_sk pero prob_random = 8.5% (más explotación)
    """
    from smac import HyperparameterOptimizationFacade
    from smac.acquisition.function import EI
    from smac.acquisition.maximizer import LocalAndSortedRandomSearch
    from smac.initial_design import SobolInitialDesign
    from smac.random_design import ProbabilityRandomDesign

    sk_model = StochasticKrigingModel(
        configspace  = cs,
        seed         = seed,
        nu           = 2.5,
        normalize_y  = True,
    )

    return HyperparameterOptimizationFacade(
        scenario              = scenario,
        target_function       = fn_obj,
        model                 = sk_model,
        acquisition_function  = EI(xi=0.0),
        acquisition_maximizer = LocalAndSortedRandomSearch(
            configspace  = cs,
            seed         = seed,
            challengers  = 1000,
        ),
        initial_design        = SobolInitialDesign(scenario),
        random_design         = ProbabilityRandomDesign(probability=0.085),
        overwrite             = True,
    )


# Builders del módulo 4 originales (sin SK) — para comparación
def _build_hpo_rf(scenario, fn_obj):
    from smac import HyperparameterOptimizationFacade
    from smac.initial_design import SobolInitialDesign
    from smac.acquisition.function import EI
    from smac.random_design import ProbabilityRandomDesign
    return HyperparameterOptimizationFacade(
        scenario=scenario, target_function=fn_obj,
        initial_design=SobolInitialDesign(scenario),
        acquisition_function=EI(),
        random_design=ProbabilityRandomDesign(probability=0.20),
        overwrite=True,
    )

def _build_blackbox_gp(scenario, fn_obj):
    from smac import BlackBoxFacade
    from smac.initial_design import SobolInitialDesign
    from smac.acquisition.function import EI
    from smac.random_design import ProbabilityRandomDesign
    return BlackBoxFacade(
        scenario=scenario, target_function=fn_obj,
        initial_design=SobolInitialDesign(scenario),
        acquisition_function=EI(),
        random_design=ProbabilityRandomDesign(probability=0.085),
        overwrite=True,
    )

def _build_multifidelity(scenario, fn_obj):
    from smac import MultiFidelityFacade
    from smac.initial_design import SobolInitialDesign
    from smac.acquisition.function import EI
    from smac.random_design import ProbabilityRandomDesign
    from smac.intensifier import Hyperband
    return MultiFidelityFacade(
        scenario=scenario, target_function=fn_obj,
        initial_design=SobolInitialDesign(scenario),
        acquisition_function=EI(),
        intensifier=Hyperband(scenario, incumbent_selection="highest_budget"),
        random_design=ProbabilityRandomDesign(probability=0.20),
        overwrite=True,
    )

def _build_random(scenario, fn_obj):
    from smac import RandomFacade
    from smac.initial_design import RandomInitialDesign
    return RandomFacade(
        scenario=scenario, target_function=fn_obj,
        initial_design=RandomInitialDesign(scenario),
        overwrite=True,
    )


# Registro de tipos
TIPOS_SK = {
    "hpo_sk":       "HPO con Stochastic Kriging (recomendado)",
    "blackbox_sk":  "Blackbox con Stochastic Kriging",
    "hpo_kgcp":     "HPO con SK + KGCP (Knowledge Gradient)",
    "blackbox_kgcp":"Blackbox con SK + KGCP (Knowledge Gradient)",
}
TIPOS_RF = {
    "hpo":        "HPO con Random Forest (módulo 4)",
    "blackbox":   "Blackbox con GP (módulo 4)",
    "multifidelity": "Multi-Fidelity RF (módulo 4)",
    "random":     "Random Search (módulo 4)",
}
TODOS_TIPOS = {**TIPOS_SK, **TIPOS_RF}


# ──────────────────────────────────────────────────────────────
# Resultado
# ──────────────────────────────────────────────────────────────
@dataclass
class ResultadoSMAC_SK:
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
    descripcion:      str   = ""
    # ── Campos nuevos módulo 8 — replicación adaptativa ───────
    adaptativo:       bool  = False
    n_min_reps:       int   = 0
    n_max_reps:       int   = 0
    n_warmup:         int   = 0
    n_reps_history:   list  = field(default_factory=list)  # n_reps por evaluación
    sigma_history:    list  = field(default_factory=list)  # σ²_sim por evaluación
    # ── Campos nuevos módulo 9 — REVI ──────────────────────────
    revi:               bool = False
    sigma2_mod_history: list = field(default_factory=list)  # σ²_mod (modelo SK)
    # ── Campos nuevos módulo 10 — KGCP ─────────────────────────
    kgcp:               bool = False
    n_kgcp_mc:          int  = 0
    n_kgcp_cand:        int  = 0
    kg_history:         list = field(default_factory=list)  # KG medio por eval
    ei_history:         list = field(default_factory=list)  # EI medio por eval



# ──────────────────────────────────────────────────────────────
# Visualización: convergencia + tiempo de proceso
# ──────────────────────────────────────────────────────────────
def graficar(
    resultado,
    guardar_png: str  = None,
    mostrar:     bool = False,
) -> str:
    """
    Genera figura de convergencia + tiempo de proceso.
    Si el resultado es adaptativo (módulo 8), añade 2 paneles extra:
      · N° réplicas por evaluación (vs. línea fija del módulo 7)
      · σ² (varianza simulador) por evaluación
    """
    if hasattr(resultado, "__dict__"):
        r = resultado.__dict__
    elif hasattr(resultado, "_asdict"):
        r = resultado._asdict()
    else:
        r = dict(resultado)

    historia       = r.get("historia_costos", [])
    tipo           = r.get("tipo", "")
    objetivo       = r.get("objetivo", "objetivo")
    t_total        = float(r.get("tiempo_seg", 0.0))
    costo_final    = float(r.get("costo_incumbente", float("inf")))
    n_corridas     = int(r.get("n_corridas_eval", 1))
    nombre_tipo    = r.get("nombre_tipo", tipo)
    adaptativo     = bool(r.get("adaptativo", False))
    revi_mode      = bool(r.get("revi", False))
    kgcp_mode      = bool(r.get("kgcp", False))
    n_reps_history = r.get("n_reps_history", [])
    sigma_history  = r.get("sigma_history", [])
    kg_history     = r.get("kg_history", [])
    ei_history     = r.get("ei_history", [])
    n_min_reps     = int(r.get("n_min_reps", n_corridas))
    n_max_reps     = int(r.get("n_max_reps", n_corridas))
    n_warmup       = int(r.get("n_warmup", 0))

    if not historia:
        log.warning("graficar(): historia_costos vacía, no se genera gráfico.")
        return ""

    n           = len(historia)
    costos      = [h.get("costo", float("nan"))             for h in historia]
    incumbentes = [h.get("mejor_hasta_ahora", float("nan")) for h in historia]
    t_secs      = [h.get("t_seg", None)                     for h in historia]

    tiene_tiempo = any(v is not None for v in t_secs)
    if not tiene_tiempo and t_total > 0:
        t_secs = [round(t_total * (i + 1) / n, 1) for i in range(n)]
    t_secs  = [v if v is not None else 0.0 for v in t_secs]
    t_delta = [t_secs[0]] + [max(0.0, t_secs[i] - t_secs[i-1]) for i in range(1, n)]
    trials  = list(range(1, n + 1))

    hay_reps  = (revi_mode or adaptativo) and n_reps_history and sigma_history
    hay_kgcp  = kgcp_mode and (kg_history or ei_history)
    n_panels  = 2 + (2 if hay_reps else 0) + (1 if hay_kgcp else 0)
    h_ratios  = ([3, 2]
                 + ([1.5, 1.5] if hay_reps else [])
                 + ([1.5]      if hay_kgcp else []))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(10, 4 + n_panels * 2.5),
        gridspec_kw={"height_ratios": h_ratios},
    )
    if n_panels == 1:
        axes = [axes]
    elif n_panels == 2:
        axes = list(axes)
    ax1, ax2 = axes[0], axes[1]

    acq_str = "KGCP" if kgcp_mode else "EI"
    rep_str_modo = "REVI" if revi_mode else ("Adaptativo" if adaptativo else "Fijo")
    reps_str = (f"n_reps [{n_min_reps}–{n_max_reps}] {rep_str_modo}"
                if (revi_mode or adaptativo) else f"{n_corridas} réplicas/config")
    fig.suptitle(
        f"SMAC+SK — {nombre_tipo}  [Acq:{acq_str}  Reps:{rep_str_modo}]\n"
        f"Objetivo: {objetivo}  ·  {n} evaluaciones  ·  "
        f"{reps_str}  ·  {t_total:.0f}s totales",
        fontsize=11, fontweight="bold", y=0.99,
    )

    # ── Panel 1: Convergencia ─────────────────────────────────
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
                label=f"Diseño inicial (~{n_init})")
    ax1.axvline(n_init + 0.5, color="#e67e22", linewidth=0.8,
                linestyle=":", alpha=0.6)
    if adaptativo and n_warmup > 0:
        ax1.axvspan(0.5, n_warmup + 0.5, alpha=0.05, color="#8e44ad",
                    label=f"Warmup ({n_warmup})")
        ax1.axvline(n_warmup + 0.5, color="#8e44ad", linewidth=0.8,
                    linestyle="--", alpha=0.5)
    ax1.set_xlabel("N° evaluación", fontsize=10)
    ax1.set_ylabel(objetivo, fontsize=10)
    ax1.set_title("Convergencia", fontsize=10, pad=4)
    ax1.legend(fontsize=8, framealpha=0.9)
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=12))
    ax1.set_xlim(0.5, n + 0.5)

    # ── Panel 2: Tiempo ───────────────────────────────────────
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
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8,
               framealpha=0.9, loc="upper left")

    if hay_reps:
        ax3, ax4 = axes[2], axes[3]
        reps_n = n_reps_history[:n]
        reps_trials = list(range(1, len(reps_n) + 1))

        # Panel 3: N° réplicas por evaluación
        colores_reps = ["#27ae60" if r <= n_min_reps else
                        "#e67e22" if r >= n_max_reps else
                        "#1a6db5" for r in reps_n]
        ax3.bar(reps_trials, reps_n, color=colores_reps, edgecolor="none",
                alpha=0.8, zorder=2)
        ax3.axhline(n_min_reps, color="#27ae60", linewidth=1.0,
                    linestyle="--", alpha=0.7, label=f"n_min={n_min_reps}")
        ax3.axhline(n_max_reps, color="#e74c3c", linewidth=1.0,
                    linestyle="--", alpha=0.7, label=f"n_max={n_max_reps}")
        if n_warmup > 0:
            ax3.axvline(n_warmup + 0.5, color="#8e44ad", linewidth=0.9,
                        linestyle="--", alpha=0.6, label=f"fin warmup")
        media_reps = float(np.mean(reps_n)) if reps_n else n_min_reps
        ax3.axhline(media_reps, color="#333333", linewidth=1.0,
                    linestyle=":", alpha=0.6, label=f"media={media_reps:.1f}")
        ax3.set_ylabel("N° réplicas", fontsize=9)
        ax3.set_title("Réplicas adaptativas por evaluación", fontsize=10, pad=4)
        ax3.set_xlim(0.5, n + 0.5)
        ax3.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=12))
        ax3.set_yticks(range(n_min_reps, n_max_reps + 1))
        ax3.legend(fontsize=7.5, framealpha=0.9, ncol=4)

        # Panel 4: σ²_sim y σ²_mod por evaluación
        sigma_n     = sigma_history[:n]           # σ²_sim
        sigma_mod_n = r.get("sigma2_mod_history", [])[:n]   # σ²_mod (solo REVI)
        sigma_trials = list(range(1, len(sigma_n) + 1))
        if sigma_n:
            ax4.fill_between(sigma_trials, sigma_n, alpha=0.20, color="#c0392b")
            ax4.plot(sigma_trials, sigma_n, color="#c0392b", linewidth=1.5,
                     label="σ²_sim (simulador)")
            sigma_med = float(np.median(sigma_n))
            ax4.axhline(sigma_med, color="#c0392b", linewidth=0.9,
                        linestyle="--", alpha=0.6,
                        label=f"mediana σ²_sim={sigma_med:.1f}")
        if sigma_mod_n:
            ax4b = ax4.twinx()
            ax4b.plot(list(range(1, len(sigma_mod_n)+1)), sigma_mod_n,
                      color="#8e44ad", linewidth=1.5, alpha=0.8,
                      label="σ²_mod (modelo SK)")
            ax4b.set_ylabel("σ²_mod (modelo)", fontsize=8, color="#8e44ad")
            ax4b.tick_params(axis="y", labelcolor="#8e44ad")
            lines_b, labels_b = ax4b.get_legend_handles_labels()
            lines_a, labels_a = ax4.get_legend_handles_labels()
            ax4.legend(lines_a + lines_b, labels_a + labels_b,
                       fontsize=7.5, framealpha=0.9, loc="upper right")
        else:
            ax4.legend(fontsize=8, framealpha=0.9)
        if n_warmup > 0:
            ax4.axvline(n_warmup + 0.5, color="#8e44ad", linewidth=0.9,
                        linestyle="--", alpha=0.5)
        ax4.set_xlabel("N° evaluación", fontsize=10)
        ax4.set_ylabel("σ²_sim (varianza simulador)", fontsize=9)
        ax4.set_title("Varianza simulador y modelo por evaluación", fontsize=10, pad=4)
        ax4.set_xlim(0.5, n + 0.5)
        ax4.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=12))

    # ── Panel KGCP: KG vs EI por iteración ───────────────────
    if hay_kgcp:
        ax_kg = axes[2 + (2 if hay_reps else 0)]
        kg_t  = list(range(1, len(kg_history) + 1))
        ei_t  = list(range(1, len(ei_history) + 1))
        if kg_history:
            ax_kg.plot(kg_t, kg_history, color="#1a6db5", linewidth=1.8,
                       label="KG (KGCP)", zorder=3)
            ax_kg.fill_between(kg_t, kg_history, alpha=0.12, color="#1a6db5")
        if ei_history:
            ax_kgr = ax_kg.twinx()
            ax_kgr.plot(ei_t, ei_history, color="#e67e22", linewidth=1.5,
                        linestyle="--", alpha=0.8, label="EI (comparable)", zorder=2)
            ax_kgr.set_ylabel("EI", fontsize=8, color="#e67e22")
            ax_kgr.tick_params(axis="y", labelcolor="#e67e22")
            lines_a, labels_a = ax_kg.get_legend_handles_labels()
            lines_b, labels_b = ax_kgr.get_legend_handles_labels()
            ax_kg.legend(lines_a + lines_b, labels_a + labels_b,
                         fontsize=8, framealpha=0.9)
        else:
            ax_kg.legend(fontsize=8, framealpha=0.9)
        ax_kg.set_xlabel("N° evaluación", fontsize=10)
        ax_kg.set_ylabel("KG", fontsize=9)
        ax_kg.set_title("Knowledge Gradient vs Expected Improvement por iteración",
                         fontsize=10, pad=4)
        ax_kg.set_xlim(0.5, n + 0.5)
        ax_kg.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=12))

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if guardar_png is None:
        sfx = "kgcp" if kgcp_mode else ("revi" if revi_mode else
              "adaptativo" if adaptativo else tipo)
        guardar_png = f"convergencia_{sfx}_{objetivo}.png"
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
    resultados: dict,
    guardar_png: str = "convergencia_comparacion_sk_rf.png",
    mostrar:     bool = False,
) -> str:
    """
    Gráfico de convergencia superpuesto para comparar SK vs RF
    (o cualquier conjunto de resultados de optimizar()).

    resultados: {nombre: ResultadoSMAC_SK_o_dict, ...}
    """
    colores = ["#1a6db5", "#c0392b", "#27ae60", "#8e44ad", "#e67e22"]
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                    gridspec_kw={"height_ratios": [3, 2]})

    objetivo = ""
    for idx, (nombre, res) in enumerate(resultados.items()):
        if hasattr(res, "__dict__"):
            r = res.__dict__
        else:
            r = dict(res)

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
        tiene_tiempo = any(v is not None for v in t_secs)
        if not tiene_tiempo and t_total > 0:
            t_secs = [round(t_total * (i + 1) / n, 1) for i in range(n)]
        t_secs = [v if v is not None else 0.0 for v in t_secs]
        trials = list(range(1, n + 1))

        inc_v = [(i, v) for i, v in zip(trials, incumbentes) if not np.isnan(v)]
        if inc_v:
            xi, yi = zip(*inc_v)
            ax1.step(xi, yi, where="post", color=color, linewidth=2.0,
                     label=f"{nombre}  (final={costo_fin:.2f})", zorder=3)

        ax2.plot(trials, t_secs, color=color, linewidth=1.8,
                 label=f"{nombre}  ({t_total:.0f}s)")

    ax1.set_xlabel("N° evaluación", fontsize=10)
    ax1.set_ylabel(objetivo, fontsize=10)
    ax1.set_title("Convergencia comparativa", fontsize=10, pad=4)
    ax1.legend(fontsize=9, framealpha=0.9)
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=12))

    ax2.set_xlabel("N° evaluación", fontsize=10)
    ax2.set_ylabel("Tiempo acumulado (s)", fontsize=10)
    ax2.set_title("Tiempo acumulado comparativo", fontsize=10, pad=4)
    ax2.legend(fontsize=9, framealpha=0.9)
    ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=12))

    fig.suptitle(f"Comparación SMAC  ·  objetivo: {objetivo}",
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


# ──────────────────────────────────────────────────────────────
# Función principal de optimización
# ──────────────────────────────────────────────────────────────
def optimizar(
    tipo:            str   = "hpo_kgcp",
    n_trials:        int   = 100,
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
    # ── Replicación ───────────────────────────────────────────
    adaptativo:      bool  = False,
    revi:            bool  = False,
    n_min_reps:      int   = 2,
    n_max_reps:      int   = 8,
    n_warmup:        int   = 10,
    # ── KGCP — NUEVO M10 ──────────────────────────────────────
    n_kgcp_mc:       int   = 64,
    n_kgcp_cand:     int   = 500,
) -> ResultadoSMAC_SK:
    """
    Ejecuta SMAC+SK con KGCP como función de adquisición (módulo 10).

    tipo='hpo_kgcp'      → HPO con SK+KGCP  (recomendado)
    tipo='blackbox_kgcp' → Blackbox con SK+KGCP
    tipo='hpo_sk'        → HPO con SK+EI    (M7, para comparación)

    Replicación: fija (default), adaptativa (M8) o REVI (M9) — combinables.
    revi tiene prioridad sobre adaptativo si ambos son True.
    """
    if tipo not in TODOS_TIPOS:
        raise ValueError(f"Tipo '{tipo}' no válido. Opciones: {list(TODOS_TIPOS.keys())}")

    output_dir   = output_dir   or f"smac_m10_output_{tipo}"
    guardar_json = guardar_json or f"resultado_m10_{tipo}.json"

    is_kgcp  = tipo in ("hpo_kgcp", "blackbox_kgcp")
    rep_modo = ("REVI" if revi else "Adaptativo" if adaptativo else "Fijo")
    acq_str  = "KGCP" if is_kgcp else "EI"
    log.info("="*65)
    log.info("SMAC+SK Módulo 10 — %s", TODOS_TIPOS[tipo])
    log.info("Acquisition : %s  |  Replicación: %s", acq_str, rep_modo)
    log.info("Objetivo    : %s", objetivo)
    log.info("n_trials    : %d  |  seed: %d", n_trials, seed)
    if is_kgcp:
        log.info("KGCP params : n_mc=%d  n_cand=%d", n_kgcp_mc, n_kgcp_cand)
    if revi or adaptativo:
        log.info("Reps params : n_min=%d  n_max=%d  n_warmup=%d",
                 n_min_reps, n_max_reps, n_warmup)
    log.info("="*65)

    _variance_store.clear()
    _n_reps_store.clear()

    cs = crear_espacio_configuracion(seed=seed)

    # Instanciar política según modo
    revi_policy      = None
    adaptive_policy  = None
    sk_model_ref     = [None]   # lista mutable para pasar por referencia al closure

    if revi:
        revi_policy = REVIReplicationPolicy(
            n_min=n_min_reps, n_max=n_max_reps,
            n_warmup=n_warmup,
        )
    elif adaptativo:
        adaptive_policy = AdaptiveReplicationPolicy(
            n_min=n_min_reps, n_max=n_max_reps,
            n_warmup=n_warmup,
        )

    _pesos   = pesos_kpi or {}
    _tracker = {"t0_opt": None, "calls": []}

    def fn_obj(config, seed: int = 0, budget: float = None) -> float:
        if _tracker["t0_opt"] is None:
            _tracker["t0_opt"] = time.time()

        # Inyectar modelo SK actualizado en la política REVI
        if revi and revi_policy is not None and sk_model_ref[0] is not None:
            revi_policy.set_sk_model(sk_model_ref[0])

        if revi and revi_policy is not None:
            costo = evaluar_configuracion_sk_revi(
                config    = config,
                policy    = revi_policy,
                seed      = seed,
                seed_base = seed_base_modelo,
                objetivo  = objetivo,
                n_workers = n_workers,
            )
        elif adaptativo and adaptive_policy is not None:
            costo = evaluar_configuracion_sk_adaptive(
                config    = config,
                policy    = adaptive_policy,
                seed      = seed,
                seed_base = seed_base_modelo,
                objetivo  = objetivo,
                n_workers = n_workers,
            )
        else:
            costo = evaluar_configuracion_sk(
                config     = config,
                seed       = seed,
                n_corridas = n_corridas_eval,
                seed_base  = seed_base_modelo,
                objetivo   = objetivo,
                pesos_kpi  = _pesos,
                budget     = budget,
                n_workers  = n_workers,
            )

        key = _config_key(config)
        _tracker["calls"].append({
            "t_seg":   round(time.time() - _tracker["t0_opt"], 2),
            "costo":   round(float(costo), 4),
            "n_reps":  int(_n_reps_store.get(key, n_corridas_eval)),
            "sigma2":  round(float(_variance_store.get(key, 0.0)), 4),
        })
        return costo

    es_mf  = tipo == "multifidelity"
    min_b  = (min_budget or 1)              if es_mf else None
    max_b  = (max_budget or n_corridas_eval) if es_mf else None

    scenario = _scenario_base(cs, n_trials, output_dir, seed, min_b, max_b)

    # Construir SMAC — para REVI necesitamos acceso al modelo SK
    if tipo == "hpo_kgcp":
        smac = _build_hpo_kgcp(scenario, fn_obj, cs, seed,
                               n_kgcp_mc=n_kgcp_mc, n_kgcp_cand=n_kgcp_cand)
    elif tipo == "blackbox_kgcp":
        smac = _build_blackbox_kgcp(scenario, fn_obj, cs, seed,
                                    n_kgcp_mc=n_kgcp_mc, n_kgcp_cand=n_kgcp_cand)
    elif tipo == "hpo_sk":
        smac = _build_hpo_sk(scenario, fn_obj, cs, seed)
        if revi and hasattr(smac, 'model'):
            sk_model_ref[0] = smac.model
    elif tipo == "blackbox_sk":
        smac = _build_blackbox_sk(scenario, fn_obj, cs, seed)
        if revi and hasattr(smac, 'model'):
            sk_model_ref[0] = smac.model
    elif tipo == "hpo":
        smac = _build_hpo_rf(scenario, fn_obj)
    elif tipo == "blackbox":
        smac = _build_blackbox_gp(scenario, fn_obj)
    elif tipo == "multifidelity":
        smac = _build_multifidelity(scenario, fn_obj)
    elif tipo == "random":
        smac = _build_random(scenario, fn_obj)
    else:
        raise ValueError(f"Tipo no reconocido: {tipo}")

    t0 = time.time()
    log.info("Iniciando optimización...")
    incumbente_config = smac.optimize()

    # Actualizar referencia al modelo SK tras optimizar (ya entrenado)
    if revi and revi_policy is not None and hasattr(smac, 'model'):
        revi_policy.set_sk_model(smac.model)

    t_total = time.time() - t0

    mejor_config = dict(incumbente_config)
    try:
        mejor_costo = float(smac.runhistory.get_cost(incumbente_config))
    except Exception:
        mejor_costo = float('inf')

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
                entrada["t_seg"]  = tracker_calls[idx]["t_seg"]
                entrada["n_reps"] = tracker_calls[idx]["n_reps"]
                entrada["sigma2"] = tracker_calls[idx]["sigma2"]
            historia.append(entrada)
        except Exception:
            pass

    mejor_hasta = float("inf")
    for entrada in historia:
        mejor_hasta = min(mejor_hasta, entrada["costo"])
        entrada["mejor_hasta_ahora"] = round(mejor_hasta, 4)

    n_reps_history  = [c.get("n_reps", n_corridas_eval) for c in tracker_calls]
    sigma_history   = [c.get("sigma2", 0.0)             for c in tracker_calls]
    sigma2_mod_hist = (revi_policy.sigma2_mod_hist
                       if revi and revi_policy else [])

    # Resumen de réplicas
    active_policy = revi_policy if revi else adaptive_policy
    if active_policy and n_reps_history:
        total_reps = sum(n_reps_history)
        reps_fijas = len(n_reps_history) * n_corridas_eval
        ahorro_pct = 100.0 * (1 - total_reps / max(reps_fijas, 1))
        log.info("─"*65)
        log.info("Réplicas totales (%s): %d  vs  fijo(%d): %d  →  ahorro: %.1f%%",
                 modo_str, total_reps, n_corridas_eval, reps_fijas, ahorro_pct)
        log.info("Media réplicas/eval: %.1f  [min=%d  max=%d]",
                 float(np.mean(n_reps_history)),
                 min(n_reps_history), max(n_reps_history))

    log.info("─"*65)
    log.info("Completado %.0fs | Mejor %s = %.3f", t_total, objetivo, mejor_costo)
    log.info("Mejor configuración:")
    for k, v in mejor_config.items():
        baseline_v = _BASELINES.get(k, "?")
        log.info("  %-30s = %-10s  (baseline=%s)", k, v, baseline_v)

    # Extraer historial KG/EI si es KGCP
    kg_hist = []
    ei_hist = []
    if is_kgcp and hasattr(smac, 'acquisition_function'):
        acq = smac.acquisition_function
        if isinstance(acq, KGCPAcquisition):
            kg_hist = list(acq.kg_history)
            ei_hist = list(acq.ei_history)

    resultado = ResultadoSMAC_SK(
        tipo               = tipo,
        nombre_tipo        = TODOS_TIPOS[tipo],
        objetivo           = objetivo,
        n_trials           = n_trials,
        n_corridas_eval    = n_corridas_eval,
        seed               = seed,
        incumbente         = {
            k: (int(v) if isinstance(v, (int, np.integer)) else round(float(v), 6))
            for k, v in mejor_config.items()
        },
        costo_incumbente   = round(mejor_costo, 4),
        n_evaluaciones     = len(historia),
        tiempo_seg         = round(t_total, 2),
        output_dir         = output_dir,
        historia_costos    = historia,
        descripcion        = TODOS_TIPOS[tipo],
        adaptativo         = bool(adaptativo and not revi),
        revi               = bool(revi),
        n_min_reps         = int(n_min_reps),
        n_max_reps         = int(n_max_reps),
        n_warmup           = int(n_warmup),
        n_reps_history     = n_reps_history,
        sigma_history      = sigma_history,
        sigma2_mod_history = sigma2_mod_hist,
        kgcp               = bool(is_kgcp),
        n_kgcp_mc          = int(n_kgcp_mc),
        n_kgcp_cand        = int(n_kgcp_cand),
        kg_history         = kg_hist,
        ei_history         = ei_hist,
    )

    out = asdict(resultado)
    with open(guardar_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log.info("Resultado guardado en '%s'", guardar_json)

    if not getattr(optimizar, "_no_plot", False):
        png_path = guardar_json.replace(".json", ".png")
        graficar(resultado, guardar_png=png_path)

    return resultado
    if tipo not in TODOS_TIPOS:
        raise ValueError(f"Tipo '{tipo}' no válido. Opciones: {list(TODOS_TIPOS.keys())}")

    output_dir   = output_dir   or f"smac_m8_output_{tipo}"
    guardar_json = guardar_json or f"resultado_m8_{tipo}.json"

    log.info("="*65)
    log.info("SMAC+SK Módulo 8 — %s", TODOS_TIPOS[tipo])
    log.info("Objetivo   : %s", objetivo)
    log.info("Adaptativo : %s  [n_min=%d  n_max=%d  n_warmup=%d]",
             adaptativo, n_min_reps, n_max_reps, n_warmup)
    log.info("n_trials   : %d  |  seed: %d", n_trials, seed)
    log.info("="*65)

    _variance_store.clear()
    _n_reps_store.clear()

    cs = crear_espacio_configuracion(seed=seed)

    # Política adaptativa
    policy = AdaptiveReplicationPolicy(
        n_min=n_min_reps, n_max=n_max_reps,
        n_warmup=n_warmup,
    ) if adaptativo else None

    _pesos   = pesos_kpi or {}
    _tracker = {"t0_opt": None, "calls": []}

    def fn_obj(config, seed: int = 0, budget: float = None) -> float:
        if _tracker["t0_opt"] is None:
            _tracker["t0_opt"] = time.time()

        if adaptativo and policy is not None:
            costo = evaluar_configuracion_sk_adaptive(
                config    = config,
                policy    = policy,
                seed      = seed,
                seed_base = seed_base_modelo,
                objetivo  = objetivo,
                n_workers = n_workers,
            )
        else:
            costo = evaluar_configuracion_sk(
                config     = config,
                seed       = seed,
                n_corridas = n_corridas_eval,
                seed_base  = seed_base_modelo,
                objetivo   = objetivo,
                pesos_kpi  = _pesos,
                budget     = budget,
                n_workers  = n_workers,
            )

        key = _config_key(config)
        _tracker["calls"].append({
            "t_seg":   round(time.time() - _tracker["t0_opt"], 2),
            "costo":   round(float(costo), 4),
            "n_reps":  int(_n_reps_store.get(key, n_corridas_eval)),
            "sigma2":  round(float(_variance_store.get(key, 0.0)), 4),
        })
        return costo

    es_mf  = tipo == "multifidelity"
    min_b  = (min_budget or 1)              if es_mf else None
    max_b  = (max_budget or n_corridas_eval) if es_mf else None

    scenario = _scenario_base(cs, n_trials, output_dir, seed, min_b, max_b)

    if tipo == "hpo_sk":
        smac = _build_hpo_sk(scenario, fn_obj, cs, seed)
    elif tipo == "blackbox_sk":
        smac = _build_blackbox_sk(scenario, fn_obj, cs, seed)
    elif tipo == "hpo":
        smac = _build_hpo_rf(scenario, fn_obj)
    elif tipo == "blackbox":
        smac = _build_blackbox_gp(scenario, fn_obj)
    elif tipo == "multifidelity":
        smac = _build_multifidelity(scenario, fn_obj)
    elif tipo == "random":
        smac = _build_random(scenario, fn_obj)
    else:
        raise ValueError(f"Tipo no reconocido: {tipo}")

    t0 = time.time()
    log.info("Iniciando optimización...")
    incumbente_config = smac.optimize()
    t_total = time.time() - t0

    mejor_config = dict(incumbente_config)
    try:
        mejor_costo = float(smac.runhistory.get_cost(incumbente_config))
    except Exception:
        mejor_costo = float('inf')

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
                entrada["t_seg"]  = tracker_calls[idx]["t_seg"]
                entrada["n_reps"] = tracker_calls[idx]["n_reps"]
                entrada["sigma2"] = tracker_calls[idx]["sigma2"]
            historia.append(entrada)
        except Exception:
            pass

    mejor_hasta = float("inf")
    for entrada in historia:
        mejor_hasta = min(mejor_hasta, entrada["costo"])
        entrada["mejor_hasta_ahora"] = round(mejor_hasta, 4)

    n_reps_history = [c.get("n_reps", n_corridas_eval) for c in tracker_calls]
    sigma_history  = [c.get("sigma2", 0.0)             for c in tracker_calls]

    # Resumen adaptativo
    if adaptativo and n_reps_history:
        total_reps = sum(n_reps_history)
        reps_fijas = len(n_reps_history) * n_corridas_eval
        ahorro_pct = 100.0 * (1 - total_reps / max(reps_fijas, 1))
        log.info("─"*65)
        log.info("Réplicas totales (adaptativo): %d  vs  fijo: %d  →  ahorro: %.1f%%",
                 total_reps, reps_fijas, ahorro_pct)
        log.info("Media réplicas por eval: %.1f  [min=%d  max=%d]",
                 float(np.mean(n_reps_history)), min(n_reps_history), max(n_reps_history))

    log.info("─"*65)
    log.info("Completado %.0fs | Mejor %s = %.3f", t_total, objetivo, mejor_costo)
    log.info("Mejor configuración:")
    for k, v in mejor_config.items():
        baseline_v = _BASELINES.get(k, "?")
        log.info("  %-30s = %-10s  (baseline=%s)", k, v, baseline_v)

    resultado = ResultadoSMAC_SK(
        tipo             = tipo,
        nombre_tipo      = TODOS_TIPOS[tipo],
        objetivo         = objetivo,
        n_trials         = n_trials,
        n_corridas_eval  = n_corridas_eval,
        seed             = seed,
        incumbente       = {
            k: (int(v) if isinstance(v, (int, np.integer)) else round(float(v), 6))
            for k, v in mejor_config.items()
        },
        costo_incumbente = round(mejor_costo, 4),
        n_evaluaciones   = len(historia),
        tiempo_seg       = round(t_total, 2),
        output_dir       = output_dir,
        historia_costos  = historia,
        descripcion      = TODOS_TIPOS[tipo],
        adaptativo       = bool(adaptativo),
        n_min_reps       = int(n_min_reps),
        n_max_reps       = int(n_max_reps),
        n_warmup         = int(n_warmup),
        n_reps_history   = n_reps_history,
        sigma_history    = sigma_history,
    )

    out = asdict(resultado)
    with open(guardar_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log.info("Resultado guardado en '%s'", guardar_json)

    if not getattr(optimizar, "_no_plot", False):
        png_path = guardar_json.replace(".json", ".png")
        graficar(resultado, guardar_png=png_path)

    return resultado
    if tipo not in TODOS_TIPOS:
        raise ValueError(f"Tipo '{tipo}' no válido. Opciones: {list(TODOS_TIPOS.keys())}")

    output_dir   = output_dir   or f"smac_m8_output_{tipo}"
    guardar_json = guardar_json or f"resultado_m8_{tipo}.json"


# ──────────────────────────────────────────────────────────────
# Comparar SK vs RF
# ──────────────────────────────────────────────────────────────
def comparar_sk_vs_rf(
    n_trials:        int  = 50,
    n_corridas_eval: int  = 3,
    seed:            int  = 42,
    seed_base_modelo:int  = 202,
    objetivo:        str  = "tts_full_days_mean",
) -> dict:
    """Corre hpo_sk y hpo (RF) con los mismos parámetros y compara."""
    resultados = {}
    for tipo in ["hpo_sk", "hpo"]:
        log.info("\n[%s]", tipo.upper())
        r = optimizar(
            tipo             = tipo,
            n_trials         = n_trials,
            n_corridas_eval  = n_corridas_eval,
            seed             = seed,
            seed_base_modelo = seed_base_modelo,
            objetivo         = objetivo,
            output_dir       = f"smac_sk_compare_{tipo}",
            guardar_json     = f"resultado_compare_{tipo}.json",
        )
        resultados[tipo] = asdict(r)

    print("\n" + "="*70)
    print("COMPARACIÓN SK vs RF")
    print(f"Objetivo: {objetivo}  |  n_trials: {n_trials}  |  n_corridas: {n_corridas_eval}")
    print("="*70)
    print(f"  {'Tipo':<20} {'Surrogate':<25} {'Mejor costo':>12} {'Tiempo':>8}")
    print("  " + "─"*68)
    surrogates = {"hpo_sk": "Stochastic Kriging", "hpo": "Random Forest"}
    ordenados = sorted(resultados.items(), key=lambda x: x[1]["costo_incumbente"])
    for tipo, r in ordenados:
        ganador = " ← GANADOR" if tipo == ordenados[0][0] else ""
        print(f"  {tipo:<20} {surrogates[tipo]:<25} "
              f"{r['costo_incumbente']:>12.2f}  {r['tiempo_seg']:>6.0f}s{ganador}")
    print("\n  Mejor configuración (ganador):")
    mejor_tipo, mejor_r = ordenados[0]
    for k, v in mejor_r["incumbente"].items():
        baseline_v = _BASELINES.get(k, "?")
        cambio = " ←" if str(v) != str(baseline_v) else ""
        print(f"    {k:<35} = {v}  (baseline={baseline_v}){cambio}")
    print("="*70)

    with open("resultado_comparacion_sk_rf.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    # Gráfico comparativo
    if not getattr(optimizar, "_no_plot", False):
        graficar_comparacion(resultados,
                             guardar_png="convergencia_comparacion_sk_rf.png")

    return resultados


# ──────────────────────────────────────────────────────────────
# Evaluar incumbente — IDÉNTICO al módulo 4
# ──────────────────────────────────────────────────────────────
def evaluar_incumbente(
    incumbente:   dict,
    n_corridas:   int  = 10,
    seed_base:    int  = 300,
    objetivo:     str  = "tts_full_days_mean",
    guardar_json: str  = "evaluacion_incumbente_sk.json",
    n_workers:    int  = 0,
) -> dict:
    """Evalúa incumbente con más corridas — versión paralelizada. API compatible con módulo 4."""
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

    mu = float(np.mean(valores))
    sd = float(np.std(valores, ddof=1)) if len(valores) > 1 else 0.0
    try:
        from scipy.stats import t
        tc = float(t.ppf(0.975, len(valores)-1))
    except Exception:
        tc = 1.96
    se = sd / np.sqrt(len(valores))

    resultado = {
        "incumbente":  incumbente,
        "objetivo":    objetivo,
        "n_corridas":  len(valores),
        "valores":     [round(v, 4) for v in valores],
        "media":       round(mu, 4),
        "sd":          round(sd, 4),
        "ic_95_bajo":  round(mu - tc*se, 4),
        "ic_95_alto":  round(mu + tc*se, 4),
    }

    with open(guardar_json, "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)
    log.info("%s = %.2f ± %.2f  IC95=[%.2f, %.2f]",
             objetivo, mu, sd, mu-tc*se, mu+tc*se)
    return resultado


# ──────────────────────────────────────────────────────────────
# Mostrar espacio
# ──────────────────────────────────────────────────────────────
def mostrar_espacio():
    cs = crear_espacio_configuracion()
    print("\n" + "="*80)
    print("ESPACIO DE CONFIGURACIÓN — 12 PARÁMETROS (SMAC+SK)")
    print("="*80)
    print(f"\n  {'Parámetro':<35} {'Tipo':<8} {'Min':>6} {'Max':>6} {'Baseline':>10}")
    print("  " + "─"*70)
    for hp in cs.get_hyperparameters():
        tipo = "Int" if hasattr(hp, 'lower') and isinstance(hp.lower, int) else "Float"
        low  = getattr(hp, 'lower', '?')
        high = getattr(hp, 'upper', '?')
        base = _BASELINES.get(hp.name, '?')
        print(f"  {hp.name:<35} {tipo:<8} {str(low):>6} {str(high):>6} {str(base):>10}")
    print(f"\n  Total: {len(cs.get_hyperparameters())} hiperparámetros")
    print("\n  Tipos disponibles:")
    for tipo, desc in TODOS_TIPOS.items():
        sk_tag = " [SK]" if "sk" in tipo else " [RF]"
        print(f"    {tipo:<20} {desc}{sk_tag}")


# ──────────────────────────────────────────────────────────────
# CLI — Módulo 10
# ──────────────────────────────────────────────────────────────
# Guard obligatorio para multiprocessing en Windows y macOS
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Módulo 10: SK + KGCP — Knowledge Gradient with Correlated Priors"
    )
    parser.add_argument("--tipo", type=str, default="hpo_kgcp",
                        choices=list(TODOS_TIPOS.keys()),
                        help="Tipo SMAC (default: hpo_kgcp)")
    parser.add_argument("--n_trials",    type=int,   default=100)
    parser.add_argument("--n_corridas",  type=int,   default=3,
                        help="Réplicas fijas (si no usa revi/adaptativo)")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--seed_modelo", type=int,   default=202)
    parser.add_argument("--output_dir",  type=str,   default=None)
    parser.add_argument("--guardar_json",type=str,   default=None)
    parser.add_argument("--min_budget",  type=float, default=None)
    parser.add_argument("--max_budget",  type=float, default=None)
    parser.add_argument("--n_workers",   type=int,   default=0)
    parser.add_argument("--no_plot",     action="store_true")
    # ── KGCP ────────────────────────────────────────────────
    parser.add_argument("--n_kgcp_mc",   type=int,   default=64,
                        help="Muestras MC para E_Z en KGCP (default: 64)")
    parser.add_argument("--n_kgcp_cand", type=int,   default=500,
                        help="Candidatos X' para max en KGCP (default: 500)")
    # ── Replicación ─────────────────────────────────────────
    parser.add_argument("--revi",          action="store_true",
                        help="Activar REVI para asignación de réplicas (M9)")
    parser.add_argument("--adaptativo",    action="store_true",
                        help="Activar replicación adaptativa (M8)")
    parser.add_argument("--n_min_reps",  type=int,   default=2)
    parser.add_argument("--n_max_reps",  type=int,   default=8)
    parser.add_argument("--n_warmup",    type=int,   default=10)
    # ── Otros ───────────────────────────────────────────────
    parser.add_argument("--mostrar_espacio", action="store_true")
    parser.add_argument("--evaluar_incumbente", type=str, default=None)
    args = parser.parse_args()

    optimizar._no_plot = args.no_plot

    if args.mostrar_espacio:
        mostrar_espacio()
        sys.exit(0)

    if args.evaluar_incumbente:
        with open(args.evaluar_incumbente) as f:
            data = json.load(f)
        evaluar_incumbente(
            incumbente   = data.get("incumbente", {}),
            n_corridas   = max(10, args.n_corridas * 3),
            objetivo     = data.get("objetivo", "tts_full_days_mean"),
            guardar_json = "evaluacion_incumbente_m10.json",
            n_workers    = args.n_workers,
        )
        sys.exit(0)

    optimizar(
        tipo             = args.tipo,
        n_trials         = args.n_trials,
        n_corridas_eval  = args.n_corridas,
        seed             = args.seed,
        seed_base_modelo = args.seed_modelo,
        objetivo         = "tts_full_days_mean",
        output_dir       = args.output_dir,
        guardar_json     = args.guardar_json,
        min_budget       = args.min_budget,
        max_budget       = args.max_budget,
        n_workers        = args.n_workers,
        revi             = args.revi,
        adaptativo       = args.adaptativo,
        n_min_reps       = args.n_min_reps,
        n_max_reps       = args.n_max_reps,
        n_warmup         = args.n_warmup,
        n_kgcp_mc        = args.n_kgcp_mc,
        n_kgcp_cand      = args.n_kgcp_cand,
    )
    parser = argparse.ArgumentParser(
        description="Módulo 9: SK + REVI — Replicated Expected Value of Information"
    )
    parser.add_argument("--tipo", type=str, default="hpo_sk",
                        choices=list(TODOS_TIPOS.keys()),
                        help="Tipo SMAC (default: hpo_sk)")
    parser.add_argument("--n_trials",    type=int,   default=100,
                        help="Evaluaciones totales (default: 100)")
    parser.add_argument("--n_corridas",  type=int,   default=3,
                        help="Réplicas base si --no_revi y --no_adaptativo (default: 3)")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--seed_modelo", type=int,   default=202)
    parser.add_argument("--output_dir",  type=str,   default=None)
    parser.add_argument("--guardar_json",type=str,   default=None)
    parser.add_argument("--min_budget",  type=float, default=None)
    parser.add_argument("--max_budget",  type=float, default=None)
    parser.add_argument("--n_workers",   type=int,   default=0,
                        help="Procesos paralelos por evaluación (0=auto)")
    parser.add_argument("--no_plot",     action="store_true",
                        help="No generar PNG")
    # ── Modo de replicación ─────────────────────────────────
    parser.add_argument("--no_revi",       action="store_true",
                        help="Desactiva REVI y usa AdaptiveReplicationPolicy (M8)")
    parser.add_argument("--no_adaptativo", action="store_true",
                        help="Desactiva replicación adaptativa (n_corridas fijo)")
    parser.add_argument("--n_min_reps",  type=int,   default=2,
                        help="Réplicas mínimas (default: 2)")
    parser.add_argument("--n_max_reps",  type=int,   default=8,
                        help="Réplicas máximas (default: 8)")
    parser.add_argument("--n_warmup",    type=int,   default=10,
                        help="Evaluaciones de warmup con n_min (default: 10)")
    # ── Otros ───────────────────────────────────────────────
    parser.add_argument("--mostrar_espacio", action="store_true")
    parser.add_argument("--evaluar_incumbente", type=str, default=None,
                        help="JSON resultado para evaluar incumbente con más corridas")
    args = parser.parse_args()

    optimizar._no_plot = args.no_plot

    if args.mostrar_espacio:
        mostrar_espacio()
        sys.exit(0)

    if args.evaluar_incumbente:
        with open(args.evaluar_incumbente) as f:
            data = json.load(f)
        evaluar_incumbente(
            incumbente   = data.get("incumbente", {}),
            n_corridas   = max(10, args.n_corridas * 3),
            objetivo     = data.get("objetivo", "tts_full_days_mean"),
            guardar_json = "evaluacion_incumbente_m9.json",
            n_workers    = args.n_workers,
        )
        sys.exit(0)

    # Determinar modo
    use_revi       = not args.no_revi
    use_adaptativo = (not use_revi) and (not args.no_adaptativo)

    optimizar(
        tipo             = args.tipo,
        n_trials         = args.n_trials,
        n_corridas_eval  = args.n_corridas,
        seed             = args.seed,
        seed_base_modelo = args.seed_modelo,
        objetivo         = "tts_full_days_mean",
        output_dir       = args.output_dir,
        guardar_json     = args.guardar_json,
        min_budget       = args.min_budget,
        max_budget       = args.max_budget,
        n_workers        = args.n_workers,
        revi             = use_revi,
        adaptativo       = use_adaptativo,
        n_min_reps       = args.n_min_reps,
        n_max_reps       = args.n_max_reps,
        n_warmup         = args.n_warmup,
    )
