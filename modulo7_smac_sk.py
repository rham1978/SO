"""
===============================================================================
MÓDULO 7 — SMAC con Stochastic Kriging (SK) como Surrogate
===============================================================================
Basado en:
  · modulo4_smac.py        — estructura SMAC, builders, evaluar_incumbente
  · simulador_clinica_baseline.py — SimConfig real con parámetros y rangos

Diferencias vs módulo 4 (SMAC estándar):
  · Surrogate: Stochastic Kriging en lugar de Random Forest
  · El SK modela σ²(x) variable por punto (heteroscedasticidad)
  · Puntos ruidosos (sistema saturado) pesan menos en la predicción
  · Mantiene TODOS los parámetros, nombres y rangos del módulo 4

Parámetros optimizados (12 — igual que módulo 4):
  1.  horas_especialista_1ra   → fixed_weekly_capacity     [8, 30]
  2.  horas_control_post       → fixed_post_control_capacity [20, 70]
  3.  cupos_laboratorio_ugd    → ugd_lab_per_week           [20, 100]
  4.  cupos_ecografia_matrona  → mat_us_per_week            [10, 50]
  5.  cupos_ecografia_ugd      → ugd_us_per_week            [10, 50]
  6.  dias_publicacion         → publish_lead_workdays      [1, 10]
  7.  pct_bloqueo_1ra          → blocked_pct                [0.05, 0.50]
  8.  pct_consultas_vacias     → empty_control_p_ugd        [0.05, 0.50]
  9.  num_matronas             → matrona_capacity           [1, 4]
  10. num_agentes_ugd          → agent_capacity             [1, 4]
  11. pct_no_contactabilidad   → not_contactable_p          [0.05, 0.50]
  12. pct_bloqueo_post_control → blocked_pct_post_control   [0.05, 0.50]

Uso:
  python modulo7_smac_sk.py --tipo hpo_sk --n_trials 100 --n_corridas 3
  python modulo7_smac_sk.py --tipo hpo_sk --n_trials 50 --comparar_rf
  python modulo7_smac_sk.py --mostrar_espacio
  python modulo7_smac_sk.py --evaluar_incumbente resultado_smac_sk_hpo_sk.json
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
) -> float:
    """
    Función objetivo para SMAC+SK.
    Igual que módulo 4 pero ADEMÁS guarda la varianza muestral
    en _variance_store para que StochasticKrigingModel la use.
    """
    global _N_REPS_GLOBAL
    _, CFG_base, run_once, _, _ = _importar_baseline()
    cfg = _aplicar_config_a_cfg(config, CFG_base)

    n = max(1, int(round(budget))) if budget is not None else n_corridas
    _N_REPS_GLOBAL = n

    valores = []
    for r in range(n):
        try:
            res = run_once(seed_offset=seed_base + seed + r, cfg=cfg)
            if objetivo == "compuesto" and pesos_kpi:
                val = sum(float(pesos_kpi.get(k, 0.0)) * float(res.get(k, 0.0))
                          for k in pesos_kpi)
            else:
                val = float(res.get(objetivo, 1e9))
            valores.append(val)
        except Exception as e:
            log.warning("Error corrida %d: %s", r, e)
            valores.append(1e9)

    media = float(np.mean(valores))
    var   = float(np.var(valores, ddof=1)) if len(valores) > 1 else 1000.0

    # Guardar varianza — usada por StochasticKrigingModel._train()
    key = _config_key(config)
    _variance_store[key] = var
    _n_reps_store[key]   = len(valores)

    log.info("  Config eval → %s=%.2f  σ=%.2f  (n=%d)", objetivo, media, np.sqrt(var), n)
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
    "hpo_sk":      "HPO con Stochastic Kriging (recomendado)",
    "blackbox_sk": "Blackbox con Stochastic Kriging",
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


# ──────────────────────────────────────────────────────────────
# Función principal de optimización
# ──────────────────────────────────────────────────────────────
def optimizar(
    tipo:            str   = "hpo_sk",
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
) -> ResultadoSMAC_SK:
    """
    Ejecuta SMAC con el surrogate especificado.
    API idéntica al módulo 4.
    """
    if tipo not in TODOS_TIPOS:
        raise ValueError(f"Tipo '{tipo}' no válido. Opciones: {list(TODOS_TIPOS.keys())}")

    output_dir   = output_dir   or f"smac_sk_output_{tipo}"
    guardar_json = guardar_json or f"resultado_smac_sk_{tipo}.json"

    log.info("="*65)
    log.info("SMAC+SK — %s", TODOS_TIPOS[tipo])
    log.info("Objetivo  : %s", objetivo)
    log.info("n_trials  : %d  |  n_corridas: %d  |  seed: %d",
             n_trials, n_corridas_eval, seed)
    log.info("="*65)

    # Limpiar almacenes globales
    _variance_store.clear()
    _n_reps_store.clear()

    cs = crear_espacio_configuracion(seed=seed)

    _pesos = pesos_kpi or {}
    def fn_obj(config, seed: int = 0, budget: float = None) -> float:
        return evaluar_configuracion_sk(
            config      = config,
            seed        = seed,
            n_corridas  = n_corridas_eval,
            seed_base   = seed_base_modelo,
            objetivo    = objetivo,
            pesos_kpi   = _pesos,
            budget      = budget,
        )

    # Multi-fidelidad
    es_mf  = tipo == "multifidelity"
    min_b  = (min_budget or 1)              if es_mf else None
    max_b  = (max_budget or n_corridas_eval) if es_mf else None

    scenario = _scenario_base(cs, n_trials, output_dir, seed, min_b, max_b)

    # Construir SMAC con surrogate correcto
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

    # Optimizar
    t0 = time.time()
    log.info("Iniciando optimización...")
    incumbente_config = smac.optimize()
    t_total = time.time() - t0

    # Extraer resultados
    mejor_config = dict(incumbente_config)
    try:
        mejor_costo = float(smac.runhistory.get_cost(incumbente_config))
    except Exception:
        mejor_costo = float('inf')

    historia = []
    for trial_key, trial_val in smac.runhistory.items():
        try:
            historia.append({
                "config_id": trial_key.config_id,
                "seed":      trial_key.seed,
                "budget":    trial_key.budget,
                "costo":     round(float(trial_val.cost), 4),
            })
        except Exception:
            pass

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
    )

    out = asdict(resultado)
    with open(guardar_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log.info("Resultado guardado en '%s'", guardar_json)
    return resultado


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
) -> dict:
    """Evalúa incumbente con más corridas. API idéntica al módulo 4."""
    _, CFG_base, run_once, _, _ = _importar_baseline()

    class _FakeConfig(dict):
        def __getitem__(self, k): return super().__getitem__(k)
        def __contains__(self, k): return super().__contains__(k)

    cfg = _aplicar_config_a_cfg(_FakeConfig(incumbente), CFG_base)
    log.info("Evaluando incumbente con %d corridas...", n_corridas)

    valores = []
    for r in range(n_corridas):
        try:
            res = run_once(seed_offset=seed_base + r, cfg=cfg)
            valores.append(float(res.get(objetivo, 1e9)))
            log.info("  Corrida %d/%d: %s=%.2f", r+1, n_corridas, objetivo, valores[-1])
        except Exception as e:
            log.warning("Error corrida %d: %s", r, e)

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
# CLI — idéntico al módulo 4
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Módulo 7: SMAC + Stochastic Kriging — DES heteroscedástico"
    )
    parser.add_argument("--tipo", type=str, default="hpo_sk",
                        choices=list(TODOS_TIPOS.keys()),
                        help="Tipo SMAC (default: hpo_sk)")
    parser.add_argument("--n_trials",    type=int,   default=100)
    parser.add_argument("--n_corridas",  type=int,   default=3)
    parser.add_argument("--objetivo",    type=str,   default="tts_full_days_mean",
                        choices=["wl_total_end","wl_first_end",
                                 "tts_first_attended_days_mean",
                                 "tts_full_days_mean","compuesto"])
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--seed_modelo", type=int,   default=202)
    parser.add_argument("--output_dir",  type=str,   default=None)
    parser.add_argument("--guardar_json",type=str,   default=None)
    parser.add_argument("--min_budget",  type=float, default=None)
    parser.add_argument("--max_budget",  type=float, default=None)
    parser.add_argument("--comparar_rf", action="store_true",
                        help="Compara hpo_sk vs hpo(RF) con los mismos parámetros")
    parser.add_argument("--mostrar_espacio", action="store_true")
    parser.add_argument("--evaluar_incumbente", type=str, default=None,
                        help="JSON resultado para evaluar incumbente con más corridas")
    args = parser.parse_args()

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
            guardar_json = "evaluacion_incumbente_sk.json",
        )
        sys.exit(0)

    if args.comparar_rf:
        comparar_sk_vs_rf(
            n_trials        = args.n_trials,
            n_corridas_eval = args.n_corridas,
            seed            = args.seed,
            seed_base_modelo= args.seed_modelo,
            objetivo        = args.objetivo,
        )
        sys.exit(0)

    pesos = None
    if args.objetivo == "compuesto":
        pesos = {
            "tts_full_days_mean":           1.0 / 279.78,
            "tts_first_attended_days_mean": 0.5 / 193.85,
            "wl_total_end":                 0.3 / 1660.0,
        }

    optimizar(
        tipo             = args.tipo,
        n_trials         = args.n_trials,
        n_corridas_eval  = args.n_corridas,
        seed             = args.seed,
        seed_base_modelo = args.seed_modelo,
        objetivo         = args.objetivo,
        pesos_kpi        = pesos,
        output_dir       = args.output_dir,
        guardar_json     = args.guardar_json,
        min_budget       = args.min_budget,
        max_budget       = args.max_budget,
    )
