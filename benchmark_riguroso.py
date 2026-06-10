#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_riguroso.py
═══════════════════════════════════════════════════════════════════════
Orquestador unificado que conecta los módulos de SO-main con el plan
de experimentos de files/. Sustituye al comparativa original para el
experimento riguroso de IFORS.

Qué hace este archivo:
  1. Reutiliza _RUNNERS y _convergencia_* de modulo_comparativa_caja_negra.py
     (no los reimplementa).
  2. Agrega la capa de N_SEEDS macro-réplicas y presupuesto igualado que
     faltaba en el comparativa.
  3. Produce para CADA módulo:
       - resultado_{mod}_seed{s}.json   (corrida individual)
       - convergencia_{mod}.png         (curva con banda IC95)
  4. Produce para el conjunto:
       - consolidado_riguroso.json      (todas las corridas)
       - comparativa_rigurosa.png       (todas las curvas en un eje)
       - tabla_rigurosa.png             (costo final re-evaluado)
       - significancia.txt              (Wilcoxon pareado)

Conexión con los dos zips:
  ┌─ SO-main/ ─────────────────────────────────────────────────────┐
  │  simulador_clinica_baseline.py   ← motor DES (no se toca)      │
  │  modulo_comparativa_caja_negra.py ← _RUNNERS, _convergencia_*  │
  │  modulo4_smac_v2.py  … modulo14_aloe.py  ← algoritmos          │
  └────────────────────────────────────────────────────────────────┘
        ↑ importados directamente
  ┌─ files/ ───────────────────────────────────────────────────────┐
  │  benchmark_riguroso.py  (este archivo)                          │
  │  analisis_resultados.py ← post-proceso del consolidado          │
  └────────────────────────────────────────────────────────────────┘

Uso:
    # Corrida completa (recomendada):
    python benchmark_riguroso.py --modulos M4 M8 M10 M13 RS \
        --n_seeds 15 --n_trials 150 --n_cores 9 --out resultados/

    # Prueba rápida (verificar que todo corre):
    python benchmark_riguroso.py --modulos M4 M13 RS \
        --n_seeds 2 --n_trials 10 --n_cores 3 --out prueba/

    # Solo random search (valida la cadena de imports):
    python benchmark_riguroso.py --modulos RS \
        --n_seeds 3 --n_trials 20 --n_cores 3 --out prueba_rs/

Dependencias (instalar con pip):
    numpy scipy matplotlib

Hardware objetivo: Pelluhue (i9-10900K, 10 cores físicos).
Recomendado: --n_cores 9 (un core libre para SO y térmica).
"""

from __future__ import annotations
import argparse
import concurrent.futures
import json
import logging
import time
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

log = logging.getLogger("benchmark_riguroso")

# ───────────────────────────────────────────────────────────────────────
# Constantes globales
# ───────────────────────────────────────────────────────────────────────

# Colores/estilos por módulo (igual que en comparativa)
ESTILO = {
    "M4":  {"label": "M1 SMAC-GP+EI",    "color": "#1f77b4", "ls": "-",  "marker": "o"},
    "M7":  {"label": "M2 SMAC+SK (EI)",  "color": "#ff7f0e", "ls": "-",  "marker": "s"},
    "M8":  {"label": "M3 SK Adaptativo", "color": "#2ca02c", "ls": "--", "marker": "D"},
    "M9":  {"label": "SK-REVI",          "color": "#d62728", "ls": "--", "marker": "v"},
    "M10": {"label": "M4 SK-KGCP",       "color": "#9467bd", "ls": "-.", "marker": "^"},
    "M11": {"label": "M5 ASTRO-DF",      "color": "#8c564b", "ls": "-.", "marker": "p"},
    "M12": {"label": "STRONG",           "color": "#e377c2", "ls": ":",  "marker": "h"},
    "M13": {"label": "M6 SPSA",          "color": "#7f7f7f", "ls": ":",  "marker": "x"},
    "M14": {"label": "ALOE",             "color": "#bcbd22", "ls": (0, (3,1,1,1)), "marker": "*"},
    "RS":  {"label": "RS",               "color": "#17becf", "ls": "--", "marker": "P"},
}

# Parámetros del runner por familia
# n_trials: evaluciones del simulador a contar para presupuesto unificado
# evals_por_config: cuántas corridas del simulador por punto para M4-M10
# evals_por_iter: para M11-M14, cuántas evaluaciones por iteración (corregido)
FAMILIA_SMAC = {"M4", "M7", "M8", "M9", "M10"}
FAMILIA_ITER = {"M11", "M12", "M13", "M14"}


def _worker_run_once(args):
    """Module-level worker for ProcessPoolExecutor (must be picklable)."""
    seed_offset, cfg_dict, objetivo, pesos_kpi = args
    from simulador_clinica_baseline import run_once, SimConfig
    import dataclasses
    cfg = SimConfig(**{k: v for k, v in cfg_dict.items()
                       if k in SimConfig.__dataclass_fields__})
    res = run_once(seed_offset=seed_offset, cfg=cfg)
    if objetivo == "compuesto" and pesos_kpi:
        return sum(float(pesos_kpi.get(k, 0.0)) * float(res.get(k, 0.0))
                   for k in pesos_kpi)
    return float(res.get(objetivo, 1e9))


# ───────────────────────────────────────────────────────────────────────
# Random Search (baseline) — implementado aquí, no depende de main
# ───────────────────────────────────────────────────────────────────────

def _random_search_runner(n_trials: int, seed: int, n_reps: int = 3,
                           pesos_kpi: dict = None) -> dict:
    """
    Evalúa n_trials puntos aleatorios en el espacio de parámetros.
    Usa la infraestructura de main (PARAM_NAMES, PARAM_RANGES, run_once, SimConfig).
    """
    import modulo_comparativa_caja_negra as comp
    from simulador_clinica_baseline import SimConfig, CFG
    import dataclasses

    PARAM_NAMES = comp.PARAM_NAMES
    PARAM_RANGES = comp.PARAM_RANGES
    ENTEROS = {
        "horas_especialista_1ra", "horas_control_post", "cupos_laboratorio_ugd",
        "cupos_ecografia_matrona", "cupos_ecografia_ugd", "dias_publicacion",
        "num_matronas", "num_agentes_ugd",
    }

    objetivo_kpi = "compuesto" if pesos_kpi else "tts_full_days_mean"
    kpi_dict = pesos_kpi or {}

    rng = np.random.RandomState(seed)
    historia = []
    mejor = float("inf")
    mejor_cfg = None
    t0 = time.time()
    seed_off = seed * 10_000  # bloque de offsets exclusivo para esta seed

    for trial in range(n_trials):
        # Generar punto aleatorio
        cfg = dataclasses.replace(CFG)
        vals = {}
        for nombre in PARAM_NAMES:
            lo, hi = PARAM_RANGES[nombre]
            v = lo + rng.random() * (hi - lo)
            if nombre in ENTEROS:
                v = int(round(v))
            vals[nombre] = v

        # Aplicar al cfg (mapeo nombre → atributo SimConfig)
        cfg.fixed_weekly_capacity        = int(vals.get("horas_especialista_1ra", 16))
        cfg.use_fixed_weekly_capacity    = True
        cfg.fixed_post_control_capacity  = int(vals.get("horas_control_post", 40))
        cfg.use_fixed_post_control_hours = True
        cfg.ugd_lab_per_week             = int(vals.get("cupos_laboratorio_ugd", 54))
        cfg.mat_us_per_week              = int(vals.get("cupos_ecografia_matrona", 25))
        cfg.ugd_us_per_week              = int(vals.get("cupos_ecografia_ugd", 25))
        cfg.publish_lead_workdays        = int(vals.get("dias_publicacion", 5))
        cfg.blocked_pct                  = float(vals.get("pct_bloqueo_1ra", 0.32))
        cfg.empty_control_p_ugd          = float(vals.get("pct_consultas_vacias", 0.30))
        cfg.matrona_capacity             = int(vals.get("num_matronas", 1))
        cfg.agent_capacity               = int(vals.get("num_agentes_ugd", 1))
        cfg.not_contactable_p            = float(vals.get("pct_no_contactabilidad", 0.30))
        cfg.blocked_pct_post_control     = float(vals.get("pct_bloqueo_post_control", 0.34))

        # Evaluar con n_reps réplicas (CRN por seed) — timeout anti-deadlock DES
        resultados_rep = []
        for r in range(n_reps):
            try:
                with concurrent.futures.ProcessPoolExecutor(max_workers=1) as _ex:
                    _fut = _ex.submit(
                        _worker_run_once,
                        (seed_off + trial * n_reps + r,
                         dataclasses.asdict(cfg), objetivo_kpi, kpi_dict)
                    )
                    res_val = _fut.result(timeout=600.0)
                resultados_rep.append(float(res_val))
            except concurrent.futures.TimeoutError:
                log.warning("RS trial=%d rep=%d descartada por timeout (600s)", trial, r)
            except Exception as e:
                log.warning("RS trial=%d rep=%d error: %s", trial, r, e)

        if not resultados_rep:
            continue

        costo = float(np.mean(resultados_rep))
        if costo < mejor:
            mejor = costo
            mejor_cfg = vals.copy()

        historia.append({
            "iter": trial,
            "costo": round(costo, 4),
            "n_reps": n_reps,
            "mejor_hasta_ahora": round(mejor, 4),
            "t_seg": round(time.time() - t0, 2),
        })

    return {
        "modulo": "RS",
        "algoritmo": "Random Search",
        "costo_incumbente": round(mejor, 4),
        "tiempo_seg": round(time.time() - t0, 2),
        "n_evaluaciones": n_trials * n_reps,
        "incumbente": mejor_cfg or {},
        "historia_costos": historia,
        "seed": seed,
    }


# ───────────────────────────────────────────────────────────────────────
# Runner unificado por módulo y seed
# ───────────────────────────────────────────────────────────────────────

def _params_para_runner(modulo: str, n_trials: int, seed: int) -> dict:
    """
    Traduce n_trials (presupuesto unificado en evaluaciones del simulador)
    a los parámetros nativos de cada runner de modulo_comparativa_caja_negra.

    SMAC (M4-M10): n_trials configs × n_corridas réplicas/config.
    Iterativos (M11-M14): max_iter con evals_por_iter corregido.
    """
    if modulo in FAMILIA_SMAC:
        n_corridas = 3   # réplicas fijas para balance calidad/costo
        n_smac = max(5, n_trials // n_corridas)
        return {"n_trials": n_smac, "n_corridas": n_corridas, "seed": seed}

    elif modulo == "M11":  # ASTRO-DF: ~30 reps/iter (fijado en el runner)
        max_iter = max(1, n_trials // 30)
        return {"max_iter": max_iter, "n_workers": 0, "seed": seed}

    elif modulo == "M12":  # STRONG: 100 eval solo para arrancar
        max_iter = max(1, n_trials // 100)
        return {"max_iter": max_iter, "n_workers": 0, "seed": seed}

    elif modulo == "M13":  # SPSA: 3 eval/iter (bug documentado; n_reps=5)
        n_reps = 5
        max_iter = max(1, n_trials // (3 * n_reps))
        return {"max_iter": max_iter, "n_reps": n_reps, "n_workers": 0, "seed": seed}

    elif modulo == "M14":  # ALOE: 48 grad + 60 arm = 108/iter
        r = 5
        max_iter = max(1, n_trials // 108)
        return {"max_iter": max_iter, "r": r, "n_workers": 0, "seed": seed}

    return {}


def _evals_por_iter(modulo: str, params: dict) -> int:
    """Evaluaciones reales del simulador por iteración/trial (para eje X)."""
    if modulo in FAMILIA_SMAC:
        return params.get("n_corridas", 3)
    elif modulo == "M11":
        return 30
    elif modulo == "M12":
        return 100
    elif modulo == "M13":
        return 3 * params.get("n_reps", 5)   # corregido: 3 eval/iter
    elif modulo == "M14":
        _d = 12
        return 2 * _d * 2 + 2 * params.get("r", 5)  # grad + armijo
    elif modulo == "RS":
        return params.get("n_reps", 3)
    return 1


def _re_evaluar_incumbente(incumbente_cfg: dict, r_final: int,
                            seed_offset_base: int = 500_000,
                            pesos_kpi: dict = None) -> dict:
    """
    Re-evalúa la configuración incumbente con r_final réplicas frescas.
    Usa offsets en un bloque separado para no solapar con la optimización.
    """
    from simulador_clinica_baseline import run_once, SimConfig, CFG
    import dataclasses

    try:
        cfg = dataclasses.replace(CFG)
        cfg.fixed_weekly_capacity        = int(round(incumbente_cfg.get("horas_especialista_1ra", 16)))
        cfg.use_fixed_weekly_capacity    = True
        cfg.fixed_post_control_capacity  = int(round(incumbente_cfg.get("horas_control_post", 40)))
        cfg.use_fixed_post_control_hours = True
        cfg.ugd_lab_per_week             = int(round(incumbente_cfg.get("cupos_laboratorio_ugd", 54)))
        cfg.mat_us_per_week              = int(round(incumbente_cfg.get("cupos_ecografia_matrona", 25)))
        cfg.ugd_us_per_week              = int(round(incumbente_cfg.get("cupos_ecografia_ugd", 25)))
        cfg.publish_lead_workdays        = int(round(incumbente_cfg.get("dias_publicacion", 5)))
        cfg.blocked_pct                  = float(incumbente_cfg.get("pct_bloqueo_1ra", 0.32))
        cfg.empty_control_p_ugd          = float(incumbente_cfg.get("pct_consultas_vacias", 0.30))
        cfg.matrona_capacity             = int(round(incumbente_cfg.get("num_matronas", 1)))
        cfg.agent_capacity               = int(round(incumbente_cfg.get("num_agentes_ugd", 1)))
        cfg.not_contactable_p            = float(incumbente_cfg.get("pct_no_contactabilidad", 0.30))
        cfg.blocked_pct_post_control     = float(incumbente_cfg.get("pct_bloqueo_post_control", 0.34))

        if pesos_kpi:
            vals = []
            for r in range(r_final):
                res = run_once(seed_offset=seed_offset_base + r, cfg=cfg)
                vals.append(sum(float(pesos_kpi.get(k, 0.0)) * float(res.get(k, 0.0))
                                for k in pesos_kpi))
        else:
            vals = [float(run_once(seed_offset=seed_offset_base + r, cfg=cfg)["tts_full_days_mean"])
                    for r in range(r_final)]
        arr = np.array(vals)
        return {
            "media":   float(arr.mean()),
            "sd":      float(arr.std(ddof=1)),
            "ic95_lo": float(arr.mean() - 1.96 * arr.std(ddof=1) / np.sqrt(r_final)),
            "ic95_hi": float(arr.mean() + 1.96 * arr.std(ddof=1) / np.sqrt(r_final)),
            "r_final": r_final,
        }
    except Exception as e:
        log.error("Re-evaluación falló: %s", e)
        return {"media": float("nan"), "sd": float("nan"),
                "ic95_lo": float("nan"), "ic95_hi": float("nan"), "r_final": 0}


def _correr_una(modulo: str, seed: int, n_trials: int, r_final: int,
                out_dir: Path, resume: bool = False,
                pesos_kpi: dict = None) -> dict:
    """
    Ejecuta un módulo con una macro-seed. Guarda JSON individual.
    Si resume=True y el JSON ya existe, lo carga sin re-ejecutar.
    Esta función corre en un proceso separado.
    """
    f_json = out_dir / f"resultado_{modulo}_seed{seed:02d}.json"
    if resume and f_json.exists():
        try:
            registro = json.loads(f_json.read_text())
            log.info("↩ %s seed=%02d cargado desde JSON (resume)", modulo, seed)
            return registro
        except Exception as e:
            log.warning("No se pudo cargar %s: %s — se re-ejecuta", f_json, e)

    t0 = time.time()
    resultado_bruto = None

    try:
        if modulo == "RS":
            resultado_bruto = _random_search_runner(n_trials=n_trials, seed=seed,
                                                    pesos_kpi=pesos_kpi)
            historia = resultado_bruto["historia_costos"]
            import modulo_comparativa_caja_negra as comp
            conv_eval = comp._convergencia_iterativa(historia)
            conv_time = comp._tiempos_convergencia(historia)

        elif modulo in FAMILIA_SMAC:
            import modulo_comparativa_caja_negra as comp
            params = _params_para_runner(modulo, n_trials, seed)
            params["pesos_kpi"] = pesos_kpi
            resultado_bruto = comp._RUNNERS[modulo](**params)
            conv_eval = resultado_bruto.get("conv_eval", [])
            conv_time = resultado_bruto.get("conv_time", [])
            historia  = resultado_bruto  # el runner ya retorna el dict

        elif modulo in FAMILIA_ITER:
            import modulo_comparativa_caja_negra as comp
            params = _params_para_runner(modulo, n_trials, seed)
            params["pesos_kpi"] = pesos_kpi
            resultado_bruto = comp._RUNNERS[modulo](**params)
            conv_eval = resultado_bruto.get("conv_eval", [])
            conv_time = resultado_bruto.get("conv_time", [])

        else:
            raise ValueError(f"Módulo '{modulo}' no reconocido.")

        # Re-evaluación honesta del incumbente
        incumbente_cfg = resultado_bruto.get("incumbente", {})
        if isinstance(incumbente_cfg, dict) and incumbente_cfg:
            reeval = _re_evaluar_incumbente(
                incumbente_cfg, r_final,
                seed_offset_base=500_000 + seed * r_final,
                pesos_kpi=pesos_kpi,
            )
        else:
            reeval = {"media": float("nan"), "sd": float("nan"),
                      "ic95_lo": float("nan"), "ic95_hi": float("nan"), "r_final": 0}

        registro = {
            "modulo":          modulo,
            "macro_seed":      seed,
            "n_trials_param":  n_trials,
            "n_eval_usadas":   resultado_bruto.get("n_evaluaciones",
                               len(resultado_bruto.get("historia_costos", [])) * _evals_por_iter(modulo, {})),
            "costo_opt":       resultado_bruto.get("costo_incumbente", float("nan")),
            "reeval":          reeval,
            "conv_eval":       conv_eval,
            "conv_time":       conv_time,
            "incumbente":      incumbente_cfg,
            "tiempo_seg":      time.time() - t0,
        }

        # Guardar JSON individual
        f = out_dir / f"resultado_{modulo}_seed{seed:02d}.json"
        f.write_text(json.dumps(registro, indent=2, default=str))
        return registro

    except Exception as exc:
        log.error("FALLO %s seed=%d: %s", modulo, seed, exc, exc_info=True)
        return {
            "modulo": modulo, "macro_seed": seed,
            "error": str(exc), "tiempo_seg": time.time() - t0,
        }


# ───────────────────────────────────────────────────────────────────────
# Gráficas
# ───────────────────────────────────────────────────────────────────────

def _graficar_convergencia_modulo(modulo: str, corridas: list[dict],
                                   out_dir: Path, baseline: float = 270.0) -> None:
    """Curva de convergencia individual: media ± IC95 sobre seeds."""
    fig, ax = plt.subplots(figsize=(9, 5))

    # Recolectar curvas de cada seed
    curvas = []
    max_eval = 0
    for r in corridas:
        cv = r.get("conv_eval", [])
        if not cv:
            continue
        evs  = np.array([p[0] for p in cv])
        best = np.array([p[1] for p in cv])
        curvas.append((evs, best))
        if len(evs) > 0:
            max_eval = max(max_eval, int(evs[-1]))

    if not curvas or max_eval == 0:
        plt.close(fig)
        return

    grid = np.arange(1, max_eval + 1)
    M = []
    for evs, best in curvas:
        y = np.full(len(grid), np.nan)
        for i, g in enumerate(grid):
            idx = np.searchsorted(evs, g, side="right") - 1
            if idx >= 0:
                y[i] = best[idx]
        M.append(y)

    M    = np.vstack(M)
    med  = np.nanmean(M, axis=0)
    n    = np.sum(~np.isnan(M), axis=0)
    sd   = np.nanstd(M, axis=0, ddof=1)
    ic   = 1.96 * sd / np.sqrt(np.maximum(n, 1))
    est  = ESTILO.get(modulo, {"label": modulo, "color": "#333333", "ls": "-"})

    ax.plot(grid, med, color=est["color"], lw=2.5, label=f"{est['label']} (n={len(curvas)})")
    ax.fill_between(grid, med - ic, med + ic, color=est["color"], alpha=0.2)
    ax.axhline(baseline, ls="--", color="red", alpha=0.6, label=f"Baseline ({baseline:.0f} d)")

    ax.set_xlabel("Evaluaciones del simulador (acumuladas)")
    ax.set_ylabel("Mejor costo incumbente — TTS total (días)")
    ax.set_title(f"{est['label']} — Convergencia\n(media ± IC95, {len(curvas)} seeds, CRN)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"convergencia_{modulo}.png", dpi=150)
    plt.close(fig)
    log.info("Gráfica: convergencia_%s.png", modulo)


def _graficar_comparativa(grupos: dict[str, list[dict]], out_dir: Path,
                           baseline: float = 270.0) -> None:
    """Todas las curvas en un solo eje a presupuesto igualado."""
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axhline(baseline, ls="--", color="red", alpha=0.5, lw=1.5,
               label=f"Baseline ({baseline:.0f} d)")

    for modulo, corridas in sorted(grupos.items()):
        curvas = []
        max_eval = 0
        for r in corridas:
            cv = r.get("conv_eval", [])
            if not cv:
                continue
            evs  = np.array([p[0] for p in cv])
            best = np.array([p[1] for p in cv])
            curvas.append((evs, best))
            if len(evs):
                max_eval = max(max_eval, int(evs[-1]))

        if not curvas or max_eval == 0:
            continue

        grid = np.arange(1, max_eval + 1)
        M = []
        for evs, best in curvas:
            y = np.full(len(grid), np.nan)
            for i, g in enumerate(grid):
                idx = np.searchsorted(evs, g, side="right") - 1
                if idx >= 0:
                    y[i] = best[idx]
            M.append(y)

        M   = np.vstack(M)
        med = np.nanmean(M, axis=0)
        n   = np.sum(~np.isnan(M), axis=0)
        sd  = np.nanstd(M, axis=0, ddof=1)
        ic  = 1.96 * sd / np.sqrt(np.maximum(n, 1))
        est = ESTILO.get(modulo, {"label": modulo, "color": "#555555", "ls": "-"})

        ax.plot(grid, med, color=est["color"], ls=est.get("ls", "-"),
                lw=2, label=f"{est['label']} (n={len(curvas)})")
        ax.fill_between(grid, med - ic, med + ic, color=est["color"], alpha=0.12)

    ax.set_xlabel("Evaluaciones del simulador (acumuladas)")
    ax.set_ylabel("Mejor costo incumbente — TTS total (días)")
    ax.set_title("Comparativa rigurosa: convergencia a presupuesto igualado\n(media ± IC95 sobre macro-réplicas, CRN)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "comparativa_rigurosa.png", dpi=150)
    plt.close(fig)
    log.info("Gráfica: comparativa_rigurosa.png")


def _graficar_boxplot(grupos: dict[str, list[dict]], out_dir: Path,
                      baseline: float = 270.0) -> None:
    """Boxplot de costo final re-evaluado por módulo."""
    metodos = sorted(grupos.keys())
    data    = []
    labels  = []
    for m in metodos:
        vals = [r["reeval"]["media"] for r in grupos[m]
                if "reeval" in r and r["reeval"].get("media") == r["reeval"].get("media")]
        if vals:
            data.append(vals)
            labels.append(ESTILO.get(m, {}).get("label", m))

    if not data:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(data, tick_labels=labels if hasattr(ax.boxplot, '__self__') else None,
                    showmeans=True, patch_artist=True)
    # fallback para mpl < 3.9
    try:
        ax.set_xticklabels(labels, rotation=30, ha="right")
    except Exception:
        pass

    for patch, m in zip(bp["boxes"], metodos):
        patch.set_facecolor(ESTILO.get(m, {}).get("color", "#999999"))
        patch.set_alpha(0.6)

    ax.axhline(baseline, ls="--", color="red", alpha=0.6,
               label=f"Baseline ({baseline:.0f} d)")
    ax.set_ylabel("Costo final re-evaluado — TTS total (días)")
    ax.set_title("Distribución del costo final por método\n(incumbente re-evaluado con réplicas frescas)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "boxplot_final.png", dpi=150)
    plt.close(fig)
    log.info("Gráfica: boxplot_final.png")


def _tabla_significancia(grupos: dict[str, list[dict]], out_dir: Path) -> str:
    """Wilcoxon pareado por seed (válido con CRN)."""
    metodos = sorted(grupos.keys())
    por_seed = {
        m: {r["macro_seed"]: r["reeval"]["media"]
            for r in grupos[m] if "reeval" in r}
        for m in metodos
    }
    lineas = ["# Test de Wilcoxon pareado por seed (CRN garantiza pareo)\n"]
    lineas.append(f"{'Comparación':<26}{'Δ medio (d)':>14}{'p-valor':>12}{'Significativo':>16}")
    lineas.append("-" * 68)

    for i, a in enumerate(metodos):
        for b in metodos[i+1:]:
            seeds = sorted(set(por_seed[a]) & set(por_seed[b]))
            xa = np.array([por_seed[a][s] for s in seeds])
            xb = np.array([por_seed[b][s] for s in seeds])
            diff = float(np.mean(xa - xb))
            if _HAY_SCIPY and len(seeds) >= 6:
                try:
                    _, p = sps.wilcoxon(xa, xb)
                except ValueError:
                    p = float("nan")
            else:
                p = float("nan")
            sig = "sí (p<0.05)" if (p == p and p < 0.05) else ("no" if p == p else "n/d")
            lineas.append(f"{a + ' vs ' + b:<26}{diff:>14.2f}{p:>12.4f}{sig:>16}")

    txt = "\n".join(lineas)
    (out_dir / "significancia.txt").write_text(txt)
    return txt


def _tabla_resumen(grupos: dict[str, list[dict]], out_dir: Path,
                   baseline: float = 270.0) -> str:
    """Tabla de texto con costo medio re-evaluado e IC95."""
    lineas = [f"# Tabla resumen — costo final re-evaluado (baseline={baseline:.0f} d)\n"]
    lineas.append(f"{'Método':<14}{'Costo medio':>14}{'IC95 lo':>10}{'IC95 hi':>10}{'vs base %':>12}{'n seeds':>9}")
    lineas.append("-" * 69)

    for m in sorted(grupos.keys()):
        reevals = [r["reeval"]["media"] for r in grupos[m]
                   if "reeval" in r and r["reeval"].get("media") == r["reeval"].get("media")]
        lo_vals = [r["reeval"]["ic95_lo"] for r in grupos[m] if "reeval" in r]
        hi_vals = [r["reeval"]["ic95_hi"] for r in grupos[m] if "reeval" in r]
        if not reevals:
            continue
        media = float(np.mean(reevals))
        lo    = float(np.mean(lo_vals)) if lo_vals else float("nan")
        hi    = float(np.mean(hi_vals)) if hi_vals else float("nan")
        pct   = (baseline - media) / baseline * 100
        label = ESTILO.get(m, {}).get("label", m)
        lineas.append(f"{label:<14}{media:>14.2f}{lo:>10.2f}{hi:>10.2f}{pct:>12.1f}%{len(reevals):>9}")

    txt = "\n".join(lineas)
    (out_dir / "tabla_resumen.txt").write_text(txt)
    return txt


# ───────────────────────────────────────────────────────────────────────
# Orquestador principal
# ───────────────────────────────────────────────────────────────────────

def ejecutar(modulos: list[str], n_seeds: int, n_trials: int,
             r_final: int, n_cores: int, out_dir: Path,
             baseline: float = 270.0, resume: bool = False,
             pesos_kpi: dict = None) -> None:

    out_dir.mkdir(parents=True, exist_ok=True)
    tareas = [(m, s) for m in modulos for s in range(n_seeds)]

    if resume:
        ya_hechas = sum(
            1 for m, s in tareas
            if (out_dir / f"resultado_{m}_seed{s:02d}.json").exists()
        )
        log.info("Modo resume: %d/%d corridas ya completadas en disco",
                 ya_hechas, len(tareas))

    t0 = time.time()
    log.info("=" * 65)
    log.info("Benchmark riguroso IFORS%s", " (RESUME)" if resume else "")
    log.info("Módulos: %s", modulos)
    log.info("Seeds: %d  |  n_trials: %d  |  r_final: %d  |  cores: %d",
             n_seeds, n_trials, r_final, n_cores)
    if pesos_kpi:
        log.info("Objetivo compuesto: %s", pesos_kpi)
    log.info("Total de corridas: %d", len(tareas))
    log.info("Salida: %s", out_dir)
    log.info("=" * 65)

    resultados = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as ex:
        futuros = {
            ex.submit(_correr_una, m, s, n_trials, r_final, out_dir, resume, pesos_kpi): (m, s)
            for m, s in tareas
        }
        for i, fut in enumerate(concurrent.futures.as_completed(futuros), 1):
            m, s = futuros[fut]
            try:
                r = fut.result()
                resultados.append(r)
                reeval_med = r.get("reeval", {}).get("media", float("nan"))
                log.info("[%d/%d] %-10s seed=%2d  costo_opt=%.2f  reeval=%.2f  t=%.0fs",
                         i, len(tareas), m, s,
                         r.get("costo_opt", float("nan")),
                         reeval_med, r.get("tiempo_seg", 0))
            except Exception as exc:
                log.error("[%d/%d] %s seed=%d FALLÓ: %s", i, len(tareas), m, s, exc)

    # Guardar consolidado
    consolidado = out_dir / "consolidado_riguroso.json"
    consolidado.write_text(json.dumps(resultados, indent=2, default=str))
    log.info("Consolidado guardado: %s", consolidado)

    # Agrupar por módulo y graficar
    grupos: dict[str, list[dict]] = {}
    for r in resultados:
        if "error" not in r:
            grupos.setdefault(r["modulo"], []).append(r)

    for modulo, corridas in grupos.items():
        _graficar_convergencia_modulo(modulo, corridas, out_dir, baseline)

    _graficar_comparativa(grupos, out_dir, baseline)
    _graficar_boxplot(grupos, out_dir, baseline)

    txt_sig   = _tabla_significancia(grupos, out_dir)
    txt_tabla = _tabla_resumen(grupos, out_dir, baseline)

    log.info("\n%s", txt_tabla)
    log.info("\n%s", txt_sig)
    log.info("=" * 65)
    log.info("Completado en %.1f h  |  %d corridas OK de %d",
             (time.time() - t0) / 3600, len(grupos and resultados), len(tareas))
    log.info("Archivos en: %s", out_dir)


def main():
    p = argparse.ArgumentParser(
        description="Benchmark riguroso SO-main + files (IFORS julio 2026).")
    p.add_argument("--modulos", nargs="+",
                   default=["M4", "M7", "M8", "M10", "M11", "M13", "RS"],
                   help="Módulos a correr (M4 M7 M8 M9 M10 M11 M12 M13 M14 RS).")
    p.add_argument("--n_seeds",  type=int, default=15,
                   help="Macro-réplicas por módulo (default 15).")
    p.add_argument("--n_trials", type=int, default=150,
                   help="Presupuesto de evaluaciones del simulador (default 150).")
    p.add_argument("--r_final",  type=int, default=30,
                   help="Réplicas para re-evaluar incumbente (default 30).")
    p.add_argument("--n_cores",  type=int, default=9,
                   help="Procesos paralelos (default 9 = cores físicos - 1).")
    p.add_argument("--baseline", type=float, default=270.0,
                   help="Valor baseline del DES (default 270 días).")
    p.add_argument("--out",      default="resultados_rigurosos",
                   help="Directorio de salida.")
    p.add_argument("--resume",   action="store_true",
                   help="Reanuda corrida: carga JSONs existentes y salta corridas ya completadas.")
    p.add_argument("--lambda_obj", type=float, default=None,
                   help="Valor λ para objetivo compuesto f=tts_full - λ·total_atenciones "
                        "(obtenido con calibrar_lambda.py). Si se omite usa sólo TTS.")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    pesos_kpi = None
    if args.lambda_obj is not None:
        pesos_kpi = {"tts_full_days_mean": 1.0, "total_atenciones": -args.lambda_obj}
        log.info("Objetivo compuesto activo: f = tts_full - %.4f · total_atenciones",
                 args.lambda_obj)

    ejecutar(
        modulos   = args.modulos,
        n_seeds   = args.n_seeds,
        n_trials  = args.n_trials,
        r_final   = args.r_final,
        n_cores   = args.n_cores,
        out_dir   = Path(args.out),
        baseline  = args.baseline,
        resume    = args.resume,
        pesos_kpi = pesos_kpi,
    )


if __name__ == "__main__":
    main()
