#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
correr_astrodf_rs.py
════════════════════════════════════════════════════════════════════════
Corre ASTRO-DF (M11) y Random Search (RS) con la MISMA configuración
que el resto de los algoritmos del benchmark IFORS:

  · 15 macro-réplicas (seeds 0-14)
  · B_SIM = 150 evaluaciones del simulador por corrida
  · r_final = 50 réplicas para re-evaluar el incumbente
  · Formato de salida idéntico a benchmark_riguroso.py

Fixes de deadlock vs. corridas anteriores:
  1. multiprocessing.set_start_method('spawn') — evita deadlocks en Linux
     cuando hay ProcessPoolExecutors anidados (fork hereda locks).
  2. M11 corre con n_workers=1 (secuencial internamente); la paralelización
     entre seeds la maneja este script con n_cores procesos independientes.

Uso:
    # Desde el directorio raíz del repositorio:
    python correr_astrodf_rs.py                      # RS + M11, 15 seeds, 9 cores
    python correr_astrodf_rs.py --modulos RS          # solo RS
    python correr_astrodf_rs.py --modulos M11         # solo M11
    python correr_astrodf_rs.py --n_cores 4           # limitar cores
    python correr_astrodf_rs.py --out mis_resultados/ # directorio de salida

Tiempo estimado (@ 224s por simulación, 4 cores):
    RS  : 15 seeds × 150 evals × 224s / 4 cores ≈ 35 h pared
    M11 : 15 seeds × 5 iter × ~30 reps × 224s / 4 cores ≈ 14 h pared
    (RS es el bottleneck — paralelizar entre seeds baja a ~9 h pared con 9 cores)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

# ── Asegurarse de que el directorio del repositorio esté en sys.path ──────────
_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

log = logging.getLogger("correr_astrodf_rs")

# ───────────────────────────────────────────────────────────────────────
# Parámetros del experimento (DEBEN coincidir con el resto del benchmark)
# ───────────────────────────────────────────────────────────────────────
N_SEEDS  = 15
B_SIM    = 150   # presupuesto de evaluaciones del simulador por corrida
R_FINAL  = 50    # réplicas para re-evaluar incumbente


# ═══════════════════════════════════════════════════════════════════════
# Helpers reutilizados de benchmark_riguroso
# ═══════════════════════════════════════════════════════════════════════

def _worker_run_once(args):
    """Worker de ProcessPoolExecutor — debe ser picklable (definido a nivel módulo)."""
    seed_offset, cfg_dict, objetivo = args
    from simulador_clinica_baseline import run_once, SimConfig
    cfg = SimConfig(**{k: v for k, v in cfg_dict.items()
                       if k in SimConfig.__dataclass_fields__})
    res = run_once(seed_offset=seed_offset, cfg=cfg)
    return float(res.get(objetivo, 1e9))


def _re_evaluar_incumbente(incumbente_cfg: dict, r_final: int,
                            seed_offset_base: int = 500_000) -> tuple:
    """Re-evalúa el incumbente con r_final réplicas frescas (sin solapar optimization)."""
    from simulador_clinica_baseline import run_once, SimConfig, CFG

    nan_reeval = {"media": float("nan"), "sd": float("nan"),
                  "ic95_lo": float("nan"), "ic95_hi": float("nan"), "r_final": 0}
    nan_kpis   = {k: float("nan") for k in [
        "tts_media", "tts_sd", "tts_ic95_lo", "tts_ic95_hi",
        "at_media", "at_sd", "at_ic95_lo", "at_ic95_hi",
        "at_first_media", "at_post_media"]}

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
        cfg.benchmark_mode               = True

        obj_vals  = []
        tts_vals  = []
        at_vals   = []
        at_f_vals = []
        at_p_vals = []

        for r in range(r_final):
            res  = run_once(seed_offset=seed_offset_base + r, cfg=cfg)
            tts  = float(res.get("tts_full_days_mean",          float("nan")))
            at   = float(res.get("total_atenciones",            float("nan")))
            at_f = float(res.get("total_atenciones_first",      float("nan")))
            at_p = float(res.get("total_atenciones_post",       float("nan")))
            tts_vals.append(tts)
            at_vals.append(at)
            at_f_vals.append(at_f)
            at_p_vals.append(at_p)
            obj_vals.append(tts)

        def _stats(arr_list, r_final=r_final):
            a = np.array(arr_list, dtype=float)
            n = len(a)
            return {"media":    float(a.mean()),
                    "sd":       float(a.std(ddof=1)),
                    "ic95_lo":  float(a.mean() - 1.96 * a.std(ddof=1) / np.sqrt(n)),
                    "ic95_hi":  float(a.mean() + 1.96 * a.std(ddof=1) / np.sqrt(n)),
                    "r_final":  r_final}

        obj_s = _stats(obj_vals)
        tts_s = _stats(tts_vals)
        at_s  = _stats(at_vals)

        reeval = obj_s
        kpis   = {
            "tts_media":   tts_s["media"],  "tts_sd":      tts_s["sd"],
            "tts_ic95_lo": tts_s["ic95_lo"],"tts_ic95_hi": tts_s["ic95_hi"],
            "at_media":    at_s["media"],   "at_sd":       at_s["sd"],
            "at_ic95_lo":  at_s["ic95_lo"], "at_ic95_hi":  at_s["ic95_hi"],
            "at_first_media": float(np.nanmean(at_f_vals)),
            "at_post_media":  float(np.nanmean(at_p_vals)),
        }
        return reeval, kpis

    except Exception as e:
        log.error("_re_evaluar_incumbente error: %s", e)
        return nan_reeval, nan_kpis


# ═══════════════════════════════════════════════════════════════════════
# Runner de Random Search
# ═══════════════════════════════════════════════════════════════════════

def _convergencia_iterativa(historia: list[dict]) -> list:
    """Convierte historia a [(n_eval_acum, mejor_hasta_ahora), ...]."""
    curva = []
    acum  = 0
    for h in historia:
        n = h.get("n_reps", 3)
        acum += n
        curva.append([acum, h.get("mejor_hasta_ahora", float("nan"))])
    return curva


def _correr_rs(macro_seed: int) -> dict:
    """Corre Random Search para una macro-seed."""
    import modulo_comparativa_caja_negra as comp
    from simulador_clinica_baseline import SimConfig, CFG

    PARAM_NAMES  = comp.PARAM_NAMES
    PARAM_RANGES = comp.PARAM_RANGES
    ENTEROS = {
        "horas_especialista_1ra", "horas_control_post", "cupos_laboratorio_ugd",
        "cupos_ecografia_matrona", "cupos_ecografia_ugd", "dias_publicacion",
        "num_matronas", "num_agentes_ugd",
    }

    objetivo = "tts_full_days_mean"
    n_reps   = 3
    n_trials = B_SIM // n_reps    # 50 puntos × 3 réplicas = 150 evaluaciones

    rng      = np.random.RandomState(macro_seed)
    historia = []
    mejor    = float("inf")
    mejor_cfg = None
    t0       = time.time()
    seed_off = macro_seed * 10_000

    for trial in range(n_trials):
        cfg = dataclasses.replace(CFG)
        vals = {}
        for nombre in PARAM_NAMES:
            lo, hi = PARAM_RANGES[nombre]
            v = lo + rng.random() * (hi - lo)
            if nombre in ENTEROS:
                v = int(round(v))
            vals[nombre] = v

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
        cfg.benchmark_mode               = True

        cfg_dict = dataclasses.asdict(cfg)
        resultados_rep = []

        for r in range(n_reps):
            try:
                # Cada llamada al simulador en proceso aislado (timeout anti-deadlock)
                with concurrent.futures.ProcessPoolExecutor(max_workers=1) as _ex:
                    _fut = _ex.submit(_worker_run_once,
                                      (seed_off + trial * n_reps + r, cfg_dict, objetivo))
                    val = _fut.result(timeout=600.0)
                resultados_rep.append(float(val))
            except concurrent.futures.TimeoutError:
                log.warning("RS seed=%d trial=%d rep=%d timeout (600s)", macro_seed, trial, r)
            except Exception as e:
                log.warning("RS seed=%d trial=%d rep=%d error: %s", macro_seed, trial, r, e)

        if not resultados_rep:
            continue

        costo = float(np.mean(resultados_rep))
        if costo < mejor:
            mejor    = costo
            mejor_cfg = vals.copy()

        historia.append({
            "iter":              trial,
            "costo":             round(costo, 4),
            "n_reps":            len(resultados_rep),
            "mejor_hasta_ahora": round(mejor, 4),
            "t_seg":             round(time.time() - t0, 2),
        })

    return {
        "modulo":           "RS",
        "costo_incumbente": round(mejor, 4),
        "tiempo_seg":       round(time.time() - t0, 2),
        "n_evaluaciones":   n_trials * n_reps,
        "incumbente":       mejor_cfg or {},
        "historia_costos":  historia,
        "seed":             macro_seed,
    }


# ═══════════════════════════════════════════════════════════════════════
# Runner de ASTRO-DF (M11) — n_workers=1 para evitar deadlock
# ═══════════════════════════════════════════════════════════════════════

def _correr_m11(macro_seed: int) -> dict:
    """
    Corre M11 ASTRO-DF para una macro-seed.
    Fuerza n_workers=1 (secuencial) para evitar deadlock cuando este proceso
    ya fue lanzado desde un ProcessPoolExecutor externo.
    """
    import modulo_11_astrodf as m11
    import modulo_comparativa_caja_negra as comp

    # ~30 reps/iter × 5 iter ≈ 150 evals del simulador
    max_iter = max(1, B_SIM // 30)

    log.info("M11 seed=%d  max_iter=%d  n_workers=1 (secuencial)", macro_seed, max_iter)

    res = m11.optimizar_astro_df(
        seed         = macro_seed,
        max_iter     = max_iter,
        n_workers    = 1,          # CRÍTICO: evita ProcessPoolExecutor anidado
        objetivo     = "tts_full_days_mean",
        guardar_json = f"resultado_m11_seed{macro_seed:02d}_tmp.json",
    )

    historia   = res.historia_costos if hasattr(res, "historia_costos") else []
    incumbente = comp._incumbente_a_dict(res.incumbente) if hasattr(res, "incumbente") else {}

    conv_eval = comp._convergencia_iterativa(historia)
    conv_time = comp._tiempos_convergencia(historia)

    return {
        "modulo":           "M11",
        "costo_incumbente": float(res.costo_incumbente) if hasattr(res, "costo_incumbente") else float("nan"),
        "tiempo_seg":       float(res.tiempo_seg)       if hasattr(res, "tiempo_seg")       else 0.0,
        "n_evaluaciones":   sum(e.get("n_reps", 1) for e in historia),
        "incumbente":       incumbente,
        "historia_costos":  historia,
        "conv_eval":        conv_eval,
        "conv_time":        conv_time,
        "seed":             macro_seed,
    }


# ═══════════════════════════════════════════════════════════════════════
# Orquestador principal
# ═══════════════════════════════════════════════════════════════════════

_RUNNERS = {
    "RS":  _correr_rs,
    "M11": _correr_m11,
}


def _correr_una(modulo: str, macro_seed: int, out_dir: Path) -> dict:
    """Ejecuta (modulo, seed), re-evalúa incumbente y guarda JSON."""
    f_out = out_dir / f"resultado_{modulo}_seed{macro_seed:02d}.json"
    if f_out.exists():
        log.info("Ya existe %s — omitiendo.", f_out.name)
        return json.loads(f_out.read_text())

    t0 = time.time()
    runner = _RUNNERS[modulo]

    try:
        bruto = runner(macro_seed)
    except Exception as exc:
        log.error("%s seed=%d FALLÓ: %s", modulo, macro_seed, exc)
        registro = {
            "modulo": modulo, "macro_seed": macro_seed,
            "error": str(exc), "tiempo_seg": time.time() - t0,
        }
        f_out.write_text(json.dumps(registro, indent=2, default=str))
        return registro

    incumbente_cfg = bruto.get("incumbente", {})
    if isinstance(incumbente_cfg, dict) and incumbente_cfg:
        reeval, kpis = _re_evaluar_incumbente(
            incumbente_cfg,
            r_final=R_FINAL,
            seed_offset_base=500_000 + macro_seed * R_FINAL,
        )
    else:
        reeval = {"media": float("nan"), "sd": float("nan"),
                  "ic95_lo": float("nan"), "ic95_hi": float("nan"), "r_final": 0}
        kpis   = {k: float("nan") for k in [
            "tts_media", "tts_sd", "tts_ic95_lo", "tts_ic95_hi",
            "at_media", "at_sd", "at_ic95_lo", "at_ic95_hi",
            "at_first_media", "at_post_media"]}

    conv_eval = bruto.get("conv_eval") or _convergencia_iterativa(
        bruto.get("historia_costos", []))

    registro = {
        "modulo":          modulo,
        "macro_seed":      macro_seed,
        "n_trials_param":  B_SIM,
        "n_eval_usadas":   bruto.get("n_evaluaciones", 0),
        "costo_opt":       bruto.get("costo_incumbente", float("nan")),
        "reeval":          reeval,
        "kpis_incumbente": kpis,
        "conv_eval":       conv_eval,
        "conv_time":       bruto.get("conv_time", []),
        "incumbente":      incumbente_cfg,
        "tiempo_seg":      time.time() - t0,
    }

    f_out.write_text(json.dumps(registro, indent=2, default=str))
    log.info("[OK] %s seed=%02d  costo_opt=%.2f  reeval=%.2f  t=%.0fs",
             modulo, macro_seed,
             registro["costo_opt"],
             registro["reeval"].get("media", float("nan")),
             registro["tiempo_seg"])
    return registro


def ejecutar(modulos: list[str], n_seeds: int, n_cores: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    tareas = [(m, s) for m in modulos for s in range(n_seeds)]

    log.info("══ IFORS Benchmark ══")
    log.info("Módulos : %s", modulos)
    log.info("Seeds   : %d  |  B_SIM: %d  |  r_final: %d  |  n_cores: %d",
             n_seeds, B_SIM, R_FINAL, n_cores)
    log.info("Corridas: %d  |  Salida: %s", len(tareas), out_dir)

    resultados = []
    t0 = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as ex:
        futuros = {
            ex.submit(_correr_una, m, s, out_dir): (m, s)
            for (m, s) in tareas
        }
        for i, fut in enumerate(concurrent.futures.as_completed(futuros), 1):
            m, s = futuros[fut]
            try:
                r = fut.result()
                resultados.append(r)
            except Exception as exc:
                log.error("[%d/%d] %s seed=%d EXCEPCIÓN: %s", i, len(tareas), m, s, exc)

    consolidado = out_dir / "consolidado.json"
    consolidado.write_text(json.dumps(resultados, indent=2, default=str))
    log.info("Listo. %d corridas en %.1f h. Consolidado: %s",
             len(resultados), (time.time() - t0) / 3600, consolidado)


# ═══════════════════════════════════════════════════════════════════════
# Punto de entrada
# ═══════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Corre M11 (ASTRO-DF) y RS con la config del benchmark IFORS.")
    p.add_argument("--modulos",  nargs="+", default=["RS", "M11"],
                   choices=["RS", "M11"],
                   help="Algoritmos a correr (por defecto: RS M11).")
    p.add_argument("--n_seeds",  type=int, default=N_SEEDS,
                   help=f"Número de macro-réplicas (default {N_SEEDS}).")
    p.add_argument("--n_cores",  type=int, default=min(9, os.cpu_count() or 4),
                   help="Procesos paralelos entre seeds (default: min(9, cpu_count)).")
    p.add_argument("--out",      default="resultados_astrodf_rs",
                   help="Directorio de salida (default: resultados_astrodf_rs/).")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    ejecutar(args.modulos, args.n_seeds, args.n_cores, Path(args.out))


if __name__ == "__main__":
    # CRÍTICO: usar spawn en lugar de fork para evitar deadlocks con
    # ProcessPoolExecutors anidados (M11 crea su propio PPE internamente).
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
