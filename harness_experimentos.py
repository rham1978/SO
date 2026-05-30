#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
harness_experimentos.py
═══════════════════════════════════════════════════════════════════════
Harness de experimentos riguroso para el benchmark de optimización de
caja negra sobre el simulador DES de la Clínica de Ginecología.

Resuelve los cuatro defectos del benchmark original:
  1. Presupuesto desigual  → presupuesto FIJO de evaluaciones del simulador
                              (B_SIM) idéntico para todos los métodos.
  2. Seed única            → N_SEEDS macro-réplicas independientes.
  3. Common Random Numbers → cada macro-réplica usa un bloque FIJO de
                              seed_offsets, COMPARTIDO entre métodos, para
                              que las diferencias método-a-método se estimen
                              con varianza mínima.
  4. Sin re-evaluación     → el incumbente final de cada (método, seed) se
                              re-evalúa con R_FINAL réplicas frescas para
                              estimar su valor esperado real.

NO modifica el simulador ni los algoritmos: los importa y los envuelve.

Paralelización: a nivel de MACRO-RÉPLICA (seed). Cada proceso corre un
algoritmo completo de forma secuencial; se lanzan hasta N_CORES procesos
simultáneos. Esto satura los cores independientemente de cuántas réplicas
internas pida cada método (random=1, SPSA=2, SK adaptativo=variable).

Uso:
    python harness_experimentos.py --metodos M10 M13 RS SMAC_GP \
        --n_seeds 15 --b_sim 150 --n_cores 9 --out resultados/

Autor: preparado para la presentación IFORS (julio 2026).
"""

from __future__ import annotations
import argparse
import concurrent.futures
import dataclasses
import json
import logging
import os
import time
from pathlib import Path

import numpy as np

# ───────────────────────────────────────────────────────────────────────
# Configuración global del experimento
# ───────────────────────────────────────────────────────────────────────

# Bloque de CRN: para CADA macro-réplica s, el conjunto de seed_offsets que
# el simulador usará para sus réplicas internas es:
#     CRN_BLOCK[s] = [s*CRN_STRIDE + 0, s*CRN_STRIDE + 1, ..., +(CRN_STRIDE-1)]
# Compartido entre TODOS los métodos. CRN_STRIDE debe ser >= máximo número
# de réplicas que cualquier método pedirá en un punto (n_max adaptativo).
CRN_STRIDE = 64          # holgado: ningún método pide >64 réplicas/punto
SEED_BASE  = 202         # debe coincidir con cfg.random_seed_base del simulador

log = logging.getLogger("harness")


# ───────────────────────────────────────────────────────────────────────
# Contador de presupuesto: envuelve run_once para contar evaluaciones reales
# del simulador y CORTAR cuando se agota el presupuesto B_SIM.
# ───────────────────────────────────────────────────────────────────────

class PresupuestoAgotado(Exception):
    """Se lanza cuando un método excede el presupuesto de evaluaciones."""
    pass


class EvaluadorPresupuestado:
    """
    Envuelve la evaluación del simulador para:
      - contar cada llamada a run_once como 1 evaluación,
      - usar CRN: las réplicas de un punto usan offsets del bloque de la seed,
      - registrar la traza (n_eval_acumulada, costo, mejor_hasta_ahora),
      - cortar limpiamente al agotar B_SIM.

    Se inyecta en los módulos que aceptan una función de evaluación externa.
    Para módulos que no la aceptan, ver adaptadores más abajo.
    """

    def __init__(self, macro_seed: int, b_sim: int, objetivo: str = "tts_full_days_mean"):
        self.macro_seed = macro_seed
        self.b_sim      = b_sim
        self.objetivo   = objetivo
        self.n_eval     = 0
        self.mejor      = float("inf")
        self.traza      = []          # lista de (n_eval, costo_punto, mejor_hasta_ahora)
        self._crn_base  = macro_seed * CRN_STRIDE

    def _run_once_cached(self, cfg_dict: dict, rep_idx: int) -> float:
        """Una réplica del simulador con offset CRN determinista."""
        from simulador_clinica_baseline import run_once, SimConfig
        # Offset CRN: mismo (macro_seed, rep_idx) → mismo offset en todos los métodos
        seed_offset = self._crn_base + rep_idx
        cfg = SimConfig(**cfg_dict) if not isinstance(cfg_dict, SimConfig) else cfg_dict
        res = run_once(seed_offset=seed_offset, cfg=cfg)
        return float(res[self.objetivo])

    def evaluar(self, cfg_dict: dict, n_reps: int) -> tuple[float, float]:
        """
        Evalúa un punto con n_reps réplicas (CRN). Devuelve (media, varianza).
        Cuenta n_reps contra el presupuesto. Lanza PresupuestoAgotado si excede.
        """
        if self.n_eval + n_reps > self.b_sim:
            raise PresupuestoAgotado(
                f"n_eval={self.n_eval}+{n_reps} > B_SIM={self.b_sim}")

        valores = [self._run_once_cached(cfg_dict, r) for r in range(n_reps)]
        self.n_eval += n_reps

        media = float(np.mean(valores))
        var   = float(np.var(valores, ddof=1)) if n_reps > 1 else float("nan")

        if media < self.mejor:
            self.mejor = media
        self.traza.append((self.n_eval, media, self.mejor))
        return media, var

    def reevaluar_incumbente(self, cfg_dict: dict, r_final: int,
                             offset_inicio: int = 100_000) -> dict:
        """
        Re-evalúa el incumbente final con r_final réplicas FRESCAS (offsets
        que NO se solaparon con la optimización, para evitar sesgo optimista).
        Devuelve estadísticos del valor esperado real.
        """
        from simulador_clinica_baseline import run_once, SimConfig
        cfg = SimConfig(**cfg_dict) if not isinstance(cfg_dict, SimConfig) else cfg_dict
        base = offset_inicio + self.macro_seed * r_final
        vals = [float(run_once(seed_offset=base + r, cfg=cfg)[self.objetivo])
                for r in range(r_final)]
        vals = np.array(vals)
        return {
            "media":   float(vals.mean()),
            "sd":      float(vals.std(ddof=1)),
            "ic95_lo": float(vals.mean() - 1.96 * vals.std(ddof=1) / np.sqrt(r_final)),
            "ic95_hi": float(vals.mean() + 1.96 * vals.std(ddof=1) / np.sqrt(r_final)),
            "r_final": r_final,
        }


# ───────────────────────────────────────────────────────────────────────
# Una corrida = (método, macro_seed). Esta es la unidad que se paraleliza.
# ───────────────────────────────────────────────────────────────────────

def correr_una(metodo: str, macro_seed: int, b_sim: int, r_final: int,
               objetivo: str = "tts_full_days_mean") -> dict:
    """
    Ejecuta UN método con UNA macro-seed, dentro del presupuesto b_sim,
    re-evalúa el incumbente y devuelve el registro completo.

    IMPORTANTE: esta función corre en su propio proceso. Toda la
    paralelización del benchmark es a este nivel; los métodos NO deben
    abrir sus propios ProcessPoolExecutor aquí (se fuerza n_workers=1
    internamente vía las funciones adaptadoras).
    """
    import importlib
    t0 = time.time()
    ev = EvaluadorPresupuestado(macro_seed, b_sim, objetivo)

    try:
        adaptador = _ADAPTADORES[metodo]
    except KeyError:
        raise ValueError(f"Método '{metodo}' sin adaptador. Disponibles: "
                         f"{list(_ADAPTADORES.keys())}")

    incumbente_cfg = adaptador(ev, macro_seed, b_sim)

    # Re-evaluación honesta del incumbente
    reeval = ev.reevaluar_incumbente(incumbente_cfg, r_final)

    return {
        "metodo":          metodo,
        "macro_seed":      macro_seed,
        "b_sim":           b_sim,
        "n_eval_usadas":   ev.n_eval,
        "mejor_opt":       ev.mejor,          # mejor durante optimización (ruidoso)
        "incumbente_cfg":  incumbente_cfg,
        "reeval":          reeval,            # valor esperado real (poco ruido)
        "traza":           ev.traza,          # curva de convergencia
        "tiempo_seg":      time.time() - t0,
    }


# ───────────────────────────────────────────────────────────────────────
# ADAPTADORES: traducen la interfaz de cada módulo al EvaluadorPresupuestado.
#
# Cada adaptador recibe (ev, macro_seed, b_sim) y debe:
#   - correr el algoritmo usando ev.evaluar(cfg_dict, n_reps) para CADA punto,
#   - capturar PresupuestoAgotado y terminar limpiamente,
#   - devolver el cfg_dict del incumbente final.
#
# NOTA: estos adaptadores son PLANTILLAS. Las marco con TODO donde hay que
# enganchar con la API real de tu módulo. La razón de hacerlo así (en vez de
# editar cada módulo) es que tus módulos ya funcionan; el harness los invoca
# punto por punto. Para SK/SMAC, lo más limpio es usar el bucle de ask/tell
# del optimizador. Ver el plan de experimentos para el detalle por método.
# ───────────────────────────────────────────────────────────────────────

def _cfg_desde_vector(x_unit: np.ndarray) -> dict:
    """
    Convierte un vector en [0,1]^12 a un cfg_dict del simulador, respetando
    enteros con int(round(...)) — CORRIGE el bug de truncamiento int().
    Usa los mismos PARAM_RANGES del comparativa.
    """
    from modulo_comparativa_caja_negra import PARAM_NAMES, PARAM_RANGES
    from simulador_clinica_baseline import SimConfig, CFG

    ENTEROS = {
        "horas_especialista_1ra", "horas_control_post", "cupos_laboratorio_ugd",
        "cupos_ecografia_matrona", "cupos_ecografia_ugd", "dias_publicacion",
        "num_matronas", "num_agentes_ugd",
    }
    cfg = dataclasses.replace(CFG)  # copia del baseline
    valores = {}
    for i, nombre in enumerate(PARAM_NAMES):
        lo, hi = PARAM_RANGES[nombre]
        v = lo + float(x_unit[i]) * (hi - lo)
        if nombre in ENTEROS:
            v = int(round(v))          # ← redondeo correcto, no truncamiento
        valores[nombre] = v

    # Mapeo nombre→atributo de SimConfig (idéntico a _aplicar_config_a_cfg)
    cfg.fixed_weekly_capacity        = int(valores["horas_especialista_1ra"])
    cfg.use_fixed_weekly_capacity    = True
    cfg.fixed_post_control_capacity  = int(valores["horas_control_post"])
    cfg.use_fixed_post_control_hours = True
    cfg.ugd_lab_per_week             = int(valores["cupos_laboratorio_ugd"])
    cfg.mat_us_per_week              = int(valores["cupos_ecografia_matrona"])
    cfg.ugd_us_per_week              = int(valores["cupos_ecografia_ugd"])
    cfg.publish_lead_workdays        = int(valores["dias_publicacion"])
    cfg.blocked_pct                  = float(valores["pct_bloqueo_1ra"])
    cfg.empty_control_p_ugd          = float(valores["pct_consultas_vacias"])
    cfg.matrona_capacity             = int(valores["num_matronas"])
    cfg.agent_capacity               = int(valores["num_agentes_ugd"])
    cfg.not_contactable_p            = float(valores["pct_no_contactabilidad"])
    cfg.blocked_pct_post_control     = float(valores["pct_bloqueo_post_control"])
    return dataclasses.asdict(cfg)


def adaptador_random_search(ev: EvaluadorPresupuestado, macro_seed: int,
                            b_sim: int, n_reps_punto: int = 3) -> dict:
    """
    Random search: PISO de comparación. Muestrea puntos uniformes en [0,1]^12,
    evalúa cada uno con n_reps_punto réplicas (CRN), hasta agotar presupuesto.
    """
    rng = np.random.RandomState(1000 + macro_seed)   # seed del ALGORITMO
    mejor_cfg = None
    mejor_val = float("inf")
    while True:
        x = rng.uniform(0, 1, size=12)
        cfg_dict = _cfg_desde_vector(x)
        try:
            media, _ = ev.evaluar(cfg_dict, n_reps_punto)
        except PresupuestoAgotado:
            break
        if media < mejor_val:
            mejor_val = media
            mejor_cfg = cfg_dict
    return mejor_cfg


# ── Adaptadores para los módulos reales (plantillas a enganchar) ──────────
# Ver PLAN_DE_EXPERIMENTOS.md sección 6 para la guía de enganche exacta.

def adaptador_sk_kgcp(ev, macro_seed, b_sim) -> dict:
    """
    TODO[enganche]: envolver modulo10_sk_kgcp para que cada evaluación pase
    por ev.evaluar(). La vía limpia es usar el bucle interno ask/tell del
    optimizador SK y reemplazar su llamada a run_once por ev.evaluar().
    Mientras no esté enganchado, lanza NotImplementedError para no producir
    resultados silenciosamente inválidos.
    """
    raise NotImplementedError(
        "adaptador_sk_kgcp: enganchar con modulo10 (ver plan, sección 6.2)")


def adaptador_spsa(ev, macro_seed, b_sim) -> dict:
    """TODO[enganche]: envolver modulo13_spsa (ver plan, sección 6.3)."""
    raise NotImplementedError(
        "adaptador_spsa: enganchar con modulo13 (ver plan, sección 6.3)")


def adaptador_smac_gp(ev, macro_seed, b_sim) -> dict:
    """TODO[enganche]: envolver modulo4_smac (ver plan, sección 6.4)."""
    raise NotImplementedError(
        "adaptador_smac_gp: enganchar con modulo4 (ver plan, sección 6.4)")


_ADAPTADORES = {
    "RS":      adaptador_random_search,   # listo para usar
    "M10":     adaptador_sk_kgcp,         # enganchar
    "M13":     adaptador_spsa,            # enganchar
    "SMAC_GP": adaptador_smac_gp,         # enganchar
}


# ───────────────────────────────────────────────────────────────────────
# Orquestación del experimento completo
# ───────────────────────────────────────────────────────────────────────

def ejecutar_experimento(metodos: list[str], n_seeds: int, b_sim: int,
                         r_final: int, n_cores: int, out_dir: Path) -> None:
    """
    Lanza (métodos × seeds) corridas, paralelizando a nivel de macro-réplica.
    Guarda un JSON por corrida y un consolidado.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tareas = [(m, s) for m in metodos for s in range(n_seeds)]
    log.info("Total de corridas: %d (%d métodos × %d seeds)",
             len(tareas), len(metodos), n_seeds)
    log.info("Presupuesto por corrida: B_SIM=%d · re-eval=%d réplicas", b_sim, r_final)
    log.info("Costo estimado: %d evaluaciones ≈ %.1f h-CPU ≈ %.1f h pared (@%d cores, 90s/corrida)",
             len(tareas) * (b_sim + r_final),
             len(tareas) * (b_sim + r_final) * 90 / 3600,
             len(tareas) * (b_sim + r_final) * 90 / 3600 / max(n_cores, 1),
             n_cores)

    resultados = []
    t0 = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as ex:
        futuros = {
            ex.submit(correr_una, m, s, b_sim, r_final): (m, s)
            for (m, s) in tareas
        }
        for i, fut in enumerate(concurrent.futures.as_completed(futuros), 1):
            m, s = futuros[fut]
            try:
                r = fut.result()
                resultados.append(r)
                f = out_dir / f"corrida_{m}_seed{s:02d}.json"
                f.write_text(json.dumps(r, indent=2, default=str))
                log.info("[%d/%d] %s seed=%d  reeval=%.2f d  (%d eval, %.0f s)",
                         i, len(tareas), m, s,
                         r["reeval"]["media"], r["n_eval_usadas"], r["tiempo_seg"])
            except Exception as exc:
                log.error("[%d/%d] %s seed=%d FALLÓ: %s", i, len(tareas), m, s, exc)

    consolidado = out_dir / "consolidado.json"
    consolidado.write_text(json.dumps(resultados, indent=2, default=str))
    log.info("Listo. %d corridas en %.1f h pared. Consolidado: %s",
             len(resultados), (time.time() - t0) / 3600, consolidado)


def main():
    p = argparse.ArgumentParser(description="Harness de experimentos riguroso (IFORS).")
    p.add_argument("--metodos", nargs="+", default=["RS"],
                   help="Métodos a correr (claves de _ADAPTADORES).")
    p.add_argument("--n_seeds", type=int, default=15, help="Macro-réplicas por método.")
    p.add_argument("--b_sim",   type=int, default=150, help="Presupuesto de evaluaciones del simulador.")
    p.add_argument("--r_final", type=int, default=50,  help="Réplicas para re-evaluar incumbente.")
    p.add_argument("--n_cores", type=int, default=9,   help="Procesos paralelos (cores físicos - 1).")
    p.add_argument("--out",     default="resultados",  help="Directorio de salida.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    ejecutar_experimento(args.metodos, args.n_seeds, args.b_sim,
                         args.r_final, args.n_cores, Path(args.out))


if __name__ == "__main__":
    main()
