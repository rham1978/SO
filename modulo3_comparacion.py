"""
===============================================================================
MÓDULO 3 — COMPARACIÓN ESTADÍSTICA: BASELINE VS ESCENARIOS
===============================================================================
Compara estadísticamente el modelo baseline contra escenarios alternativos.

Tests disponibles:
  - U de Mann-Whitney  : distribuciones de KPIs continuas (no paramétrico)
  - t de Welch         : comparación de medias (paramétrico, varianzas ≠)
  - Kruskal-Wallis     : comparación de 3+ escenarios simultáneamente

KPIs comparados:
  - TTS primera consulta (días)
  - TTS post-consulta (días)
  - TTS total (días)
  - Lista de espera final 1ra consulta
  - Lista de espera final control post
  - Atenciones primera consulta
  - Atenciones control post

Uso:
    python modulo3_comparacion.py --demo
    python modulo3_comparacion.py --escenarios config_escenarios.json

O importar:
    from modulo3_comparacion import comparar_escenarios, EscenarioConfig
===============================================================================
"""

import numpy as np
import json
import logging
import argparse
from dataclasses import dataclass, field, asdict
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("comparacion")


# ──────────────────────────────────────────────────────────────────────────────
# KPIs a comparar
# ──────────────────────────────────────────────────────────────────────────────
KPIS_COMPARAR = [
    "tts_first_attended_days_mean",
    "tts_post_days_mean",
    "tts_full_days_mean",
    "wl_first_end",
    "wl_control_end",
    "wl_total_end",
    "first_attended",
    "post_ctrl_attended",
    "post_completed",
    "slots_expired_first",
    "slots_expired_post",
]

KPI_LABELS = {
    "tts_first_attended_days_mean": "TTS 1ra consulta (días)",
    "tts_post_days_mean":           "TTS post-consulta (días)",
    "tts_full_days_mean":           "TTS total (días)",
    "wl_first_end":                 "Lista espera 1ra final",
    "wl_control_end":               "Lista espera control final",
    "wl_total_end":                 "Lista espera total final",
    "first_attended":               "Atenciones 1ra consulta",
    "post_ctrl_attended":           "Atenciones control post",
    "post_completed":               "Completan post-consulta",
    "slots_expired_first":          "Slots expirados 1ra",
    "slots_expired_post":           "Slots expirados control",
}


# ──────────────────────────────────────────────────────────────────────────────
# Configuración de escenario
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class EscenarioConfig:
    """
    Define un escenario: nombre + modificaciones al SimConfig baseline.
    Los parámetros no especificados heredan del baseline.
    """
    nombre:      str  = "escenario"
    descripcion: str  = ""
    seed_base:   int  = 202
    n_corridas:  int  = 10

    # Parámetros modificables (None = usar baseline)
    fixed_weekly_capacity:        Optional[int]   = None  # slots 1ra/sem
    fixed_post_control_hours:     Optional[float] = None  # horas control post/sem
    ugd_lab_per_week:             Optional[int]   = None  # cupos lab UGD
    mat_us_per_week:              Optional[int]   = None  # cupos eco matrona
    ugd_us_per_week:              Optional[int]   = None  # cupos eco UGD
    publish_lead_workdays:        Optional[int]   = None  # días publicación anticipada
    blocked_pct:                  Optional[float] = None  # % bloqueo 1ra
    blocked_pct_post_control:     Optional[float] = None  # % bloqueo control post
    empty_control_p_ugd:          Optional[float] = None  # % consultas vacías
    not_contactable_p:            Optional[float] = None  # % no contactabilidad
    matrona_capacity:             Optional[int]   = None  # número de matronas
    agent_capacity:               Optional[int]   = None  # número agentes UGD


# ──────────────────────────────────────────────────────────────────────────────
# Estadísticos descriptivos de una muestra
# ──────────────────────────────────────────────────────────────────────────────
def _desc(arr: list[float]) -> dict:
    a = np.array(arr, dtype=float)
    n = len(a)
    if n == 0:
        return {"n": 0, "media": 0.0, "sd": 0.0, "ic_low": 0.0, "ic_high": 0.0}
    mu = float(np.mean(a))
    sd = float(np.std(a, ddof=1)) if n > 1 else 0.0
    try:
        from scipy.stats import t
        tc = float(t.ppf(0.975, n-1))
    except ImportError:
        tc = 1.96
    se = sd / np.sqrt(n)
    return {
        "n":       n,
        "media":   round(mu, 3),
        "sd":      round(sd, 3),
        "mediana": round(float(np.median(a)), 3),
        "ic_low":  round(mu - tc*se, 3),
        "ic_high": round(mu + tc*se, 3),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Tests estadísticos par a par
# ──────────────────────────────────────────────────────────────────────────────
def _test_mannwhitney(a: list[float], b: list[float], alpha: float = 0.05) -> dict:
    try:
        from scipy.stats import mannwhitneyu
        U, p = mannwhitneyu(a, b, alternative='two-sided')
        return {"test": "Mann-Whitney U", "statistic": round(float(U), 4),
                "p_valor": round(float(p), 6), "rechaza_h0": bool(p < alpha)}
    except ImportError:
        return {"test": "Mann-Whitney U", "statistic": None,
                "p_valor": None, "rechaza_h0": None,
                "nota": "scipy no disponible"}

def _test_welch(a: list[float], b: list[float], alpha: float = 0.05) -> dict:
    try:
        from scipy.stats import ttest_ind
        stat, p = ttest_ind(a, b, equal_var=False)
        return {"test": "t Welch", "statistic": round(float(stat), 4),
                "p_valor": round(float(p), 6), "rechaza_h0": bool(p < alpha)}
    except ImportError:
        return {"test": "t Welch", "statistic": None,
                "p_valor": None, "rechaza_h0": None,
                "nota": "scipy no disponible"}

def _test_kruskal(*grupos: list[float], alpha: float = 0.05) -> dict:
    try:
        from scipy.stats import kruskal
        stat, p = kruskal(*grupos)
        return {"test": "Kruskal-Wallis", "statistic": round(float(stat), 4),
                "p_valor": round(float(p), 6), "rechaza_h0": bool(p < alpha)}
    except ImportError:
        return {"test": "Kruskal-Wallis", "statistic": None,
                "p_valor": None, "rechaza_h0": None,
                "nota": "scipy no disponible"}


# ──────────────────────────────────────────────────────────────────────────────
# Ejecutar corridas de un escenario
# ──────────────────────────────────────────────────────────────────────────────
def correr_escenario(escenario: EscenarioConfig) -> dict[str, list[float]]:
    """
    Corre n_corridas del modelo con la configuración del escenario.
    Retorna dict {kpi_name: [valor_corrida1, valor_corrida2, ...]}
    """
    import sys, os, random
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    try:
        import simpy
        from simulador_clinica_baseline import ClinicModelAdjusted, SimConfig, CFG, run_once
    except ImportError as e:
        log.error("No se pudo importar baseline: %s", e)
        return {}

    # Construir config modificada
    import copy
    cfg = copy.deepcopy(CFG)

    # Aplicar overrides del escenario
    if escenario.fixed_weekly_capacity is not None:
        cfg.fixed_weekly_capacity = escenario.fixed_weekly_capacity
        cfg.use_fixed_weekly_capacity = True
    if escenario.fixed_post_control_hours is not None:
        cfg.fixed_post_control_hours = escenario.fixed_post_control_hours
        cfg.use_fixed_post_control_hours = True
    if escenario.ugd_lab_per_week is not None:
        cfg.ugd_lab_per_week = escenario.ugd_lab_per_week
    if escenario.mat_us_per_week is not None:
        cfg.mat_us_per_week = escenario.mat_us_per_week
    if escenario.ugd_us_per_week is not None:
        cfg.ugd_us_per_week = escenario.ugd_us_per_week
    if escenario.publish_lead_workdays is not None:
        cfg.publish_lead_workdays = escenario.publish_lead_workdays
    if escenario.blocked_pct is not None:
        cfg.blocked_pct = escenario.blocked_pct
    if escenario.blocked_pct_post_control is not None:
        cfg.blocked_pct_post_control = escenario.blocked_pct_post_control
    if escenario.empty_control_p_ugd is not None:
        cfg.empty_control_p_ugd = escenario.empty_control_p_ugd
    if escenario.not_contactable_p is not None:
        cfg.not_contactable_p = escenario.not_contactable_p
    if escenario.agent_capacity is not None:
        cfg.agent_capacity = escenario.agent_capacity

    resultados = {k: [] for k in KPIS_COMPARAR}

    log.info("Corriendo escenario '%s' (%d corridas)...", escenario.nombre, escenario.n_corridas)
    for r in range(escenario.n_corridas):
        res = run_once(seed_offset=escenario.seed_base - 202 + r, cfg=cfg)
        for k in KPIS_COMPARAR:
            if k in res:
                resultados[k].append(float(res[k]))
        log.info("  [%s] corrida %d/%d completada", escenario.nombre, r+1, escenario.n_corridas)

    return resultados


# ──────────────────────────────────────────────────────────────────────────────
# Comparación completa
# ──────────────────────────────────────────────────────────────────────────────
def comparar_escenarios(
    escenarios:    list[EscenarioConfig],
    alpha:         float = 0.05,
    guardar_json:  str   = "resultado_comparacion.json",
    verbose:       bool  = True,
) -> dict:
    """
    Corre cada escenario, compara par a par contra el primero (baseline)
    y entre todos usando Kruskal-Wallis.

    El primer elemento de `escenarios` se trata como la línea base.

    Retorna dict con resultados completos.
    """
    assert len(escenarios) >= 2, "Se necesitan al menos 2 escenarios (baseline + 1)"

    # Correr todos los escenarios
    datos_por_escenario = {}
    for esc in escenarios:
        datos_por_escenario[esc.nombre] = correr_escenario(esc)

    nombre_baseline = escenarios[0].nombre
    datos_baseline  = datos_por_escenario[nombre_baseline]

    resultado = {
        "baseline":    nombre_baseline,
        "alpha":       alpha,
        "escenarios":  {},
        "comparaciones_vs_baseline": {},
        "kruskal_wallis_global": {},
    }

    # Estadísticos por escenario
    for esc in escenarios:
        datos = datos_por_escenario[esc.nombre]
        resultado["escenarios"][esc.nombre] = {
            "descripcion": esc.descripcion,
            "n_corridas":  esc.n_corridas,
            "kpis": {k: _desc(datos.get(k, [])) for k in KPIS_COMPARAR},
        }

    # Comparaciones par a par vs baseline
    for esc in escenarios[1:]:
        datos_esc = datos_por_escenario[esc.nombre]
        comp = {}
        for k in KPIS_COMPARAR:
            a = datos_baseline.get(k, [])
            b = datos_esc.get(k, [])
            if not a or not b:
                continue
            mu_a = np.mean(a)
            mu_b = np.mean(b)
            diff_abs = round(float(mu_b - mu_a), 3)
            diff_pct = round(float((mu_b - mu_a) / mu_a * 100) if mu_a != 0 else 0.0, 2)
            comp[k] = {
                "label":        KPI_LABELS.get(k, k),
                "baseline_media": round(float(mu_a), 3),
                "escenario_media":round(float(mu_b), 3),
                "diff_absoluta":  diff_abs,
                "diff_pct":       diff_pct,
                "mann_whitney":   _test_mannwhitney(a, b, alpha),
                "welch_t":        _test_welch(a, b, alpha),
            }
        resultado["comparaciones_vs_baseline"][esc.nombre] = comp

    # Kruskal-Wallis global (todos los escenarios)
    for k in KPIS_COMPARAR:
        grupos = [datos_por_escenario[esc.nombre].get(k, []) for esc in escenarios]
        grupos = [g for g in grupos if len(g) > 0]
        if len(grupos) >= 2:
            resultado["kruskal_wallis_global"][k] = _test_kruskal(*grupos, alpha=alpha)

    if verbose:
        _print_comparacion(resultado, escenarios)

    # Guardar
    with open(guardar_json, "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)
    log.info("Resultado guardado en '%s'", guardar_json)

    return resultado


def _print_comparacion(resultado: dict, escenarios: list[EscenarioConfig]):
    print("\n" + "="*80)
    print("COMPARACIÓN ESTADÍSTICA: BASELINE VS ESCENARIOS")
    print("="*80)
    print(f"Baseline: {resultado['baseline']}")
    print(f"Alpha   : {resultado['alpha']}")

    for esc in escenarios[1:]:
        nombre = esc.nombre
        comp   = resultado["comparaciones_vs_baseline"].get(nombre, {})
        print(f"\n── Escenario: {nombre} ({esc.descripcion}) ──")
        print(f"  {'KPI':<35} {'Base':>10} {'Escen':>10} {'Δ%':>8} {'p-MW':>10} {'Sig':>5}")
        print("  " + "-"*75)
        for k, datos in comp.items():
            mw    = datos["mann_whitney"]
            p     = mw.get("p_valor")
            sig   = "***" if p is not None and p < 0.001 else \
                    "**"  if p is not None and p < 0.01  else \
                    "*"   if p is not None and p < 0.05  else ""
            p_str = f"{p:.4f}" if p is not None else "N/A"
            label = datos["label"][:33]
            print(f"  {label:<35} {datos['baseline_media']:>10.1f} {datos['escenario_media']:>10.1f} "
                  f"{datos['diff_pct']:>+7.1f}% {p_str:>10} {sig:>5}")

    print("\n── Kruskal-Wallis global (todos los escenarios) ──")
    for k, res in resultado["kruskal_wallis_global"].items():
        label = KPI_LABELS.get(k, k)
        p     = res.get("p_valor")
        sig   = "***" if p is not None and p < 0.001 else \
                "**"  if p is not None and p < 0.01  else \
                "*"   if p is not None and p < 0.05  else ""
        p_str = f"{p:.4f}" if p is not None else "N/A"
        print(f"  {label:<40} H={res.get('statistic','N/A'):>8}  p={p_str:>8}  {sig}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Módulo 3: Comparación baseline vs escenarios")
    parser.add_argument("--demo",       action="store_true")
    parser.add_argument("--alpha",      type=float, default=0.05)
    parser.add_argument("--n_corridas", type=int,   default=5)
    args = parser.parse_args()

    if args.demo:
        print("\n[MODO DEMO] Corriendo baseline + 2 escenarios de ejemplo\n")

        escenarios_demo = [
            EscenarioConfig(
                nombre      = "baseline",
                descripcion = "Línea base sin cambios",
                n_corridas  = args.n_corridas,
            ),
            EscenarioConfig(
                nombre                    = "mas_capacidad_1ra",
                descripcion               = "+4 slots/sem primera consulta (16→20)",
                n_corridas                = args.n_corridas,
                fixed_weekly_capacity     = 20,
            ),
            EscenarioConfig(
                nombre                    = "menor_bloqueo",
                descripcion               = "Bloqueo 1ra reducido (32%→20%)",
                n_corridas                = args.n_corridas,
                blocked_pct               = 0.20,
            ),
        ]

        comparar_escenarios(
            escenarios   = escenarios_demo,
            alpha        = args.alpha,
            guardar_json = "resultado_comparacion_demo.json",
            verbose      = True,
        )
