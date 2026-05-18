"""
===============================================================================
MÓDULO 1 — DETERMINACIÓN NÚMERO DE CORRIDAS
===============================================================================
Conectado con: simulador_clinica_baseline.py

Métodos implementados:
  1. Fórmula analítica para error absoluto
  2. Fórmula analítica para error relativo
  3. Método secuencial para error relativo

Cada método obtiene los datos directamente corriendo run_once() del baseline.

KPIs disponibles:
  - tts_first_attended_days_mean  : TTS primera consulta atendida (días)
  - tts_post_days_mean            : TTS post-consulta (días)
  - tts_full_days_mean            : TTS total ingreso→alta (días)
  - wl_first_end                  : Lista espera 1ra al final
  - wl_control_end                : Lista espera control al final
  - wl_total_end                  : Lista espera total al final
  - first_attended                : N° atenciones primera consulta
  - post_ctrl_attended            : N° atenciones control post
  - tts_first_process_days_mean   : TTS proceso 1ra consulta (días)
  - slots_expired_first           : Slots expirados 1ra consulta
  - slots_expired_post            : Slots expirados control post

Uso:
    python modulo1_num_corridas.py
    python modulo1_num_corridas.py --kpi wl_total_end --n_piloto 10 --eps_rel 0.05
    python modulo1_num_corridas.py --solo_analiticos
    python modulo1_num_corridas.py --listar_kpis

O importar:
    from modulo1_num_corridas import reporte_num_corridas_desde_modelo

CAMBIOS RESPECTO A LA VERSIÓN ANTERIOR:
  [BUG]  Default de kpi en reporte_num_corridas_desde_modelo era el label, no la clave.
  [BUG]  n_recomendado podía ser menor que n_piloto (ya se tienen esos datos).
  [BUG]  Semillas del método secuencial se solapaban con las del piloto compartido.
  [BUG]  CV infinito / división por cero cuando media ≈ 0 no estaba bien manejado.
  [MEJ]  Validación de KPI al inicio, antes de correr ninguna corrida del modelo.
  [MEJ]  ETA estimado durante corridas piloto largas.
  [MEJ]  IC relativo final mostrado en el resumen del método 3.
===============================================================================
"""

import math
import numpy as np
import json
import logging
import argparse
import time
import sys
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("num_corridas")

# ──────────────────────────────────────────────────────────────────────────────
# KPIs disponibles del modelo
# ──────────────────────────────────────────────────────────────────────────────
KPIS_DISPONIBLES = {
    "tts_first_attended_days_mean":  "TTS primera consulta atendida (días)",
    "tts_first_closed_days_mean":    "TTS primera consulta cerrada (días)",
    "tts_post_days_mean":            "TTS post-consulta (días)",
    "tts_full_days_mean":            "TTS total ingreso→alta (días)",
    "wl_first_end":                  "Lista espera 1ra al final horizonte",
    "wl_control_end":                "Lista espera control post al final",
    "wl_total_end":                  "Lista espera total al final",
    "first_attended":                "Atenciones primera consulta",
    "post_ctrl_attended":            "Atenciones control post",
    "post_completed":                "Completan ruta post",
    "tts_first_process_days_mean":   "TTS proceso primera consulta (días)",
    "slots_expired_first":           "Slots expirados primera consulta",
    "slots_expired_post":            "Slots expirados control post",
}

# ──────────────────────────────────────────────────────────────────────────────
# t de Student bilateral 95%
# ──────────────────────────────────────────────────────────────────────────────
_T_TABLE = {
    1:12.706, 2:4.303, 3:3.182, 4:2.776, 5:2.571,
    6:2.447, 7:2.365, 8:2.306, 9:2.262, 10:2.228,
    11:2.201, 12:2.179, 13:2.160, 14:2.145, 15:2.131,
    16:2.120, 17:2.110, 18:2.101, 19:2.093, 20:2.086,
    25:2.060, 30:2.042, 40:2.021, 60:2.000, 120:1.980,
}

def _t_crit(df: int) -> float:
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, max(1, df)))
    except ImportError:
        pass
    if df <= 0: return 12.706
    if df in _T_TABLE: return float(_T_TABLE[df])
    keys = sorted(_T_TABLE.keys())
    if df > max(keys): return 1.96
    lower = max(k for k in keys if k <= df)
    upper = min(k for k in keys if k > df)
    return float(_T_TABLE[lower] + (df-lower)/(upper-lower)*(_T_TABLE[upper]-_T_TABLE[lower]))

# ──────────────────────────────────────────────────────────────────────────────
# Conexión con el modelo baseline
# ──────────────────────────────────────────────────────────────────────────────
_run_once_cache = None
_cfg_cache      = None

def _importar_modelo():
    """Importa run_once y CFG del baseline (con caché para no reimportar)."""
    global _run_once_cache, _cfg_cache
    if _run_once_cache is not None:
        return _run_once_cache, _cfg_cache

    directorio = os.path.dirname(os.path.abspath(__file__))
    if directorio not in sys.path:
        sys.path.insert(0, directorio)

    try:
        from simulador_clinica_baseline import run_once, CFG
        _run_once_cache = run_once
        _cfg_cache      = CFG
        log.info("Modelo baseline importado: simulador_clinica_baseline.py")
        return run_once, CFG
    except ImportError as e:
        log.error("No se pudo importar simulador_clinica_baseline.py: %s", e)
        log.error("Debe estar en el mismo directorio: %s", directorio)
        raise


def _validar_kpi(kpi: str) -> None:
    """
    [MEJ] Valida que el KPI existe antes de correr cualquier corrida.
    Lanza KeyError con mensaje claro si no es válido.
    """
    if kpi not in KPIS_DISPONIBLES:
        opciones = "\n  ".join(f"{k:<45} {v}" for k, v in KPIS_DISPONIBLES.items())
        raise KeyError(
            f"KPI '{kpi}' no existe.\n"
            f"Opciones disponibles:\n  {opciones}"
        )


def correr_una_vez(kpi: str, seed_offset: int, cfg=None) -> float:
    """
    Ejecuta una corrida completa del modelo y retorna el valor del KPI.

    Parámetros
    ----------
    kpi         : clave del KPI (ver KPIS_DISPONIBLES)
    seed_offset : semilla de esta corrida
    cfg         : SimConfig opcional (None = usa CFG baseline)
    """
    run_once, CFG_base = _importar_modelo()
    cfg_usar = cfg if cfg is not None else CFG_base
    res = run_once(seed_offset=seed_offset, cfg=cfg_usar)
    if kpi not in res:
        raise KeyError(f"KPI '{kpi}' no fue devuelto por run_once(). Usa --listar_kpis.")
    return float(res[kpi])


def correr_piloto(kpi: str, n: int, seed_base: int = 0, cfg=None) -> list:
    """
    Corre n corridas piloto del modelo y retorna lista de valores del KPI.

    [MEJ] Muestra ETA estimado basado en el tiempo promedio de corridas previas.
    """
    # [MEJ] Validar KPI antes de correr ninguna corrida
    _validar_kpi(kpi)

    log.info("Corriendo %d corridas piloto para KPI='%s'...", n, kpi)
    valores  = []
    tiempos  = []

    for i in range(n):
        t0  = time.time()
        val = correr_una_vez(kpi, seed_offset=seed_base + i, cfg=cfg)
        dt  = time.time() - t0
        valores.append(val)
        tiempos.append(dt)

        eta = np.mean(tiempos) * (n - i - 1)
        log.info(
            "  Corrida %2d/%d: %s = %.3f  (%.0fs | ETA ~%.0fs)",
            i + 1, n, kpi, val, dt, eta
        )

    return valores

# ──────────────────────────────────────────────────────────────────────────────
# Dataclasses de resultado
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class ResultadoAbsoluto:
    metodo:           str   = "Formula analitica — error absoluto"
    kpi:              str   = ""
    n_piloto:         int   = 0
    valores_piloto:   list  = field(default_factory=list)
    media_piloto:     float = 0.0
    sd_piloto:        float = 0.0
    cv_piloto:        float = 0.0
    t_crit:           float = 0.0
    epsilon_abs:      float = 0.0
    alpha:            float = 0.05
    n_recomendado:    int   = 0
    intervalo_bajo:   float = 0.0
    intervalo_alto:   float = 0.0
    tiempo_total_seg: float = 0.0

@dataclass
class ResultadoRelativo:
    metodo:           str   = "Formula analitica — error relativo"
    kpi:              str   = ""
    n_piloto:         int   = 0
    valores_piloto:   list  = field(default_factory=list)
    media_piloto:     float = 0.0
    sd_piloto:        float = 0.0
    cv_piloto:        float = 0.0
    t_crit:           float = 0.0
    epsilon_rel:      float = 0.0
    alpha:            float = 0.05
    n_recomendado:    int   = 0
    intervalo_bajo:   float = 0.0
    intervalo_alto:   float = 0.0
    tiempo_total_seg: float = 0.0

@dataclass
class ResultadoSecuencial:
    metodo:           str   = "Metodo secuencial — error relativo"
    kpi:              str   = ""
    n_piloto:         int   = 0
    epsilon_rel:      float = 0.0
    alpha:            float = 0.05
    n_max:            int   = 0
    historia:         list  = field(default_factory=list)
    valores:          list  = field(default_factory=list)
    n_final:          int   = 0
    media_final:      float = 0.0
    sd_final:         float = 0.0
    ic_bajo:          float = 0.0
    ic_alto:          float = 0.0
    ic_rel_final:     float = 0.0
    convergido:       bool  = False
    tiempo_total_seg: float = 0.0

# ──────────────────────────────────────────────────────────────────────────────
# Método 1: Fórmula analítica — error absoluto
# ──────────────────────────────────────────────────────────────────────────────
def calcular_n_absoluto(
    kpi:            str,
    n_piloto:       int   = 10,
    epsilon_abs:    float = 5.0,
    alpha:          float = 0.05,
    seed_base:      int   = 0,
    cfg             = None,
    valores_piloto: list  = None,
) -> ResultadoAbsoluto:
    """
    Fórmula: n = ceil( (t_{α/2, n0-1} × s / ε)² )

    Corre n_piloto corridas del modelo baseline para estimar la variabilidad
    del KPI, luego calcula cuántas corridas son necesarias para garantizar
    que el error de estimación sea ≤ epsilon_abs con nivel (1-alpha).

    Si valores_piloto se provee, usa esos datos sin correr el modelo.

    [BUG] n_recomendado ahora es siempre ≥ n_piloto.
    """
    t0 = time.time()

    vals = valores_piloto if valores_piloto is not None else \
           correr_piloto(kpi, n_piloto, seed_base=seed_base, cfg=cfg)

    n0  = len(vals)
    mu  = float(np.mean(vals))
    sd  = float(np.std(vals, ddof=1)) if n0 > 1 else 0.0
    cv  = (sd / abs(mu)) if abs(mu) > 1e-9 else float('inf')
    tc  = _t_crit(n0 - 1)
    se  = sd / math.sqrt(n0) if n0 > 0 else 0.0

    if sd == 0 or epsilon_abs == 0:
        n_rec = n0
    else:
        # [BUG] max() garantiza que n_rec nunca sea menor que n_piloto
        n_rec = max(n0, math.ceil((tc * sd / epsilon_abs) ** 2))

    elapsed = time.time() - t0
    log.info("[M1] n_piloto=%d  media=%.3f  sd=%.3f  n_rec=%d  (%.0fs)",
             n0, mu, sd, n_rec, elapsed)

    return ResultadoAbsoluto(
        kpi=kpi, n_piloto=n0, valores_piloto=[round(v, 4) for v in vals],
        media_piloto=round(mu, 4), sd_piloto=round(sd, 4), cv_piloto=round(cv, 4),
        t_crit=round(tc, 4), epsilon_abs=epsilon_abs, alpha=alpha,
        n_recomendado=n_rec,
        intervalo_bajo=round(mu - tc * se, 4), intervalo_alto=round(mu + tc * se, 4),
        tiempo_total_seg=round(elapsed, 2),
    )

# ──────────────────────────────────────────────────────────────────────────────
# Método 2: Fórmula analítica — error relativo
# ──────────────────────────────────────────────────────────────────────────────
def calcular_n_relativo(
    kpi:            str,
    n_piloto:       int   = 10,
    epsilon_rel:    float = 0.05,
    alpha:          float = 0.05,
    seed_base:      int   = 0,
    cfg             = None,
    valores_piloto: list  = None,
) -> ResultadoRelativo:
    """
    Fórmula: n = ceil( (t_{α/2, n0-1} × CV / ε_rel)² )
    donde CV = s / |μ|  (coeficiente de variación)

    Corre n_piloto corridas del modelo baseline y calcula cuántas corridas
    garantizan que el error relativo de estimación sea ≤ epsilon_rel.

    [BUG] n_recomendado ahora es siempre ≥ n_piloto.
    [BUG] media ≈ 0 genera advertencia explícita en vez de CV=inf silencioso.
    """
    t0 = time.time()

    vals = valores_piloto if valores_piloto is not None else \
           correr_piloto(kpi, n_piloto, seed_base=seed_base, cfg=cfg)

    n0  = len(vals)
    mu  = float(np.mean(vals))
    sd  = float(np.std(vals, ddof=1)) if n0 > 1 else 0.0
    se  = sd / math.sqrt(n0) if n0 > 0 else 0.0
    tc  = _t_crit(n0 - 1)

    # [BUG] Guard explícito para media ≈ 0
    if abs(mu) < 1e-9:
        log.warning(
            "[M2] Media del KPI '%s' es aproximadamente 0 (μ=%.6f). "
            "El error relativo no es aplicable. Se retorna n_rec = n_piloto.",
            kpi, mu
        )
        cv    = float('inf')
        n_rec = n0
    elif sd == 0 or epsilon_rel == 0:
        cv    = 0.0
        n_rec = n0
    else:
        cv = sd / abs(mu)
        # [BUG] max() garantiza que n_rec nunca sea menor que n_piloto
        n_rec = max(n0, math.ceil((tc * cv / epsilon_rel) ** 2))

    elapsed = time.time() - t0
    log.info("[M2] n_piloto=%d  CV=%.4f  n_rec=%d  (%.0fs)", n0, cv, n_rec, elapsed)

    return ResultadoRelativo(
        kpi=kpi, n_piloto=n0, valores_piloto=[round(v, 4) for v in vals],
        media_piloto=round(mu, 4), sd_piloto=round(sd, 4), cv_piloto=round(cv, 4),
        t_crit=round(tc, 4), epsilon_rel=epsilon_rel, alpha=alpha,
        n_recomendado=n_rec,
        intervalo_bajo=round(mu - tc * se, 4), intervalo_alto=round(mu + tc * se, 4),
        tiempo_total_seg=round(elapsed, 2),
    )

# ──────────────────────────────────────────────────────────────────────────────
# Método 3: Secuencial — error relativo
# ──────────────────────────────────────────────────────────────────────────────
def metodo_secuencial(
    kpi:         str,
    n_piloto:    int   = 5,
    epsilon_rel: float = 0.05,
    alpha:       float = 0.05,
    n_max:       int   = 100,
    seed_base:   int   = 200,
    cfg          = None,
) -> ResultadoSecuencial:
    """
    Método secuencial de Wald adaptado para error relativo.

    Criterio de parada: IC_rel = t × s / (√n × |μ|) ≤ epsilon_rel

    Proceso:
    1. Corre n_piloto corridas del modelo (arranque)
    2. Evalúa convergencia con el criterio IC_rel
    3. Si no converge → corre 1 corrida adicional del modelo → repite paso 2
    4. Para cuando converge o alcanza n_max corridas

    [BUG] seed_actual arranca en seed_base (no en seed_base + n_piloto),
          evitando solapamiento de semillas con corridas compartidas externas.
          El llamador es responsable de pasar un seed_base diferenciado.
    [MEJ] ETA mostrado durante corridas adicionales.
    """
    # [MEJ] Validar KPI al inicio
    _validar_kpi(kpi)

    t0          = time.time()
    vals        = []
    historia    = []
    # [BUG] seed_actual parte desde seed_base; el offset con el piloto
    #        compartido se maneja en reporte_num_corridas_desde_modelo
    seed_actual = seed_base
    tiempos     = []

    # Corridas piloto iniciales
    log.info("[M3] Corriendo %d corridas piloto iniciales (seed_base=%d)...",
             n_piloto, seed_base)
    for i in range(n_piloto):
        t_iter = time.time()
        val = correr_una_vez(kpi, seed_offset=seed_actual, cfg=cfg)
        dt  = time.time() - t_iter
        vals.append(val)
        tiempos.append(dt)
        seed_actual += 1
        log.info("  Piloto %2d/%d: %.3f  (%.0fs)", i + 1, n_piloto, val, dt)

    convergido = False
    while len(vals) < n_max:
        n    = len(vals)
        mu   = float(np.mean(vals))
        sd   = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        tc   = _t_crit(n - 1)
        se   = sd / math.sqrt(n) if n > 0 else 0.0
        ic_l = mu - tc * se
        ic_h = mu + tc * se

        # [BUG] Guard media ≈ 0 en el loop secuencial
        if abs(mu) < 1e-9:
            ic_rel  = float('inf')
            converge = False
            log.warning("[M3] Media ≈ 0 en n=%d, IC relativo no aplicable.", n)
        else:
            ic_rel   = (tc * se / abs(mu))
            converge = bool(ic_rel <= epsilon_rel)

        historia.append({
            "n": n, "media": round(mu, 4), "sd": round(sd, 4),
            "ic_low": round(ic_l, 4), "ic_high": round(ic_h, 4),
            "ic_rel": round(ic_rel, 6), "converge": converge,
        })

        log.info("  n=%3d | media=%8.3f | sd=%7.3f | IC_rel=%6.3f%% | obj≤%.1f%% | %s",
                 n, mu, sd, ic_rel * 100, epsilon_rel * 100,
                 "✓ CONVERGE" if converge else "...")

        if converge:
            convergido = True
            break

        # Correr 1 corrida adicional con ETA
        t_iter = time.time()
        val    = correr_una_vez(kpi, seed_offset=seed_actual, cfg=cfg)
        dt     = time.time() - t_iter
        tiempos.append(dt)
        vals.append(val)
        seed_actual += 1

        eta = np.mean(tiempos) * max(0, n_max - len(vals))
        log.info("  → Nueva corrida: %.3f  (%.0fs | ETA máx ~%.0fs)", val, dt, eta)

    n_fin  = len(vals)
    mu     = float(np.mean(vals))
    sd     = float(np.std(vals, ddof=1)) if n_fin > 1 else 0.0
    tc     = _t_crit(n_fin - 1)
    se     = sd / math.sqrt(n_fin)
    ic_rel_fin = (tc * se / abs(mu)) if abs(mu) > 1e-9 else float('inf')
    elapsed    = time.time() - t0

    if not convergido:
        log.warning("[M3] NO convergió en %d corridas. IC_rel final=%.2f%% > %.1f%%",
                    n_max, ic_rel_fin * 100, epsilon_rel * 100)
    else:
        log.info("[M3] Convergió en %d corridas. IC_rel=%.2f%%  (%.0fs)",
                 n_fin, ic_rel_fin * 100, elapsed)

    return ResultadoSecuencial(
        kpi=kpi, n_piloto=n_piloto, epsilon_rel=epsilon_rel,
        alpha=alpha, n_max=n_max, historia=historia,
        valores=[round(v, 4) for v in vals],
        n_final=n_fin, media_final=round(mu, 4), sd_final=round(sd, 4),
        ic_bajo=round(mu - tc * se, 4), ic_alto=round(mu + tc * se, 4),
        ic_rel_final=round(ic_rel_fin, 6),
        convergido=convergido, tiempo_total_seg=round(elapsed, 2),
    )

# ──────────────────────────────────────────────────────────────────────────────
# Reporte combinado — conectado con el modelo
# ──────────────────────────────────────────────────────────────────────────────
def reporte_num_corridas_desde_modelo(
    kpi:              str   = "wl_total_end",   # [BUG] era "Lista espera total al final"
    n_piloto:         int   = 10,
    epsilon_abs:      float = 5.0,
    epsilon_rel:      float = 0.05,
    alpha:            float = 0.05,
    n_max_secuencial: int   = 100,
    seed_base:        int   = 0,
    cfg               = None,
    solo_analiticos:  bool  = False,
    guardar_json:     str   = "resultado_num_corridas.json",
    reutilizar_piloto:bool  = True,
) -> dict:
    """
    Ejecuta los 3 métodos obteniendo datos directamente del modelo baseline.

    Con reutilizar_piloto=True (default), los métodos 1 y 2 comparten las
    mismas corridas piloto (más eficiente). El método 3 siempre corre
    con semillas distintas para independencia.

    Parámetros
    ----------
    kpi               : KPI a analizar (clave de KPIS_DISPONIBLES)
    n_piloto          : corridas piloto para métodos 1 y 2
    epsilon_abs       : error absoluto máximo (método 1)
    epsilon_rel       : error relativo máximo (métodos 2 y 3)
    alpha             : nivel de significancia
    n_max_secuencial  : máximo corridas para método secuencial
    seed_base         : semilla base
    cfg               : SimConfig opcional (None = usa CFG baseline)
    solo_analiticos   : si True, omite el método secuencial
    guardar_json      : archivo JSON de salida
    reutilizar_piloto : si True, métodos 1 y 2 usan las mismas corridas

    [BUG] kpi default corregido: ahora es la clave 'wl_total_end'.
    [BUG] seed del método 3 separado del piloto compartido (seed_base + n_piloto).
    [MEJ] Validación de KPI antes de correr nada.
    """
    # [MEJ] Validar KPI al inicio, antes de cualquier corrida
    _validar_kpi(kpi)

    log.info("="*60)
    log.info("MÓDULO 1 — NÚMERO DE CORRIDAS — KPI: %s", kpi)
    log.info("="*60)

    # Corridas piloto compartidas (si aplica)
    vals_compartidos = None
    if reutilizar_piloto:
        log.info("Generando corridas piloto compartidas (n=%d, seed=%d)...",
                 n_piloto, seed_base)
        vals_compartidos = correr_piloto(kpi, n_piloto, seed_base=seed_base, cfg=cfg)

    # Método 1
    log.info("\n── Método 1: error absoluto (ε=%.2f) ──", epsilon_abs)
    r1 = calcular_n_absoluto(
        kpi=kpi, n_piloto=n_piloto, epsilon_abs=epsilon_abs,
        alpha=alpha, seed_base=seed_base, cfg=cfg,
        valores_piloto=vals_compartidos,
    )

    # Método 2
    log.info("\n── Método 2: error relativo (ε=%.1f%%) ──", epsilon_rel * 100)
    r2 = calcular_n_relativo(
        kpi=kpi, n_piloto=n_piloto, epsilon_rel=epsilon_rel,
        alpha=alpha, seed_base=seed_base, cfg=cfg,
        valores_piloto=vals_compartidos,
    )

    # Método 3 (secuencial)
    r3 = None
    if not solo_analiticos:
        # [BUG] seed separado: seed_base + n_piloto evita solapamiento
        seed_seq = seed_base + n_piloto
        log.info("\n── Método 3: secuencial (ε=%.1f%%, n_max=%d, seed=%d) ──",
                 epsilon_rel * 100, n_max_secuencial, seed_seq)
        r3 = metodo_secuencial(
            kpi=kpi, n_piloto=min(n_piloto, 5),
            epsilon_rel=epsilon_rel, alpha=alpha,
            n_max=n_max_secuencial, seed_base=seed_seq, cfg=cfg,
        )

    # Resumen
    candidatos = [r1.n_recomendado, r2.n_recomendado] + ([r3.n_final] if r3 else [])
    n_final    = max(candidatos)
    mas_conservador = (
        "metodo1_absoluto"   if r1.n_recomendado == n_final else
        "metodo2_relativo"   if r2.n_recomendado == n_final else
        "metodo3_secuencial"
    )

    resultado = {
        "kpi":           kpi,
        "kpi_label":     KPIS_DISPONIBLES.get(kpi, kpi),
        "parametros": {
            "n_piloto":         n_piloto,
            "epsilon_abs":      epsilon_abs,
            "epsilon_rel":      epsilon_rel,
            "alpha":            alpha,
            "n_max_secuencial": n_max_secuencial,
            "seed_base":        seed_base,
        },
        "metodo1_absoluto":   asdict(r1),
        "metodo2_relativo":   asdict(r2),
        "metodo3_secuencial": asdict(r3) if r3 else None,
        "resumen": {
            "n_metodo1_abs":          r1.n_recomendado,
            "n_metodo2_rel":          r2.n_recomendado,
            "n_metodo3_seq":          r3.n_final if r3 else None,
            "n_recomendado_final":    n_final,
            "metodo_mas_conservador": mas_conservador,
            # [MEJ] IC relativo final del método 3 en el resumen
            "ic_rel_final_m3":        round(r3.ic_rel_final * 100, 2) if r3 else None,
        },
    }

    with open(guardar_json, "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)
    log.info("\nResultado guardado en '%s'", guardar_json)

    return resultado

# ──────────────────────────────────────────────────────────────────────────────
# Imprimir resultado
# ──────────────────────────────────────────────────────────────────────────────
def imprimir_resultado(resultado: dict):
    kpi   = resultado["kpi"]
    label = resultado["kpi_label"]
    p     = resultado["parametros"]
    res   = resultado["resumen"]
    r1    = resultado["metodo1_absoluto"]
    r2    = resultado["metodo2_relativo"]
    r3    = resultado.get("metodo3_secuencial")

    print("\n" + "="*70)
    print("MÓDULO 1 — DETERMINACIÓN NÚMERO DE CORRIDAS")
    print(f"KPI  : {label}")
    print(f"Clave: {kpi}")
    print("="*70)

    print(f"\n── Método 1: Error absoluto  (ε = {p['epsilon_abs']}) ──")
    print(f"  n piloto      : {r1['n_piloto']}")
    print(f"  Media         : {r1['media_piloto']:.3f}")
    print(f"  SD            : {r1['sd_piloto']:.3f}")
    print(f"  CV            : {r1['cv_piloto']*100:.1f}%")
    print(f"  t crítico     : {r1['t_crit']:.3f}")
    print(f"  IC 95%        : [{r1['intervalo_bajo']:.3f}, {r1['intervalo_alto']:.3f}]")
    print(f"  Tiempo        : {r1['tiempo_total_seg']:.0f}s")
    print(f"  ► n RECOMENDADO: {r1['n_recomendado']}")

    print(f"\n── Método 2: Error relativo  (ε = {p['epsilon_rel']*100:.1f}%) ──")
    print(f"  n piloto      : {r2['n_piloto']}")
    print(f"  CV            : {r2['cv_piloto']*100:.1f}%")
    print(f"  IC 95%        : [{r2['intervalo_bajo']:.3f}, {r2['intervalo_alto']:.3f}]")
    print(f"  Tiempo        : {r2['tiempo_total_seg']:.0f}s")
    print(f"  ► n RECOMENDADO: {r2['n_recomendado']}")

    if r3:
        print(f"\n── Método 3: Secuencial  (ε = {p['epsilon_rel']*100:.1f}%, n_max={p['n_max_secuencial']}) ──")
        print(f"  n final       : {r3['n_final']}")
        conv_str = "✓ SÍ" if r3["convergido"] else "✗ NO (alcanzó n_max)"
        print(f"  Convergió     : {conv_str}")
        print(f"  Media final   : {r3['media_final']:.3f}")
        print(f"  IC 95%        : [{r3['ic_bajo']:.3f}, {r3['ic_alto']:.3f}]")
        print(f"  IC_rel final  : {r3['ic_rel_final']*100:.2f}%")
        print(f"  Tiempo        : {r3['tiempo_total_seg']:.0f}s")
        print(f"  ► n FINAL      : {r3['n_final']}")

        print(f"\n  Historia convergencia (últimas 8 iter.):")
        print(f"  {'n':>5} {'media':>10} {'sd':>8} {'IC_rel%':>9} {'ok':>5}")
        print("  " + "-"*42)
        for h in r3["historia"][-8:]:
            print(f"  {h['n']:>5} {h['media']:>10.3f} {h['sd']:>8.3f} "
                  f"{h['ic_rel']*100:>8.2f}%  {'✓' if h['converge'] else ''}")

    print("\n" + "─"*70)
    print("RESUMEN")
    print("─"*70)
    print(f"  Método 1 (abs) : {res['n_metodo1_abs']:>5} corridas")
    print(f"  Método 2 (rel) : {res['n_metodo2_rel']:>5} corridas")
    if r3:
        conv  = "✓" if r3["convergido"] else "✗ no convergió"
        ic_r  = res.get("ic_rel_final_m3")
        ic_str = f"  IC_rel={ic_r:.2f}%" if ic_r is not None else ""
        print(f"  Método 3 (seq) : {res['n_metodo3_seq']:>5} corridas  {conv}{ic_str}")
    print(f"\n  ► N RECOMENDADO FINAL : {res['n_recomendado_final']}")
    print(f"    (método más conservador: {res['metodo_mas_conservador']})")
    print("="*70)

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Módulo 1: Número de corridas — conectado con simulador_clinica_baseline.py"
    )
    parser.add_argument("--kpi", type=str, default="wl_total_end",
                        choices=list(KPIS_DISPONIBLES.keys()),
                        help="KPI a analizar (clave, ver --listar_kpis)")
    parser.add_argument("--n_piloto",  type=int,   default=10)
    parser.add_argument("--eps_abs",   type=float, default=5.0,
                        help="Error absoluto máximo (unidades del KPI)")
    parser.add_argument("--eps_rel",   type=float, default=0.05,
                        help="Error relativo máximo (fracción, ej. 0.05 = 5%%)")
    parser.add_argument("--alpha",     type=float, default=0.05)
    parser.add_argument("--n_max",     type=int,   default=100,
                        help="Máximo corridas método secuencial")
    parser.add_argument("--seed",      type=int,   default=0)
    parser.add_argument("--solo_analiticos", action="store_true",
                        help="Solo métodos 1 y 2 (omite secuencial)")
    parser.add_argument("--no_reutilizar",   action="store_true",
                        help="Usa semillas distintas para cada método")
    parser.add_argument("--guardar_json", type=str, default="resultado_num_corridas.json")
    parser.add_argument("--listar_kpis",  action="store_true",
                        help="Lista los KPIs disponibles y sale")
    args = parser.parse_args()

    if args.listar_kpis:
        print("\nKPIs disponibles:")
        for k, v in KPIS_DISPONIBLES.items():
            print(f"  {k:<45} {v}")
        sys.exit(0)

    resultado = reporte_num_corridas_desde_modelo(
        kpi               = args.kpi,
        n_piloto          = args.n_piloto,
        epsilon_abs       = args.eps_abs,
        epsilon_rel       = args.eps_rel,
        alpha             = args.alpha,
        n_max_secuencial  = args.n_max,
        seed_base         = args.seed,
        cfg               = None,
        solo_analiticos   = args.solo_analiticos,
        guardar_json      = args.guardar_json,
        reutilizar_piloto = not args.no_reutilizar,
    )

    imprimir_resultado(resultado)
