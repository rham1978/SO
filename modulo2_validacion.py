"""
===============================================================================
MÓDULO 2 — VALIDACIÓN DEL MODELO VS REALIDAD  (versión corregida)
===============================================================================
Correcciones aplicadas:
  1. diagnostico_backlog() ya no usa stats hardcodeadas — lee el Excel dinámicamente
  2. Import del simulador robusto: busca en directorio del script Y en cwd
  3. Mejor manejo de errores con mensajes claros
  4. Compatible con datos_reales.xlsx (n=626, media=94.6, mediana=71.0)
===============================================================================
"""

import numpy as np
import json
import logging
import argparse
import time
import sys
import os
from dataclasses import dataclass, field, asdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("validacion")

MODOS = {
    "sin_backlog": "_tts_first_sin_backlog_min",
    "nuevos":      "_tts_first_nuevos_min",
    "proceso":     "_tts_first_proceso_min",
    "total":       "_tts_first_total_min",
    "backlog":     "_tts_first_backlog_solo_min",
}
MODO_RECOMENDADO = "sin_backlog"

DESCRIPCIONES_MODO = {
    "sin_backlog": "Pacientes nuevos (enqueued_at>0) — comparable directo con Excel  ← RECOMENDADO",
    "nuevos":      "Pacientes con enqueued_at≥0 (nuevos + exactamente t=0)",
    "proceso":     "Todos los atendidos, tiempo solo dentro de la simulación",
    "total":       "Todos los atendidos, tiempo completo incl. backlog histórico",
    "backlog":     "Solo pacientes con backlog histórico (enqueued_at<0)",
}

# ──────────────────────────────────────────────────────────────
# Import robusto del simulador
# ──────────────────────────────────────────────────────────────
_cache = None

def _importar_baseline():
    global _cache
    if _cache:
        return _cache

    # Buscar en: 1) directorio del script, 2) cwd, 3) sys.path actual
    candidatos = [
        os.path.dirname(os.path.abspath(__file__)),
        os.getcwd(),
    ]
    for d in candidatos:
        if d not in sys.path:
            sys.path.insert(0, d)

    try:
        from simulador_clinica_baseline import run_once, CFG
        _cache = (run_once, CFG)
        log.info("✓ Simulador importado correctamente.")
        return _cache
    except ImportError as e:
        msg = (
            f"\n{'='*60}\n"
            f"ERROR: No se encontró 'simulador_clinica_baseline.py'\n"
            f"{'='*60}\n"
            f"Buscado en:\n"
            + "\n".join(f"  · {d}" for d in candidatos) +
            f"\n\nSoluciones:\n"
            f"  1. Asegúrate que ambos archivos estén en la misma carpeta\n"
            f"  2. Corre el script desde esa carpeta:\n"
            f"     cd 'C:\\Users\\raraneda\\Desktop\\Proyecto CRS'\n"
            f"     python modulo2_validacion.py --excel datos_reales.xlsx\n"
            f"  3. Error original: {e}\n"
            f"{'='*60}"
        )
        print(msg)
        raise SystemExit(1)


# ──────────────────────────────────────────────────────────────
# Verificar campo duplicado en SimConfig antes de correr
# ──────────────────────────────────────────────────────────────
def _verificar_simulador():
    """
    Verifica que el simulador no tenga el campo duplicado
    ugd_us_duration_min que causa TypeError.
    """
    try:
        run_once, CFG = _importar_baseline()
        # Intenta instanciar — si hay campo duplicado falla aquí
        from simulador_clinica_baseline import SimConfig
        _ = SimConfig()
        log.info("✓ SimConfig instancia correctamente — sin campos duplicados.")
        return True
    except TypeError as e:
        msg = (
            f"\n{'='*60}\n"
            f"ERROR en SimConfig: {e}\n"
            f"{'='*60}\n"
            f"Causa probable: campo 'ugd_us_duration_min' duplicado.\n\n"
            f"Solución — en simulador_clinica_baseline.py busca:\n\n"
            f"    ugd_lab_per_week:   int   = 54\n"
            f"    ugd_us_duration_min: float = 60.0   ← ELIMINAR ESTA LÍNEA\n"
            f"    ugd_lab_duration_min: float = 60.0\n\n"
            f"Deja solo:\n\n"
            f"    ugd_lab_per_week:   int   = 54\n"
            f"    ugd_lab_duration_min: float = 60.0\n"
            f"{'='*60}"
        )
        print(msg)
        raise SystemExit(1)


# ──────────────────────────────────────────────────────────────
# Carga de datos reales
# ──────────────────────────────────────────────────────────────
def cargar_datos_reales(path_excel: str) -> list:
    """Lee días de espera reales desde Excel — columna única sin header."""
    if not os.path.exists(path_excel):
        # Buscar en directorio del script
        alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), path_excel)
        if os.path.exists(alt):
            path_excel = alt
        else:
            print(f"\nERROR: No se encontró '{path_excel}'")
            print(f"  Buscado en: {os.getcwd()}")
            print(f"  Y en:       {os.path.dirname(os.path.abspath(__file__))}")
            raise SystemExit(1)

    try:
        import pandas as pd
        df    = pd.read_excel(path_excel, header=None)
        col   = df.iloc[:, 0].dropna()
        datos = [float(v) for v in col if isinstance(v, (int, float)) and not np.isnan(float(v))]
        if not datos:
            print(f"\nERROR: El Excel '{path_excel}' no contiene datos numéricos en columna 1.")
            raise SystemExit(1)
        log.info("✓ Excel '%s': n=%d  media=%.1f  mediana=%.1f  sd=%.1f  min=%d  max=%d",
                 path_excel, len(datos),
                 np.mean(datos), np.median(datos),
                 np.std(datos, ddof=1), min(datos), max(datos))
        return datos
    except ImportError:
        print("\nERROR: Falta pandas. Instala con:  pip install pandas openpyxl")
        raise SystemExit(1)
    except Exception as e:
        print(f"\nERROR al leer '{path_excel}': {e}")
        raise SystemExit(1)


# ──────────────────────────────────────────────────────────────
# Correr modelo y extraer esperas
# ──────────────────────────────────────────────────────────────
def correr_y_extraer(n_corridas: int = 1, seed_base: int = 202,
                     modo: str = MODO_RECOMENDADO, cfg=None) -> list:
    if modo not in MODOS:
        raise ValueError(f"Modo '{modo}' no válido. Opciones: {list(MODOS.keys())}")
    clave = MODOS[modo]
    run_once, CFG_base = _importar_baseline()
    cfg_usar = cfg or CFG_base

    todas = []
    for r in range(n_corridas):
        t0  = time.time()
        res = run_once(seed_offset=seed_base + r, cfg=cfg_usar)
        lst = res.get(clave, [])
        if not lst:
            log.warning("  Corrida %d: lista '%s' vacía — verifica el modo.", r+1, clave)
        dias = [v / (60 * 24.0) for v in lst if v >= 0]
        todas.extend(dias)
        log.info("  Corrida %d/%d: %d esperas (modo='%s') %.0fs",
                 r+1, n_corridas, len(dias), modo, time.time()-t0)

    if not todas:
        print(f"\nADVERTENCIA: No se obtuvieron datos en modo='{modo}'.")
        print(f"  Prueba con --modo proceso  o  --modo total")

    log.info("Total esperas simuladas: %d (modo='%s')", len(todas), modo)
    return todas


def correr_todas_las_listas(n_corridas: int = 1, seed_base: int = 202, cfg=None) -> dict:
    run_once, CFG_base = _importar_baseline()
    cfg_usar = cfg or CFG_base
    acum = {m: [] for m in MODOS}
    for r in range(n_corridas):
        res = run_once(seed_offset=seed_base + r, cfg=cfg_usar)
        for modo, clave in MODOS.items():
            lst = res.get(clave, [])
            acum[modo].extend([v / (60*24.0) for v in lst if v >= 0])
    return acum


# ──────────────────────────────────────────────────────────────
# Estadísticas descriptivas
# ──────────────────────────────────────────────────────────────
def _desc(arr: list) -> dict:
    a = np.array(arr, dtype=float)
    a = a[~np.isnan(a)]
    n = len(a)
    if n == 0:
        return {k: 0 for k in ["n","media","mediana","sd","min","p25","p75","p90","max"]}
    return {
        "n":       int(n),
        "media":   round(float(np.mean(a)), 2),
        "mediana": round(float(np.median(a)), 2),
        "sd":      round(float(np.std(a, ddof=1)) if n > 1 else 0.0, 2),
        "min":     round(float(np.min(a)), 2),
        "p25":     round(float(np.percentile(a, 25)), 2),
        "p75":     round(float(np.percentile(a, 75)), 2),
        "p90":     round(float(np.percentile(a, 90)), 2),
        "max":     round(float(np.max(a)), 2),
    }


# ──────────────────────────────────────────────────────────────
# Test U Mann-Whitney
# ──────────────────────────────────────────────────────────────
def _mannwhitney(x: list, y: list) -> tuple:
    try:
        from scipy.stats import mannwhitneyu
        U, p = mannwhitneyu(np.array(x), np.array(y), alternative='two-sided')
        return float(U), float(p), "scipy"
    except ImportError:
        pass
    import math
    a, b   = np.array(x, float), np.array(y, float)
    n1, n2 = len(a), len(b)
    comb   = np.concatenate([a, b])
    labels = np.array([0]*n1 + [1]*n2)
    order  = np.argsort(comb, kind='stable')
    ranks  = np.empty(len(comb))
    ranks[order] = np.arange(1, len(comb)+1, dtype=float)
    sv = comb[order]; i = 0
    while i < len(sv):
        j = i
        while j < len(sv) and sv[j] == sv[i]: j += 1
        ranks[order[i:j]] = (i+1+j)/2.0; i = j
    R1  = ranks[labels==0].sum()
    U1  = R1 - n1*(n1+1)/2.0
    U   = min(U1, n1*n2-U1)
    mu  = n1*n2/2.0
    sig = math.sqrt(n1*n2*(n1+n2+1)/12.0)
    z   = (U-mu)/sig if sig > 0 else 0.0
    def phi(z):
        t    = 1/(1+0.2316419*abs(z))
        poly = t*(0.319381530+t*(-0.356563782+t*(1.781477937+t*(-1.821255978+t*1.330274429))))
        return 1-(1/math.sqrt(2*math.pi))*math.exp(-0.5*z**2)*poly
    return float(U), float(2*(1-phi(abs(z)))), "manual"


# ──────────────────────────────────────────────────────────────
# Dataclass resultado
# ──────────────────────────────────────────────────────────────
@dataclass
class ResultadoValidacion:
    modo:                str   = "sin_backlog"
    descripcion_modo:    str   = ""
    n_corridas_modelo:   int   = 1
    alpha:               float = 0.05
    desc_real:           dict  = field(default_factory=dict)
    desc_simulado:       dict  = field(default_factory=dict)
    u_statistic:         float = 0.0
    p_valor:             float = 0.0
    implementacion:      str   = ""
    h0_rechazada:        bool  = False
    conclusion:          str   = ""
    diferencia_medias:   float = 0.0
    diferencia_medianas: float = 0.0
    pct_error_media:     float = 0.0
    pct_error_mediana:   float = 0.0
    tiempo_total_seg:    float = 0.0


# ──────────────────────────────────────────────────────────────
# Validar vs realidad
# ──────────────────────────────────────────────────────────────
def validar_vs_realidad(datos_reales: list, datos_sim: list,
                        alpha: float = 0.05, modo: str = MODO_RECOMENDADO,
                        n_corridas: int = 1, verbose: bool = True) -> ResultadoValidacion:
    if not datos_sim:
        print(f"\nERROR: No hay datos simulados para comparar (modo='{modo}').")
        print("  Prueba: --modo proceso  o  --modo total")
        raise SystemExit(1)

    t0         = time.time()
    U, p, impl = _mannwhitney(datos_reales, datos_sim)
    h0         = bool(p < alpha)
    dr, ds     = _desc(datos_reales), _desc(datos_sim)
    d_med      = round(ds["media"]   - dr["media"],   2)
    d_mdn      = round(ds["mediana"] - dr["mediana"], 2)
    pct_med    = round((d_med/dr["media"]*100)   if dr["media"]   != 0 else 0.0, 2)
    pct_mdn    = round((d_mdn/dr["mediana"]*100) if dr["mediana"] != 0 else 0.0, 2)

    if h0:
        dir_ = "sobreestima" if d_med > 0 else "subestima"
        concl = (f"SE RECHAZA H0 (p={p:.6f} < α={alpha}): distribuciones significativamente "
                 f"distintas. El modelo {dir_} la espera real en {abs(pct_med):.1f}% (media) "
                 f"y {abs(pct_mdn):.1f}% (mediana). Considerar recalibrar parámetros.")
    else:
        concl = (f"NO SE RECHAZA H0 (p={p:.6f} ≥ α={alpha}): sin evidencia estadística "
                 "de diferencia. El modelo es compatible con los datos reales.")

    r = ResultadoValidacion(
        modo=modo, descripcion_modo=DESCRIPCIONES_MODO.get(modo,""),
        n_corridas_modelo=n_corridas, alpha=alpha,
        desc_real=dr, desc_simulado=ds,
        u_statistic=round(U,2), p_valor=round(p,8),
        implementacion=impl, h0_rechazada=h0, conclusion=concl,
        diferencia_medias=d_med, diferencia_medianas=d_mdn,
        pct_error_media=pct_med, pct_error_mediana=pct_mdn,
        tiempo_total_seg=round(time.time()-t0, 2),
    )
    if verbose:
        _imprimir(r)
    return r


def _imprimir(r: ResultadoValidacion):
    print("\n" + "="*72)
    print("MÓDULO 2 — VALIDACIÓN MODELO VS REALIDAD")
    print(f"KPI  : Días espera primera consulta")
    print(f"Modo : {r.modo} — {r.descripcion_modo}")
    print("="*72)
    campos = [("n","N"),("media","Media (días)"),("mediana","Mediana (días)"),
              ("sd","SD"),("min","Mínimo"),("p25","P25"),
              ("p75","P75"),("p90","P90"),("max","Máximo")]
    print(f"\n  {'Estadístico':<22} {'Real':>10} {'Simulado':>10} {'Δ':>10} {'Δ%':>8}")
    print("  " + "─"*64)
    for key, label in campos:
        vr = r.desc_real.get(key, 0)
        vs = r.desc_simulado.get(key, 0)
        if key == "n":
            print(f"  {label:<22} {vr:>10} {vs:>10}")
        else:
            diff = vs - vr
            pct  = (diff/vr*100) if vr != 0 else 0.0
            print(f"  {label:<22} {vr:>10.1f} {vs:>10.1f} {diff:>+9.1f} {pct:>+7.1f}%")
    print(f"\n  Error relativo media   : {r.pct_error_media:+.1f}%")
    print(f"  Error relativo mediana : {r.pct_error_mediana:+.1f}%")
    print(f"\n  {'─'*64}")
    print(f"  Test U Mann-Whitney  (bilateral, α={r.alpha})")
    print(f"  {'─'*64}")
    print(f"  U estadístico   : {r.u_statistic:,.0f}")
    print(f"  p-valor         : {r.p_valor:.8f}")
    print(f"  Implementación  : {r.implementacion}")
    print(f"  Corridas modelo : {r.n_corridas_modelo}")
    print(f"  Tiempo          : {r.tiempo_total_seg:.0f}s")
    sym = "⚠  RECHAZA H0" if r.h0_rechazada else "✓  NO RECHAZA H0"
    print(f"\n  {sym}")
    print(f"\n  → {r.conclusion}")
    print("="*72)


# ──────────────────────────────────────────────────────────────
# Diagnóstico de backlog — CORREGIDO: usa Excel dinámicamente
# ──────────────────────────────────────────────────────────────
def diagnostico_backlog(seed_base: int = 202, cfg=None, path_excel: str = None):
    """
    Corre el modelo 1 vez y muestra descomposición backlog vs nuevos.
    Si se provee path_excel, usa esas stats como referencia real.
    """
    run_once, CFG_base = _importar_baseline()
    cfg_usar = cfg or CFG_base
    log.info("Corriendo diagnóstico de backlog (1 corrida)...")
    res = run_once(seed_offset=seed_base, cfg=cfg_usar)

    listas = {
        "sin_backlog (enqueued>0)":    [v/(60*24) for v in res.get("_tts_first_sin_backlog_min",[])  if v>=0],
        "nuevos (enqueued≥0)":         [v/(60*24) for v in res.get("_tts_first_nuevos_min",[])       if v>=0],
        "proceso (todos, solo sim)":   [v/(60*24) for v in res.get("_tts_first_proceso_min",[])      if v>=0],
        "total (todos, incl.bk hist)": [v/(60*24) for v in res.get("_tts_first_total_min",[])        if v>=0],
        "backlog solo (enqueued<0)":   [v/(60*24) for v in res.get("_tts_first_backlog_solo_min",[]) if v>=0],
    }

    # Stats reales — desde Excel si disponible, sino placeholder
    if path_excel and os.path.exists(path_excel):
        dr = cargar_datos_reales(path_excel)
        real_n   = len(dr)
        real_med = round(np.mean(dr), 1)
        real_mdn = round(np.median(dr), 1)
        real_sd  = round(np.std(dr, ddof=1), 1)
        real_p25 = round(np.percentile(dr, 25), 1)
        real_p75 = round(np.percentile(dr, 75), 1)
    else:
        real_n, real_med, real_mdn = "?", "?", "?"
        real_sd, real_p25, real_p75 = "?", "?", "?"
        log.warning("No se especificó --excel — stats reales no disponibles para diagnóstico.")

    print("\n" + "="*80)
    print("DIAGNÓSTICO DE BACKLOG — PRIMERA CONSULTA")
    print("="*80)
    print(f"\n  {'Grupo':<38} {'n':>5} {'media':>8} {'mediana':>9} {'sd':>8} {'p25':>7} {'p75':>7}")
    print("  " + "─"*82)
    for nombre, arr in listas.items():
        if not arr:
            print(f"  {nombre:<38} {'(vacío)':>5}")
            continue
        a = np.array(arr)
        print(f"  {nombre:<38} {len(a):>5} {np.mean(a):>8.1f} {np.median(a):>9.1f} "
              f"{np.std(a,ddof=1):>8.1f} {np.percentile(a,25):>7.1f} {np.percentile(a,75):>7.1f}")

    print(f"  {'─'*82}")
    print(f"  {'REAL Excel (referencia)':<38} {str(real_n):>5} {str(real_med):>8} "
          f"{str(real_mdn):>9} {str(real_sd):>8} {str(real_p25):>7} {str(real_p75):>7}")
    print("="*80)

    sin_bk = listas.get("sin_backlog (enqueued>0)", [])
    if sin_bk and isinstance(real_med, float):
        err_med = (np.mean(sin_bk) - real_med) / real_med * 100
        err_mdn = (np.median(sin_bk) - real_mdn) / real_mdn * 100 if real_mdn else 0
        print(f"\n  Modo RECOMENDADO (sin_backlog):")
        print(f"  Media simulada   : {np.mean(sin_bk):.1f} días  (real: {real_med} días)  error: {err_med:+.1f}%")
        print(f"  Mediana simulada : {np.median(sin_bk):.1f} días  (real: {real_mdn} días)  error: {err_mdn:+.1f}%")
        n_bk  = len(listas.get("backlog solo (enqueued<0)",[]))
        n_tot = len(listas.get("total (todos, incl.bk hist)",[]))
        if n_tot > 0:
            print(f"\n  El {n_bk/n_tot*100:.0f}% de atendidos traían backlog histórico (enqueued_at<0).")
    print("="*80)


# ──────────────────────────────────────────────────────────────
# Comparar modos
# ──────────────────────────────────────────────────────────────
def comparar_modos(path_excel: str, seed_base: int = 202,
                   alpha: float = 0.05, cfg=None) -> dict:
    datos_reales = cargar_datos_reales(path_excel)
    listas       = correr_todas_las_listas(n_corridas=1, seed_base=seed_base, cfg=cfg)
    dr           = _desc(datos_reales)

    print("\n" + "="*80)
    print("COMPARACIÓN DE MODOS vs DATOS REALES")
    print(f"Excel: n={dr['n']}  media={dr['media']:.1f}  mediana={dr['mediana']:.1f}  sd={dr['sd']:.1f}")
    print("="*80)
    print(f"\n  {'Modo':<18} {'n':>5} {'media':>8} {'mediana':>9} {'err%media':>10} {'p-valor':>12} {'H0':>12}")
    print("  " + "─"*78)

    res = {}
    for modo in MODOS:
        datos = listas[modo]
        if not datos:
            print(f"  {modo:<18} {'(vacío)':>5}")
            continue
        ds  = _desc(datos)
        U, p, _ = _mannwhitney(datos_reales, datos)
        err = (ds["media"]-dr["media"])/dr["media"]*100 if dr["media"] != 0 else 0
        h0str = "rechaza ✗" if p < alpha else "acepta  ✓"
        recom = " ← RECOMENDADO" if modo == MODO_RECOMENDADO else ""
        print(f"  {modo:<18} {ds['n']:>5} {ds['media']:>8.1f} {ds['mediana']:>9.1f} "
              f"{err:>+9.1f}% {p:>12.6f} {h0str:>12}{recom}")
        res[modo] = {"desc": ds, "U": U, "p": p,
                     "rechaza_h0": p < alpha, "pct_error_media": round(err,1)}

    print(f"\n  {'REAL':<18} {dr['n']:>5} {dr['media']:>8.1f} {dr['mediana']:>9.1f}")
    print("="*80)
    print("\n  INTERPRETACIÓN:")
    print("  · Si todos los modos rechazan H0 → modelo subestima/sobreestima sistemáticamente")
    print("  · El modo con menor |err%media| y mayor p-valor es el más compatible")
    print("  · sin_backlog es el más comparable con datos reales de pacientes nuevos")
    return res


# ──────────────────────────────────────────────────────────────
# Pipeline completo
# ──────────────────────────────────────────────────────────────
def validacion_completa(
    path_excel:   str,
    n_corridas:   int   = 1,
    seed_base:    int   = 202,
    alpha:        float = 0.05,
    modo:         str   = MODO_RECOMENDADO,
    cfg           = None,
    guardar_json: str   = "resultado_validacion.json",
) -> ResultadoValidacion:
    t0 = time.time()
    log.info("="*60)
    log.info("MÓDULO 2 — VALIDACIÓN  modo='%s'  n_corridas=%d", modo, n_corridas)
    log.info("="*60)

    log.info("[1/3] Cargando datos reales del Excel...")
    datos_reales = cargar_datos_reales(path_excel)

    log.info("[2/3] Corriendo modelo y extrayendo TTS primera consulta...")
    datos_sim = correr_y_extraer(n_corridas, seed_base, modo, cfg)

    log.info("[3/3] Test U Mann-Whitney...")
    resultado = validar_vs_realidad(datos_reales, datos_sim, alpha, modo, n_corridas, verbose=True)

    out = asdict(resultado)
    out.update({"archivo_excel": path_excel, "seed_base": seed_base})
    with open(guardar_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log.info("Guardado en '%s' (%.0fs total)", guardar_json, time.time()-t0)
    return resultado


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Módulo 2: Validación modelo vs realidad — Test U Mann-Whitney"
    )
    parser.add_argument("--excel",          type=str,  default=None)
    parser.add_argument("--n_corridas",     type=int,  default=1)
    parser.add_argument("--seed",           type=int,  default=202)
    parser.add_argument("--alpha",          type=float, default=0.05)
    parser.add_argument("--modo",           type=str,  default=MODO_RECOMENDADO,
                        choices=list(MODOS.keys()),
                        help=f"Modo de extracción (default: {MODO_RECOMENDADO})")
    parser.add_argument("--guardar_json",   type=str,  default="resultado_validacion.json")
    parser.add_argument("--comparar_modos", action="store_true")
    parser.add_argument("--diagnostico",    action="store_true")
    parser.add_argument("--demo",           action="store_true",
                        help="Modo demo con datos sintéticos (no requiere simulador)")
    args = parser.parse_args()

    if args.demo:
        print("\n[MODO DEMO] Datos sintéticos — no requiere simulador")
        np.random.seed(42)
        real = list(np.random.exponential(scale=94.6, size=626))
        sim  = list(np.abs(np.random.normal(loc=80,   scale=55,  size=400)))
        r = validar_vs_realidad(real, sim, args.alpha, "sin_backlog", 1, verbose=True)
        with open("resultado_validacion_demo.json","w") as f:
            json.dump(asdict(r), f, ensure_ascii=False, indent=2)
        print("\n✓ Demo completado. Resultado en resultado_validacion_demo.json")

    elif args.diagnostico:
        _verificar_simulador()
        diagnostico_backlog(seed_base=args.seed, path_excel=args.excel)

    elif args.excel is None:
        parser.print_help()
        print(f"\n⚠  Especifica --excel o usa --demo")
        print(f"\n   Ejemplo rápido:")
        print(f"   python modulo2_validacion.py --excel datos_reales.xlsx")
        print(f"\n   Sin simulador (demo):")
        print(f"   python modulo2_validacion.py --demo")

    elif args.comparar_modos:
        _verificar_simulador()
        comparar_modos(args.excel, seed_base=args.seed, alpha=args.alpha)

    else:
        _verificar_simulador()
        validacion_completa(
            path_excel   = args.excel,
            n_corridas   = args.n_corridas,
            seed_base    = args.seed,
            alpha        = args.alpha,
            modo         = args.modo,
            guardar_json = args.guardar_json,
        )