"""
Políticas de referencia:
  - aleatoria: elige una acción uniforme del catálogo cada semana (referencia trivial).
  - fija(a): usa siempre la misma configuración a.
  - mejor fija: búsqueda exhaustiva sobre el catálogo en las semillas de VALIDACIÓN,
    con el mismo presupuesto de simulación que los agentes (ver presupuesto.py):
    las 28 acciones con las N_TAMIZ semillas (todas las de validación; si N_TAMIZ < 100,
    confirmación de las TOP_K mejores con las restantes). Se elige por L (métrica principal).
  - SMAC-SK con el mismo presupuesto: ver smac_sk_presupuesto.py (compite con los demás).
  - SMAC-SK de la tesis (solo contexto): el incumbente de SMAC con Stochastic
    Kriging (modulo7_smac_sk.py), aplicado tal cual todo el año. Es una referencia
    de CONTEXTO: usa más cupos que el presupuesto del catálogo y además cambia
    parámetros de gestión (bloqueos, contactabilidad, agentes) que el agente no controla.

Uso:
  python politicas_referencia.py                 # presupuesto completo, validación
  python politicas_referencia.py --humo          # corrida pequeña en semillas de humo
Escribe <out>/referencias.json
"""
import argparse, functools, json, os, time
import numpy as np
from catalogo import CATALOGO, N_ACCIONES, ACCION_BASE, OPERACION_ACTUAL
from entorno import SEMILLAS_VAL, SEMILLAS_HUMO
from evaluar import evaluar, resumen
import presupuesto as P

# Incumbente SMAC-SK de la tesis (RESUMEN_MODELOS_IFORS.md, re-evaluado n = 50).
# Nombres de modulo7_smac_sk._aplicar_config_a_cfg → campos de SimConfig.
SMAC_SK_CFG = {
    "fixed_weekly_capacity": 30,         # horas_especialista_1ra (cupos 1ra/sem)
    "fixed_post_control_capacity": 70,   # horas_control_post
    "ugd_lab_per_week": 100,             # cupos_laboratorio_ugd (sin efecto en el DES)
    "mat_us_per_week": 50,               # cupos_ecografia_matrona
    "ugd_us_per_week": 50,               # cupos_ecografia_ugd
    "publish_lead_workdays": 1,          # dias_publicacion
    "matrona_capacity": 1,               # num_matronas
    "agent_capacity": 4,                 # num_agentes_ugd
    "blocked_pct": 0.05,                 # pct_bloqueo_1ra
    "empty_control_p_ugd": 0.05,         # pct_consultas_vacias
    "not_contactable_p": 0.05,           # pct_no_contactabilidad
    "blocked_pct_post_control": 0.05,    # pct_bloqueo_post_control
}
# Versión "solo cupos": las 4 palancas de capacidad del incumbente, gestión como en la base.
SMAC_SK_CUPOS = {k: SMAC_SK_CFG[k] for k in
                 ("fixed_weekly_capacity", "fixed_post_control_capacity", "mat_us_per_week", "ugd_us_per_week")}


def _fija(a, obs):
    return a


def hacer_fija(a, seed):
    return functools.partial(_fija, a)


def _aleatoria(rng, obs):
    return int(rng.integers(N_ACCIONES))


def hacer_aleatoria(seed):
    return functools.partial(_aleatoria, np.random.default_rng(10**6 + seed))


def evaluar_smac_sk(semillas, solo_cupos=False):
    cfg = SMAC_SK_CUPOS if solo_cupos else SMAC_SK_CFG
    return evaluar(functools.partial(hacer_fija, ACCION_BASE), semillas,
                   cfg_overrides=cfg, aplicar_acciones=False)


def evaluar_operacion_actual(semillas):
    """Operación actual del CRS (16/40/50/25): no cumple la regla 3:1, se evalúa fuera del catálogo."""
    return evaluar(functools.partial(hacer_fija, ACCION_BASE), semillas,
                   cfg_overrides=OPERACION_ACTUAL, aplicar_acciones=False)


def mejor_fija(semillas, n_tamiz, top_k):
    """Búsqueda exhaustiva con presupuesto 42·n_tamiz + top_k·(len(semillas) − n_tamiz)."""
    semillas = list(semillas)
    tamiz = {}
    for a in range(N_ACCIONES):
        tamiz[a] = evaluar(functools.partial(hacer_fija, a), semillas[:n_tamiz])
        print(f"fija {a:3d} {CATALOGO[a]} L {resumen(tamiz[a])[0]:.1f}", flush=True)
    resto = semillas[n_tamiz:]
    if not resto or top_k == 0:
        mejor = min(tamiz, key=lambda a: resumen(tamiz[a])[0])
        return mejor, tamiz, {mejor: tamiz[mejor]}, n_tamiz * N_ACCIONES
    top = sorted(tamiz, key=lambda a: resumen(tamiz[a])[0])[:top_k]
    final = {}
    for a in top:
        final[a] = tamiz[a] + evaluar(functools.partial(hacer_fija, a), semillas[n_tamiz:])
        print(f"finalista {a} L {resumen(final[a])[0]:.2f} ± {resumen(final[a])[1]:.2f}", flush=True)
    mejor = min(top, key=lambda a: resumen(final[a])[0])
    gasto = n_tamiz * N_ACCIONES + top_k * (len(semillas) - n_tamiz)
    return mejor, tamiz, final, gasto


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--humo", action="store_true")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    if a.humo:
        H = P.HUMO
        semillas = list(SEMILLAS_HUMO[:H["N_TAMIZ"] + H["N_CONF"]])
        n_tamiz, top_k, out = H["N_TAMIZ"], H["TOP_K"], a.out or "resultados/humo"
    else:
        semillas, n_tamiz, top_k, out = list(SEMILLAS_VAL), P.N_TAMIZ, P.TOP_K, a.out or "resultados/validacion"
    os.makedirs(out, exist_ok=True)
    t = time.time()
    mejor, tamiz, final, gasto = mejor_fija(semillas, n_tamiz, top_k)
    res = {"catalogo": CATALOGO, "semillas": semillas, "presupuesto_mejor_fija": gasto,
           "tamizaje": tamiz, "finalistas": final, "mejor_fija": mejor}
    res["base"] = evaluar_operacion_actual(semillas)
    res["aleatoria"] = evaluar(hacer_aleatoria, semillas)
    res["smac_sk"] = evaluar_smac_sk(semillas)
    res["smac_sk_solo_cupos"] = evaluar_smac_sk(semillas, solo_cupos=True)
    for k in ("base", "aleatoria", "smac_sk", "smac_sk_solo_cupos"):
        print(f"{k:20s} L {resumen(res[k])[0]:.1f} ± {resumen(res[k])[1]:.1f}   "
              f"TTS {resumen(res[k], 'tts_full_days_mean')[0]:.1f}   "
              f"atenciones {resumen(res[k], 'total_atenciones')[0]:.0f}")
    print(f"mejor fija: {mejor} {CATALOGO[mejor]}  (presupuesto {gasto} años simulados, {time.time()-t:.0f}s)")
    P.guardar_json(res, os.path.join(out, "referencias.json"), indent=1, default=str)
