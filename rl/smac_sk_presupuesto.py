"""
SMAC-SK (optimizador de la tesis, ../modulo7_smac_sk.py) re-optimizado con las MISMAS
reglas que los demás métodos:

  - Mismo presupuesto de cupos: 1ra consulta + control = 56, ecografía + laboratorio de
    matrona = 75, cirugía 20. Se optimizan 2 enteros: cupos de 1ra (8–28) y de ecografía
    (41–59), con los mismos límites del catálogo pero en pasos de 1 (399 configuraciones).
  - Misma métrica: L (pacientes promedio en el sistema).
  - Mismas semillas: solo validación (10000–10099), con números aleatorios comunes.
  - Mismo presupuesto de simulación B (presupuesto.py):
        búsqueda   N_TRIALS configuraciones × N_REPS semillas de validación (las mismas para todas)
        confirmación las TOP_K mejores × las 100 − N_REPS semillas restantes
        total      N_TRIALS·N_REPS + TOP_K·(100 − N_REPS) ≤ B
  - Surrogate Stochastic Kriging, EI y diseño inicial LHS, igual que `_build_hpo_sk` de la tesis.

Uso:
  python smac_sk_presupuesto.py            # presupuesto completo
  python smac_sk_presupuesto.py --humo     # corrida pequeña
Escribe <out>/smac_sk.json
"""
from __future__ import annotations

import argparse
import functools
import json
import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import modulo7_smac_sk as m7  # noqa: E402

from catalogo import BASE, CUPOS_ESPECIALISTA, CUPOS_EXAMENES, SURG_FIJO, ACCION_BASE  # noqa: E402
from entorno import SEMILLAS_VAL, SEMILLAS_HUMO  # noqa: E402
from evaluar import evaluar, resumen  # noqa: E402
import presupuesto as P  # noqa: E402

logging.getLogger("smac_sk").setLevel(logging.WARNING)

N_REPS, N_TRIALS, TOP_K = 10, 250, 5   # 250·10 + 5·90 = 2 950 ≤ B
assert N_TRIALS * N_REPS + TOP_K * (100 - N_REPS) <= P.B


def cfg_cupos(first: int, eco: int) -> dict:
    """Configuración del simulador para un reparto (first, eco) dentro de los presupuestos."""
    return {"fixed_weekly_capacity": int(first),
            "fixed_post_control_capacity": int(CUPOS_ESPECIALISTA - first),
            "ugd_us_per_week": int(eco) // 2, "mat_us_per_week": int(eco) - int(eco) // 2,
            "mat_lab_per_week": int(CUPOS_EXAMENES - eco),
            "surg_throughput_per_week": SURG_FIJO}


def _fija(obs):
    return ACCION_BASE  # se ignora: aplicar_acciones=False deja la configuración fija


def _hacer(seed):
    return _fija


def evaluar_cupos(first, eco, semillas):
    return evaluar(_hacer, semillas, cfg_overrides=cfg_cupos(first, eco), aplicar_acciones=False)


def espacio(seed):
    from ConfigSpace import ConfigurationSpace
    from ConfigSpace.hyperparameters import UniformIntegerHyperparameter
    cs = ConfigurationSpace(seed=seed)
    cs.add([UniformIntegerHyperparameter("cupos_1ra", 8, 28, default_value=BASE["first"]),
            UniformIntegerHyperparameter("cupos_eco", 41, 59, default_value=BASE["eco"])])
    return cs


def optimizar(semillas, n_reps, n_trials, top_k, out, seed=0):
    from smac import Scenario
    semillas = list(semillas)
    busq, resto = semillas[:n_reps], semillas[n_reps:]
    m7._variance_store.clear(); m7._n_reps_store.clear()
    historia = {}

    def objetivo(config, seed: int = 0) -> float:
        f, e = int(config["cupos_1ra"]), int(config["cupos_eco"])
        r = evaluar_cupos(f, e, busq)
        L = np.array([x["L"] for x in r])
        key = m7._config_key(config)
        m7._variance_store[key] = float(L.var(ddof=1)) if len(L) > 1 else 1000.0
        m7._n_reps_store[key] = len(L)
        m7._N_REPS_GLOBAL = len(L)
        historia[(f, e)] = r
        print(f"  SMAC-SK eval {len(historia):3d}: 1ra {f:2d} eco {e:2d} → L {L.mean():.1f}", flush=True)
        return float(L.mean())

    cs = espacio(seed)
    sc = Scenario(configspace=cs, n_trials=n_trials, deterministic=True, seed=seed,
                  output_directory=os.path.join(out, "smac_sk_salida"), name="smac_sk_presupuesto")
    smac = m7._build_hpo_sk(sc, objetivo, cs, seed)
    t0 = time.time()
    smac.optimize()
    # Confirmación con el resto de las semillas de validación (igual que la mejor fija)
    top = sorted(historia, key=lambda k: resumen(historia[k])[0])[:top_k]
    final = {k: historia[k] + evaluar_cupos(*k, resto) for k in top}
    mejor = min(top, key=lambda k: resumen(final[k])[0])
    gasto = len(historia) * len(busq) + len(top) * len(resto)
    res = {"mejor": {"cupos_1ra": mejor[0], "cupos_eco": mejor[1], "cfg": cfg_cupos(*mejor)},
           "L_val": resumen(final[mejor]), "presupuesto_usado": gasto, "n_configs": len(historia),
           "semillas_busqueda": busq, "semillas_confirmacion": resto,
           "historia": [{"cupos_1ra": k[0], "cupos_eco": k[1], "L": resumen(v)[0]} for k, v in historia.items()],
           "finalistas": {f"{k[0]}/{k[1]}": resumen(v) for k, v in final.items()},
           "seg": round(time.time() - t0)}
    print(f"SMAC-SK mejor: 1ra {mejor[0]} / control {CUPOS_ESPECIALISTA - mejor[0]} / eco {mejor[1]} / "
          f"lab {CUPOS_EXAMENES - mejor[1]}  L val {res['L_val'][0]:.1f} ± {res['L_val'][1]:.1f}  "
          f"(presupuesto {gasto} años simulados, {res['seg']} s)", flush=True)
    json.dump(res, open(os.path.join(out, "smac_sk.json"), "w"), indent=1, default=str)
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--humo", action="store_true")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    if a.humo:
        H = P.HUMO
        out = a.out or "resultados/humo"
        semillas = list(SEMILLAS_HUMO[:H["N_TAMIZ"] + H["N_CONF"]])
        args = (H["N_TAMIZ"], 12, H["TOP_K"])
    else:
        out, semillas, args = a.out or "resultados/validacion", list(SEMILLAS_VAL), (N_REPS, N_TRIALS, TOP_K)
    os.makedirs(out, exist_ok=True)
    optimizar(semillas, *args, out=out)
