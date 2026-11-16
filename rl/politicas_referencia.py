"""
Políticas de referencia:
  - aleatoria: elige una acción uniforme del catálogo cada semana (referencia trivial).
  - fija(a): usa siempre la misma configuración a.
  - mejor fija: búsqueda exhaustiva sobre el catálogo en las semillas de VALIDACIÓN
    (el catálogo es pequeño, así que la búsqueda exhaustiva reemplaza a SMAC/kriging
    y da la mejor configuración fija posible dentro del mismo presupuesto).

Uso:  python politicas_referencia.py [n_semillas_tamizaje]
Escribe resultados/referencias_val.json
"""
import functools, json, os, sys, time
import numpy as np
from catalogo import CATALOGO, N_ACCIONES, ACCION_BASE
from entorno import SEMILLAS_VAL
from evaluar import evaluar, resumen


def _fija(a, obs):
    return a


def hacer_fija(a, seed):
    return functools.partial(_fija, a)


def _aleatoria(rng, obs):
    return int(rng.integers(N_ACCIONES))


def hacer_aleatoria(seed):
    return functools.partial(_aleatoria, np.random.default_rng(10**6 + seed))


if __name__ == "__main__":
    os.makedirs("resultados", exist_ok=True)
    n_tamiz = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    out = {"catalogo": CATALOGO}
    t = time.time()
    # 1) tamizaje: todas las configuraciones fijas en las primeras n_tamiz semillas de validación
    tamiz = {}
    for a in range(N_ACCIONES):
        r = evaluar(functools.partial(hacer_fija, a), SEMILLAS_VAL[:n_tamiz])
        tamiz[a] = resumen(r)
        print(f"fija {a:3d} {CATALOGO[a]} TTS {tamiz[a][0]:.1f} ± {tamiz[a][1]:.1f}", flush=True)
    out["tamizaje"] = tamiz
    # 2) las 5 mejores + la base, con las 100 semillas de validación
    top = sorted(tamiz, key=lambda a: tamiz[a][0])[:5]
    if ACCION_BASE not in top:
        top.append(ACCION_BASE)
    out["finalistas"] = {}
    for a in top:
        r = evaluar(functools.partial(hacer_fija, a), SEMILLAS_VAL)
        out["finalistas"][a] = r
        print(f"finalista {a} TTS {resumen(r)[0]:.2f} ± {resumen(r)[1]:.2f}", flush=True)
    mejor = min(top, key=lambda a: resumen(out["finalistas"][a])[0])
    out["mejor_fija"] = mejor
    # 3) aleatoria en validación
    out["aleatoria"] = evaluar(hacer_aleatoria, SEMILLAS_VAL)
    print("aleatoria TTS %.2f ± %.2f" % resumen(out["aleatoria"]))
    print("mejor fija:", mejor, CATALOGO[mejor], "base:", ACCION_BASE, f"({time.time()-t:.0f}s)")
    json.dump(out, open("resultados/referencias_val.json", "w"), indent=1, default=str)
