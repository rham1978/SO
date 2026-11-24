"""
Prueba de equivalencia simulador original vs acelerado (sim_fast.py), con la configuración
por defecto del DES (la validada en la tesis) y la misma semilla en ambos (pareado).

Para cada KPI: diferencia media (acelerado − original), IC90 y IC95 de la diferencia, y
prueba TOST con margen ±MARGEN·media del original (equivalencia si el IC90 cae dentro).
Uso: python validar_equivalencia.py [n_semillas] [margen]   → validacion_equivalencia.json
"""
import json, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy import stats
from validar_estadistico import job, KEYS

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    margen = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
    jobs = [(s, f) for s in range(n) for f in (False, True)]
    with ProcessPoolExecutor(4) as ex:
        res = list(ex.map(job, jobs, chunksize=1))
    out = {"n": n, "margen": margen, "kpis": {}, "crudo": res}
    for fast in (False, True):
        print(("acelerado" if fast else "original "), f"s/año: {np.mean([r[2] for r in res if r[1] == fast]):.1f}")
    print(f"{'KPI':22s} {'original':>9s} {'dif':>8s} {'IC95 dif (%)':>18s} {'TOST ±'+str(int(margen*100))+'%':>10s}")
    for k in KEYS:
        a = {r[0]: r[3][k] for r in res if not r[1]}
        b = {r[0]: r[3][k] for r in res if r[1]}
        d = np.array([b[s] - a[s] for s in a]); m = float(np.mean(list(a.values())))
        se = d.std(ddof=1) / np.sqrt(len(d)); t95, t90 = stats.t.ppf(0.975, len(d) - 1), stats.t.ppf(0.95, len(d) - 1)
        lo, hi = d.mean() - t95 * se, d.mean() + t95 * se
        eq = (d.mean() - t90 * se > -margen * m) and (d.mean() + t90 * se < margen * m)
        out["kpis"][k] = {"media_original": m, "dif": float(d.mean()), "ic95": [float(lo), float(hi)],
                          "ic95_pct": [100 * lo / m, 100 * hi / m], "equivalente_tost": bool(eq),
                          "sd_original": float(np.std(list(a.values()), ddof=1))}
        print(f"{k:22s} {m:9.1f} {d.mean():+8.2f} [{100*lo/m:+5.1f}%, {100*hi/m:+5.1f}%]  {'sí' if eq else 'NO':>10s}")
    json.dump(out, open("validacion_equivalencia.json", "w"), indent=1)
