"""Validación estadística: original vs rápido, año completo, N semillas (CRN)."""
import logging, sys, json, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy import stats
KEYS = ["tts_full_days_mean", "tts_first_days_mean", "tts_post_days_mean", "total_atenciones", "post_completed", "first_attended"]
def job(args):
    seed, fast = args
    logging.disable(logging.CRITICAL)
    import sim_fast
    from sim_fast import sim
    if fast:
        sim_fast.apply_patch()
    # Cada tarea corre en un proceso nuevo (max_tasks_per_child=1): verificar que se usa la versión pedida
    usa_rapido = sim.ClinicModelAdjusted._book_from_pool is sim_fast._book_from_pool_fast
    assert usa_rapido == fast, (seed, fast, usa_rapido)
    t = time.time()
    r = sim.run_once(seed, sim.SimConfig(benchmark_mode=True))
    return seed, fast, time.time() - t, {k: float(r[k]) for k in KEYS}
if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    jobs = [(s, f) for s in range(n) for f in (False, True)]
    with ProcessPoolExecutor(__import__("presupuesto").nucleos(), max_tasks_per_child=1) as ex:
        res = list(ex.map(job, jobs))
    json.dump(res, open("validacion_sim_fast.json", "w"), indent=1)
    for fast in (False, True):
        ts = [r[2] for r in res if r[1] == fast]
        print(("rapido  " if fast else "original"), f"tiempo medio por año: {np.mean(ts):.1f}s")
    for k in KEYS:
        a = np.array([r[3][k] for r in res if not r[1]]); b = np.array([r[3][k] for r in res if r[1]])
        d = b - a
        print(f"{k:24s} orig {a.mean():9.2f}±{a.std(ddof=1):6.2f}  rapido {b.mean():9.2f}±{b.std(ddof=1):6.2f}  "
              f"dif {d.mean():+7.2f} (IC95 ±{1.96*d.std(ddof=1)/np.sqrt(len(d)):.2f})  Wilcoxon pareado p={stats.wilcoxon(d).pvalue if np.any(d) else 1.0:.2f}")
