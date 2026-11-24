"""Evaluación paralela de políticas en un conjunto de semillas (mismas llegadas para todas: CRN)."""
from __future__ import annotations
import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np

KPIS = ("L", "tts_full_days_mean", "total_atenciones", "tts_first_days_mean", "tts_post_days_mean",
        "en_sistema_final")


def _job(args):
    make_policy, seed, kw = args
    from entorno import rollout
    pol = make_policy(seed)
    tr = rollout(pol, seed, **kw)
    return {"seed": seed, "retorno": float(tr["rewards"].sum()),
            **{k: float(tr["kpis"][k]) for k in KPIS}, "acciones": tr["actions"].tolist()}


def evaluar(make_policy, semillas, n_workers: int | None = None, **rollout_kw) -> list[dict]:
    """make_policy(seed) -> policy_fn; debe ser pickleable (función de módulo o functools.partial).
    rollout_kw se pasa a entorno.rollout (cfg_overrides, aplicar_acciones)."""
    n_workers = n_workers or os.cpu_count()
    with ProcessPoolExecutor(n_workers) as ex:
        return list(ex.map(_job, [(make_policy, s, rollout_kw) for s in semillas], chunksize=2))


def resumen(res: list[dict], k: str = "L") -> tuple[float, float]:
    x = np.array([r[k] for r in res])
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))
