"""
Ejecutor de trabajos largos del agente (corre como proceso aparte).

El agente nunca corre el simulador "dentro" de una llamada a herramienta:
las corridas tardan minutos u horas. En vez de eso, `herramientas.py` lanza
este script en segundo plano con una carpeta de trabajo que contiene
`spec.json`, y este script deja ahí `resultado.json` o `error.txt`.

Dos tipos de trabajo:
  - "evaluacion": corre run_once para UNA configuración con n réplicas,
    usando offsets fijos (números aleatorios comunes entre evaluaciones).
  - "algoritmo": corre UN módulo del benchmark (M4, M11, SA, ...) vía
    benchmark_riguroso.py con 1 macro-semilla y presupuesto acotado.

Uso (lo invoca herramientas.py, no tú):
    python3 agentes/ejecutor.py <carpeta_trabajo>
"""

import concurrent.futures
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Bloque de offsets reservado para el agente. benchmark_riguroso usa
# 0..n para optimizar y 500_000+ para re-evaluar incumbentes; el agente usa
# 900_000+ para no solapar. Todas las evaluaciones del agente comparten
# estos offsets -> comparaciones pareadas (CRN) entre configuraciones.
OFFSET_BASE_AGENTE = 900_000


def config_a_simconfig(config: dict, semanas: int = 52):
    """Traduce los 12 parámetros canónicos a SimConfig.

    Copia exacta del mapeo de benchmark_riguroso._re_evaluar_incumbente,
    para que el agente evalúe configuraciones igual que el benchmark.
    """
    import dataclasses
    from simulador_clinica_baseline import CFG

    cfg = dataclasses.replace(CFG)
    cfg.fixed_weekly_capacity        = int(round(config["horas_especialista_1ra"]))
    cfg.use_fixed_weekly_capacity    = True
    cfg.fixed_post_control_capacity  = int(round(config["horas_control_post"]))
    cfg.use_fixed_post_control_hours = True
    cfg.ugd_lab_per_week             = int(round(config["cupos_laboratorio_ugd"]))
    cfg.mat_us_per_week              = int(round(config["cupos_ecografia_matrona"]))
    cfg.ugd_us_per_week              = int(round(config["cupos_ecografia_ugd"]))
    cfg.publish_lead_workdays        = int(round(config["dias_publicacion"]))
    cfg.blocked_pct                  = float(config["pct_bloqueo_1ra"])
    cfg.empty_control_p_ugd          = float(config["pct_consultas_vacias"])
    cfg.matrona_capacity             = int(round(config["num_matronas"]))
    cfg.agent_capacity               = int(round(config["num_agentes_ugd"]))
    cfg.not_contactable_p            = float(config["pct_no_contactabilidad"])
    cfg.blocked_pct_post_control     = float(config["pct_bloqueo_post_control"])
    cfg.benchmark_mode               = True
    cfg.weeks_to_simulate            = int(semanas)
    return cfg


def _replica(args):
    """Worker a nivel de módulo (pickleable): una réplica del DES."""
    offset, config, semanas = args
    from simulador_clinica_baseline import run_once
    res = run_once(seed_offset=offset, cfg=config_a_simconfig(config, semanas))
    return {
        "offset": offset,
        "tts_full_days_mean": float(res["tts_full_days_mean"]),
        "total_atenciones": float(res["total_atenciones"]),
    }


def correr_evaluacion(spec: dict) -> dict:
    n_reps = int(spec["n_reps"])
    semanas = int(spec.get("semanas", 52))
    timeout_seg = float(spec.get("timeout_replica_seg", 3600))
    n_workers = max(1, min(n_reps, int(spec.get("n_workers", os.cpu_count() or 1))))
    tareas = [(OFFSET_BASE_AGENTE + r, spec["config"], semanas) for r in range(n_reps)]

    replicas, descartadas = [], []
    ex = concurrent.futures.ProcessPoolExecutor(max_workers=n_workers)
    try:
        futuros = {ex.submit(_replica, t): t[0] for t in tareas}
        limite = time.time() + timeout_seg * (n_reps / n_workers + 1)
        for fut, offset in futuros.items():
            restante = max(1.0, limite - time.time())
            try:
                replicas.append(fut.result(timeout=min(timeout_seg, restante)))
            except concurrent.futures.TimeoutError:
                descartadas.append({"offset": offset, "motivo": "timeout (posible deadlock DES)"})
            except Exception as e:  # una réplica rota no debe tumbar el trabajo
                descartadas.append({"offset": offset, "motivo": repr(e)})
    finally:
        ex.shutdown(wait=False, cancel_futures=True)

    replicas.sort(key=lambda d: d["offset"])
    return {"tipo": "evaluacion", "config": spec["config"], "semanas": semanas,
            "replicas": replicas, "descartadas": descartadas}


def correr_algoritmo(spec: dict, carpeta: Path) -> dict:
    salida = carpeta / "salida"
    cmd = [sys.executable, str(REPO / "benchmark_riguroso.py"),
           "--modulos", spec["modulo"],
           "--n_seeds", "1",
           "--n_trials", str(spec["n_trials"]),
           "--r_final", str(spec["r_final"]),
           "--n_cores", "1",
           "--max_seed_horas", str(spec.get("max_horas", 24)),
           "--out", str(salida)]
    if spec.get("lambda_obj") is not None:
        cmd += ["--lambda_obj", str(spec["lambda_obj"])]
    with open(carpeta / "benchmark.log", "w") as log:
        proc = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    f = salida / f"resultado_{spec['modulo']}_seed00.json"
    if not f.exists():
        raise RuntimeError(f"benchmark_riguroso terminó (código {proc.returncode}) "
                           f"sin {f.name}; ver benchmark.log")
    r = json.loads(f.read_text())
    if "error" in r:
        raise RuntimeError(f"el módulo falló: {r['error']}")
    return {"tipo": "algoritmo", "modulo": spec["modulo"], "n_trials": spec["n_trials"],
            "r_final": spec["r_final"], "lambda_obj": spec.get("lambda_obj"),
            "incumbente": r.get("incumbente"), "costo_opt": r.get("costo_opt"),
            "reeval": r.get("reeval"), "kpis_incumbente": r.get("kpis_incumbente"),
            "n_eval_usadas": r.get("n_eval_usadas"), "tiempo_seg": r.get("tiempo_seg")}


def main(carpeta: Path) -> None:
    spec = json.loads((carpeta / "spec.json").read_text())
    t0 = time.time()
    try:
        if spec["tipo"] == "evaluacion":
            resultado = correr_evaluacion(spec)
        elif spec["tipo"] == "algoritmo":
            resultado = correr_algoritmo(spec, carpeta)
        else:
            raise ValueError(f"tipo de trabajo desconocido: {spec['tipo']}")
        resultado["duracion_seg"] = round(time.time() - t0, 1)
        tmp = carpeta / "resultado.json.tmp"
        tmp.write_text(json.dumps(resultado, indent=2, default=str))
        tmp.rename(carpeta / "resultado.json")   # escritura atómica
    except Exception:
        (carpeta / "error.txt").write_text(traceback.format_exc())


if __name__ == "__main__":
    main(Path(sys.argv[1]))
