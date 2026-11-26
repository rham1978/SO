"""
Selección del indicador principal sin modificar el DES.

Idea: un buen indicador debe moverse en la dirección correcta ante cambios cuyo
efecto conocemos de antemano (más capacidad ⇒ mejor, más demanda ⇒ peor), y
hacerlo con buena relación señal/ruido. Se corre el DES tal como está (solo se
observan los pacientes, no se cambia su lógica) en escenarios controlados con
la misma semilla y se compara cada escenario con la base, pareado.

Indicadores candidatos:
  TTS_comp   tts_full_days_mean de la tesis: tiempo medio de quienes terminan su ruta en el año
  TTS_1ra    tts_first_days_mean: tiempo hasta cerrar la etapa de 1ra consulta
  L          pacientes en el sistema, promedio diario (= pacientes·día / 364)
  TTS_cens   tiempo medio en el sistema de TODOS los pacientes vistos en el año,
             contando hasta el cierre a quienes siguen dentro (media restringida)
  Atenciones total_atenciones (más es mejor)

Uso: python indicadores.py [n_semillas]   → resultados/indicadores.json
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

DAY = 24 * 60
TERMINALES = {"post_completed", "first_canceled", "removed_unreachable_first"}

ESCENARIOS = {
    # nombre: (cambios a la config, dirección esperada: -1 = el sistema mejora)
    "base": ({}, 0),
    "1ra +25%": ({"fixed_weekly_capacity": 20}, -1),
    "control +25%": ({"fixed_post_control_capacity": 50}, -1),
    "cirugía +25%": ({"surg_throughput_per_week": 25}, -1),
    "eco +25%": ({"ugd_us_per_week": 31, "mat_us_per_week": 31}, -1),
    "demanda +15%": ({"nb_arrivals_n": 8.99 * 1.15}, +1),
    "anticipación 10 días": ({"publish_lead_workdays": 10}, None),  # efecto en discusión
}
INDICADORES = ("TTS_comp", "TTS_1ra", "L", "TTS_cens", "Atenciones")
MEJOR_ES_MENOS = {"TTS_comp": True, "TTS_1ra": True, "L": True, "TTS_cens": True, "Atenciones": False}


def _correr(args):
    nombre, seed = args
    logging.disable(logging.CRITICAL)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import sim_fast
    from sim_fast import sim
    from entorno import config_base
    sim_fast.apply_patch()

    class Observado(sim.ClinicModelAdjusted):
        """Solo registra entradas y salidas; la lógica del DES no cambia."""
        def __init__(self, env, rng, cfg=None):
            self._entrada, self._salida, self._L = {}, {}, []
            super().__init__(env, rng, cfg)
            env.process(self._monitor())

        def _register_patient(self, p):
            self._entrada[p.pid] = float(p.enqueued_at)
            return super()._register_patient(p)

        def _set_patient_state(self, pid, state):
            if state in TERMINALES and pid not in self._salida:
                self._salida[pid] = self.env.now
            return super()._set_patient_state(pid, state)

        def _monitor(self):
            while True:
                yield self.env.timeout(DAY)
                self._L.append(self.total_in_system_now())

    cfg = dataclasses.replace(config_base(), **ESCENARIOS[nombre][0])
    rng = np.random.RandomState(cfg.random_seed_base + seed)
    random.seed(cfg.random_seed_base + seed)
    env = sim.simpy.Environment()
    m = Observado(env, rng, cfg)
    env.run(until=cfg.sim_time_min)
    T = cfg.sim_time_min
    tiempos = [(m._salida.get(pid, T) - max(t_in, -1e18)) / DAY for pid, t_in in m._entrada.items()]
    k = m.kpis
    full = k.total_time_full_all
    first = k.first_closed_total_time_all
    return nombre, seed, {
        "TTS_comp": float(np.mean(full)) / DAY if full else float("nan"),
        "TTS_1ra": float(np.mean(first)) / DAY if first else float("nan"),
        "L": float(np.mean(m._L)),
        "TTS_cens": float(np.mean(tiempos)),
        "Atenciones": float(k.attended + k.post_completed),
    }


def main(n=30):
    semillas = range(10_000, 10_000 + n)  # validación
    jobs = [(e, s) for e in ESCENARIOS for s in semillas]
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        res = list(ex.map(_correr, jobs, chunksize=2))
    tab = {e: {s: v for (ee, s, v) in res if ee == e} for e in ESCENARIOS}
    out = {}
    print(f"{'escenario':22s} " + " ".join(f"{i:>22s}" for i in INDICADORES))
    for e, (_, esperado) in ESCENARIOS.items():
        if e == "base":
            continue
        fila = {}
        for ind in INDICADORES:
            d = np.array([tab[e][s][ind] - tab["base"][s][ind] for s in semillas])
            media, ic = d.mean(), 1.96 * d.std(ddof=1) / np.sqrt(len(d))
            signo_mejor = -1 if MEJOR_ES_MENOS[ind] else +1
            # ¿se movió en la dirección esperada y de forma significativa?
            if esperado is None:
                ok = None
            else:
                mejora = np.sign(media) == signo_mejor
                sig = abs(media) > ic
                ok = bool(sig and (mejora if esperado == -1 else not mejora))
            fila[ind] = {"dif": float(media), "ic95": float(ic), "snr": float(abs(media) / (ic / 1.96) if ic else 0), "ok": ok}
        out[e] = fila
        print(f"{e:22s} " + " ".join(
            f"{fila[i]['dif']:+8.1f}±{fila[i]['ic95']:5.1f} {'✓' if fila[i]['ok'] else ('✗' if fila[i]['ok'] is False else '·'):>2s}   "
            for i in INDICADORES))
    os.makedirs("resultados", exist_ok=True)
    json.dump({"diferencias_vs_base": out,
               "base": {i: float(np.mean([tab['base'][s][i] for s in semillas])) for i in INDICADORES}},
              open("resultados/indicadores.json", "w"), indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
