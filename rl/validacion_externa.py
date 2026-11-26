"""
Validación externa del DES del CMI frente a datos observados del CRS Cordillera.

Compara las salidas del simulador con lo observado en un PERÍODO DISTINTO del de
calibración (la calibración usó 52 semanas de llegadas y el corte de lista de
espera de septiembre de 2026). El DES no se modifica: solo se observa.

Dos modos de simulación (se reportan ambos):
  entradas_observadas  (principal) el modelo recibe las entradas del período
                       observado: tasa de llegadas, cupos ofrecidos por semana y la
                       lista inicial (por etapa y prioridad) en la fecha de inicio.
                       Valida la ESTRUCTURA del modelo: dadas las mismas entradas,
                       ¿reproduce las salidas?
  tesis                configuración calibrada de la tesis tal cual (operación
                       actual 16/40/50/25, cirugía 20). Valida la capacidad
                       PREDICTIVA sin re-ajustar nada.

Datos de entrada (carpeta --datos, CSV UTF-8, separador "," o ";", fechas AAAA-MM-DD).
El detalle de cada campo está en DATOS_VALIDACION_EXTERNA.md.
  pacientes.csv          una fila por interconsulta de ginecología CMI (anonimizada)
  eventos.csv            una fila por atención (1ra, control, eco, lab, cirugía)
  cortes_lista.csv       lista de espera por etapa y prioridad al cierre de cada mes
  capacidad_semanal.csv  cupos ofrecidos por semana (entrada del modelo)
Ningún archivo es obligatorio por sí solo: el script valida lo que pueda con lo que
reciba y lista lo que quedó sin validar.

Indicadores (todos calculados igual en lo observado y en lo simulado):
  Flujos mensuales      ingresos a lista, 1ras consultas, controles, ecografías,
                        laboratorios, cirugías, altas CMI
  Stocks (cortes)       lista de 1ra consulta (total y por prioridad), pacientes en
                        control, en prequirúrgico y en lista quirúrgica, L total
  Esperas               días de ingreso a 1ra consulta: mediana, media, p90 de
                        quienes ingresan y se atienden en el período, y P(atendida
                        ≤ d días) para d = 30, 60, 90, 180 entre quienes tienen al
                        menos d días de seguimiento (misma censura en ambos lados)

Criterios (fijados ANTES de ver los datos; editar CRITERIOS si se acuerdan otros):
  · total del período de cada flujo dentro del intervalo de predicción 95 % del DES
    y con sesgo relativo |sim − obs| / obs ≤ 10 %
  · ≥ 80 % de los meses de cada flujo dentro de la banda 95 % del DES
  · ≥ 80 % de los cortes de cada lista dentro de la banda 95 % y sesgo absoluto
    medio ≤ 15 %
  · cada estadístico de espera dentro del intervalo de predicción 95 %

Uso:
  python validacion_externa.py --datos datos_crs --inicio 2025-09-01 --fin 2026-08-31
  python validacion_externa.py --demo                 # prueba con datos sintéticos
  python validacion_externa.py --demo --demo_demanda 1.25   # ¿detecta un desajuste?
Opciones: --reps 50 (réplicas), --nucleos N, --out resultados/validacion_externa,
          --modos entradas_observadas,tesis
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import logging
import math
import os
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DAY = 24 * 60
PRI_OBS = {"alta": "high", "media": "mid", "baja": "low",
           "high": "high", "mid": "mid", "low": "low", "a": "high", "m": "mid", "b": "low"}
FLUJOS = ("ingresos", "primeras", "controles", "ecografias", "laboratorios", "cirugias", "altas")
ETAPAS = ("primera", "control", "prequirurgico", "quirurgico")
D_ESPERA = (30, 60, 90, 180)
SEMILLA_BASE_VAL = 700_000  # bloque propio, disjunto de entrenamiento, validación y prueba del RL
CRITERIOS = {"sesgo_flujo_total": 0.10, "cobertura_mensual": 0.80, "sesgo_corte": 0.15, "nivel": 0.95}

# ═══════════════════════════ simulación observada ═══════════════════════════


def _tipo_examen(key: str) -> str | None:
    if "us_lead" in key:
        return "ecografias"
    if "lab_lead" in key:
        return "laboratorios"
    return None


def _correr(args):
    """Una réplica del DES con observadores. Devuelve eventos (en días) y stocks diarios."""
    seed, cfg_over = args
    logging.disable(logging.CRITICAL)
    import sim_fast
    from sim_fast import sim
    from entorno import config_base
    sim_fast.apply_patch()

    class Observado(sim.ClinicModelAdjusted):
        def __init__(self, env, rng, cfg=None):
            self._pri, self._enq, self._open1 = {}, {}, {}
            self._ev = defaultdict(list)
            self._esperas = {}            # pid nuevo -> (t_ingreso, t_1ra o None, causal)
            self._diario = []
            self._etapa, self._n_etapa = {}, defaultdict(int)
            super().__init__(env, rng, cfg)
            env.process(self._monitor())

        def _register_patient(self, p):
            pr = str(getattr(p.priority, "value", p.priority))
            self._pri[p.pid] = pr
            self._enq[p.pid] = float(p.enqueued_at)
            if not p.is_backlog:
                self._ev["ingresos"].append(float(p.enqueued_at))
                self._esperas[p.pid] = [float(p.enqueued_at), None, None, pr]
            return super()._register_patient(p)

        # Etapa posterior a la 1ra consulta = ruta en que está la paciente (incluye la espera de
        # exámenes previa al control, que el DES no marca con un estado propio).
        def _mover(self, pid, nueva):
            vieja = self._etapa.get(pid)
            if vieja == nueva:
                return
            if vieja:
                self._n_etapa[vieja] -= 1
            if nueva:
                self._n_etapa[nueva] += 1
            self._etapa[pid] = nueva

        def route_control(self, patient):
            self._mover(patient.pid, "control")
            return (yield from super().route_control(patient))

        def route_preq(self, patient):
            self._mover(patient.pid, "prequirurgico")
            return (yield from super().route_preq(patient))

        def route_quir(self, patient, from_preq):
            self._mover(patient.pid, "quirurgico")
            return (yield from super().route_quir(patient, from_preq))

        def route_surgery_waitlist(self, patient):
            self._mover(patient.pid, "quirurgico")
            return (yield from super().route_surgery_waitlist(patient))

        def _after_surgery(self, patient):
            self._mover(patient.pid, "postop")
            return (yield from super()._after_surgery(patient))

        def _set_patient_state(self, pid, state):
            if state == "post_completed":
                self._mover(pid, None)
            if state in ("first_backlog", "first_arrival"):
                self._open1[pid] = self._pri.get(pid, "high")
            elif state == "after_surgery":
                self._ev["cirugias"].append(self.env.now)
            elif state == "post_completed":
                self._ev["altas"].append(self.env.now)
            return super()._set_patient_state(pid, state)

        def _record_first_closure(self, patient, total_minutes, reason, attended=False):
            self._open1.pop(patient.pid, None)
            if attended:
                self._ev["primeras"].append(self.env.now)
            else:
                self._ev["egreso_sin_atencion"].append(self.env.now)
            if patient.pid in self._esperas:
                self._esperas[patient.pid][1] = self.env.now if attended else None
                self._esperas[patient.pid][2] = str(reason)
            return super()._record_first_closure(patient, total_minutes, reason, attended)

        def _book_from_pool(self, pool, lock_res, not_before, waitslot_key, pid=None):
            r = yield from sim_fast._book_from_pool_fast(self, pool, lock_res, not_before, waitslot_key, pid)
            tipo = _tipo_examen(waitslot_key)
            if r is not None and tipo:
                self._ev[tipo].append(float(r))
            return r

        def _monitor(self):
            while True:
                yield self.env.timeout(DAY)
                por_pri = defaultdict(int)
                for pr in self._open1.values():
                    por_pri[pr] += 1
                self._diario.append((
                    por_pri["high"], por_pri["mid"], por_pri["low"],
                    self._n_etapa["control"], self._n_etapa["prequirurgico"], self._n_etapa["quirurgico"],
                    self.total_in_system_now(), self.kpis.post_control_attended,
                ))

    cfg = config_base(**cfg_over)
    rng = np.random.RandomState(cfg.random_seed_base + seed)
    random.seed(cfg.random_seed_base + seed)
    env = sim.simpy.Environment()
    m = Observado(env, rng, cfg)
    env.run(until=cfg.sim_time_min)

    d = np.array(m._diario, dtype=float)
    ev = {k: (np.array(v) / DAY).tolist() for k, v in m._ev.items()}
    # controles: el DES cuenta controles atendidos acumulados; se pasan a eventos diarios
    ctrl_dia = np.diff(np.concatenate([[0.0], d[:, 7]]))
    ev["controles"] = np.repeat(np.arange(len(ctrl_dia)) + 0.5, ctrl_dia.astype(int)).tolist()
    esperas = [(a / DAY, (b / DAY if b is not None else None), c, pr) for a, b, c, pr in m._esperas.values()]
    return {
        "seed": seed,
        "eventos": ev,
        "stock": {  # índice i = cierre del día i (desde la fecha de inicio)
            "primera_alta": d[:, 0].tolist(), "primera_media": d[:, 1].tolist(),
            "primera_baja": d[:, 2].tolist(),
            "primera_total": (d[:, 0] + d[:, 1] + d[:, 2]).tolist(),
            "control_total": d[:, 3].tolist(), "prequirurgico_total": d[:, 4].tolist(),
            "quirurgico_total": d[:, 5].tolist(), "L_total": d[:, 6].tolist(),
        },
        "esperas": esperas,
        "dias": int(cfg.sim_time_min // DAY),
    }


def simular(cfg_over: dict, reps: int, nucleos: int, semilla0: int = SEMILLA_BASE_VAL):
    jobs = [(semilla0 + r, cfg_over) for r in range(reps)]
    with ProcessPoolExecutor(nucleos) as ex:
        return list(ex.map(_correr, jobs))


# ═══════════════════════════ entradas del modelo ═══════════════════════════


def _media_efectiva_llegadas(n: float, p: float, aps_reject: float) -> float:
    """Llegadas aceptadas por día en el DES: binomial negativa truncada en 0, menos rechazo APS."""
    p0 = p ** n
    return n * (1 - p) / p / (1 - p0) * (1 - aps_reject)


def p_para_tasa(tasa_dia: float, n: float = 8.99, aps_reject: float = 0.05) -> float:
    """p de la binomial negativa (con n fijo) que da la tasa de ingresos observada."""
    lo, hi = 1e-4, 0.9999
    for _ in range(200):
        mid = (lo + hi) / 2
        if _media_efectiva_llegadas(n, mid, aps_reject) > tasa_dia:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def config_tesis() -> dict:
    from catalogo import OPERACION_ACTUAL
    return dict(OPERACION_ACTUAL)


def config_desde_datos(obs: dict, inicio: dt.date, fin: dt.date) -> tuple[dict, list[str]]:
    """Traduce las entradas observadas del período a parámetros del DES."""
    over, notas = config_tesis(), []
    dias = (fin - inicio).days + 1
    if obs.get("ingresos_dia") is not None:
        tasa = obs["ingresos_dia"]
        over["nb_arrivals_p"] = p_para_tasa(tasa)
        notas.append(f"llegadas: {tasa:.2f} ingresos/día observados → nb_arrivals_p = {over['nb_arrivals_p']:.4f} (n = 8,99 fijo)")
    cap = obs.get("capacidad_media")
    if cap:
        mapa = {"cupos_primera": "fixed_weekly_capacity", "cupos_control": "fixed_post_control_capacity",
                "cupos_eco_ugd": "ugd_us_per_week", "cupos_eco_matrona": "mat_us_per_week",
                "cupos_lab_matrona": "mat_lab_per_week", "cirugias": "surg_throughput_per_week"}
        for col, par in mapa.items():
            if col in cap and not math.isnan(cap[col]):
                over[par] = int(round(cap[col]))
        notas.append("cupos semanales: media observada del período " + ", ".join(
            f"{k} = {v:.1f}" for k, v in cap.items()))
    ini = dict(obs.get("corte_inicial") or {})
    if "primera_total" in ini and not any(k in ini for k in ("primera_alta", "primera_media", "primera_baja")):
        from simulador_clinica_baseline import SimConfig
        b = SimConfig()
        tot = b.backlog_high_n + b.backlog_mid_n + b.backlog_low_n
        ini["primera_alta"] = round(ini["primera_total"] * b.backlog_high_n / tot)
        ini["primera_baja"] = round(ini["primera_total"] * b.backlog_low_n / tot)
        ini["primera_media"] = ini["primera_total"] - ini["primera_alta"] - ini["primera_baja"]
        notas.append("lista inicial de 1ra sin prioridad: se repartió con las proporciones de la tesis")
    if ini:
        m = {"primera_alta": "backlog_high_n", "primera_media": "backlog_mid_n", "primera_baja": "backlog_low_n",
             "control_total": "post_init_control_n", "prequirurgico_total": "post_init_preq_n",
             "quirurgico_total": "post_init_surgery_waitlist_n"}
        for k, par in m.items():
            if k in ini:
                over[par] = int(ini[k])
        notas.append("lista inicial: " + ", ".join(f"{k} = {v}" for k, v in ini.items()))
    over["weeks_to_simulate"] = int(math.ceil(dias / 7))
    return over, notas


# ═══════════════════════════ datos observados ═══════════════════════════


def _leer(path):
    import pandas as pd
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def _fecha(s):
    import pandas as pd
    f = pd.to_datetime(s, errors="coerce", dayfirst=False, format="mixed")
    return pd.Series([x.date() if not pd.isna(x) else None for x in f], index=s.index, dtype=object)


def cargar_observado(carpeta: str, inicio: dt.date, fin: dt.date) -> tuple[dict, list[str]]:
    """Lee los CSV y los lleva a la misma forma que la salida del simulador (días desde el inicio)."""
    obs = {"eventos": {}, "cortes": {}, "esperas": None}
    faltan = []
    d0 = lambda f: (f - inicio).days  # noqa: E731

    pac = _leer(os.path.join(carpeta, "pacientes.csv"))
    eve = _leer(os.path.join(carpeta, "eventos.csv"))
    cor = _leer(os.path.join(carpeta, "cortes_lista.csv"))
    cap = _leer(os.path.join(carpeta, "capacidad_semanal.csv"))

    if pac is not None:
        pac["ing"] = _fecha(pac["fecha_ingreso_lista"])
        pac["att"] = _fecha(pac["fecha_primera_consulta"]) if "fecha_primera_consulta" in pac else None
        pac["egr"] = _fecha(pac["fecha_egreso_lista"]) if "fecha_egreso_lista" in pac else pac["att"]
        pac["pri"] = pac["prioridad"].astype(str).str.strip().str.lower().map(PRI_OBS) if "prioridad" in pac else "high"
        nuevos = pac[pac["ing"].apply(lambda x: x is not None and inicio <= x <= fin)]
        obs["eventos"]["ingresos"] = [d0(f) + 0.5 for f in nuevos["ing"]]
        obs["ingresos_dia"] = len(nuevos) / ((fin - inicio).days + 1)
        obs["esperas"] = [(d0(a) + 0.5, (d0(b) + 0.5) if isinstance(b, dt.date) and b <= fin else None)
                          for a, b in zip(nuevos["ing"], nuevos["att"])]
        obs["mix_prioridad"] = nuevos["pri"].value_counts(normalize=True).round(3).to_dict()
        if "fecha_alta_cmi" in pac:
            alt = _fecha(pac["fecha_alta_cmi"])
            obs["eventos"]["altas"] = [d0(f) + 0.5 for f in alt if isinstance(f, dt.date) and inicio <= f <= fin]
        # lista de 1ra derivada de pacientes (si no vienen cortes): abiertos al cierre de cada día pedido
        obs["_pac"] = pac
    else:
        faltan.append("pacientes.csv (ingresos, esperas hasta 1ra consulta, lista de 1ra)")

    if eve is not None:
        eve["f"] = _fecha(eve["fecha"])
        if "estado" in eve:
            eve = eve[eve["estado"].astype(str).str.lower().str.startswith("realiz")]
        tipos = {"primera": "primeras", "control": "controles", "ecografia": "ecografias",
                 "laboratorio": "laboratorios", "cirugia": "cirugias"}
        t = eve["tipo"].astype(str).str.lower().str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii")
        for k, v in tipos.items():
            sel = eve[t.str.startswith(k) & eve["f"].apply(lambda x: x is not None and inicio <= x <= fin)]
            obs["eventos"][v] = [d0(f) + 0.5 for f in sel["f"]]
    else:
        faltan.append("eventos.csv (1ras, controles, ecografías, laboratorios, cirugías por mes)")
    if "primeras" not in obs["eventos"] and pac is not None and pac["att"] is not None:
        obs["eventos"]["primeras"] = [d0(f) + 0.5 for f in pac["att"] if isinstance(f, dt.date) and inicio <= f <= fin]

    if cor is not None:
        cor["f"] = _fecha(cor["fecha_corte"])
        cor["etapa"] = cor["etapa"].astype(str).str.lower().str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii")
        cor["prioridad"] = cor.get("prioridad", "total")
        cor["prioridad"] = cor["prioridad"].fillna("total").astype(str).str.lower()
        for (f, et, pr), n in cor.groupby(["f", "etapa", "prioridad"])["n_pacientes"].sum().items():
            et = next((e for e in ETAPAS if et.startswith(e[:5])), et)
            clave = f"{et}_{'total' if pr == 'total' else {'high': 'alta', 'mid': 'media', 'low': 'baja'}.get(PRI_OBS.get(pr, pr), pr)}"
            obs["cortes"].setdefault(clave, {})[f] = float(n)
        # la 1ra total se arma con las prioridades si no viene explícita
        for f in {f for k in obs["cortes"] if k.startswith("primera_") for f in obs["cortes"][k]}:
            if f not in obs["cortes"].get("primera_total", {}):
                s = sum(obs["cortes"].get(f"primera_{p}", {}).get(f, 0) for p in ("alta", "media", "baja"))
                obs["cortes"].setdefault("primera_total", {})[f] = s
    elif pac is not None:
        # sin cortes: lista de 1ra por prioridad al cierre de cada mes, reconstruida desde pacientes.csv
        for f in _cierres_de_mes(inicio, fin) + [inicio - dt.timedelta(days=1)]:
            abiertos = pac[pac["ing"].apply(lambda x: x is not None and x <= f)
                           & pac["egr"].apply(lambda x: x is None or x > f)]
            for p_es, p in (("alta", "high"), ("media", "mid"), ("baja", "low")):
                obs["cortes"].setdefault(f"primera_{p_es}", {})[f] = float((abiertos["pri"] == p).sum())
            obs["cortes"].setdefault("primera_total", {})[f] = float(len(abiertos))
        faltan.append("cortes_lista.csv (control, prequirúrgico y lista quirúrgica; la lista de 1ra se reconstruyó desde pacientes.csv)")
    else:
        faltan.append("cortes_lista.csv (stocks de lista por etapa)")

    # corte inicial = el más cercano a la víspera del inicio (± 7 días)
    ini = {}
    for k, serie in obs["cortes"].items():
        cand = [f for f in serie if abs((f - (inicio - dt.timedelta(days=1))).days) <= 7]
        if cand:
            ini[k] = serie[min(cand, key=lambda f: abs((f - inicio).days))]
    obs["corte_inicial"] = ini or None

    if cap is not None:
        cap["s"] = _fecha(cap["semana_inicio"])
        cap = cap[cap["s"].apply(lambda x: x is not None and inicio - dt.timedelta(days=6) <= x <= fin)]
        cols = [c for c in ("cupos_primera", "cupos_control", "cupos_eco_ugd", "cupos_eco_matrona",
                            "cupos_lab_matrona", "cirugias") if c in cap]
        obs["capacidad_media"] = {c: float(cap[c].mean()) for c in cols}
    else:
        faltan.append("capacidad_semanal.csv (cupos ofrecidos: sin esto el modo entradas_observadas usa los cupos de la tesis)")
    obs.pop("_pac", None)
    return obs, faltan


# ═══════════════════════════ indicadores comunes ═══════════════════════════


def _cierres_de_mes(inicio: dt.date, fin: dt.date) -> list[dt.date]:
    out, f = [], inicio
    while f <= fin:
        nxt = (f.replace(day=1) + dt.timedelta(days=32)).replace(day=1)
        cierre = nxt - dt.timedelta(days=1)
        if cierre <= fin:
            out.append(cierre)
        f = nxt
    return out


def meses_completos(inicio: dt.date, fin: dt.date) -> list[tuple[str, int, int]]:
    """(etiqueta, día_desde, día_hasta_excl) de cada mes calendario completo dentro del período."""
    out, f = [], inicio
    while f <= fin:
        a = f.replace(day=1)
        b = (a + dt.timedelta(days=32)).replace(day=1)
        if a >= inicio and b - dt.timedelta(days=1) <= fin:
            out.append((a.strftime("%Y-%m"), (a - inicio).days, (b - inicio).days))
        f = b
    return out


def flujos_mensuales(eventos: dict, meses) -> dict:
    out = {}
    for k in FLUJOS:
        if k not in eventos:
            continue
        x = np.asarray(eventos[k], dtype=float)
        out[k] = [int(((x >= a) & (x < b)).sum()) for _, a, b in meses]
    return out


def flujos_totales(eventos: dict, n_dias: int) -> dict:
    return {k: int(((np.asarray(eventos[k]) >= 0) & (np.asarray(eventos[k]) < n_dias)).sum())
            for k in FLUJOS if k in eventos}


def stats_espera(esperas, n_dias: int) -> dict:
    """Mismas reglas para observado y simulado: solo ingresos dentro del período."""
    ing = [(a, b) for a, b, *_ in esperas if 0 <= a < n_dias]
    if not ing:
        return {}
    w = np.array([b - a for a, b in ing if b is not None and b < n_dias])
    out = {"frac_atendida_en_periodo": float(len(w) / len(ing))}
    if len(w):
        out.update({"mediana_dias": float(np.median(w)), "media_dias": float(np.mean(w)),
                    "p90_dias": float(np.percentile(w, 90))})
    for d in D_ESPERA:
        elegibles = [(a, b) for a, b in ing if a <= n_dias - d]
        if len(elegibles) >= 20:
            out[f"P_atendida_<={d}d"] = float(np.mean([b is not None and b - a <= d for a, b in elegibles]))
    return out


def _rango_pct(x: float, muestra) -> float:
    s = np.sort(np.asarray(muestra, float))
    return float((np.searchsorted(s, x, "left") + np.searchsorted(s, x, "right")) / 2 / len(s))


def comparar_escalar(obs: float, sims, tol_sesgo: float | None = None) -> dict:
    s = np.asarray(sims, float)
    s = s[~np.isnan(s)]
    a = (1 - CRITERIOS["nivel"]) / 2
    lo, hi = np.quantile(s, [a, 1 - a])
    q = _rango_pct(obs, s)
    sesgo = (s.mean() - obs) / obs if obs else float("nan")
    ok = bool(lo <= obs <= hi) and (tol_sesgo is None or (not math.isnan(sesgo) and abs(sesgo) <= tol_sesgo))
    return {"obs": obs, "sim_media": float(s.mean()), "sim_de": float(s.std(ddof=1)),
            "ip95": [float(lo), float(hi)], "rango_percentil_obs": q, "p_dos_colas": float(min(1, 2 * min(q, 1 - q))),
            "sesgo_rel": float(sesgo), "cumple": ok}


# ═══════════════════════════ validación ═══════════════════════════


def validar(obs: dict, sims: list, inicio: dt.date, fin: dt.date) -> dict:
    n_dias = (fin - inicio).days + 1
    meses = meses_completos(inicio, fin)
    res = {"meses": [m[0] for m in meses], "flujos_total": {}, "flujos_mensual": {}, "cortes": {}, "esperas": {}}

    # flujos
    f_obs_m = flujos_mensuales(obs["eventos"], meses)
    f_obs_t = flujos_totales(obs["eventos"], n_dias)
    f_sim_m = [flujos_mensuales(s["eventos"], meses) for s in sims]
    f_sim_t = [flujos_totales(s["eventos"], n_dias) for s in sims]
    for k in FLUJOS:
        if k not in f_obs_t or k not in f_sim_t[0]:
            continue
        res["flujos_total"][k] = comparar_escalar(f_obs_t[k], [t[k] for t in f_sim_t], CRITERIOS["sesgo_flujo_total"])
        S = np.array([m[k] for m in f_sim_m], float)
        o = np.array(f_obs_m[k], float)
        a = (1 - CRITERIOS["nivel"]) / 2
        lo, hi = np.quantile(S, a, axis=0), np.quantile(S, 1 - a, axis=0)
        dentro = (o >= lo) & (o <= hi)
        mape = float(np.mean(np.abs(S.mean(0) - o) / np.maximum(o, 1)))
        res["flujos_mensual"][k] = {
            "obs": o.tolist(), "sim_media": S.mean(0).tolist(), "banda_lo": lo.tolist(), "banda_hi": hi.tolist(),
            "cobertura": float(dentro.mean()) if len(o) else float("nan"), "mape": mape,
            "cumple": bool(len(o) and dentro.mean() >= CRITERIOS["cobertura_mensual"]),
        }

    # cortes de lista
    for clave, serie in obs["cortes"].items():
        if clave not in sims[0]["stock"]:
            continue
        filas = []
        for f, n in sorted(serie.items()):
            i = (f - inicio).days
            if not (0 <= i < min(len(s["stock"][clave]) for s in sims)) or i < 7:
                continue  # el corte inicial es entrada del modelo, no se valida
            c = comparar_escalar(n, [s["stock"][clave][i] for s in sims])
            c["fecha"] = f.isoformat()
            filas.append(c)
        if filas:
            cob = float(np.mean([c["cumple"] for c in filas]))
            ses = [abs(c["sesgo_rel"]) for c in filas if not math.isnan(c["sesgo_rel"])]
            ses = float(np.mean(ses)) if ses else float("nan")
            res["cortes"][clave] = {"filas": filas, "cobertura": cob, "sesgo_abs_medio": ses,
                                    "cumple": bool(cob >= CRITERIOS["cobertura_mensual"]
                                                   and (math.isnan(ses) or ses <= CRITERIOS["sesgo_corte"]))}

    # esperas hasta 1ra consulta
    if obs.get("esperas"):
        eo = stats_espera(obs["esperas"], n_dias)
        es = [stats_espera(s["esperas"], n_dias) for s in sims]
        for k, v in eo.items():
            vals = [e.get(k, np.nan) for e in es]
            if np.isfinite(vals).sum() >= 10:
                res["esperas"][k] = comparar_escalar(v, vals)
        # Mann-Whitney observado vs cada réplica (como en la tesis): mediana de p y % de réplicas con p > 0,05
        try:
            from scipy.stats import mannwhitneyu, ks_2samp
            wo = [b - a for a, b in obs["esperas"] if b is not None and 0 <= a < n_dias and b < n_dias]
            ps_mw, ps_ks = [], []
            for s in sims:
                ws = [b - a for a, b, *_ in s["esperas"] if b is not None and 0 <= a < n_dias and b < n_dias]
                if len(wo) > 5 and len(ws) > 5:
                    ps_mw.append(mannwhitneyu(wo, ws).pvalue)
                    ps_ks.append(ks_2samp(wo, ws).pvalue)
            if ps_mw:
                res["esperas"]["_pruebas_distribucion"] = {
                    "mann_whitney_p_mediana": float(np.median(ps_mw)), "ks_p_mediana": float(np.median(ps_ks)),
                    "frac_replicas_mw_p>0.05": float(np.mean(np.array(ps_mw) > 0.05)),
                    "frac_replicas_ks_p>0.05": float(np.mean(np.array(ps_ks) > 0.05)),
                }
        except ImportError:
            pass

    # resumen
    chequeos = ([("flujo total", k, v["cumple"]) for k, v in res["flujos_total"].items()]
                + [("flujo mensual", k, v["cumple"]) for k, v in res["flujos_mensual"].items()]
                + [("lista", k, v["cumple"]) for k, v in res["cortes"].items()]
                + [("espera", k, v["cumple"]) for k, v in res["esperas"].items() if not k.startswith("_")])
    res["resumen"] = {"n_chequeos": len(chequeos), "n_cumple": sum(c[2] for c in chequeos),
                      "no_cumple": [f"{a}: {b}" for a, b, c in chequeos if not c]}
    return res


# ═══════════════════════════ salida ═══════════════════════════


def _fmt(x, d=1):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def informe_md(salida: dict) -> str:
    L = [f"# Validación externa del DES — {salida['inicio']} a {salida['fin']}", "",
         f"Réplicas por modo: {salida['reps']}. Criterios: {json.dumps(CRITERIOS)}.", ""]
    if salida["faltan"]:
        L += ["**Datos no recibidos (indicadores sin validar):**", ""] + [f"- {f}" for f in salida["faltan"]] + [""]
    if salida.get("mix_prioridad_obs"):
        L += [f"Mezcla de prioridades observada en ingresos: {salida['mix_prioridad_obs']} (DES: alta 0,65, media 0,35).", ""]
    for modo, r in salida["modos"].items():
        L += [f"## Modo {modo}", ""] + [f"- {n}" for n in r.get("notas", [])] + [""]
        v = r["validacion"]
        L += [f"**{v['resumen']['n_cumple']} de {v['resumen']['n_chequeos']} chequeos cumplen.**", ""]
        if v["flujos_total"]:
            L += ["| Flujo (total del período) | Observado | DES media | IP95 DES | Sesgo | Meses en banda | Cumple |",
                  "|---|---:|---:|---|---:|---:|:---:|"]
            for k, c in v["flujos_total"].items():
                fm = v["flujos_mensual"].get(k, {})
                L.append(f"| {k} | {_fmt(c['obs'], 0)} | {_fmt(c['sim_media'], 0)} | [{_fmt(c['ip95'][0], 0)}; {_fmt(c['ip95'][1], 0)}] | "
                         f"{_fmt(100 * c['sesgo_rel'])} % | {_fmt(100 * fm.get('cobertura', float('nan')), 0)} % | "
                         f"{'sí' if c['cumple'] and fm.get('cumple', True) else 'no'} |")
            L.append("")
        if v["cortes"]:
            L += ["| Lista (serie de cortes) | Cortes | En banda IP95 | Sesgo abs. medio | Cumple |", "|---|---:|---:|---:|:---:|"]
            for k, cs in v["cortes"].items():
                L.append(f"| {k} | {len(cs['filas'])} | {_fmt(100 * cs['cobertura'], 0)} % | {_fmt(100 * cs['sesgo_abs_medio'])} % | "
                         f"{'sí' if cs['cumple'] else 'no'} |")
            L += ["", "<details><summary>Detalle por corte</summary>", "",
                  "| Lista | Fecha | Observado | DES media | IP95 DES | Sesgo |", "|---|---|---:|---:|---|---:|"]
            for k, cs in v["cortes"].items():
                for c in cs["filas"]:
                    L.append(f"| {k} | {c['fecha']} | {_fmt(c['obs'], 0)} | {_fmt(c['sim_media'], 0)} | "
                             f"[{_fmt(c['ip95'][0], 0)}; {_fmt(c['ip95'][1], 0)}] | {_fmt(100 * c['sesgo_rel'])} % |")
            L += ["", "</details>", ""]
        if v["esperas"]:
            L += ["| Espera ingreso → 1ra consulta | Observado | DES media | IP95 DES | Percentil obs. | Cumple |", "|---|---:|---:|---|---:|:---:|"]
            for k, c in v["esperas"].items():
                if k.startswith("_"):
                    continue
                d = 3 if k.startswith(("P_", "frac")) else 1
                L.append(f"| {k} | {_fmt(c['obs'], d)} | {_fmt(c['sim_media'], d)} | [{_fmt(c['ip95'][0], d)}; {_fmt(c['ip95'][1], d)}] | "
                         f"{_fmt(c['rango_percentil_obs'], 2)} | {'sí' if c['cumple'] else 'no'} |")
            pr = v["esperas"].get("_pruebas_distribucion")
            if pr:
                L += ["", "Informativo, no entra en los criterios (las esperas de una misma réplica no son independientes, "
                      "así que esta prueba rechaza de más):",
                      f"Distribución completa (observado vs cada réplica): Mann-Whitney p mediana = {_fmt(pr['mann_whitney_p_mediana'], 3)} "
                      f"({_fmt(100 * pr['frac_replicas_mw_p>0.05'], 0)} % de réplicas con p > 0,05); KS p mediana = {_fmt(pr['ks_p_mediana'], 3)}."]
            L.append("")
        if v["resumen"]["no_cumple"]:
            L += ["No cumplen: " + "; ".join(v["resumen"]["no_cumple"]), ""]
    return "\n".join(L)


def figuras(salida: dict, out: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    modo = "entradas_observadas" if "entradas_observadas" in salida["modos"] else next(iter(salida["modos"]))
    v = salida["modos"][modo]["validacion"]
    ks = list(v["flujos_mensual"])
    if ks:
        n = len(ks)
        fig, axs = plt.subplots((n + 2) // 3, 3, figsize=(13, 3.2 * ((n + 2) // 3)), squeeze=False)
        for ax, k in zip(axs.flat, ks):
            f = v["flujos_mensual"][k]
            x = np.arange(len(f["obs"]))
            ax.fill_between(x, f["banda_lo"], f["banda_hi"], color="#9bb7d4", alpha=.5, label="DES IP95")
            ax.plot(x, f["sim_media"], color="#2b5d8a", label="DES media")
            ax.plot(x, f["obs"], "o-", color="#c0392b", ms=4, label="Observado")
            ax.set_title(f"{k} (cobertura {100 * f['cobertura']:.0f} %)", fontsize=10)
            ax.set_xticks(x, [m[2:] for m in v["meses"]], rotation=60, fontsize=7)
        for ax in list(axs.flat)[n:]:
            ax.axis("off")
        axs.flat[0].legend(fontsize=7)
        fig.suptitle(f"Flujos mensuales — modo {modo}")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig_flujos.png"), dpi=150)
        plt.close(fig)
    if v["cortes"]:
        n = len(v["cortes"])
        fig, axs = plt.subplots(1, n, figsize=(3.6 * n, 3.2), squeeze=False)
        for ax, (k, cs) in zip(axs.flat, v["cortes"].items()):
            fs = cs["filas"]
            x = np.arange(len(fs))
            ax.fill_between(x, [c["ip95"][0] for c in fs], [c["ip95"][1] for c in fs], color="#9bb7d4", alpha=.5)
            ax.plot(x, [c["sim_media"] for c in fs], color="#2b5d8a")
            ax.plot(x, [c["obs"] for c in fs], "o-", color="#c0392b", ms=4)
            ax.set_title(k, fontsize=10)
            ax.set_xticks(x, [c["fecha"][2:7] for c in fs], rotation=60, fontsize=7)
        fig.suptitle(f"Listas al cierre de mes — modo {modo}")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig_cortes.png"), dpi=150)
        plt.close(fig)


# ═══════════════════════════ datos sintéticos (demo) ═══════════════════════════


def generar_demo(carpeta: str, inicio: dt.date, factor_demanda: float = 1.0, semilla: int = 123_456):
    """Escribe los 4 CSV con el formato pedido al CRS, a partir de UNA corrida del DES.

    Con factor_demanda = 1 el "mundo real" es el propio modelo: la validación debe pasar.
    Con factor_demanda ≠ 1 la realidad tiene más (o menos) demanda: debe fallar.
    """
    import csv
    os.makedirs(carpeta, exist_ok=True)
    over = config_tesis()
    over["nb_arrivals_n"] = 8.99 * factor_demanda  # media ∝ n con p fijo
    r = _correr((semilla, over))
    fecha = lambda x: (inicio + dt.timedelta(days=math.floor(x))).isoformat()  # noqa: E731
    with open(os.path.join(carpeta, "pacientes.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id_paciente", "prioridad", "fecha_ingreso_lista", "fecha_primera_consulta",
                    "fecha_egreso_lista", "causal_egreso"])
        nombre = {"high": "alta", "mid": "media", "low": "baja"}
        for i, (a, b, c, pr) in enumerate(r["esperas"]):
            pr = nombre.get(pr, pr)
            w.writerow([f"P{i:05d}", pr, fecha(a), fecha(b) if b is not None else "",
                        fecha(b) if b is not None else "", "atendida" if b is not None else (c or "")])
    with open(os.path.join(carpeta, "eventos.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fecha", "tipo", "estado"])
        for k, tipo in (("primeras", "primera"), ("controles", "control"), ("ecografias", "ecografia"),
                        ("laboratorios", "laboratorio"), ("cirugias", "cirugia")):
            for x in r["eventos"].get(k, []):
                w.writerow([fecha(x), tipo, "realizada"])
    with open(os.path.join(carpeta, "cortes_lista.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fecha_corte", "etapa", "prioridad", "n_pacientes"])
        from simulador_clinica_baseline import SimConfig
        base = SimConfig()
        v = inicio - dt.timedelta(days=1)
        for et, pr, n in (("primera", "alta", base.backlog_high_n), ("primera", "media", base.backlog_mid_n),
                          ("primera", "baja", base.backlog_low_n), ("control", "total", base.post_init_control_n),
                          ("prequirurgico", "total", base.post_init_preq_n), ("quirurgico", "total", base.post_init_surgery_waitlist_n)):
            w.writerow([v.isoformat(), et, pr, n])
        fin = inicio + dt.timedelta(days=r["dias"] - 1)
        for c in _cierres_de_mes(inicio, fin):
            i = (c - inicio).days
            st = r["stock"]
            for et, pr, k in (("primera", "alta", "primera_alta"), ("primera", "media", "primera_media"),
                              ("primera", "baja", "primera_baja"), ("control", "total", "control_total"),
                              ("prequirurgico", "total", "prequirurgico_total"), ("quirurgico", "total", "quirurgico_total")):
                w.writerow([c.isoformat(), et, pr, int(st[k][i])])
    with open(os.path.join(carpeta, "capacidad_semanal.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["semana_inicio", "cupos_primera", "cupos_control", "cupos_eco_ugd", "cupos_eco_matrona",
                    "cupos_lab_matrona", "cirugias"])
        for s in range(52):
            w.writerow([(inicio + dt.timedelta(weeks=s)).isoformat(), 16, 40, 25, 25, 25, 20])
    return inicio + dt.timedelta(days=r["dias"] - 1)


# ═══════════════════════════ main ═══════════════════════════


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datos", default="datos_crs")
    ap.add_argument("--inicio", help="primer día del período de validación (idealmente lunes), AAAA-MM-DD")
    ap.add_argument("--fin", help="último día del período (máx. 52 semanas recomendado)")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--nucleos", type=int, default=int(os.environ.get("RL_NUCLEOS", os.cpu_count() or 1)))
    ap.add_argument("--modos", default="entradas_observadas,tesis")
    ap.add_argument("--out", default=os.path.join("resultados", "validacion_externa"))
    ap.add_argument("--demo", action="store_true", help="genera datos sintéticos con el formato pedido y los valida")
    ap.add_argument("--demo_demanda", type=float, default=1.0)
    a = ap.parse_args()

    if a.demo:
        a.inicio = a.inicio or "2025-09-01"
        a.datos = os.path.join(a.out, f"datos_demo_x{a.demo_demanda:g}")
        fin = generar_demo(a.datos, dt.date.fromisoformat(a.inicio), a.demo_demanda)
        a.fin = a.fin or fin.isoformat()
        a.out = os.path.join(a.out, f"demo_x{a.demo_demanda:g}")
        print(f"Datos sintéticos en {a.datos} (demanda real × {a.demo_demanda:g})")
    if not (a.inicio and a.fin):
        ap.error("--inicio y --fin son obligatorios (o usa --demo)")
    inicio, fin = dt.date.fromisoformat(a.inicio), dt.date.fromisoformat(a.fin)
    if inicio.weekday() != 0:
        print(f"Aviso: {inicio} no es lunes; el DES parte la semana en lunes, se comparan fechas tal cual.")
    if (fin - inicio).days + 1 > 364:
        print("Aviso: período > 52 semanas; el DES se corre por todo el período.")
    os.makedirs(a.out, exist_ok=True)

    obs, faltan = cargar_observado(a.datos, inicio, fin)
    if not obs["eventos"] and not obs["cortes"]:
        sys.exit(f"No se encontraron datos en {a.datos}")
    salida = {"inicio": a.inicio, "fin": a.fin, "reps": a.reps, "faltan": faltan,
              "mix_prioridad_obs": obs.get("mix_prioridad"), "criterios": CRITERIOS, "modos": {}}
    for modo in a.modos.split(","):
        if modo == "entradas_observadas":
            over, notas = config_desde_datos(obs, inicio, fin)
        elif modo == "tesis":
            over = config_tesis()
            over["weeks_to_simulate"] = int(math.ceil(((fin - inicio).days + 1) / 7))
            notas = ["configuración calibrada de la tesis, sin usar ningún dato del período"]
        else:
            sys.exit(f"modo desconocido: {modo}")
        print(f"[{modo}] simulando {a.reps} réplicas con {a.nucleos} núcleos …", flush=True)
        sims = simular(over, a.reps, a.nucleos)
        salida["modos"][modo] = {"config": over, "notas": notas, "validacion": validar(obs, sims, inicio, fin)}
        rs = salida["modos"][modo]["validacion"]["resumen"]
        print(f"[{modo}] {rs['n_cumple']}/{rs['n_chequeos']} chequeos cumplen", flush=True)

    with open(os.path.join(a.out, "validacion_externa.json"), "w", encoding="utf-8") as f:
        json.dump(salida, f, ensure_ascii=False, indent=1, default=str)
    md = informe_md(salida)
    with open(os.path.join(a.out, "validacion_externa.md"), "w", encoding="utf-8") as f:
        f.write(md)
    figuras(salida, a.out)
    print(md)
    print(f"\nResultados en {a.out}/")


if __name__ == "__main__":
    main()
