"""
Entorno semanal de decisión sobre el simulador de la clínica CMI.

Cada episodio = 1 año (52 semanas). Antes de cada semana w, el agente elige una
configuración de cupos del catálogo (ver `catalogo.py`). La decisión se toma el
viernes 00:00 de la semana anterior, antes de que el simulador publique la agenda
(publish_lead_workdays = 1 → publicación viernes 8:00) y antes de generar los
cupos de exámenes y cirugía (lunes 8:00).

Recompensa semanal: -(pacientes en el sistema × días) / 1000, muestreado a diario.

Uso:
    from entorno import rollout
    traj = rollout(policy_fn, seed=0)     # policy_fn(obs: np.ndarray) -> int
    traj["obs"], traj["actions"], traj["rewards"], traj["kpis"]

`rollout` reutiliza `run_once` del simulador original, de modo que los KPIs
finales (tts_full_days_mean, total_atenciones, ...) se calculan exactamente
igual que en la tesis.
"""
from __future__ import annotations

import dataclasses
import logging
import math
from collections import Counter

import numpy as np
import simpy

import sim_fast
from sim_fast import sim
from catalogo import CATALOGO, aplicar_accion, ACCION_BASE

sim_fast.apply_patch()
logging.disable(logging.WARNING)

DAY = 24 * 60
WEEK = 7 * DAY
N_WEEKS = 52

# Particiones por semilla del simulador (fijadas antes de entrenar)
SEMILLAS_TRAIN = range(0, 10_000)
SEMILLAS_VAL = range(10_000, 10_100)
SEMILLAS_TEST = range(500_000, 500_030)  # mismo bloque de re-evaluación de incumbentes de benchmark_riguroso.py (SMAC)
# Semillas de humo: solo para probar el pipeline de punta a punta (corrida pequeña).
# Disjuntas de las tres particiones, para no gastar la prueba antes de tiempo.
SEMILLAS_HUMO = range(900_000, 900_100)

# ── Agrupación de estados del paciente para la observación ──────────────────
def _grupo(state: str) -> str | None:
    if state in ("post_completed", "first_canceled", "removed_unreachable_first"):
        return None
    if state.startswith(("booked_first", "in_first")):
        return "first_booked"
    if ("first" in state or "reconversion" in state or state in ("local_adjust_wait", "special_prefirst_route")):
        return "first_wait"
    if "control" in state or "monthly" in state:
        return "control"
    if "preq" in state:
        return "preq"
    if "quir" in state or "surgery" in state:
        return "quir"
    return "post_other"  # post_started, after_surgery, postop...

GRUPOS_FIRST = ("first_wait",)
GRUPOS_POST = ("first_booked", "control", "preq", "quir", "post_other")
PRIORIDADES = ("high", "mid", "low")

OBS_NAMES = (
    [f"first_wait_{p}" for p in PRIORIDADES]
    + list(GRUPOS_POST)
    + ["first_wait_age_days", "exam_waiters", "post_ctrl_blocked_week", "sin_week", "cos_week"]
    + ["prev_first", "prev_control", "prev_eco", "prev_lab"]
)
OBS_DIM = len(OBS_NAMES)

_CURRENT_MODEL = None


class _ModelHook(sim.ClinicModelAdjusted):
    """Registra la prioridad y llegada de cada paciente y cuenta esperas por exámenes."""

    def __init__(self, env, rng, cfg=None):
        global _CURRENT_MODEL
        self._pinfo = {}
        self._exam_waiters = 0
        super().__init__(env, rng, cfg)
        _CURRENT_MODEL = self

    def _register_patient(self, p):
        self._pinfo[p.pid] = (str(getattr(p.priority, "value", p.priority)), float(p.enqueued_at))
        return super()._register_patient(p)

    def _book_from_pool(self, *a, **kw):
        self._exam_waiters += 1
        try:
            r = yield from sim_fast._book_from_pool_fast(self, *a, **kw)
        finally:
            self._exam_waiters -= 1
        return r


def observar(model, prev_action: int, blocked_week: int, week: int) -> np.ndarray:
    now = model.env.now
    c = Counter()
    age_sum, age_n = 0.0, 0
    for pid, st in model.patient_state.items():
        g = _grupo(st)
        if g is None:
            continue
        if g == "first_wait":
            pr, enq = model._pinfo.get(pid, ("mid", now))
            c[f"first_wait_{pr}"] += 1
            age_sum += (now - enq) / DAY
            age_n += 1
        else:
            c[g] += 1
    prev = CATALOGO[prev_action]
    ang = 2 * math.pi * (week % 52) / 52
    v = [c[f"first_wait_{p}"] for p in PRIORIDADES] + [c[g] for g in GRUPOS_POST]
    v += [age_sum / age_n if age_n else 0.0, model._exam_waiters, blocked_week, math.sin(ang), math.cos(ang)]
    v += [prev["first"], prev["control"], prev["eco"], prev["lab"]]
    return np.asarray(v, dtype=np.float32)


class _DecisionEnv(simpy.Environment):
    """Environment que intercala decisiones semanales y mide pacientes-día."""

    def __init__(self, policy_fn, cfg, traj):
        super().__init__()
        self._policy = policy_fn
        self._cfg = cfg
        self._traj = traj

    def run(self, until=None):
        model = _CURRENT_MODEL
        cfg, traj = self._cfg, self._traj
        prev_a = traj["actions"][0]
        blocked_prev = 0
        for w in range(1, N_WEEKS + 1):
            t_dec = until if w == N_WEEKS else min(w * WEEK - 3 * DAY, until)  # viernes 00:00 previo a la semana w
            patient_days = 0.0
            # avanzar día a día hasta la próxima decisión, muestreando pacientes en sistema
            t = self.now
            while t < t_dec:
                t = min(t + DAY, t_dec)
                super().run(until=t)
                patient_days += model.total_in_system_now()  # 1 día × pacientes
            traj["rewards"].append(-patient_days / 1000.0)
            if w == N_WEEKS or t_dec >= until:
                break
            blocked = model.kpis.post_control_blocked
            obs = observar(model, prev_a, blocked - blocked_prev, w)
            blocked_prev = blocked
            a = int(self._policy(obs))
            if not getattr(cfg, "_sin_acciones", False):
                aplicar_accion(cfg, a)
            traj["obs"].append(obs)
            traj["actions"].append(a)
            prev_a = a
        if self.now < until:
            super().run(until=until)


def config_base(**overrides) -> "sim.SimConfig":
    cfg = sim.SimConfig(
        benchmark_mode=True,
        use_fixed_weekly_capacity=True,
        use_fixed_post_control_hours=True,
        publish_lead_workdays=1,
        n_workers=1,
    )
    return dataclasses.replace(cfg, **overrides)


def rollout(policy_fn, seed: int, cfg_overrides: dict | None = None,
            aplicar_acciones: bool = True) -> dict:
    """Simula un año con la política dada. Devuelve trayectoria y KPIs finales.

    aplicar_acciones=False deja la configuración tal como viene en cfg_overrides
    durante todo el año (reglas externas al catálogo, p. ej. el incumbente SMAC-SK).
    """
    cfg = config_base(**(cfg_overrides or {}))
    if not aplicar_acciones:
        cfg._sin_acciones = True
    traj = {"obs": [], "actions": [], "rewards": [], "seed": seed}

    # Decisión de la semana 0 (t = 0): observación inicial "vacía" del sistema
    # se construye después de crear el modelo; aquí usamos la política sobre un
    # modelo recién inicializado.
    orig_env, orig_model = sim.simpy.Environment, sim.ClinicModelAdjusted
    holder = {}

    def make_env():
        e = _DecisionEnv(policy_fn, cfg, traj)
        holder["env"] = e
        return e

    class _Hook0(_ModelHook):
        def __init__(self, env, rng, c=None):
            super().__init__(env, rng, c)
            obs0 = observar(self, ACCION_BASE, 0, 0)
            a0 = int(policy_fn(obs0))
            if not getattr(cfg, "_sin_acciones", False):
                aplicar_accion(cfg, a0)
            traj["obs"].append(obs0)
            traj["actions"].append(a0)

    sim.simpy.Environment = make_env
    sim.ClinicModelAdjusted = _Hook0
    try:
        kpis = sim.run_once(seed, cfg)
    finally:
        sim.simpy.Environment = orig_env
        sim.ClinicModelAdjusted = orig_model
    traj["kpis"] = {k: v for k, v in kpis.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    # Métrica principal: L = pacientes promedio en el sistema (pacientes·día / días),
    # que por la ley de Little es proporcional al tiempo medio en el sistema.
    dias = cfg.weeks_to_simulate * 7
    traj["kpis"]["L"] = -1000.0 * float(np.sum(traj["rewards"])) / dias
    # Fin del horizonte: pacientes que siguen dentro al cierre del año (detecta
    # políticas que "empujan" trabajo más allá del horizonte).
    traj["kpis"]["en_sistema_final"] = float(_CURRENT_MODEL.total_in_system_now())
    traj["obs"] = np.stack(traj["obs"])
    traj["actions"] = np.asarray(traj["actions"], dtype=np.int64)
    traj["rewards"] = np.asarray(traj["rewards"], dtype=np.float32)
    assert len(traj["obs"]) == len(traj["actions"]) == len(traj["rewards"]) == N_WEEKS, (
        len(traj["obs"]), len(traj["actions"]), len(traj["rewards"]))
    return traj
