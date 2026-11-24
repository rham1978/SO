"""
Piezas comunes a los tres agentes (REINFORCE, PPO, Double DQN), para que difieran
solo en el algoritmo: misma red (MLP 2×64, tanh), mismo estado normalizado, misma
forma de muestrear episodios y mismo punto de control en validación.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from concurrent.futures import ProcessPoolExecutor

from catalogo import N_ACCIONES
from entorno import OBS_DIM, SEMILLAS_TRAIN, SEMILLAS_VAL, rollout

torch.set_num_threads(1)

N_NORM = 200                       # episodios con política aleatoria para ajustar la normalización
ESCALA_R = 10.0                    # recompensa / ESCALA_R (≈ −1,2 por semana): solo para estabilidad numérica


# ── Normalización del estado (ajustada SOLO con episodios de entrenamiento) ──
class Normalizador:
    def __init__(self, media, desv):
        self.media = np.asarray(media, np.float32)
        self.desv = np.maximum(np.asarray(desv, np.float32), 1e-6)

    def __call__(self, obs):
        return ((obs - self.media) / self.desv).astype(np.float32)

    @staticmethod
    def ajustar(n_episodios=N_NORM, ex=None):
        """Media y desviación del estado bajo política aleatoria, semillas de entrenamiento
        (bloque reservado al final del rango, que los agentes no usan para entrenar)."""
        semillas = list(SEMILLAS_TRAIN[-n_episodios:])
        own = ex is None
        ex = ex or ProcessPoolExecutor(os.cpu_count())
        try:
            trajs = list(ex.map(_rollout_aleatorio, semillas))
        finally:
            if own:
                ex.shutdown()
        X = np.concatenate([t["obs"] for t in trajs])
        return Normalizador(X.mean(0), X.std(0))


def normalizador_compartido(n_episodios=N_NORM, ex=None, carpeta="resultados"):
    """El mismo normalizador para todos los métodos: se calcula una vez y se guarda."""
    path = os.path.join(carpeta, f"normalizador_{n_episodios}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return Normalizador(z["media"], z["desv"])
    norm = Normalizador.ajustar(n_episodios, ex)
    os.makedirs(carpeta, exist_ok=True)
    np.savez(path, media=norm.media, desv=norm.desv)
    return norm


def _rollout_aleatorio(seed):
    rng = np.random.default_rng(seed)
    return rollout(lambda o: int(rng.integers(N_ACCIONES)), seed)


def semillas_entrenamiento(semilla_init: int) -> np.ndarray:
    """Cada inicialización recorre un orden distinto del rango de entrenamiento."""
    rng = np.random.default_rng(semilla_init)
    return rng.permutation(np.arange(SEMILLAS_TRAIN.start, SEMILLAS_TRAIN.stop - N_NORM))


def mlp(salida, obs_dim=OBS_DIM, h=64):
    return nn.Sequential(nn.Linear(obs_dim, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, salida))


# ── Rollout en un proceso trabajador (la red viaja como state_dict) ─────────
def rollout_red(args):
    """args = (tipo, state_dict, norm, seed, modo, eps)
    tipo 'pi': red de logits (REINFORCE, PPO); modo 'muestrear' o 'greedy'.
    tipo 'q':  red de valores Q (DQN);         modo 'eps' (ε-greedy) o 'greedy'."""
    tipo, state_dict, norm, seed, modo, eps = args
    torch.set_num_threads(1)
    red = mlp(N_ACCIONES)
    red.load_state_dict(state_dict)
    red.eval()
    rng = np.random.default_rng(seed + 7_777_777)

    def policy_fn(obs):
        with torch.no_grad():
            z = red(torch.as_tensor(norm(obs)))
        if modo == "greedy":
            return int(z.argmax())
        if tipo == "pi":
            p = torch.softmax(z, -1).numpy().astype(np.float64)
            return int(rng.choice(N_ACCIONES, p=p / p.sum()))
        if rng.random() < eps:
            return int(rng.integers(N_ACCIONES))
        return int(z.argmax())

    tr = rollout(policy_fn, seed)
    return {k: tr[k] for k in ("obs", "actions", "rewards")} | {
        "L": tr["kpis"]["L"], "tts": tr["kpis"]["tts_full_days_mean"],
        "atenciones": tr["kpis"]["total_atenciones"]}


class PuntoControl:
    """Evalúa la política greedy en las mismas N_EVAL semillas de validación y guarda la mejor por L."""

    def __init__(self, out, tipo, norm, n_eval, config):
        os.makedirs(out, exist_ok=True)
        self.out, self.tipo, self.norm, self.n_eval, self.config = out, tipo, norm, n_eval, config
        self.curva, self.mejor, self.t0 = [], np.inf, time.time()

    def __call__(self, ex, red, vistos, retorno_train):
        sd = {k: v.detach().clone() for k, v in red.state_dict().items()}
        ev = list(ex.map(rollout_red, [(self.tipo, sd, self.norm, s, "greedy", 0.0)
                                       for s in SEMILLAS_VAL[:self.n_eval]]))
        L = float(np.mean([e["L"] for e in ev]))
        acc = np.concatenate([e["actions"] for e in ev])
        self.curva.append({"episodios": vistos, "L_val": L,
                           "tts_val": float(np.mean([e["tts"] for e in ev])),
                           "retorno_train": retorno_train,
                           "acciones_distintas": int(len(np.unique(acc))),
                           "seg": round(time.time() - self.t0)})
        print(f"[{vistos:5d}] L val {L:.1f}  TTS val {self.curva[-1]['tts_val']:.1f}  "
              f"retorno train {retorno_train:.1f}  acciones distintas {self.curva[-1]['acciones_distintas']}",
              flush=True)
        if L < self.mejor:
            self.mejor = L
            torch.save({"tipo": self.tipo, "red": sd, "norm_media": self.norm.media,
                        "norm_desv": self.norm.desv, "episodios": vistos, "L_val": L},
                       os.path.join(self.out, "mejor.pt"))

    def guardar(self, gasto):
        json.dump({"curva": self.curva, "config": self.config, "presupuesto_usado": gasto},
                  open(os.path.join(self.out, "curva.json"), "w"), indent=1)


def cargar_politica(path):
    """Devuelve (tipo, red, norm) desde un mejor.pt."""
    ck = torch.load(path, weights_only=False)
    red = mlp(N_ACCIONES)
    red.load_state_dict(ck["red"])
    red.eval()
    return ck["tipo"], red, Normalizador(ck["norm_media"], ck["norm_desv"])
