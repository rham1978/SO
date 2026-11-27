"""
Double DQN sobre el mismo entorno, red y presupuesto.

  - Q(s, ·): MLP 2×64 → 28 valores. Pérdida de Huber.
  - Objetivo Double DQN: y = r + γ·Q_obj(s', argmax_a Q(s', a)); terminal en la semana 52.
  - Buffer de repetición de 100 000 transiciones; ε lineal de 1 a 0,05 en el primer
    `frac_eps` del presupuesto de episodios.
  - Los episodios se simulan en paralelo con la red actual (ε-greedy); después de cada
    lote se hacen `pasos_por_trans` actualizaciones por transición nueva.

Uso:
    python ddqn.py --semilla 0 --lr 1e-3 --obj_cada 500 --out resultados/ddqn_s0
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from catalogo import N_ACCIONES
import presupuesto as P
from comun import ESCALA_R, N_NORM, normalizador_compartido, PuntoControl, mlp, rollout_red, semillas_entrenamiento, guardar_estado, cargar_estado


class Buffer:
    def __init__(self, cap, obs_dim):
        self.o = np.zeros((cap, obs_dim), np.float32); self.o2 = np.zeros_like(self.o)
        self.a = np.zeros(cap, np.int64); self.r = np.zeros(cap, np.float32); self.d = np.zeros(cap, np.float32)
        self.cap, self.n, self.i = cap, 0, 0

    def agregar(self, o, a, r, o2, d):
        self.o[self.i], self.a[self.i], self.r[self.i], self.o2[self.i], self.d[self.i] = o, a, r, o2, d
        self.i = (self.i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def muestra(self, rng, k):
        idx = rng.integers(self.n, size=k)
        return (torch.as_tensor(self.o[idx]), torch.as_tensor(self.a[idx]), torch.as_tensor(self.r[idx]),
                torch.as_tensor(self.o2[idx]), torch.as_tensor(self.d[idx]))


def entrenar(episodios=P.B_TRAIN, lote=8, lr=1e-3, gamma=0.99, obj_cada=500, buffer=100_000,
             minilote=64, pasos_por_trans=0.25, eps_fin=0.05, frac_eps=0.5, semilla=0,
             eval_cada=P.EVAL_CADA, n_eval=P.N_EVAL, out="resultados/ddqn", n_workers=None, n_norm=N_NORM):
    torch.manual_seed(semilla)
    rng = np.random.default_rng(semilla)
    config = dict(metodo="ddqn", episodios=episodios, lote=lote, lr=lr, gamma=gamma, obj_cada=obj_cada,
                  buffer=buffer, minilote=minilote, pasos_por_trans=pasos_por_trans, eps_fin=eps_fin,
                  frac_eps=frac_eps, semilla=semilla, eval_cada=eval_cada, n_eval=n_eval)
    with ProcessPoolExecutor(n_workers or P.nucleos()) as ex:
        norm = normalizador_compartido(n_norm, ex)
        q, q_obj = mlp(N_ACCIONES), mlp(N_ACCIONES)
        q_obj.load_state_dict(q.state_dict())
        opt = torch.optim.Adam(q.parameters(), lr=lr)
        buf = None
        sem_train = semillas_entrenamiento(semilla, episodios)
        pc = PuntoControl(out, "q", norm, n_eval, config)
        vistos, prox_eval, pasos = 0, eval_cada, 0
        est = cargar_estado(out)
        if est:
            q.load_state_dict(est["q"]); q_obj.load_state_dict(est["q_obj"])
            opt.load_state_dict(est["opt"]); pc.restaurar(est["pc"]); buf = est["buf"]
            vistos, prox_eval, pasos = est["vistos"], est["prox_eval"], est["pasos"]
            torch.set_rng_state(est["torch_rng"]); rng.bit_generator.state = est["rng"]
        while vistos < episodios:
            eps = max(eps_fin, 1.0 - (1.0 - eps_fin) * vistos / max(1, frac_eps * episodios))
            sd = {k: v.clone() for k, v in q.state_dict().items()}
            seeds = sem_train[vistos:vistos + min(lote, episodios - vistos)]
            trajs = list(ex.map(rollout_red, [("q", sd, norm, int(s), "eps", eps) for s in seeds]))
            vistos += len(trajs)
            nuevas = 0
            for t in trajs:
                o = norm(t["obs"])
                if buf is None:
                    buf = Buffer(buffer, o.shape[1])
                T = len(o)
                for k in range(T):
                    fin = k == T - 1
                    buf.agregar(o[k], t["actions"][k], t["rewards"][k] / ESCALA_R,
                                o[k] if fin else o[k + 1], float(fin))
                    nuevas += 1
            for _ in range(int(nuevas * pasos_por_trans)):
                O, A, R, O2, D = buf.muestra(rng, minilote)
                with torch.no_grad():
                    a2 = q(O2).argmax(1, keepdim=True)
                    y = R + gamma * (1 - D) * q_obj(O2).gather(1, a2).squeeze(1)
                loss = F.smooth_l1_loss(q(O).gather(1, A[:, None]).squeeze(1), y)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 10.0)
                opt.step()
                pasos += 1
                if pasos % obj_cada == 0:
                    q_obj.load_state_dict(q.state_dict())

            if vistos >= prox_eval or vistos >= episodios:
                pc(ex, q, vistos, float(np.mean([t["rewards"].sum() for t in trajs])))
                prox_eval += eval_cada
                guardar_estado(out, {"q": q.state_dict(), "q_obj": q_obj.state_dict(), "opt": opt.state_dict(),
                                     "pc": pc.estado(), "buf": buf, "vistos": vistos, "prox_eval": prox_eval,
                                     "pasos": pasos, "torch_rng": torch.get_rng_state(),
                                     "rng": rng.bit_generator.state})
    pc.guardar({"entrenamiento": vistos, "validacion": len(pc.curva) * n_eval})
    return pc.curva


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodios", type=int, default=P.B_TRAIN)
    p.add_argument("--lote", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--obj_cada", type=int, default=500)
    p.add_argument("--semilla", type=int, default=0)
    p.add_argument("--eval_cada", type=int, default=P.EVAL_CADA)
    p.add_argument("--n_eval", type=int, default=P.N_EVAL)
    p.add_argument("--n_norm", type=int, default=N_NORM)
    p.add_argument("--out", default="resultados/ddqn_s0")
    a = p.parse_args()
    entrenar(a.episodios, a.lote, a.lr, obj_cada=a.obj_cada, semilla=a.semilla, eval_cada=a.eval_cada,
             n_eval=a.n_eval, out=a.out, n_norm=a.n_norm)
