"""
REINFORCE con línea base conectado al simulador DES.

`entorno.rollout(policy_fn, seed)` corre UN año del simulador. Cada viernes 00:00
el simulador se detiene, arma el estado (17 variables) y llama a `policy_fn(obs)`.
La política devuelve un índice del catálogo de 28 repartos de cupos y el DES sigue
hasta el viernes siguiente. Al final, `rollout` devuelve las 52 tuplas
(estado, acción, recompensa) y los KPIs del año.

  1. Muestrea `lote` episodios en paralelo con la política actual π_θ.
  2. Retorno descontado G_t de cada semana (horizonte finito: no hay valor terminal).
  3. Línea base b_t = media de G_t entre los episodios del lote en la misma semana.
  4. Pérdida = −Σ_t log π_θ(a_t|s_t)·(G_t − b_t)/σ − β·entropía.

Uso:
    python reinforce.py --semilla 0 --lr 1e-3 --out resultados/reinforce_s0
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.nn as nn

from catalogo import N_ACCIONES
import presupuesto as P
from comun import ESCALA_R, N_NORM, normalizador_compartido, PuntoControl, mlp, rollout_red, semillas_entrenamiento, guardar_estado, cargar_estado


def retornos(r, gamma):
    G = np.zeros_like(r)
    acc = 0.0
    for t in range(len(r) - 1, -1, -1):
        acc = r[t] + gamma * acc
        G[t] = acc
    return G


def entrenar(episodios=P.B_TRAIN, lote=8, lr=1e-3, gamma=0.99, beta_ent=0.01, semilla=0,
             eval_cada=P.EVAL_CADA, n_eval=P.N_EVAL, out="resultados/reinforce", n_workers=None,
             n_norm=N_NORM):
    torch.manual_seed(semilla)
    np.random.seed(semilla)
    config = dict(metodo="reinforce", episodios=episodios, lote=lote, lr=lr, gamma=gamma,
                  beta_ent=beta_ent, semilla=semilla, eval_cada=eval_cada, n_eval=n_eval)
    with ProcessPoolExecutor(n_workers or P.nucleos()) as ex:
        norm = normalizador_compartido(n_norm, ex)
        pol = mlp(N_ACCIONES)
        opt = torch.optim.Adam(pol.parameters(), lr=lr)
        sem_train = semillas_entrenamiento(semilla, episodios)
        pc = PuntoControl(out, "pi", norm, n_eval, config)
        vistos, prox_eval = 0, eval_cada
        est = cargar_estado(out)
        if est:
            pol.load_state_dict(est["pol"]); opt.load_state_dict(est["opt"]); pc.restaurar(est["pc"])
            vistos, prox_eval = est["vistos"], est["prox_eval"]
            torch.set_rng_state(est["torch_rng"]); np.random.set_state(est["np_rng"])
        while vistos < episodios:
            sd = {k: v.clone() for k, v in pol.state_dict().items()}
            seeds = sem_train[vistos:vistos + min(lote, episodios - vistos)]
            trajs = list(ex.map(rollout_red, [("pi", sd, norm, int(s), "muestrear", 0.0) for s in seeds]))
            vistos += len(trajs)

            G = np.stack([retornos(t["rewards"] / ESCALA_R, gamma) for t in trajs])  # (lote, 52)
            adv = G - G.mean(0, keepdims=True)
            adv = adv / (adv.std() + 1e-8)

            obs = torch.as_tensor(np.stack([norm(t["obs"]) for t in trajs]))
            acc = torch.as_tensor(np.stack([t["actions"] for t in trajs]))
            dist = torch.distributions.Categorical(logits=pol(obs))
            loss = -(dist.log_prob(acc) * torch.as_tensor(adv)).mean() - beta_ent * dist.entropy().mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
            opt.step()

            if vistos >= prox_eval or vistos >= episodios:
                pc(ex, pol, vistos, float(np.mean([t["rewards"].sum() for t in trajs])))
                prox_eval += eval_cada
                guardar_estado(out, {"pol": pol.state_dict(), "opt": opt.state_dict(), "pc": pc.estado(),
                                     "vistos": vistos, "prox_eval": prox_eval,
                                     "torch_rng": torch.get_rng_state(), "np_rng": np.random.get_state()})
    pc.guardar({"entrenamiento": vistos, "validacion": len(pc.curva) * n_eval})
    return pc.curva


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodios", type=int, default=P.B_TRAIN)
    p.add_argument("--lote", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--beta_ent", type=float, default=0.01)
    p.add_argument("--semilla", type=int, default=0)
    p.add_argument("--eval_cada", type=int, default=P.EVAL_CADA)
    p.add_argument("--n_eval", type=int, default=P.N_EVAL)
    p.add_argument("--n_norm", type=int, default=N_NORM)
    p.add_argument("--out", default="resultados/reinforce_s0")
    a = p.parse_args()
    entrenar(a.episodios, a.lote, a.lr, a.gamma, a.beta_ent, a.semilla, a.eval_cada, a.n_eval, a.out,
             n_norm=a.n_norm)
