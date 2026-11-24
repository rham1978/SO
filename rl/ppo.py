"""
PPO (actor-crítico con recorte) sobre el mismo entorno, red y presupuesto que REINFORCE.

  - Actor: MLP 2×64 → logits de las 42 acciones. Crítico: MLP 2×64 → V(s).
  - Ventaja con GAE(γ, λ). Horizonte finito: la semana 52 es terminal (V = 0 después),
    porque la métrica se mide en el año; el estado incluye la semana (seno/coseno),
    así que V puede distinguir cuánto falta para el cierre.
  - `epocas` pasadas de minilotes por cada lote de episodios, con recorte `clip`.

Uso:
    python ppo.py --semilla 0 --lr 3e-4 --clip 0.2 --out resultados/ppo_s0
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
from comun import ESCALA_R, N_NORM, normalizador_compartido, PuntoControl, mlp, rollout_red, semillas_entrenamiento


def gae(r, v, gamma, lam):
    """r, v: (T,). Terminal al final del episodio (v_{T} = 0)."""
    T = len(r)
    adv = np.zeros(T, np.float32)
    acc = 0.0
    for t in range(T - 1, -1, -1):
        v_sig = v[t + 1] if t + 1 < T else 0.0
        delta = r[t] + gamma * v_sig - v[t]
        acc = delta + gamma * lam * acc
        adv[t] = acc
    return adv, adv + v


def entrenar(episodios=P.B_TRAIN, lote=8, lr=3e-4, clip=0.2, gamma=0.99, lam=0.95, epocas=4,
             minilote=104, beta_ent=0.01, coef_v=0.5, semilla=0, eval_cada=P.EVAL_CADA,
             n_eval=P.N_EVAL, out="resultados/ppo", n_workers=None, n_norm=N_NORM):
    torch.manual_seed(semilla)
    np.random.seed(semilla)
    config = dict(metodo="ppo", episodios=episodios, lote=lote, lr=lr, clip=clip, gamma=gamma, lam=lam,
                  epocas=epocas, minilote=minilote, beta_ent=beta_ent, coef_v=coef_v, semilla=semilla,
                  eval_cada=eval_cada, n_eval=n_eval)
    rng = np.random.default_rng(semilla)
    with ProcessPoolExecutor(n_workers or os.cpu_count()) as ex:
        norm = normalizador_compartido(n_norm, ex)
        actor, critico = mlp(N_ACCIONES), mlp(1)
        opt = torch.optim.Adam(list(actor.parameters()) + list(critico.parameters()), lr=lr)
        sem_train = semillas_entrenamiento(semilla)
        pc = PuntoControl(out, "pi", norm, n_eval, config)
        vistos, prox_eval = 0, eval_cada
        while vistos < episodios:
            sd = {k: v.clone() for k, v in actor.state_dict().items()}
            seeds = sem_train[vistos:vistos + min(lote, episodios - vistos)]
            trajs = list(ex.map(rollout_red, [("pi", sd, norm, int(s), "muestrear", 0.0) for s in seeds]))
            vistos += len(trajs)

            O = torch.as_tensor(np.concatenate([norm(t["obs"]) for t in trajs]))
            A = torch.as_tensor(np.concatenate([t["actions"] for t in trajs]))
            with torch.no_grad():
                logp_old = torch.distributions.Categorical(logits=actor(O)).log_prob(A)
                V = critico(O).squeeze(-1).numpy()
            advs, rets, i = [], [], 0
            for t in trajs:
                n = len(t["rewards"])
                a_, r_ = gae(t["rewards"] / ESCALA_R, V[i:i + n], gamma, lam)
                advs.append(a_); rets.append(r_); i += n
            ADV = torch.as_tensor(np.concatenate(advs))
            ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
            RET = torch.as_tensor(np.concatenate(rets))

            for _ in range(epocas):
                perm = torch.as_tensor(rng.permutation(len(A)))
                for j in range(0, len(A), minilote):
                    idx = perm[j:j + minilote]
                    dist = torch.distributions.Categorical(logits=actor(O[idx]))
                    ratio = torch.exp(dist.log_prob(A[idx]) - logp_old[idx])
                    l_pi = -torch.min(ratio * ADV[idx], torch.clamp(ratio, 1 - clip, 1 + clip) * ADV[idx]).mean()
                    l_v = ((critico(O[idx]).squeeze(-1) - RET[idx]) ** 2).mean()
                    loss = l_pi + coef_v * l_v - beta_ent * dist.entropy().mean()
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critico.parameters()), 0.5)
                    opt.step()

            if vistos >= prox_eval or vistos >= episodios:
                pc(ex, actor, vistos, float(np.mean([t["rewards"].sum() for t in trajs])))
                prox_eval += eval_cada
    pc.guardar({"entrenamiento": vistos, "validacion": len(pc.curva) * n_eval})
    return pc.curva


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodios", type=int, default=P.B_TRAIN)
    p.add_argument("--lote", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--semilla", type=int, default=0)
    p.add_argument("--eval_cada", type=int, default=P.EVAL_CADA)
    p.add_argument("--n_eval", type=int, default=P.N_EVAL)
    p.add_argument("--n_norm", type=int, default=N_NORM)
    p.add_argument("--out", default="resultados/ppo_s0")
    a = p.parse_args()
    entrenar(a.episodios, a.lote, a.lr, a.clip, semilla=a.semilla, eval_cada=a.eval_cada,
             n_eval=a.n_eval, out=a.out, n_norm=a.n_norm)
