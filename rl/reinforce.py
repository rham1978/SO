"""
REINFORCE con línea base conectado al simulador DES.

Cómo se conecta con el DES
--------------------------
`entorno.rollout(policy_fn, seed)` corre UN año del simulador. Cada viernes
00:00 el simulador se detiene, arma el estado (16 variables) y llama a
`policy_fn(obs)`. La política devuelve un índice del catálogo de 42 repartos de
cupos, `aplicar_accion` lo escribe en la configuración del simulador y el DES
sigue corriendo hasta el viernes siguiente. Al final, `rollout` devuelve las 52
tuplas (estado, acción, recompensa) y los KPIs del año.

REINFORCE usa esas trayectorias completas:
  1. Muestrea N episodios en paralelo con la política actual (π_θ).
  2. Calcula el retorno descontado G_t de cada semana.
  3. Línea base b_t = media de G_t entre los N episodios del lote en la misma semana
     (reduce varianza sin sesgar el gradiente).
  4. Pérdida = −Σ_t log π_θ(a_t|s_t) · (G_t − b_t) / σ  − β · entropía.

Uso:
    python reinforce.py --episodios 3000 --semilla 0 --lr 1e-3 --out resultados/reinforce_s0
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.nn as nn

from catalogo import N_ACCIONES
from entorno import OBS_DIM, SEMILLAS_TRAIN, SEMILLAS_VAL, rollout

torch.set_num_threads(1)


# ── Normalización del estado (ajustada SOLO con episodios de entrenamiento) ──
class Normalizador:
    def __init__(self, media, desv):
        self.media = np.asarray(media, np.float32)
        self.desv = np.maximum(np.asarray(desv, np.float32), 1e-6)

    def __call__(self, obs):
        return (obs - self.media) / self.desv

    @staticmethod
    def ajustar(n_episodios=200, n_workers=None):
        """Media y desviación del estado bajo política aleatoria, semillas de entrenamiento."""
        semillas = list(SEMILLAS_TRAIN[-n_episodios:])  # bloque reservado al final del rango de entrenamiento
        with ProcessPoolExecutor(n_workers or os.cpu_count()) as ex:
            trajs = list(ex.map(_rollout_aleatorio, semillas))
        X = np.concatenate([t["obs"] for t in trajs])
        return Normalizador(X.mean(0), X.std(0))


def _rollout_aleatorio(seed):
    rng = np.random.default_rng(seed)
    return rollout(lambda o: int(rng.integers(N_ACCIONES)), seed)


# ── Política: MLP 2×64 con salida softmax sobre las 42 acciones ─────────────
class Politica(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, n_acc=N_ACCIONES, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, h), nn.Tanh(),
                                 nn.Linear(h, h), nn.Tanh(),
                                 nn.Linear(h, n_acc))

    def forward(self, x):
        return torch.distributions.Categorical(logits=self.net(x))


# ── Rollout en un proceso trabajador (la política viaja como state_dict) ────
def _rollout_politica(args):
    state_dict, norm, seed, greedy = args
    torch.set_num_threads(1)
    pol = Politica()
    pol.load_state_dict(state_dict)
    pol.eval()
    rng = torch.Generator().manual_seed(seed)

    def policy_fn(obs):
        with torch.no_grad():
            dist = pol(torch.as_tensor(norm(obs)))
            if greedy:
                return int(dist.probs.argmax())
            return int(torch.multinomial(dist.probs, 1, generator=rng))

    tr = rollout(policy_fn, seed)
    return {k: tr[k] for k in ("obs", "actions", "rewards")} | {
        "tts": tr["kpis"]["tts_full_days_mean"], "atenciones": tr["kpis"]["total_atenciones"]}


def retornos(r, gamma):
    G = np.zeros_like(r)
    acc = 0.0
    for t in range(len(r) - 1, -1, -1):
        acc = r[t] + gamma * acc
        G[t] = acc
    return G


def entrenar(episodios=3000, lote=8, lr=1e-3, gamma=0.99, beta_ent=0.01,
             semilla=0, eval_cada=250, n_eval=20, out="resultados/reinforce", n_workers=None):
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(semilla)
    np.random.seed(semilla)
    n_workers = n_workers or os.cpu_count()

    norm = Normalizador.ajustar(n_workers=n_workers)
    pol = Politica()
    opt = torch.optim.Adam(pol.parameters(), lr=lr)

    # Semillas de entrenamiento: cada inicialización recorre un orden distinto del rango
    rng = np.random.default_rng(semilla)
    sem_train = rng.permutation(np.arange(SEMILLAS_TRAIN.start, SEMILLAS_TRAIN.stop - 200))
    curva, mejor = [], (np.inf, None)
    t0, vistos = time.time(), 0

    with ProcessPoolExecutor(n_workers) as ex:
        while vistos < episodios:
            # 1) muestrear un lote de años con la política actual
            sd = {k: v.clone() for k, v in pol.state_dict().items()}
            seeds = sem_train[vistos:vistos + lote]
            trajs = list(ex.map(_rollout_politica, [(sd, norm, int(s), False) for s in seeds]))
            vistos += len(trajs)

            # 2) retornos y 3) línea base por semana (media del lote)
            G = np.stack([retornos(t["rewards"], gamma) for t in trajs])     # (lote, 52)
            adv = G - G.mean(0, keepdims=True)
            adv = adv / (adv.std() + 1e-8)

            # 4) gradiente de política
            obs = torch.as_tensor(np.stack([norm(t["obs"]) for t in trajs]))  # (lote, 52, obs)
            acc = torch.as_tensor(np.stack([t["actions"] for t in trajs]))
            dist = pol(obs)
            logp = dist.log_prob(acc)
            loss = -(logp * torch.as_tensor(adv)).mean() - beta_ent * dist.entropy().mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
            opt.step()

            # 5) punto de control en validación (política greedy, mismas semillas siempre)
            if vistos % eval_cada < lote or vistos >= episodios:
                sd = {k: v.clone() for k, v in pol.state_dict().items()}
                ev = list(ex.map(_rollout_politica, [(sd, norm, s, True) for s in SEMILLAS_VAL[:n_eval]]))
                tts = float(np.mean([e["tts"] for e in ev]))
                curva.append({"episodios": vistos, "tts_val": tts,
                              "retorno_train": float(np.mean([t["rewards"].sum() for t in trajs])),
                              "seg": round(time.time() - t0)})
                print(f"[{vistos:5d}] TTS val {tts:.1f}  retorno train {curva[-1]['retorno_train']:.1f}", flush=True)
                if tts < mejor[0]:
                    mejor = (tts, sd)
                    torch.save({"politica": sd, "norm_media": norm.media, "norm_desv": norm.desv},
                               os.path.join(out, "mejor.pt"))

    json.dump({"curva": curva, "config": dict(episodios=episodios, lote=lote, lr=lr, gamma=gamma,
                                              beta_ent=beta_ent, semilla=semilla)},
              open(os.path.join(out, "curva.json"), "w"), indent=1)
    return curva


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodios", type=int, default=3000)
    p.add_argument("--lote", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--beta_ent", type=float, default=0.01)
    p.add_argument("--semilla", type=int, default=0)
    p.add_argument("--eval_cada", type=int, default=250)
    p.add_argument("--n_eval", type=int, default=20)
    p.add_argument("--out", default="resultados/reinforce_s0")
    a = p.parse_args()
    entrenar(a.episodios, a.lote, a.lr, a.gamma, a.beta_ent, a.semilla, a.eval_cada, a.n_eval, a.out)
