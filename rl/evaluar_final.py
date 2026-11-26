"""
Evaluación final: todas las políticas con las MISMAS semillas (no son llegadas comunes: el DES usa un solo generador),
una sola vez por política final.

Reporta para cada política:
  - L, TTS de la tesis, atenciones/año, pacientes en el sistema al cierre (media ± IC95).
  - Diferencia PAREADA POR SEMILLA frente a la mejor configuración fija:
        Δ_s = L_π(s) − L_fija(s),  media ± IC95 (t de Student) y prueba de Wilcoxon.
    Para los agentes, primero se promedia Δ_s entre las inicializaciones y además se
    reporta la dispersión entre inicializaciones.
  - Interpretación: árbol de decisión de profundidad 3 que imita a cada agente
    (fidelidad = % de semanas en que el árbol elige la misma acción), y acción elegida
    vs. tamaño de la lista de espera de 1ra consulta.

Uso:
  python evaluar_final.py --semillas prueba --refs resultados/validacion/referencias.json \
         --agentes "resultados/*_s*/mejor.pt" --out resultados/prueba
  python evaluar_final.py --semillas humo ...        (corrida pequeña: no toca prueba)
"""
from __future__ import annotations

import argparse
import functools
import glob
import json
import os
import re

import numpy as np
import torch
from scipy import stats

from catalogo import CATALOGO, ACCION_BASE
from entorno import SEMILLAS_TEST, SEMILLAS_HUMO, OBS_NAMES
from evaluar import evaluar, KPIS
from politicas_referencia import hacer_fija, hacer_aleatoria, SMAC_SK_CFG, evaluar_operacion_actual
from comun import cargar_politica


def _red(path, obs):
    tipo, red, norm = _CACHE[path]
    with torch.no_grad():
        return int(red(torch.as_tensor(norm(obs))).argmax())


_CACHE = {}


def hacer_red(path, seed):
    if path not in _CACHE:
        torch.set_num_threads(1)
        _CACHE[path] = cargar_politica(path)
    return functools.partial(_red, path)


def ic95(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan")
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def pareada(a, b, k="L"):
    """Δ_s = a_s − b_s sobre semillas comunes."""
    da = {r["seed"]: r[k] for r in a}
    d = np.array([da[r["seed"]] - r[k] for r in b if r["seed"] in da])
    p = float(stats.wilcoxon(d).pvalue) if len(d) >= 6 and np.any(d != 0) else float("nan")
    return d, {"media": float(d.mean()), "ic95": ic95(d), "wilcoxon_p": p, "n": int(len(d))}


def interpretar(path, semillas, out):
    """Árbol de decisión de profundidad 3 que imita a la política (estado → acción)."""
    from sklearn.tree import DecisionTreeClassifier, export_text
    from entorno import rollout
    pol = hacer_red(path, 0)
    X, y = [], []
    for s in semillas:
        tr = rollout(pol, s)
        X.append(tr["obs"]); y.append(tr["actions"])
    X, y = np.concatenate(X), np.concatenate(y)
    arbol = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)
    first = np.array([CATALOGO[a]["first"] for a in y])
    espera = X[:, :3].sum(1)
    return {"fidelidad_arbol": float(arbol.score(X, y)),
            "arbol": export_text(arbol, feature_names=list(OBS_NAMES)),
            "acciones_usadas": {int(a): int(c) for a, c in zip(*np.unique(y, return_counts=True))},
            "corr_espera1ra_vs_cupos1ra": float(np.corrcoef(espera, first)[0, 1]) if first.std() > 0 else 0.0}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--semillas", choices=["prueba", "humo"], required=True)
    p.add_argument("--refs", required=True)
    p.add_argument("--agentes", default="resultados/*_s*/mejor.pt")
    p.add_argument("--n_interp", type=int, default=5)
    p.add_argument("--smac", default=None, help="smac_sk.json (por defecto, junto a --refs)")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    semillas = list(SEMILLAS_TEST) if a.semillas == "prueba" else list(SEMILLAS_HUMO[50:60])
    os.makedirs(a.out, exist_ok=True)
    refs = json.load(open(a.refs))
    a_fija = int(refs["mejor_fija"])

    res = {
        "mejor_fija": evaluar(functools.partial(hacer_fija, a_fija), semillas),
        "base": evaluar_operacion_actual(semillas),
        "aleatoria": evaluar(hacer_aleatoria, semillas),
        "smac_sk_tesis (contexto)": evaluar(functools.partial(hacer_fija, ACCION_BASE), semillas,
                                      cfg_overrides=SMAC_SK_CFG, aplicar_acciones=False),
    }
    smac_path = a.smac or os.path.join(os.path.dirname(a.refs), "smac_sk.json")
    if a.smac and not os.path.exists(a.smac):
        raise SystemExit(f"No existe {a.smac}")
    if os.path.exists(smac_path):  # SMAC-SK re-optimizado con el mismo presupuesto y semillas
        from smac_sk_presupuesto import cfg_cupos
        m = json.load(open(smac_path))["mejor"]
        res["smac_sk"] = evaluar(functools.partial(hacer_fija, ACCION_BASE), semillas,
                                 cfg_overrides=cfg_cupos(m["cupos_1ra"], m["cupos_eco"]), aplicar_acciones=False)
    agentes = sorted(glob.glob(a.agentes))
    por_metodo = {}
    for path in agentes:
        nombre = os.path.basename(os.path.dirname(path))
        res[nombre] = evaluar(functools.partial(hacer_red, path), semillas)
        por_metodo.setdefault(re.sub(r"_s\d+$", "", nombre), []).append(nombre)

    tabla = {}
    for nombre, r in res.items():
        fila = {k: {"media": float(np.mean([x[k] for x in r])), "ic95": ic95([x[k] for x in r])} for k in KPIS}
        if nombre != "mejor_fija":
            fila["dif_pareada_L_vs_mejor_fija"] = pareada(r, res["mejor_fija"])[1]
        tabla[nombre] = fila
    # Agentes: promedio entre inicializaciones de Δ_s, y dispersión entre inicializaciones
    for metodo, nombres in por_metodo.items():
        D = np.stack([pareada(res[n], res["mejor_fija"])[0] for n in nombres])  # (inits, semillas)
        d = D.mean(0)
        tabla[f"{metodo} (prom. {len(nombres)} inic.)"] = {"dif_pareada_L_vs_mejor_fija": {
            "media": float(d.mean()), "ic95": ic95(d),
            "wilcoxon_p": float(stats.wilcoxon(d).pvalue) if len(d) >= 6 and np.any(d != 0) else float("nan"),
            "entre_inic_media": [float(x) for x in D.mean(1)]}}

    interp = {}
    for path in agentes:
        interp[os.path.basename(os.path.dirname(path))] = interpretar(path, semillas[:a.n_interp], a.out)


    import csv
    with open(os.path.join(a.out, "por_semilla.csv.tmp"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["politica", "seed"] + list(KPIS) + ["dif_L_vs_mejor_fija"])
        base_fija = {x["seed"]: x["L"] for x in res["mejor_fija"]}
        for nombre, r in res.items():
            for x in r:
                w.writerow([nombre, x["seed"]] + [x[k] for k in KPIS] + [x["L"] - base_fija[x["seed"]]])
    os.replace(os.path.join(a.out, "por_semilla.csv.tmp"), os.path.join(a.out, "por_semilla.csv"))

    from presupuesto import guardar_json
    guardar_json({"semillas": semillas, "mejor_fija": a_fija, "agentes": agentes, "tabla": tabla,
                  "por_semilla": res, "interpretacion": interp}, os.path.join(a.out, "final.json"), indent=1)

    print(f"\nSemillas: {a.semillas} ({len(semillas)}). Mejor fija = acción {a_fija} {CATALOGO[a_fija]}")
    print(f"{'política':28s} {'L':>14s} {'TTS':>8s} {'aten.':>6s} {'fin':>6s} {'ΔL vs fija (pareada)':>26s}")
    for nombre, f in tabla.items():
        d = f.get("dif_pareada_L_vs_mejor_fija")
        ds = f"{d['media']:+7.1f} ± {d['ic95']:5.1f} (p={d['wilcoxon_p']:.2f})" if d else ""
        if "L" in f:
            print(f"{nombre:28s} {f['L']['media']:7.1f} ± {f['L']['ic95']:4.1f} {f['tts_full_days_mean']['media']:8.1f} "
                  f"{f['total_atenciones']['media']:6.0f} {f['en_sistema_final']['media']:6.0f} {ds:>26s}")
        else:
            print(f"{nombre:28s} {'':>37s}{ds:>26s}")
    for n, it in interp.items():
        print(f"\n{n}: fidelidad árbol prof. 3 = {it['fidelidad_arbol']:.2f}, acciones usadas {it['acciones_usadas']}")
