"""
Sensibilidad a la congestión: backlog inicial ×0,5 y ×1,5 (1ra consulta y post), mismas semillas
de prueba, sin reentrenar. Diferencia pareada de L frente a la mejor fija en cada escenario.
Uso: python sensibilidad.py --dir resultados/completa   → <dir>/sensibilidad/sensibilidad.json
"""
from __future__ import annotations
import argparse, functools, glob, json, os, re
import numpy as np
from catalogo import ACCION_BASE, OPERACION_ACTUAL
from entorno import SEMILLAS_TEST, config_base
from evaluar import evaluar, KPIS
from evaluar_final import hacer_red, pareada, ic95
from politicas_referencia import hacer_fija
from presupuesto import guardar_json
from smac_sk_presupuesto import cfg_cupos
from scipy import stats

CAMPOS = ("backlog_low_n", "backlog_mid_n", "backlog_high_n",
          "post_init_preq_n", "post_init_control_n", "post_init_surgery_waitlist_n")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--factores", default="0.5,1.5")
    a = p.parse_args()
    semillas = list(SEMILLAS_TEST)
    refs = json.load(open(os.path.join(a.dir, "referencias.json")))
    smac = json.load(open(os.path.join(a.dir, "smac_sk.json")))["mejor"]
    agentes = sorted(glob.glob(os.path.join(a.dir, "*_s*", "mejor.pt")))
    base = config_base()
    out = {}
    for f in [float(x) for x in a.factores.split(",")]:
        ov = {c: int(round(getattr(base, c) * f)) for c in CAMPOS}
        res = {"mejor_fija": evaluar(functools.partial(hacer_fija, int(refs["mejor_fija"])), semillas, cfg_overrides=ov),
               "base": evaluar(functools.partial(hacer_fija, ACCION_BASE), semillas,
                               cfg_overrides={**OPERACION_ACTUAL, **ov}, aplicar_acciones=False),
               "smac_sk": evaluar(functools.partial(hacer_fija, ACCION_BASE), semillas,
                                  cfg_overrides={**cfg_cupos(smac["cupos_1ra"], smac["cupos_eco"]), **ov}, aplicar_acciones=False)}
        por_metodo = {}
        for path in agentes:
            n = os.path.basename(os.path.dirname(path))
            res[n] = evaluar(functools.partial(hacer_red, path), semillas, cfg_overrides=ov)
            por_metodo.setdefault(re.sub(r"_s\d+$", "", n), []).append(n)
        tabla = {}
        for n, r in res.items():
            fila = {k: {"media": float(np.mean([x[k] for x in r])), "ic95": ic95([x[k] for x in r])} for k in KPIS}
            if n != "mejor_fija":
                fila["dif_pareada_L_vs_mejor_fija"] = pareada(r, res["mejor_fija"])[1]
            tabla[n] = fila
        for m, ns in por_metodo.items():
            D = np.stack([pareada(res[n], res["mejor_fija"])[0] for n in ns]); d = D.mean(0)
            tabla[f"{m} (prom. {len(ns)} inic.)"] = {"dif_pareada_L_vs_mejor_fija": {
                "media": float(d.mean()), "ic95": ic95(d), "wilcoxon_p": float(stats.wilcoxon(d).pvalue),
                "entre_inic_media": [float(x) for x in D.mean(1)]}}
        out[f"backlog x{f}"] = {"overrides": ov, "tabla": tabla}
        print(f"== backlog ×{f}")
        for n, fl in tabla.items():
            d = fl.get("dif_pareada_L_vs_mejor_fija", {})
            Ls = fl.get("L", {}).get("media", float("nan"))
            print(f"{n:28s} L={Ls:8.1f}  Δ={d.get('media', 0):+7.1f} ± {d.get('ic95', 0):5.1f}  p={d.get('wilcoxon_p', float('nan')):.3f}")
    os.makedirs(os.path.join(a.dir, "sensibilidad"), exist_ok=True)
    guardar_json(out, os.path.join(a.dir, "sensibilidad", "sensibilidad.json"), indent=1)
