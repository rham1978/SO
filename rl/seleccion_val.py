"""
Selección final del punto de control con TODAS las semillas de validación.

Durante el entrenamiento cada punto de control se evalúa con 20 semillas de validación
(ruido ±~10 pacientes), y elegir el mínimo de 41 puntos así favorece al que tuvo suerte.
Aquí se toman los TOP_K mejores según esas 20 semillas y se re-evalúan en las 100 semillas
de validación (10 000–10 099), las mismas con que se eligió la mejor configuración fija.
El elegido se guarda como mejor_val100.pt. Costo: TOP_K × 100 años simulados por agente e init.

Uso: python seleccion_val.py resultados/larga_2M/ppo_s0 [--top_k 5]
"""
import argparse, functools, glob, json, os, shutil

import numpy as np

from entorno import SEMILLAS_VAL
from evaluar import evaluar
from evaluar_final import hacer_red
from presupuesto import guardar_json

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("carpeta")
    p.add_argument("--top_k", type=int, default=5)
    a = p.parse_args()
    curva = json.load(open(os.path.join(a.carpeta, "curva.json")))["curva"]
    top = sorted(curva, key=lambda c: c["L_val"])[:a.top_k]
    semillas = list(SEMILLAS_VAL)
    filas = []
    for c in top:
        path = os.path.join(a.carpeta, "puntos", f"pc_{int(c['episodios']):06d}.pt")
        r = evaluar(functools.partial(hacer_red, path), semillas)
        L = [x["L"] for x in r]
        filas.append({"episodios": c["episodios"], "L_val20": c["L_val"], "L_val100": float(np.mean(L)),
                      "ic95": float(1.96 * np.std(L, ddof=1) / np.sqrt(len(L))), "archivo": path})
        print(f"[{c['episodios']:6d}] L val20 {c['L_val']:.1f}  L val100 {filas[-1]['L_val100']:.1f}", flush=True)
    mejor = min(filas, key=lambda f: f["L_val100"])
    shutil.copy(mejor["archivo"], os.path.join(a.carpeta, "mejor_val100.pt"))
    guardar_json({"candidatos": filas, "elegido": mejor, "presupuesto": a.top_k * len(semillas)},
                 os.path.join(a.carpeta, "seleccion.json"), indent=1)
    print("elegido:", mejor["episodios"], f"L val100 {mejor['L_val100']:.1f}")
