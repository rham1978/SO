"""
Compara dos corridas completas (p. ej. nube vs PC) a partir de sus resultados.

Uso: python comparar_corridas.py <carpeta_A> <carpeta_B>
     (cada carpeta es un resultados/completa con referencias.json y, si existe, prueba/por_semilla.csv)
"""
import csv, json, os, sys
import numpy as np


def L_medio(res):
    return float(np.mean([x["L"] for x in res]))


A, B = sys.argv[1], sys.argv[2]
ra, rb = (json.load(open(os.path.join(d, "referencias.json"))) for d in (A, B))
print("Referencias en validación (L medio):")
print(f"  mejor fija   A: acción {ra['mejor_fija']}   B: acción {rb['mejor_fija']}")
for k in ("base", "aleatoria"):
    print(f"  {k:12s} A {L_medio(ra[k]):8.1f}   B {L_medio(rb[k]):8.1f}")
for d, n in ((A, "A"), (B, "B")):
    p = os.path.join(d, "hp_elegidos.json")
    if os.path.exists(p):
        print(f"  hiperparámetros {n}: {json.load(open(p))}")

pa, pb = (os.path.join(d, "prueba", "por_semilla.csv") for d in (A, B))
if os.path.exists(pa) and os.path.exists(pb):
    def leer(p):
        t = {}
        for r in csv.DictReader(open(p)):
            t.setdefault(r["politica"], {})[int(r["seed"])] = float(r["L"])
        return t
    ta, tb = leer(pa), leer(pb)
    print("\nPrueba (L medio por política; diferencia B − A pareada por semilla):")
    for pol in sorted(set(ta) & set(tb)):
        s = sorted(set(ta[pol]) & set(tb[pol]))
        d = np.array([tb[pol][x] - ta[pol][x] for x in s])
        print(f"  {pol:28s} A {np.mean([ta[pol][x] for x in s]):8.1f}  B {np.mean([tb[pol][x] for x in s]):8.1f}  "
              f"B−A {d.mean():+6.1f} ± {1.96 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0:.1f}")
