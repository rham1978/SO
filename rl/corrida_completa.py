"""
Corrida completa (≈ 6 h en 4 núcleos). Etapas, cada una se salta si su salida ya existe:

  1. Referencias en validación: mejor fija (28 × 100 semillas), operación actual, aleatoria.
  2. SMAC-SK con el mismo presupuesto (smac_sk_presupuesto.py).
  3. Ajuste de hiperparámetros: 4 configuraciones por agente, 1 inicialización, 1 000 episodios,
     elegidas por el mejor L de validación (20 semillas). Igual costo para los tres agentes.
  4. Entrenamiento final: 3 agentes × 5 inicializaciones con el presupuesto B (presupuesto.py).
  5. Evaluación final en PRUEBA (una sola vez): evaluar_final.py.

Uso: python corrida_completa.py [--out resultados/completa]
"""
import argparse, itertools, json, os, subprocess, sys, time

HP = {
    "reinforce": [{"lr": lr, "beta_ent": b} for lr, b in itertools.product([1e-3, 3e-4], [0.0, 0.01])],
    "ppo": [{"lr": lr, "clip": c} for lr, c in itertools.product([3e-4, 1e-4], [0.1, 0.2])],
    "ddqn": [{"lr": lr, "obj_cada": t} for lr, t in itertools.product([1e-3, 3e-4], [500, 2000])],
}
EP_HP, EVAL_HP = 1000, 250
INITS = range(5)


COPIA = None  # carpeta persistente donde se replica `out` después de cada etapa


def sincronizar(out):
    if COPIA:
        import shutil
        try:
            shutil.copytree(out, COPIA, dirs_exist_ok=True, ignore=shutil.ignore_patterns("smac_sk_salida"))
        except Exception as e:  # la copia no debe detener la corrida
            print("aviso: no se pudo copiar a", COPIA, e, flush=True)


def run(cmd, log):
    print(time.strftime("%H:%M:%S"), " ".join(cmd), flush=True)
    with open(log, "a") as f:
        subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)
    sincronizar(os.path.dirname(log))


def flags(d):
    return [x for k, v in d.items() for x in (f"--{k}", str(v))]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="resultados/completa")
    p.add_argument("--copia", default=None, help="carpeta persistente para replicar resultados")
    p.add_argument("--sin_smac", action="store_true", help="omite SMAC-SK (p. ej. en Windows sin SMAC3)")
    p.add_argument("--inits", default="0,1,2,3,4", help="inicializaciones a entrenar, p. ej. 3,4")
    a = p.parse_args()
    out, COPIA = a.out, a.copia
    os.makedirs(out, exist_ok=True)
    log = os.path.join(out, "corrida.log")
    py = sys.executable

    if not os.path.exists(f"{out}/referencias.json"):
        run([py, "politicas_referencia.py", "--out", out], log)
    if not a.sin_smac and not os.path.exists(f"{out}/smac_sk.json"):
        run([py, "smac_sk_presupuesto.py", "--out", out], log)

    elegidos = {}
    for m, grid in HP.items():
        mejores = []
        for i, hp in enumerate(grid):
            d = f"{out}/hp/{m}_{i}"
            if not os.path.exists(f"{d}/curva.json"):
                run([py, f"{m}.py", "--episodios", str(EP_HP), "--eval_cada", str(EVAL_HP),
                     "--semilla", "100", "--out", d] + flags(hp), log)
            c = json.load(open(f"{d}/curva.json"))["curva"]
            mejores.append((min(x["L_val"] for x in c), i))
        elegidos[m] = HP[m][min(mejores)[1]]
        print(m, "hiperparámetros elegidos:", elegidos[m], "(L val por config:", mejores, ")", flush=True)
    json.dump(elegidos, open(f"{out}/hp_elegidos.json", "w"), indent=1)

    for s in [int(x) for x in a.inits.split(",")]:
        for m in HP:
            d = f"{out}/{m}_s{s}"
            if not os.path.exists(f"{d}/curva.json"):
                run([py, f"{m}.py", "--semilla", str(s), "--out", d] + flags(elegidos[m]), log)

    if not os.path.exists(f"{out}/prueba/final.json"):
        run([py, "evaluar_final.py", "--semillas", "prueba", "--refs", f"{out}/referencias.json",
             "--agentes", f"{out}/*_s[0-9]/mejor.pt", "--out", f"{out}/prueba"], log)
    print("listo", flush=True)
