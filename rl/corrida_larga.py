"""
Corrida larga: los tres agentes con 2 000 000 de pasos de entrenamiento (decisiones semanales)
por agente e inicialización, es decir 38 462 años simulados de entrenamiento cada uno.

Qué cambia respecto de corrida_completa.py (B = 3 000 años):
  - Entrenamiento: PASOS / 52 episodios (2 000 000 / 52 → 38 462), con semillas 0–9 799 y
    luego el bloque extra 100 000–199 999 (entorno.SEMILLAS_TRAIN_EXTRA), que no se cruza con
    validación (10 000–10 099) ni prueba (500 000–500 029).
  - Selección: N_CHECK puntos de control con N_EVAL semillas de validación (por defecto 40 × 20);
    después, los TOP_K = 5 mejores se re-evalúan en las 100 semillas de validación y se elige
    el mejor (seleccion_val.py → mejor_val100.pt), igual que la mejor fija (28 × 100 semillas).
  - Presupuesto por agente e inicialización: B_largo = episodios + (N_CHECK + 1) × N_EVAL + TOP_K × 100.
  - Prueba: 100 semillas (500 000–500 099) en vez de 30, para todas las políticas, una sola vez.
  - Hiperparámetros, mejor fija, operación actual, aleatoria y SMAC-SK se toman de la corrida
    completa (--base), sin volver a simularlos. El presupuesto B es un TOPE: la mejor fija
    (2 800 años) y SMAC-SK (2 960) usan menos que los agentes, lo que favorece a los agentes;
    si aun así ninguno supera a la mejor fija, la conclusión es conservadora.
  - Cada agente guarda su estado en cada punto de control (estado.pt): si la corrida se corta,
    se vuelve a lanzar el mismo comando y sigue desde el último punto de control.

Uso:
  python corrida_larga.py --base resultados/completa --out resultados/larga_2M
  python corrida_larga.py --base ... --out ... --inits 0,1      # repartir inicializaciones entre equipos
  python corrida_larga.py --base ... --out ... --solo_evaluar   # evaluación final cuando están todos
"""
import argparse, glob, json, math, os, shutil, subprocess, sys, time

AGENTES = ("reinforce", "ppo", "ddqn")


def run(cmd, log):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), " ".join(cmd), flush=True)
    with open(log, "a") as f:
        subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)


def flags(d):
    return [x for k, v in d.items() for x in (f"--{k}", str(v))]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="carpeta de la corrida completa (referencias, SMAC-SK, hp)")
    p.add_argument("--out", default="resultados/larga_2M")
    p.add_argument("--pasos", type=int, default=2_000_000, help="pasos semanales de entrenamiento por agente e init.")
    p.add_argument("--n_check", type=int, default=40)
    p.add_argument("--n_eval", type=int, default=20)
    p.add_argument("--inits", default="0,1,2,3,4")
    p.add_argument("--agentes", default=",".join(AGENTES))
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--n_prueba", type=int, default=100)
    p.add_argument("--solo_evaluar", action="store_true")
    a = p.parse_args()

    out = a.out
    os.makedirs(out, exist_ok=True)
    log = os.path.join(out, "corrida.log")
    py = sys.executable
    episodios = math.ceil(a.pasos / 52)
    eval_cada = episodios // a.n_check
    for f in ("referencias.json", "smac_sk.json", "hp_elegidos.json"):
        if not os.path.exists(os.path.join(out, f)):
            shutil.copy(os.path.join(a.base, f), os.path.join(out, f))
    elegidos = json.load(open(os.path.join(out, "hp_elegidos.json")))
    plan = {"pasos": a.pasos, "episodios_entrenamiento": episodios, "eval_cada": eval_cada,
            "n_eval": a.n_eval, "presupuesto_por_agente_init": episodios + (a.n_check + 1) * a.n_eval + a.top_k * 100,
            "top_k_val100": a.top_k, "semillas_prueba": a.n_prueba,
            "base": os.path.abspath(a.base), "hp": elegidos}
    json.dump(plan, open(os.path.join(out, "plan.json"), "w"), indent=1)
    print("plan:", plan, flush=True)

    inits = [int(x) for x in a.inits.split(",")]
    agentes = a.agentes.split(",")
    if not a.solo_evaluar:
        for s in inits:
            for m in agentes:
                d = f"{out}/{m}_s{s}"
                if not os.path.exists(f"{d}/curva.json"):
                    run([py, f"{m}.py", "--semilla", str(s), "--episodios", str(episodios),
                         "--eval_cada", str(eval_cada), "--n_eval", str(a.n_eval), "--out", d]
                        + flags(elegidos[m]), log)
                if not os.path.exists(f"{d}/seleccion.json"):
                    run([py, "seleccion_val.py", d, "--top_k", str(a.top_k)], log)

    esperados = {f"{out}/{m}_s{s}/mejor_val100.pt" for m in AGENTES for s in range(5)}
    listos = {x for x in esperados if os.path.exists(x)}
    if listos != esperados:
        print(f"Entrenados {len(listos)}/15. Falta: {sorted(esperados - listos)}", flush=True)
        print("Cuando estén todos (copia aquí las carpetas del otro equipo), corre con --solo_evaluar.")
        sys.exit(0)
    agentes_pt = sorted(esperados)
    fj = f"{out}/prueba/final.json"
    if not (os.path.exists(fj) and json.load(open(fj)).get("agentes") == agentes_pt):
        run([py, "evaluar_final.py", "--semillas", "prueba", "--refs", f"{out}/referencias.json",
             "--agentes", f"{out}/*_s*/mejor_val100.pt", "--n_prueba", str(a.n_prueba),
             "--out", f"{out}/prueba"], log)
    print("listo", flush=True)
