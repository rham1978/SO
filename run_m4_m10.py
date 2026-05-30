"""
Corre módulos M4, M7, M8, M9, M10 (algoritmos Bayesianos SMAC).
Detecta cuáles ya tienen JSON y solo corre los pendientes.
Guarda JSON individual por módulo + resultados_comparativa_m4_m10.json + gráficas.

Uso:
    python run_m4_m10.py                     # corre pendientes (auto-detecta)
    python run_m4_m10.py --forzar            # re-corre todos aunque tengan JSON
    python run_m4_m10.py --modulos M8 M9     # corre solo los indicados
    python run_m4_m10.py --n_trials 50       # más evaluaciones (default: 20)
"""
import sys, os, json, time, argparse, logging

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

import modulo_comparativa_caja_negra as comp

# ── Parámetros por defecto ────────────────────────────────────────────────────
N_TRIALS   = 20
N_CORRIDAS = 2
SEED       = 42

MODULOS_SMAC = ["M4", "M7", "M8", "M9", "M10"]

META = {
    "M4":  "resultado_comparativa_m4.json",
    "M7":  "resultado_comparativa_m7.json",
    "M8":  "resultado_comparativa_m8.json",
    "M9":  "resultado_comparativa_m9.json",
    "M10": "resultado_comparativa_m10.json",
}

ALGO_LABEL = {
    "M4":  "SMAC BlackBox (GP+EI)",
    "M7":  "SMAC + SK (EI)",
    "M8":  "SK Adaptativo",
    "M9":  "SK-REVI",
    "M10": "SK-KGCP",
}


def cargar_desde_json(path: str, modulo_id: str) -> dict:
    """Carga un JSON individual y lo convierte al formato comparativa."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    historia = d.get("historia_costos", [])
    conv_eval = comp._convergencia_smac(historia, N_CORRIDAS)
    conv_time = comp._tiempos_smac(historia, d.get("tiempo_seg", 0.0))
    return {
        "modulo":           modulo_id,
        "algoritmo":        ALGO_LABEL[modulo_id],
        "costo_incumbente": d["costo_incumbente"],
        "tiempo_seg":       d["tiempo_seg"],
        "n_evaluaciones":   d.get("n_evaluaciones", len(historia) * N_CORRIDAS),
        "incumbente":       d.get("incumbente", {}),
        "conv_eval":        conv_eval,
        "conv_time":        conv_time,
    }


def guardar_json_individual(res: dict, modulo_id: str):
    """Guarda el resultado en el JSON estándar del módulo."""
    path = os.path.join(_DIR, META[modulo_id])
    historia = []
    for ev, cf in zip(res.get("conv_eval", [[], []])[0],
                      res.get("conv_eval", [[], []])[1]):
        historia.append({"n_eval": ev, "f_incumbente": cf})
    salida = {
        "modulo":           modulo_id,
        "algoritmo":        ALGO_LABEL[modulo_id],
        "costo_incumbente": res["costo_incumbente"],
        "tiempo_seg":       res["tiempo_seg"],
        "n_evaluaciones":   res["n_evaluaciones"],
        "incumbente":       res["incumbente"],
        "n_trials":         N_TRIALS,
        "n_corridas":       N_CORRIDAS,
        "seed":             SEED,
        "historia_costos":  res.get("historia_costos", historia),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)
    log.info("✓ JSON guardado: %s", path)


def main():
    parser = argparse.ArgumentParser(description="Corre M4–M10 (SMAC Bayesianos)")
    parser.add_argument("--modulos",   nargs="+", choices=MODULOS_SMAC,
                        default=None,  help="Módulos a correr (default: pendientes)")
    parser.add_argument("--forzar",   action="store_true",
                        help="Re-correr aunque ya exista JSON")
    parser.add_argument("--n_trials", type=int, default=N_TRIALS,
                        help=f"Evaluaciones del simulador (default: {N_TRIALS})")
    parser.add_argument("--seed",     type=int, default=SEED)
    args = parser.parse_args()

    n_trials   = args.n_trials
    seed       = args.seed
    candidatos = args.modulos or MODULOS_SMAC

    # ── Detectar cuáles ya tienen JSON ───────────────────────────────────────
    resultados = {}
    pendientes = []

    for m in candidatos:
        path = os.path.join(_DIR, META[m])
        if not args.forzar and os.path.exists(path):
            try:
                resultados[m] = cargar_desde_json(path, m)
                costo = resultados[m]['costo_incumbente']
                log.info("✓ %-4s  cargado desde JSON  (costo=%.2f días)", m, costo)
            except Exception as e:
                log.warning("%-4s  JSON inválido (%s) — se re-correrá", m, e)
                pendientes.append(m)
        else:
            pendientes.append(m)

    if not pendientes:
        log.info("Todos los módulos ya tienen resultados. Usa --forzar para re-correr.")
    else:
        log.info("Módulos a correr: %s", pendientes)
        log.info("Parámetros: n_trials=%d, seed=%d, n_corridas=%d", n_trials, seed, N_CORRIDAS)

    t_total = time.time()

    for m in pendientes:
        log.info("=" * 60)
        log.info("▶ Iniciando %s — %s", m, ALGO_LABEL[m])
        t0 = time.time()
        try:
            res = comp._RUNNERS[m](
                n_trials   = n_trials,
                n_corridas = N_CORRIDAS,
                seed       = seed,
            )
            res["tiempo_seg"] = time.time() - t0
            resultados[m] = res

            # Guardar JSON individual inmediatamente
            try:
                path_ind = os.path.join(_DIR, META[m])
                salida = {
                    "modulo":           m,
                    "algoritmo":        ALGO_LABEL[m],
                    "costo_incumbente": res["costo_incumbente"],
                    "tiempo_seg":       res["tiempo_seg"],
                    "n_evaluaciones":   res.get("n_evaluaciones", n_trials * N_CORRIDAS),
                    "incumbente":       res.get("incumbente", {}),
                    "n_trials":         n_trials,
                    "n_corridas":       N_CORRIDAS,
                    "seed":             seed,
                    "historia_costos":  res.get("historia_costos", []),
                }
                with open(path_ind, "w", encoding="utf-8") as f:
                    json.dump(salida, f, indent=2, ensure_ascii=False)
                log.info("✓ JSON individual guardado: %s", path_ind)
            except Exception as e:
                log.warning("No se pudo guardar JSON individual de %s: %s", m, e)

            log.info("✓ %-4s  completado: costo=%.3f días  tiempo=%.1f min",
                     m, res['costo_incumbente'], res['tiempo_seg']/60)

        except Exception as e:
            import traceback
            log.error("✗ %s falló: %s", m, e)
            traceback.print_exc()

    # ── Resumen final ─────────────────────────────────────────────────────────
    if not resultados:
        log.error("No hay resultados. Revisar errores arriba.")
        return

    log.info("=" * 60)
    log.info("RESUMEN (orden por costo):")
    orden = sorted(resultados, key=lambda m: resultados[m]['costo_incumbente'])
    for i, m in enumerate(orden, 1):
        r = resultados[m]
        log.info("  %d. %-10s  costo=%7.3f días  tiempo=%.1f h  eval=%s",
                 i, m, r['costo_incumbente'],
                 r['tiempo_seg'] / 3600,
                 r.get('n_evaluaciones', '?'))

    # ── Guardar JSON comparativo ──────────────────────────────────────────────
    out_json = os.path.join(_DIR, "resultados_comparativa_m4_m10.json")
    comp._guardar_resultados(resultados, out_json)
    log.info("✓ JSON comparativo: %s", out_json)

    # ── Graficar ─────────────────────────────────────────────────────────────
    try:
        comp.graficar_todo(resultados, prefijo="comparativa_m4_m10")
        log.info("✓ Gráficas generadas: comparativa_m4_m10_*.png")
    except Exception as e:
        log.warning("No se generaron gráficas: %s", e)

    log.info("Tiempo total: %.1f min", (time.time() - t_total) / 60)


if __name__ == "__main__":
    main()
