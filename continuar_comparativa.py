"""
Continúa la comparativa desde donde se quedó:
  - Carga M4 y M7 desde sus JSONs individuales ya existentes
  - Corre M8, M9, M10, M11, M12, M13, M14
  - Combina todo y genera gráficas
"""
import sys, os
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

import json
import modulo_comparativa_caja_negra as comp

N_TRIALS  = 20
MAX_ITER  = 30
N_CORRIDAS = 2
N_REPS    = 5
SEED      = 42


def cargar_modulo_desde_json(path: str, modulo_id: str, algoritmo: str,
                              n_corridas: int = 2) -> dict:
    """Convierte JSON individual (formato módulo) al formato comparativa."""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)

    historia = d.get("historia_costos", [])
    conv_eval = comp._convergencia_smac(historia, n_corridas)
    conv_time = comp._tiempos_smac(historia, d.get("tiempo_seg", 0.0))

    return {
        "modulo":           modulo_id,
        "algoritmo":        algoritmo,
        "costo_incumbente": d["costo_incumbente"],
        "tiempo_seg":       d["tiempo_seg"],
        "n_evaluaciones":   d.get("n_evaluaciones", len(historia) * n_corridas),
        "incumbente":       d.get("incumbente", {}),
        "conv_eval":        conv_eval,
        "conv_time":        conv_time,
    }


if __name__ == "__main__":
    resultados = {}

    # ── Cargar M4 y M7 ya existentes ────────────────────────────
    print("Cargando M4 desde resultado_comparativa_m4.json...")
    resultados["M4"] = cargar_modulo_desde_json(
        "resultado_comparativa_m4.json", "M4", "SMAC BlackBox (GP+EI)", N_CORRIDAS)
    print(f"  M4: costo={resultados['M4']['costo_incumbente']:.3f}  "
          f"tiempo={resultados['M4']['tiempo_seg']:.0f}s")

    print("Cargando M7 desde resultado_comparativa_m7.json...")
    resultados["M7"] = cargar_modulo_desde_json(
        "resultado_comparativa_m7.json", "M7", "SMAC + SK (EI)", N_CORRIDAS)
    print(f"  M7: costo={resultados['M7']['costo_incumbente']:.3f}  "
          f"tiempo={resultados['M7']['tiempo_seg']:.0f}s")

    # ── Correr módulos restantes ─────────────────────────────────
    pendientes_smac = ["M8", "M9", "M10"]
    pendientes_iter = ["M11", "M12", "M13", "M14"]

    for m in pendientes_smac:
        print(f"\n{'='*60}\nEjecutando {m}\n{'='*60}")
        try:
            res = comp._RUNNERS[m](n_trials=N_TRIALS, n_corridas=N_CORRIDAS, seed=SEED)
            resultados[m] = res
            print(f"✓ {m}: costo={res['costo_incumbente']:.3f}  tiempo={res['tiempo_seg']:.0f}s")
        except Exception as e:
            print(f"✗ Error en {m}: {e}")

    for m in pendientes_iter:
        print(f"\n{'='*60}\nEjecutando {m}\n{'='*60}")
        try:
            if m == "M13":
                res = comp._RUNNERS[m](max_iter=MAX_ITER, n_reps=N_REPS,
                                       n_workers=0, seed=SEED)
            elif m == "M14":
                res = comp._RUNNERS[m](max_iter=MAX_ITER, r=N_REPS,
                                       n_workers=0, seed=SEED)
            else:
                res = comp._RUNNERS[m](max_iter=MAX_ITER, n_workers=0, seed=SEED)
            resultados[m] = res
            print(f"✓ {m}: costo={res['costo_incumbente']:.3f}  tiempo={res['tiempo_seg']:.0f}s")
        except Exception as e:
            print(f"✗ Error en {m}: {e}")

    # ── Guardar y graficar ───────────────────────────────────────
    if resultados:
        comp._guardar_resultados(resultados, "resultados_comparativa.json")
        comp.graficar_todo(resultados, prefijo="comparativa")
        print("\n✓ Archivos generados:")
        print("  · resultados_comparativa.json")
        print("  · comparativa_convergencia.png")
        print("  · comparativa_configuraciones.png")
        print("  · comparativa_tabla_resumen.png")
