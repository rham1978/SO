"""
Continúa la comparativa desde donde se quedó:
  - Detecta automáticamente qué módulos ya tienen JSON guardado y los carga
  - Corre solo los módulos pendientes
  - Combina todo y genera gráficas
"""
import sys, os
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

import json
import modulo_comparativa_caja_negra as comp

N_TRIALS  = 20
MAX_ITER  = 15
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


MODULO_META = {
    "M4":  ("resultado_comparativa_m4.json",  "SMAC BlackBox (GP+EI)", "smac"),
    "M7":  ("resultado_comparativa_m7.json",  "SMAC + SK (EI)",        "smac"),
    "M8":  ("resultado_comparativa_m8.json",  "SK Adaptativo",         "smac"),
    "M9":  ("resultado_comparativa_m9.json",  "SK-REVI",               "smac"),
    "M10": ("resultado_comparativa_m10.json", "SK-KGCP",               "smac"),
    "M11": ("resultado_comparativa_m11.json", "ASTRO-DF",              "iter"),
    "M12": ("resultado_comparativa_m12.json", "STRONG",                "iter"),
    "M13": ("resultado_comparativa_m13.json", "SPSA",                  "iter"),
    "M14": ("resultado_comparativa_m14.json", "ALOE",                  "iter"),
}


if __name__ == "__main__":
    resultados = {}

    # ── Cargar módulos ya existentes ─────────────────────────────
    for m, (json_path, algoritmo, tipo) in MODULO_META.items():
        if os.path.exists(json_path):
            print(f"Cargando {m} desde {json_path}...")
            resultados[m] = cargar_modulo_desde_json(json_path, m, algoritmo, N_CORRIDAS)
            print(f"  {m}: costo={resultados[m]['costo_incumbente']:.3f}  "
                  f"tiempo={resultados[m]['tiempo_seg']:.0f}s")
        else:
            print(f"{m}: JSON no encontrado, se ejecutará.")

    # ── Correr módulos pendientes ────────────────────────────────
    pendientes_smac = [m for m in ["M8", "M9", "M10"] if m not in resultados]
    pendientes_iter = [m for m in ["M11", "M12", "M13", "M14"] if m not in resultados]

    print(f"\nPendientes SMAC: {pendientes_smac}")
    print(f"Pendientes iter: {pendientes_iter}")

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
