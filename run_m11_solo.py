"""
Corre M11 ASTRO-DF con parámetros reducidos.
max_iter=5, sin workers paralelos → ~1-2h estimados.
"""
import sys, os, json, time, logging

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S")

import modulo_11_astrodf as m11

MAX_ITER = 5
SEED     = 42

print(f"Iniciando M11 ASTRO-DF — max_iter={MAX_ITER}, seed={SEED}", flush=True)
t0 = time.time()

try:
    res = m11.optimizar_astro_df(
        max_iter  = MAX_ITER,
        seed      = SEED,
        n_workers = 0,
    )

    historia = res.historia_costos if hasattr(res, 'historia_costos') else []
    resultado = {
        "modulo":           "M11",
        "algoritmo":        "ASTRO-DF",
        "costo_incumbente": res.f_incumbente if hasattr(res, 'f_incumbente') else 9999,
        "incumbente":       res.incumbente.__dict__ if hasattr(res.incumbente, '__dict__') else {},
        "tiempo_seg":       time.time() - t0,
        "n_evaluaciones":   len(historia),
        "historia_costos":  historia,
        "max_iter":         MAX_ITER,
        "seed":             SEED,
    }
    out = os.path.join(_DIR, "resultado_comparativa_m11.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)
    print(f"\n✓ M11 completado: costo={resultado['costo_incumbente']:.3f}  tiempo={resultado['tiempo_seg']/3600:.2f}h", flush=True)
    print(f"✓ Guardado en: {out}", flush=True)

    # Generar gráfica
    try:
        png = m11.graficar_astro_df(res,
            guardar_png=os.path.join(_DIR, "resultado_comparativa_m11.png"),
            mostrar=False)
        print(f"✓ Gráfica: {png}", flush=True)
    except Exception as eg:
        print(f"⚠ Gráfica no generada: {eg}", flush=True)

except Exception as e:
    import traceback
    print(f"\n✗ Error en M11: {e}", flush=True)
    traceback.print_exc()
