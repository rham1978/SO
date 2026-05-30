"""
Corre solo M12 STRONG con parámetros reducidos para terminar en ~1-2h.
n0=5, n_r=5 → 25 eval para x0 (vs 100 con defaults)
"""
import sys, os, json, time, logging

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

import modulo_comparativa_caja_negra as comp

MAX_ITER = 3
SEED     = 42
N0       = 5   # réplicas para evaluar el centro (default=10)
N_R      = 5   # réplicas para puntos del modelo cuadrático (default=10)

print(f"Iniciando M12 STRONG — n0={N0}, n_r={N_R}, max_iter={MAX_ITER}", flush=True)
print(f"Evaluaciones estimadas para x0: {N0} → ~{N0*4:.0f} min", flush=True)
print(f"Evaluaciones estimadas por iter: n0*n_r={N0*N_R} → ~{N0*N_R*4:.0f} min", flush=True)

t0 = time.time()
try:
    import modulo12_strong as m12
    res = m12.optimizar_strong(
        max_iter  = MAX_ITER,
        seed      = SEED,
        n0        = N0,
        n_r       = N_R,
        n_workers = 0,
    )

    # Empaquetar en formato comparativa
    historia = res.historia_costos if hasattr(res, 'historia_costos') else []
    resultado = {
        "modulo":           "M12",
        "algoritmo":        "STRONG",
        "costo_incumbente": res.f_incumbente if hasattr(res, 'f_incumbente') else res.get('f_incumbente', 9999),
        "incumbente":       res.incumbente.__dict__ if hasattr(res, 'incumbente') and hasattr(res.incumbente, '__dict__') else (res.incumbente if hasattr(res, 'incumbente') else {}),
        "tiempo_seg":       time.time() - t0,
        "n_evaluaciones":   sum(e.get("n_reps", 1) for e in historia) if historia else 0,
        "historia_costos":  historia,
        "max_iter":         MAX_ITER,
        "n0":               N0,
        "n_r":              N_R,
        "seed":             SEED,
    }
    out = os.path.join(_DIR, "resultado_comparativa_m12.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)
    print(f"\n✓ M12 completado: costo={resultado['costo_incumbente']:.3f}  tiempo={resultado['tiempo_seg']:.0f}s", flush=True)
    print(f"✓ Guardado en: {out}", flush=True)

except Exception as e:
    import traceback
    print(f"\n✗ Error en M12: {e}", flush=True)
    traceback.print_exc()
