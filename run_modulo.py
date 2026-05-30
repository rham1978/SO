"""Corre un único módulo iterativo y guarda su JSON."""
import sys, os
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

import modulo_comparativa_caja_negra as comp

SEED = 42
MAX_ITER = 3
N_REPS = 5

modulo = sys.argv[1]  # e.g. "M12"

print(f"Iniciando {modulo}...")
try:
    if modulo == "M13":
        res = comp._RUNNERS[modulo](max_iter=MAX_ITER, n_reps=N_REPS, n_workers=0, seed=SEED)
    elif modulo == "M14":
        res = comp._RUNNERS[modulo](max_iter=MAX_ITER, r=N_REPS, n_workers=0, seed=SEED)
    else:
        res = comp._RUNNERS[modulo](max_iter=MAX_ITER, n_workers=0, seed=SEED)
    print(f"✓ {modulo}: costo={res['costo_incumbente']:.3f}  tiempo={res['tiempo_seg']:.0f}s")
except Exception as e:
    print(f"✗ {modulo}: {e}")
    import traceback; traceback.print_exc()
