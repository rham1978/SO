#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Re-ejecuta M11 ASTRO-DF sobre el simulador completo (52 semanas) y guarda el
resultado en formato comparable al resto del benchmark.

Parámetros configurables por variables de entorno:
    M11_MAX_ITER   (def. 8)   iteraciones del trust-region
    M11_WORKERS    (def. 4)   procesos paralelos para las réplicas
    M11_SEED       (def. 42)  semilla
    M11_OUT        (def. resultado_comparativa_m11.json)

Uso:
    M11_MAX_ITER=8 M11_WORKERS=4 python run_m11_rerun.py
"""
import os, sys, json, time, logging

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S")

import modulo_11_astrodf as m11

MAX_ITER = int(os.environ.get("M11_MAX_ITER", "8"))
WORKERS  = int(os.environ.get("M11_WORKERS", "4"))
SEED     = int(os.environ.get("M11_SEED", "42"))
OUT      = os.environ.get("M11_OUT", os.path.join(_DIR, "resultado_comparativa_m11.json"))
LAMBDA   = float(os.environ.get("M11_LAMBDA", "0.082"))

print(f"▶ M11 ASTRO-DF — max_iter={MAX_ITER}  workers={WORKERS}  seed={SEED}",
      flush=True)
t0 = time.time()

res = m11.optimizar_astro_df(
    max_iter  = MAX_ITER,
    seed      = SEED,
    n_workers = WORKERS,
    objetivo  = "tts_full_days_mean",
    guardar_json = OUT.replace(".json", "_raw.json"),
)

historia = getattr(res, "historia_costos", []) or []
f_inc    = getattr(res, "f_incumbente", getattr(res, "costo_incumbente", 9999))
inc      = res.incumbente
inc_dict = inc.__dict__ if hasattr(inc, "__dict__") else dict(inc)

resultado = {
    "modulo":           "M11",
    "algoritmo":        "ASTRO-DF",
    "costo_opt":        float(f_inc),
    "costo_incumbente": float(f_inc),
    "incumbente":       inc_dict,
    "tiempo_seg":       time.time() - t0,
    "n_evaluaciones":   len(historia),
    "n_eval_usadas":    len(historia),
    "historia_costos":  historia,
    "conv_eval":        [[i + 1, c] for i, c in enumerate(historia)],
    "max_iter":         MAX_ITER,
    "seed":             SEED,
    "lambda_obj":       LAMBDA,
}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(resultado, f, indent=2, ensure_ascii=False)

dt = time.time() - t0
print(f"\n✓ M11 completado: costo={f_inc:.3f}  tiempo={dt/3600:.2f}h  "
      f"evals={len(historia)}", flush=True)
print(f"✓ Guardado en: {OUT}", flush=True)

try:
    png = m11.graficar_astro_df(
        res, guardar_png=OUT.replace(".json", ".png"), mostrar=False)
    print(f"✓ Gráfica: {png}", flush=True)
except Exception as eg:
    print(f"⚠ Gráfica no generada: {eg}", flush=True)
