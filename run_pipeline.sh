#!/bin/bash
# =========================================================
# Pipeline completo IFORS 2026
# Uso: nohup bash run_pipeline.sh > IFORS/pipeline.log 2>&1 &
# =========================================================
set -e

BASE="$(cd "$(dirname "$0")" && pwd)"
OUT_BASE="$BASE/IFORS"

cd "$BASE"

echo "======================================================="
echo "PIPELINE IFORS — $(date)"
echo "======================================================="

# ── Paso 0: actualizar código ────────────────────────────
echo ""
echo ">>> Paso 0: git pull"
git pull origin claude/wonderful-hopper-Nsrsu

# ── Paso 1: caracterización heterocedasticidad (~6h) ─────
echo ""
echo ">>> Paso 1: Caracterización heterocedasticidad — $(date)"
python3 caracterizacion_heterocedasticidad.py \
    --n_cores 9 \
    --out "$OUT_BASE/heteroced/"
echo "    Paso 1 completado — $(date)"

# ── Paso 2: calibrar lambda (~3h) ────────────────────────
echo ""
echo ">>> Paso 2: Calibrar lambda — $(date)"
python3 calibrar_lambda.py \
    --n_cores 9 \
    --out "$OUT_BASE/lambda/"
echo "    Paso 2 completado — $(date)"

# Leer lambda recomendado del JSON
LAMBDA=$(python3 -c "
import json
d = json.load(open('$OUT_BASE/lambda/lambdas.json'))
print(f\"{d['lambda_med']:.4f}\")
")
echo "    Lambda calibrado: $LAMBDA"

# ── Paso 3: benchmark (~8 días) ──────────────────────────
echo ""
echo ">>> Paso 3: Benchmark con lambda=$LAMBDA — $(date)"
python3 benchmark_riguroso.py \
    --modulos M4 M7 M8 M10 M11 M13 RS SA \
    --n_seeds 10 \
    --n_trials 150 \
    --r_final 30 \
    --n_cores 9 \
    --lambda_obj "$LAMBDA" \
    --max_seed_horas 6 \
    --out "$OUT_BASE/resultados_v2/"
echo "    Paso 3 completado — $(date)"

# ── Paso 4: análisis de resultados ───────────────────────
echo ""
echo ">>> Paso 4: Análisis — $(date)"
python3 analisis_resultados.py \
    --consolidado "$OUT_BASE/resultados_v2/consolidado_riguroso.json" \
    --baseline 270 \
    --out "$OUT_BASE/figuras_v2/"
echo "    Paso 4 completado — $(date)"

echo ""
echo "======================================================="
echo "PIPELINE COMPLETO — $(date)"
echo "======================================================="
