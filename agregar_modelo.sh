#!/bin/bash
# ============================================================================
# Agregar uno o más modelos al benchmark y regenerar la comparación + PPT,
# consistente con el deck entregado (n=18, CRN pareado).
#
# USO:
#   bash agregar_modelo.sh M11                 # agrega ASTRO-DF
#   bash agregar_modelo.sh M4RF M11            # agrega SMAC-RF y ASTRO-DF
#   bash agregar_modelo.sh M9 M12 M14 SA       # agrega SK-REVI, STRONG, ALOE, SA
#
# En segundo plano (recomendado, tarda horas):
#   nohup bash agregar_modelo.sh M11 > agregar_M11.log 2>&1 &
#   tail -f agregar_M11.log
#
# Variables de entorno (opcionales):
#   MODE       CRN (re-simula todo, def.) | reeval (rápido: modelos del JSON n=50)
#   R_COMPARA  réplicas de la comparación (def. 18)
#   N_CORES    procesos paralelos (def. nproc-1)
#   OUT_RES    carpeta de resultados (def. pipeline_out/resultados)
#   OUT_CMP    carpeta de salida de la comparación (def. comparacion_crn18)
#
# Modelos con runner disponible (se pueden agregar sin programar):
#   M4 M4RF M7 M8 M9 M10 M11 M12 M13 M14 SA
# Modelos que requieren IMPLEMENTACIÓN previa (no tienen runner):
#   COMPASS, R-SPLINE  → hay que codificar su runner en
#                        modulo_comparativa_caja_negra.py (_RUNNERS).
# ============================================================================
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
PY="${PYTHON:-python3}"

MODELOS="$*"
if [ -z "$MODELOS" ]; then
    echo "Uso: bash agregar_modelo.sh M11 [M9 ...]"
    echo "Modelos con runner: M4 M4RF M7 M8 M9 M10 M11 M12 M13 M14 SA"
    exit 1
fi

N_CORES="${N_CORES:-$(( $(nproc) > 1 ? $(nproc) - 1 : 1 ))}"
OUT_RES="${OUT_RES:-pipeline_out/resultados}"
OUT_CMP="${OUT_CMP:-comparacion_crn18}"
R_COMPARA="${R_COMPARA:-18}"
MODE="${MODE:-CRN}"

REEVAL_FLAG=""
if [ "$MODE" = "reeval" ]; then
    REEVAL_FLAG="--usar-reeval-modulos"
fi

echo "============================================================"
echo "AGREGAR MODELO(S): $MODELOS"
echo "  modo=$MODE  r=$R_COMPARA  n_cores=$N_CORES"
echo "  resultados=$OUT_RES  salida=$OUT_CMP"
echo "============================================================"

# ── PASO 1: generar incumbentes del/los modelo(s) nuevo(s) ──────────────────
# --resume salta los módulos que YA tengan resultados (no recalcula).
echo ">>> PASO 1: optimización de [$MODELOS]  (genera incumbentes; puede tardar horas)"
$PY benchmark_riguroso.py \
    --modulos $MODELOS \
    --n_seeds 15 --n_trials 150 --r_final 50 \
    --n_cores "$N_CORES" --resume \
    --out "$OUT_RES"

# ── PASO 2: comparación con TODOS (manuales + modelos) en n=$R_COMPARA ───────
echo ">>> PASO 2: comparación (re-simula todos, mismas seeds = CRN)"
$PY comparar_manual_vs_optimo.py \
    --res "$OUT_RES" \
    --r "$R_COMPARA" --timeout 900 \
    --n_workers "$N_CORES" \
    $REEVAL_FLAG \
    --out "$OUT_CMP"

# ── PASO 3: regenerar el PPT (mismo formato del deck entregado) ──────────────
echo ">>> PASO 3: regenerar PPT (12 slides, todos los modelos, n=$R_COMPARA)"
$PY generar_reporte_ppt.py \
    --json "$OUT_CMP/comparacion_manual.json" \
    --res "$OUT_RES" \
    --out report_comparison.pptx

echo "============================================================"
echo "LISTO. Salidas:"
echo "  $OUT_CMP/comparacion_manual.json   ← datos de la comparación"
echo "  $OUT_CMP/pareto_tts_atenciones.png ← Pareto"
echo "  report_comparison.pptx             ← deck actualizado con el modelo nuevo"
echo "============================================================"
