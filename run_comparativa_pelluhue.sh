#!/bin/bash
# ============================================================================
# Comparativa lista para correr — Pelluhue
#
# Compara explícitamente los dos surrogados de SMAC y agrega ASTRO-DF:
#   · M4   → SMAC BlackBox / Bayesian Optimization (Gaussian Process + EI)
#   · M4RF → SMAC con Random Forest (HPO facade: RF + EI + Sobol)   [nuevo]
#   · M11  → ASTRO-DF (trust-region con muestreo adaptativo)
# más M7, M8, M10, M13 y RS, y luego genera el análisis y las gráficas.
#
# USO (desde la carpeta del repo):
#   bash run_comparativa_pelluhue.sh                    # corrida completa
#   QUICK=1 bash run_comparativa_pelluhue.sh            # prueba rápida (humo)
#   MODULOS="M4 M4RF M11" bash run_comparativa_pelluhue.sh  # solo esos módulos
#
# En segundo plano (recomendado para corrida larga):
#   nohup bash run_comparativa_pelluhue.sh > comparativa_pelluhue.log 2>&1 &
#
# Variables de entorno (con valores por defecto):
#   MODULOS   módulos a correr          (def. "M4 M4RF M7 M8 M10 M11 M13 RS")
#   N_SEEDS   macro-réplicas por módulo (def. 15 ; QUICK→2)
#   N_TRIALS  presupuesto de evals      (def. 150 ; QUICK→30)
#   R_FINAL   réplicas reevaluación     (def. 50 ; QUICK→10)
#   N_CORES   procesos paralelos        (def. nproc-1)
#   LAMBDA    λ del objetivo combinado  (def. lee lambdas.json o 0.082)
#   MAX_H     timeout por seed [horas]  (def. 24)
#   OUT       carpeta de salida         (def. pipeline_out/resultados_rf_astrodf)
# ============================================================================
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

PY="${PYTHON:-python3}"

# ── Parámetros ──────────────────────────────────────────────────────────────
if [[ "${QUICK:-0}" == "1" ]]; then
    N_SEEDS="${N_SEEDS:-2}"; N_TRIALS="${N_TRIALS:-30}"; R_FINAL="${R_FINAL:-10}"
else
    N_SEEDS="${N_SEEDS:-15}"; N_TRIALS="${N_TRIALS:-150}"; R_FINAL="${R_FINAL:-50}"
fi
MODULOS="${MODULOS:-M4 M4RF M7 M8 M10 M11 M13 RS}"
N_CORES="${N_CORES:-$(( $(nproc) > 1 ? $(nproc) - 1 : 1 ))}"
MAX_H="${MAX_H:-24}"
OUT="${OUT:-pipeline_out/resultados_rf_astrodf}"

# Objetivo: por DEFECTO TTS puro (tiempo medio en sistema, días) — igual que el
# lote de resultados existente, para que la comparación sea homogénea.
# El objetivo compuesto (f = TTS − λ·atenciones) queda DESACTIVADO salvo que
# exportes explícitamente LAMBDA (p.ej. LAMBDA=0.082). Ojo: si lo activas, los
# resultados quedan en otra escala y NO son comparables con el lote TTS actual.
LAMBDA_ARG=""
OBJ_DESC="TTS puro (tiempo medio en sistema, días)"
if [[ -n "${LAMBDA:-}" ]]; then
    LAMBDA_ARG="--lambda_obj $LAMBDA"
    OBJ_DESC="COMPUESTO f = TTS − ${LAMBDA}·atenciones  (¡escala distinta!)"
fi

echo "============================================================"
echo "COMPARATIVA — Pelluhue — $(date)"
echo "  Módulos : $MODULOS"
echo "  M4 = SMAC BO (GP+EI) | M4RF = SMAC Random Forest | M11 = ASTRO-DF"
echo "  Objetivo: $OBJ_DESC"
echo "  n_seeds=$N_SEEDS  n_trials=$N_TRIALS  r_final=$R_FINAL"
echo "  n_cores=$N_CORES  max_seed_horas=$MAX_H"
echo "  salida  : $OUT"
echo "============================================================"

# ── 0. Verificación de dependencias críticas ────────────────────────────────
echo ">>> Verificando dependencias (smac + scikit-learn 1.6.x para el RF)…"
$PY - <<'PYCHECK'
import importlib, sys
faltan = []
for m in ("numpy","pandas","matplotlib","scipy","simpy","smac","ConfigSpace","sklearn"):
    try: importlib.import_module(m)
    except Exception as e: faltan.append((m, str(e)))
try:
    import sklearn; v = sklearn.__version__
    from sklearn.tree._tree import DTYPE          # falla en sklearn >= 1.7
    from sklearn.utils.validation import validate_data  # falla en sklearn < 1.6
    print(f"   scikit-learn {v} compatible con el RF de SMAC ✓")
except Exception as e:
    print(f"   ⚠ scikit-learn incompatible con el RF de SMAC: {e}")
    print("     → pip install 'scikit-learn==1.6.1'")
    faltan.append(("sklearn-rf", "version"))
if faltan:
    print("   ⚠ Falta(n):", ", ".join(m for m,_ in faltan))
    print("     → pip install -r requirements_analisis.txt")
    sys.exit(3)
print("   Todas las dependencias OK ✓")
PYCHECK

# ── 1. Benchmark (incluye M4-RF y M11-ASTRO-DF) ─────────────────────────────
echo ">>> Ejecutando benchmark…  (puede tardar horas; usa --resume para reanudar)"
$PY benchmark_riguroso.py \
    --modulos $MODULOS \
    --n_seeds "$N_SEEDS" \
    --n_trials "$N_TRIALS" \
    --r_final "$R_FINAL" \
    --n_cores "$N_CORES" \
    $LAMBDA_ARG \
    --max_seed_horas "$MAX_H" \
    --resume \
    --out "$OUT"

# ── 2. Análisis + gráficas ──────────────────────────────────────────────────
echo ">>> Generando análisis y gráficas…"
$PY analisis_salidas.py --res "$OUT" --out "${OUT}_analisis"

echo "============================================================"
echo "LISTO — $(date)"
echo "  Resultados : $OUT"
echo "  Análisis   : ${OUT}_analisis  (gráficas + resumen_analisis.json)"
echo "============================================================"
