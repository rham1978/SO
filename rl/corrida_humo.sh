#!/usr/bin/env bash
# Corrida pequeña: prueba el pipeline completo de punta a punta con ~1/30 del presupuesto.
# Usa semillas de humo (900000+) y las primeras de validación; NO toca las semillas de prueba.
set -euo pipefail
cd "$(dirname "$0")"
OUT=resultados/humo
mkdir -p $OUT
EP=96; EVAL=48; NEVAL=4; NNORM=16

echo "== 0) Verificación: simulador original vs acelerado (4 semillas)"
python validar_estadistico.py 4 | tee $OUT/verificacion_sim.log
mv -f validacion_sim_fast.json $OUT/ 2>/dev/null || true

echo "== 1) Referencias: mejor fija (búsqueda exhaustiva), base, aleatoria, SMAC-SK"
python politicas_referencia.py --humo --out $OUT | tee $OUT/referencias.log

echo "== 1b) SMAC-SK re-optimizado con el mismo presupuesto de cupos y semillas"
python smac_sk_presupuesto.py --humo --out $OUT 2>&1 | grep -v -i warn | tee $OUT/smac_sk.log

echo "== 2) Agentes: 3 métodos × 2 inicializaciones, $EP episodios c/u"
for s in 0 1; do
  python reinforce.py --episodios $EP --eval_cada $EVAL --n_eval $NEVAL --n_norm $NNORM --semilla $s --out $OUT/reinforce_s$s
  python ppo.py       --episodios $EP --eval_cada $EVAL --n_eval $NEVAL --n_norm $NNORM --semilla $s --out $OUT/ppo_s$s
  python ddqn.py      --episodios $EP --eval_cada $EVAL --n_eval $NEVAL --n_norm $NNORM --semilla $s --out $OUT/ddqn_s$s
done 2>&1 | tee $OUT/agentes.log

echo "== 3) Evaluación final pareada (semillas de humo) e interpretación"
python evaluar_final.py --semillas humo --refs $OUT/referencias.json \
       --agentes "$OUT/*_s*/mejor.pt" --n_interp 2 --out $OUT | tee $OUT/final.log
