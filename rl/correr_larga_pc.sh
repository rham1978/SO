#!/bin/bash
# Corrida larga (2 000 000 pasos) en el PC de Pelluhue (Ubuntu, i9-10900K: 10 núcleos / 20 hilos).
# Lanza las inicializaciones 2, 3 y 4 EN PARALELO, una por proceso, cada una con 8 trabajadores
# (cada lote de entrenamiento tiene 8 episodios, así que más de 8 por proceso no acelera).
# Cada una se relanza sola si se corta y sigue desde su último punto de control.
# Uso:  bash correr_larga_pc.sh          (vuelve a la terminal; se puede cerrar la ventana)
#       bash correr_larga_pc.sh estado   (muestra el avance)
cd "$(dirname "$0")"
OUT=resultados/larga_2M
mkdir -p "$OUT"
if [ "$1" = "estado" ]; then
  for i in 2 3 4; do echo "== init $i"; tail -n 2 "$OUT/corrida_i$i.log" 2>/dev/null; done
  ls -d "$OUT"/*_s*/ 2>/dev/null; ls "$OUT"/*_s*/mejor_val100.pt 2>/dev/null | wc -l | xargs echo "agentes terminados:"
  exit 0
fi
for i in 2 3 4; do
  if pgrep -f "corrida_larga.py .*--inits $i( |$)" >/dev/null; then echo "init $i ya está corriendo"; continue; fi
  nohup bash -c "
    source .venv/bin/activate
    export RL_NUCLEOS=\${RL_NUCLEOS:-8}
    until python corrida_larga.py --base resultados/completa --out $OUT --inits $i --log corrida_i$i.log; do
      echo 'La corrida se cortó. Reintentando en 60 s...'; sleep 60
    done" > "$OUT/lanzador_i$i.txt" 2>&1 &
  echo "init $i lanzada (pid $!)"
done
echo "Avance: bash correr_larga_pc.sh estado"
