#!/bin/bash
# Corrida larga 2 000 000 pasos: inicializaciones 2, 3 y 4 en este PC (macOS/Linux).
cd "$(dirname "$0")"; source .venv/bin/activate
until python corrida_larga.py --base resultados/completa --out resultados/larga_2M --inits 2,3,4; do
  echo "La corrida se cortó. Reintentando en 60 s..."; sleep 60
done
