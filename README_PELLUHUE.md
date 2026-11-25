# Cómo correr la comparativa mañana (Pelluhue)

Todo queda listo en esta rama. La comparativa ahora distingue **los dos surrogados
de SMAC por separado** y suma ASTRO-DF:

- **M4 = SMAC BO** (Bayesian Optimization, Gaussian Process + EI) — el que ya tenías.
- **M4RF = SMAC con Random Forest** (HPO facade: RF + EI + Sobol) — **nuevo, aparte**.
- **M11 = ASTRO-DF** (trust-region con muestreo adaptativo).

Así puedes comparar directamente BO (GP) vs RF dentro de SMAC, más el resto de métodos.

## 1. Preparar entorno (una sola vez)

```bash
pip install -r requirements_analisis.txt
```

> ⚠️ **Crítico:** el surrogado Random Forest de SMAC 2.4 necesita
> `scikit-learn==1.6.1`. Con ≥1.7 falla (`DTYPE`) y con <1.6 también
> (`validate_data`). El `requirements_analisis.txt` ya lo fija.

## 2. Correr la comparativa completa

```bash
# en segundo plano, deja log:
nohup bash run_comparativa_pelluhue.sh > comparativa_pelluhue.log 2>&1 &
tail -f comparativa_pelluhue.log
```

Esto corre el benchmark con `M4 M4RF M7 M8 M10 M11 M13 RS`, reevalúa cada incumbente
(r=50) y al final genera análisis + gráficas automáticamente.

### Variantes útiles

```bash
# Prueba de humo rápida (2 seeds, 30 trials) para validar que todo arranca:
QUICK=1 bash run_comparativa_pelluhue.sh

# Solo los dos módulos nuevos:
MODULOS="M4 M11" bash run_comparativa_pelluhue.sh

# Ajustar tamaño / cores:
N_SEEDS=10 N_TRIALS=150 N_CORES=8 bash run_comparativa_pelluhue.sh
```

- El benchmark usa `--resume`: si se corta, vuelve a lanzarlo y retoma donde quedó.
- `MAX_H` (def. 24) descarta una seed que exceda ese tiempo y sigue.
- λ se toma de `pipeline_out/calibracion/lambdas.json` (0.082) si existe.

## 3. Salidas

- `pipeline_out/resultados_rf_astrodf/` → JSON por método/seed.
- `pipeline_out/resultados_rf_astrodf_analisis/` → gráficas + `resumen_analisis.json`.

## Correr solo ASTRO-DF (M11) por separado

```bash
M11_MAX_ITER=8 M11_WORKERS=4 python3 run_m11_rerun.py
```

## Tiempos (referencia)

Cada réplica del simulador (52 semanas) tarda ~3 min. Una corrida completa de
15 seeds × 7 métodos es de **varias horas a días** según núcleos — planificar
con tiempo y/o reducir `N_SEEDS`/`N_TRIALS`.
