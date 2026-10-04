# CLAUDE.md — Repo SO (simulación-optimización CRS Cordillera)

Repo de la tesis doctoral de Raúl Araneda (PUC, Ing. y Ciencias con la Industria).
Un simulador DES validado de la unidad de ginecología mínimamente invasiva de
CRS Cordillera (SSMO) y un benchmark de algoritmos DFO/SO que lo optimizan como
caja negra. Los resultados se presentaron en IFORS 2026.

Quien lee esto conoce SimPy, SMAC3, ASTRO-DF, Stochastic Kriging y la teoría de
convergencia DFO. No expliques conceptos básicos; ve al punto.

## Reglas duras (no negociables)

1. **`simulador_clinica_baseline.py` está validado contra datos reales.**
   No lo modifiques salvo que Raúl lo pida explícitamente en la tarea.
   Si una tarea parece requerirlo, detente y explica el cambio propuesto antes.
   - Prohibido cambiar lógica de flujo, distribuciones, probabilidades de ruta,
     calendarios o defaults de `SimConfig` como "optimización" o "limpieza".
   - Cambios de rendimiento solo si dejan **idénticos** los KPIs para la misma
     semilla (comparar `run_once(seed_offset=s)` antes y después, varias semillas).
   - Precedente: en junio se revirtió un cambio de retry en `_book_from_pool`
     porque alteraba el comportamiento del modelo. No repetir.
   - Cualquier cambio aceptado al simulador obliga a re-correr
     `modulo2_validacion.py` (Mann-Whitney contra los datos reales) y reportar el resultado.
2. **Reproducibilidad.** `SimConfig.random_seed_base = 202`. `run_once(seed_offset, cfg)`
   siembra `np.random.RandomState` y `random` con `random_seed_base + seed_offset`.
   Comparaciones entre configuraciones usan números aleatorios comunes (mismos offsets).
   No introduzcas fuentes de aleatoriedad no sembradas.
3. **Espacio de búsqueda canónico** = `PARAM_NAMES`, `PARAM_RANGES`, `PARAM_BASELINE`
   en `modulo_comparativa_caja_negra.py` (12 parámetros). Todo módulo debe usar
   exactamente esos nombres y rangos. No agregues ni quites parámetros sin pedirlo.
4. **Mapeo parámetro → `SimConfig`.** En particular:
   `horas_control_post` → `cfg.fixed_post_control_capacity = int(valor)` y
   `cfg.use_fixed_post_control_hours = True`.
   `fixed_post_control_hours` **no existe** en `SimConfig`; asignarlo se ignora en
   silencio y deja el parámetro sin efecto.
5. **Objetivo.** KPI base `tts_full_days_mean`. Con `--lambda_obj λ` el objetivo es
   `f = tts_full_days_mean − λ · total_atenciones` (λ desde `calibrar_lambda.py`).
   Reportar siempre TTS y atenciones por separado además de `f`.
6. **Paralelismo.** Réplicas vía `ProcessPoolExecutor` con worker definido a nivel de
   módulo (pickleable) y guard `if __name__ == "__main__":` en todo script ejecutable.
7. **Anti-deadlock.** El DES puede colgarse con ciertas configuraciones. Toda llamada
   paralela a `run_once` usa `fut.result(timeout=...)` y descarta/registra la réplica
   que expira; el benchmark además tiene `--max_seed_horas` y `--resume`.
8. **Presupuesto de cómputo.** Las corridas completas tardan horas o días
   (benchmark ≈ 8 días en Pelluhue, 9 cores). Nunca lances el benchmark completo ni
   `run_pipeline.sh`. Para probar, usa `benchmark_mode=True`, pocas semanas, 1–2 semillas
   y pocas réplicas, y dilo explícitamente en el resumen.

## Mapa del repo

- `simulador_clinica_baseline.py` — DES (SimPy). `SimConfig`, `ClinicModelAdjusted`, `run_once`.
- `modulo1_num_corridas.py` — número de réplicas (de aquí sale `r_final = 30`, error relativo 5%).
- `modulo2_validacion.py` — validación contra datos reales (Mann-Whitney U).
- `modulo3_comparacion.py` — comparación estadística baseline vs escenarios.
- **Algoritmos activos** (los que importa `modulo_comparativa_caja_negra.py`):
  `modulo4_smac_v2.py` (M4 SMAC-GP+EI), `modulo7_smac_sk.py` (M7 SMAC+SK),
  `modulo8_sk_adaptativo_paralelizado.py` (M8), `modulo9_sk_revi.py` (M9),
  `modulo10_sk_kgcp.py` (M10), `modulo_11_astrodf.py` (M11 ASTRO-DF),
  `modulo12_strong.py` (M12), `modulo13_spsa.py` (M13 SPSA), `modulo14_aloe.py` (M14),
  `modulo_sa_alrefaei.py` (SA, Alrefaei & Andradóttir 1999).
- **Legacy, no usar como base:** `modulo4_smac.py` y `modulo4_smac_paralelizado.py`
  (todavía asignan `fixed_post_control_hours`, que no existe). No los "arregles" sin pedirlo;
  no los importes. `modulo7_smac_sk_paralelizado.py` tiene el mapeo correcto pero no está en
  el pipeline activo.
- **Problema conocido abierto:** `modulo3_comparacion.py` (escenario
  `fixed_post_control_hours`) tiene el mismo mapeo erróneo: ese escenario no cambia nada en
  el simulador. Pendiente de decisión de Raúl.
- `modulo_comparativa_caja_negra.py` — `_RUNNERS`, convergencia, espacio canónico.
- `benchmark_riguroso.py` — benchmark multi-semilla (CLI: `--modulos --n_seeds --n_trials
  --r_final --n_cores --lambda_obj --max_seed_horas --resume --out`).
- `calibrar_lambda.py`, `caracterizacion_heterocedasticidad.py`, `analisis_resultados.py`,
  `harness_experimentos.py` — pipeline IFORS.
- `run_pipeline.sh` — pipeline completo; asume la ruta de Pelluhue
  (`/home/raul/Documents/Codigos Sim`). No ejecutar en la nube.
- En las figuras del benchmark los módulos se renumeran (M4→"M1", M7→"M2", …) vía `MODULOS`
  en `benchmark_riguroso.py`. En código y commits usa siempre los nombres de archivo (M4, M7…).

## Entorno

Python 3 en Ubuntu. Dependencias principales: `simpy`, `numpy`, `scipy`, `matplotlib`,
`smac` (incluye `ConfigSpace`), `openpyxl`. Los outputs (`*.json`, `*.png`, `*.xlsx`, …)
están en `.gitignore`: no los commitees.

## Forma de trabajar

- Antes de editar un módulo, lee cómo lo invoca `modulo_comparativa_caja_negra.py`.
- Cambios pequeños y separados por commit, con mensaje que diga qué y por qué.
- Al terminar, resume: qué cambió, qué probaste (con qué tamaño de corrida) y qué
  quedó sin verificar. Si algo puede afectar resultados ya reportados en IFORS, dilo
  en la primera línea.
- Para auditar módulos contra estas reglas, usa el subagente `auditor-modulo`
  (uno por archivo, en paralelo).
