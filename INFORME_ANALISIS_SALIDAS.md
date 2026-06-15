# Análisis de las salidas actuales — Benchmark IFORS (tt_mean_time)

Generado a partir de `pipeline_out/resultados/` (90 archivos = 6 métodos × 15 seeds).
Script reproducible: `analisis_salidas.py` → carpeta `analisis_out/`.

---

## 1. ¿Qué objetivo se alcanza?

La función objetivo que **todos** los métodos minimizan es la combinada calibrada en el Paso 1
del pipeline (`pipeline_out/calibracion`):

```
f(x) = TTS_full_days_mean(x)  −  λ · total_atenciones(x)        λ = 0.082
```

- `TTS_full_days_mean` = tiempo medio de espera del paciente (días).
- `total_atenciones`   = atenciones totales producidas en el horizonte (52 semanas).
- λ = 0.082 ⇒ “una atención adicional vale 0.082 días menos de espera”.
- **Menor f(x) = mejor.** Baseline de referencia ≈ 270; TTS base ≈ 254.6 días.

Cada método reporta además una **reevaluación honesta** del incumbente con `r_final = 50`
réplicas independientes (`reeval.media` ± IC95), que es la métrica justa para comparar.

### Resultado por método (reevaluación honesta, ordenado de mejor a peor)

| Método (algoritmo)        | seeds | costo_opt μ±σ | reeval μ (±sd) | mejor seed | tiempo [h] | n_eval |
|---------------------------|:----:|:-------------:|:--------------:|:----------:|:----------:|:------:|
| **SPSA** (M13)            | 15   | 207.2 ± 22.6  | **207.1 ± 5.1**| **174.5**  | 17.2       | 110    |
| **SMAC-GP+EI** (M4)       | 15   | 208.6 ± 16.9  | 209.3 ± 5.0    | 175.0      | 19.1       | 150    |
| SMAC+SK (EI) (M7)         | 15   | 214.4 ± 15.8  | 215.2 ± 4.4    | 176.9      | 17.0       | 150    |
| SK Adaptativo (M8)        | 15   | 215.5 ± 15.1  | 216.5 ± 4.7    | 176.6      | 21.6       | 197    |
| SK-KGCP (M10)             | 15   | 217.5 ± 12.0  | 217.8 ± 4.6    | 192.0      | 17.1       | 150    |
| Random Search (RS)        | 15   | — (inf)       | — (sin datos)  | —          | 0.0        | 450    |

**Conclusiones del objetivo**

- El benchmark **mejora claramente el baseline** (~270 → ~207–218 en reevaluación honesta),
  es decir reduce el tiempo de espera neto del paciente acreditando atenciones.
- **SPSA** logra el mejor objetivo promedio (207.1) y la mejor solución individual (174.5),
  con *menos* evaluaciones (110). **SMAC-GP+EI** es el segundo y el que más baja al final
  del presupuesto (mejor cola de convergencia).
- La dispersión entre seeds es moderada (sd≈4–5 en la reevaluación); SPSA es el más variable
  (sd_seed alta) pero con el mejor extremo.
- ⚠️ **Random Search (RS) no produjo datos válidos**: `costo_opt = inf`, incumbente vacío,
  `conv_eval = []`, `tiempo ≈ 0.25 s`. Las 15 corridas RS quedaron sin ejecutar — coherente
  con que el `benchmark_riguroso.py` terminó con **código −9 (OOM / kill)** según `pipeline.log`.
  RS debe relanzarse.

---

## 2. ¿Qué variables se mueven?

Hay 12 variables de decisión. La tabla muestra el **incumbente medio** entre seeds y la barra
ordena por |Δ| respecto al baseline, normalizado al rango admisible.

| variable | baseline | rango | hacia dónde se mueve | |Δ| del rango |
|----------|:-------:|:-----:|----------------------|:-------------:|
| `num_agentes_ugd`          | 1    | [1,4]      | **↑ ~3** (al tope)        | **64 %** |
| `num_matronas`             | 1    | [1,4]      | **↑ ~2.5**                | **50 %** |
| `horas_especialista_1ra`   | 16   | [8,30]     | **↑ ~24–30**              | **44 %** |
| `horas_control_post`       | 40   | [20,70]    | **↑ ~57–70**              | **40 %** |
| `pct_bloqueo_post_control` | 0.34 | [0.05,0.5] | **↓ ~0.18–0.21**          | 33 %     |
| `pct_bloqueo_1ra`          | 0.32 | [0.05,0.5] | ↓ ~0.24                   | 20 %     |
| `pct_consultas_vacias`     | 0.30 | [0.05,0.5] | ↓ ~0.21                   | 20 %     |
| `cupos_laboratorio_ugd`    | 54   | [20,100]   | ↑ ~70                     | 20 %     |
| `cupos_ecografia_matrona`  | 25   | [10,50]    | ↑ ~27–28                  | 14 %     |
| `cupos_ecografia_ugd`      | 25   | [10,50]    | ↑ ~28                     | 13 %     |
| `pct_no_contactabilidad`   | 0.30 | [0.05,0.5] | ↓ ~0.24 (↑ en SPSA)       | 12 %     |
| `dias_publicacion`         | 5    | [1,10]     | ≈ sin cambio (~4)         | 7 %      |

**Lectura operativa (lo que recomienda la optimización):**

1. **Aumentar capacidad de recursos humanos**: agentes UGD al máximo (≈3–4) y matronas (≈2–3).
   Estas son las palancas dominantes.
2. **Subir horas de especialista 1.ª consulta** (24→30) y **horas de control post** (57→70).
3. **Reducir bloqueos y consultas vacías** (`pct_bloqueo_post_control`, `pct_bloqueo_1ra`,
   `pct_consultas_vacias` bajan ~0.2), aprovechando mejor la agenda.
4. **Días de publicación** apenas influye (variable poco sensible).
5. **SPSA difiere del resto**: lleva horas y cupos de ecografía a los topes, pero mantiene
   `pct_no_contactabilidad` más alto — solución de esquina, coherente con su naturaleza de
   gradiente.

El mapa `analisis_out/movimiento_variables.png` muestra esto como heatmap (posición 0=mín, 1=máx).

---

## 3. Gráficas generadas (`analisis_out/`)

| archivo | contenido |
|---------|-----------|
| `convergencia_evaluaciones.png` | mejor f(x) vs. nº de evaluaciones (media ± σ entre seeds) |
| `convergencia_tiempo.png`       | mejor f(x) vs. tiempo de cómputo [h] (media ± σ) |
| `tiempo_ejecucion_box.png`      | boxplot del tiempo de ejecución por método |
| `objetivo_box.png`              | boxplot del objetivo reevaluado (r=50) por método |
| `movimiento_variables.png`      | heatmap incumbente medio normalizado por variable |
| `resumen_analisis.json`         | tabla resumen + ranking + desplazamientos en JSON |

**Tiempo de ejecución:** ~17–22 h por seed. SK Adaptativo es el más caro (μ≈21.6 h, usa
n_eval≈197); SMAC+SK, SK-KGCP y SPSA son los más baratos (~17 h). El coste por seed es alto
porque cada evaluación corre el simulador de eventos discretos a 52 semanas.

**Convergencia:** SPSA desciende más rápido a presupuesto medio; SMAC-GP+EI alcanza el valor
más bajo al final del presupuesto (su banda inferior es la mejor a 150 evals).

---

## 4. Acciones solicitadas

- **Re-correr ASTRO-DF (M11):** no estaba en este lote (`pipeline.log` lo excluyó del Paso 3).
  Script nuevo `run_m11_rerun.py` (52 semanas, paralelizado). Ver estado en
  `pipeline_out/m11_rerun/`.
- **Ajustar SMAC para RF (Random Forest):** el runner M4 usaba `tipo="blackbox"`
  (Gaussian Process). Cambiado a `tipo="hpo"` → **HyperparameterOptimizationFacade
  (Random Forest + EI + Sobol)** en `modulo_comparativa_caja_negra.py`; etiqueta del benchmark
  actualizada a `SMAC-RF+EI`. Validado de extremo a extremo (la fachada RF construye y optimiza).
  Requiere `scikit-learn==1.6.1` (ver `requirements_analisis.txt`); con sklearn ≥1.7 el modelo
  RF de SMAC 2.4 falla al importar (`DTYPE` / `validate_data`).
