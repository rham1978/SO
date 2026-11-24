# Asignación adaptativa de cupos con aprendizaje reforzado (IEE3031)

Proyecto final de IEE3031 Aprendizaje de Máquina. Usa el simulador de eventos
discretos de la tesis (`../simulador_clinica_baseline.py`) como entorno.

**Pregunta:** ¿cuánto reduce el tiempo promedio en el sistema una política que
ajusta los cupos cada semana con aprendizaje reforzado, frente a la mejor
configuración fija y a una política aleatoria, con el mismo presupuesto de cupos?

## Archivos

| Archivo | Qué hace |
|---|---|
| `sim_fast.py` | Parche que acelera ~14× el simulador sin cambiar su lógica (elimina el sondeo cada 10 min en `_book_from_pool`). |
| `verificar_sim_fast.py`, `validar_estadistico.py` | Comparan simulador original y parcheado (KPIs por semilla). |
| `catalogo.py` | Catálogo de 42 acciones (6 × 7): cómo repartir los 56 cupos semanales del especialista entre 1ra consulta y control, y los 75 cupos de exámenes entre ecografía y laboratorio de matrona. La cirugía queda fija (caja negra del CRS, ~2,5 meses de demora). |
| `entorno.py` | Entorno semanal: estado, acción, recompensa, particiones por semilla y `rollout(policy, seed)`. |
| `evaluar.py` | Evaluación paralela de una política en un conjunto de semillas. |
| `politicas_referencia.py` | Política aleatoria, búsqueda exhaustiva de la mejor configuración fija (con el presupuesto común) y regla SMAC-SK de la tesis (`modulo7_smac_sk.py`) como referencia de contexto. |
| `presupuesto.py` | Presupuesto de simulación igual para todos: B = 3 000 años simulados por método e inicialización. |
| `comun.py` | Red, normalización, muestreo de episodios y punto de control compartidos por los tres agentes. |
| `reinforce.py`, `ppo.py`, `ddqn.py` | Los tres agentes (gradiente de política, actor-crítico, valor). |
| `evaluar_final.py` | Evaluación final con mismas semillas: diferencia pareada por semilla vs la mejor fija (IC95, Wilcoxon) e interpretación de la política (árbol de decisión de profundidad 3). |
| `corrida_humo.sh` | Corrida pequeña de punta a punta (~1/30 del presupuesto) en semillas de humo, sin tocar prueba. |

## Diseño

- **Episodio:** 52 semanas. Decisión el viernes 00:00 previo a cada semana (antes de publicar agenda y generar cupos).
- **Condición inicial:** backlog calibrado de la tesis en t = 0: 388 esperando 1ra consulta (248 alta, 138 media, 2 baja) y 1 070 en etapas post (700 control, 200 prequirúrgico, 170 lista quirúrgica); antigüedades y llegadas fijadas por la semilla.
- **Fin del horizonte:** la semana 52 es terminal (sin valor de arranque). Se reporta `en_sistema_final` para detectar políticas que posterguen trabajo.
- **Estado (17):** pacientes esperando 1ra consulta por prioridad, pacientes por etapa post (agendado 1ra, control, prequirúrgico, quirúrgico, otros), antigüedad media de la espera, pacientes esperando cupo de examen, controles bloqueados la semana anterior, semana del año (seno/coseno), cupos de la semana anterior (4 palancas).
- **Recompensa:** −(pacientes en el sistema × días) / 1000 en la semana, muestreado al cierre de cada día.
- **Métrica principal:** L = pacientes promedio en el sistema = pacientes·día / 364 (ley de Little).
- **Particiones (semillas del simulador):** entrenamiento 0–9999 (las últimas 200 solo para la normalización), validación 10000–10099, prueba 500000–500029 (mismo bloque con que `benchmark_riguroso.py` re-evaluó los incumbentes de SMAC y demás métodos), humo 900000+ (solo pruebas del pipeline).
- **Supuestos fijos:** `publish_lead_workdays = 1`, capacidades fijas (no aleatorias) para 1ra consulta y control, cirugía = 20/semana.

## Reproducir

```bash
pip install -r requirements.txt
python validar_estadistico.py 12        # simulador original vs rápido (~5 min, 4 núcleos)
./corrida_humo.sh                       # corrida pequeña de punta a punta (~25 min, 4 núcleos)
python politicas_referencia.py          # referencias en validación (presupuesto completo)
```
