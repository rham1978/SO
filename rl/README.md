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
| `catalogo.py` | Catálogo de 42 acciones: cómo repartir los 56 cupos semanales del especialista entre 1ra consulta y control, y los 104 cupos de exámenes entre ecografía y laboratorio. La cirugía queda fija (caja negra del CRS, ~2,5 meses de demora). |
| `entorno.py` | Entorno semanal: estado, acción, recompensa, particiones por semilla y `rollout(policy, seed)`. |
| `evaluar.py` | Evaluación paralela de una política en un conjunto de semillas. |
| `politicas_referencia.py` | Política aleatoria y búsqueda exhaustiva de la mejor configuración fija en validación. |

## Diseño

- **Episodio:** 52 semanas. Decisión el viernes 00:00 previo a cada semana (antes de publicar agenda y generar cupos).
- **Estado (16):** pacientes esperando 1ra consulta por prioridad, pacientes por etapa post (agendado 1ra, control, prequirúrgico, quirúrgico, otros), antigüedad media de la espera, pacientes esperando cupo de examen, controles bloqueados la semana anterior, semana del año (seno/coseno), cupos de la semana anterior (4 palancas).
- **Recompensa:** −(pacientes en el sistema × días) / 1000 en la semana, muestreado al cierre de cada día.
- **Particiones (semillas del simulador):** entrenamiento 0–9999, validación 10000–10099, prueba 500000–500029 (mismo bloque con que `benchmark_riguroso.py` re-evaluó los incumbentes de SMAC y demás métodos).
- **Supuestos fijos:** `publish_lead_workdays = 1`, capacidades fijas (no aleatorias) para 1ra consulta y control, cirugía = 20/semana.

## Reproducir

```bash
pip install -r requirements.txt
python validar_estadistico.py 12        # simulador original vs rápido (~5 min, 4 núcleos)
python politicas_referencia.py 20       # referencias en validación
```
