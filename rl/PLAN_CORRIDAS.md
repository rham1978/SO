# Plan de corridas

Costo medido: 1 episodio (1 año, 52 decisiones) ≈ 1,3 s en un núcleo. Con 4 núcleos, ≈ 11 000 episodios por hora.

## Paso 1 — Referencias (hecho, 16/11)

`python politicas_referencia.py 20` (≈ 13 min). Validación (semillas 10000–10099, 100 semillas, media ± IC95):

| Política | TTS (días) | Atenciones/año |
|---|---|---|
| Aleatoria | 252,6 ± 1,2 | 1 215 |
| Operación base (acción 24) | 247,7 ± 1,1 | 1 216 |
| **Mejor fija (acción 42)**: 1ra 20, control 50, eco 62, lab 40, cirugía 15 | **209,7 ± 1,2** | 1 371 |

Observación: la mejor fija lleva 1ra consulta, control y ecografía a su nivel máximo y laboratorio y cirugía al mínimo. Laboratorio y cirugía parecen no ser cuello de botella con estos parámetros; esto depende del supuesto `HORAS_CIRUGIA = 2`.

## Paso 2 — Agentes

Todos los métodos: misma red (MLP 2×64, tanh), mismo estado normalizado con media y desviación de 200 episodios de entrenamiento con política aleatoria, mismo presupuesto.

| Etapa | Corridas | Episodios | Tiempo (4 núcleos) |
|---|---|---|---|
| Ajuste de hiperparámetros (validación, 20 semillas, 1 inicialización) | 3 métodos × 3 configuraciones | 1 000 c/u | ≈ 1 h |
| Entrenamiento final | 3 métodos × 5 inicializaciones | 3 000 c/u | ≈ 4 h |
| Punto de control | cada 250 episodios, 20 semillas de validación | — | incluido |
| Prueba (una vez por política final) | 15 agentes + aleatoria + mejor fija + base, 30 semillas | — | ≈ 5 min |

Hiperparámetros a ajustar (solo en validación):
- **REINFORCE con línea base:** tasa de aprendizaje {1e-3, 3e-4}, coeficiente de entropía {0, 0,01}.
- **PPO:** tasa de aprendizaje {3e-4, 1e-4}, clip {0,1, 0,2}; γ = 0,99, GAE λ = 0,95.
- **Double DQN:** tasa de aprendizaje {1e-3, 3e-4}, actualización de red objetivo {500, 2 000} pasos; pérdida de Huber, buffer 100 000, ε lineal 1 → 0,05.

## Análisis adicional (requisito 6)

1. **Curvas de aprendizaje** con banda entre las 5 inicializaciones (salen gratis de los puntos de control).
2. **Sensibilidad a la congestión:** evaluar las políticas finales con el backlog inicial ×0,5 y ×1,5 (fuera de la distribución de entrenamiento) para ver si la ventaja de la política adaptativa crece con la congestión.
3. **Interpretabilidad:** qué configuración elige la política según el tamaño de la lista de espera (gráfico acción vs. estado).

## Reproducibilidad (< 1 h)

Se entregan los pesos entrenados y `evaluar_prueba.py`, que regenera la tabla final y las figuras desde los pesos (≈ 5 min). El entrenamiento completo (≈ 5 h) queda documentado pero no es necesario para reproducir.
