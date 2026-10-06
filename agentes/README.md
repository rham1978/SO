# Agentes sobre el DES: paso a paso

Tutorial para construir un agente que corre varios algoritmos de optimización
sobre el simulador, compara sus configuraciones con números aleatorios comunes y
recomienda una. Va en cuatro pasos y cada uno se puede correr por separado.

## Qué es un agente y qué papel cumple aquí

Un agente tiene tres partes:

1. **Un modelo** que decide qué hacer.
2. **Herramientas**: funciones tuyas que el modelo puede pedir que se ejecuten.
3. **Un loop**: tu código le muestra al modelo el estado, ejecuta lo que pide, le
   devuelve el resultado y repite hasta que termina.

El modelo nunca ejecuta nada por sí mismo: solo pide. Todo lo que ocurre de verdad
(qué se simula, con qué semillas, cuánto cuesta) lo decide el código de las
herramientas.

**El agente no reemplaza a tus algoritmos.** Un LLM probando configuraciones no
tiene garantías de convergencia ni trata el ruido heterocedástico; M4, M11 o SA
siguen siendo el motor de optimización. El agente hace el trabajo de
investigador que rodea al motor:

- decide qué algoritmos correr y con cuánto presupuesto;
- espera mientras corren;
- lee los resultados;
- confirma los incumbentes contra el baseline con comparaciones pareadas;
- escribe la conclusión con su nivel de evidencia.

## Arquitectura

```
 modelo (Claude)
    │  pide herramientas / recibe resultados
    ▼
 loop  ── paso 2: escrito a mano (API)      paso 3/4: Agent SDK (+ subagentes)
    │
    ▼
 herramientas.py   validación · presupuesto · máx. simultáneos · CRN · resúmenes
    │  lanza en segundo plano (no bloquea al modelo)
    ▼
 ejecutor.py ──► run_once (evaluaciones)  |  benchmark_riguroso.ejecutar (algoritmos)
    │
    ▼
 agentes/trabajos/<job_id>/  spec.json · resultado.json · error.txt · logs
                             presupuesto.json · bitacora.jsonl
```

| Archivo | Qué es |
|---|---|
| `herramientas.py` | Las 8 herramientas: funciones Python normales con su esquema. Aquí viven todos los límites. |
| `ejecutor.py` | Proceso en segundo plano que corre una evaluación (`run_once`) o un algoritmo (`benchmark_riguroso.ejecutar`) y deja `resultado.json`. |
| `paso1_herramientas.py` | Usa las herramientas a mano, sin modelo. |
| `paso2_loop_manual.py` | El loop de un agente escrito a mano con la API. |
| `paso3_agente_sdk.py` | El mismo agente con el Claude Agent SDK, con controles y bitácora. |
| `paso4_multiagente.py` | Orquestador con subagentes "corredor" y "evaluador". |

## Paso 0: instalar

```bash
pip install simpy numpy scipy matplotlib smac    # lo que ya usa el repo
pip install anthropic claude-agent-sdk           # pasos 2-4
export ANTHROPIC_API_KEY=sk-ant-...               # pasos 2-4 (console.anthropic.com)
```

El paso 1 no necesita API key. Los pasos 2 a 4 consumen tokens de la API, que se
cobran aparte de tu plan de Claude. Antes de una corrida larga, revisa en la
documentación del SDK cómo autenticarte y cuánto cuesta.

## Paso 1: las herramientas, sin modelo

```bash
python3 agentes/paso1_herramientas.py     # 8 semanas, 3 réplicas: ~3 min
```

Este paso hace a mano lo que después hará el agente: evalúa el baseline y una
candidata con +6 slots de primera consulta, y luego las compara.

Las herramientas son la parte más importante del agente porque fijan qué puede
hacer el modelo y cómo. Estas son las decisiones de diseño, y por qué:

- **Trabajos largos en segundo plano.** Una réplica de 52 semanas tarda minutos y
  un algoritmo, horas. Por eso no hay una herramienta `simular()` que bloquee: hay
  `lanzar_*` → `esperar` → `resultado_trabajo`. `esperar` duerme hasta que algo
  termina, así el modelo no gasta turnos preguntando "¿ya terminó?".
- **Límites duros en el código, no en el prompt.** El presupuesto de réplicas
  (`AGENTE_PRESUPUESTO_REPLICAS`, 300 por defecto), el máximo de trabajos
  simultáneos y los rangos de los 12 parámetros se validan en Python. El prompt
  puede pedir prudencia; el código la garantiza.
- **Errores como mensajes.** Si el modelo pide `num_matronas=9`, recibe
  `"num_matronas=9 fuera de [1, 4]"` y corrige. Un error no detiene el loop.
- **Números aleatorios comunes dentro de la herramienta.** Toda evaluación usa los
  offsets `900_000 + r`, apartados de los que usa el benchmark, así que dos
  evaluaciones son siempre comparables de forma pareada. El modelo no tiene que
  acordarse de esto.
- **Salidas resumidas.** El modelo recibe media, sd e IC95, no 30 números crudos.
  Cada token que lee consume contexto y dinero.
- **Desconfiar de lo que escribe el modelo.** Los `job_id` se validan para que no
  apunten fuera de `trabajos/`; lanzar un experimento idéntico a uno existente
  (misma configuración y semillas) devuelve el existente sin costo; varios
  subagentes lanzando a la vez pasan por un candado, así que el presupuesto y el
  máximo de simultáneos se respetan siempre.
- **Algoritmos: costo estimado, tiempo acotado.** Para un algoritmo, el costo en
  réplicas se estima antes de lanzar y se ajusta al terminar con lo que reporta el
  benchmark. El límite duro es `max_horas` (12 por defecto): si se supera, el
  trabajo se detiene.

En la prueba de humo (8 semanas, 3 réplicas) se ve por qué la comparación
pareada importa:

| | TTS medio (días) | sd |
|---|---|---|
| Baseline | 170,1 | 11,8 |
| +6 slots 1ra | 158,2 | 10,7 |
| **Diferencia pareada (B−A)** | **−11,9** | **1,3** |

La desviación de la diferencia pareada es casi 10 veces menor que la de cada
configuración. Con medias sueltas los IC95 se traslapan; con CRN la diferencia
es nítida. El agente siempre compara así.

## Paso 2: el loop a mano

```bash
AGENTE_SEMANAS=8 python3 agentes/paso2_loop_manual.py
```

Lee `main()` completo: son unas 30 líneas y es todo lo que un agente es.

1. `messages.create(system=..., tools=catalogo, messages=historial)`.
2. Si `stop_reason == "tool_use"`, la respuesta trae bloques `tool_use` con
   `id`, `name` e `input`.
3. Ejecutas cada uno con `herramientas.ejecutar(name, input)` y devuelves un
   bloque `tool_result` con el mismo `tool_use_id` (y `is_error` si falló).
4. Agregas todo al historial y vuelves a 1. Cuando `stop_reason` ya no es
   `tool_use`, el texto final es la respuesta.

Fíjate en que el modelo solo ve `name`, `description` e `input_schema`. La
descripción es lo que usa para decidir cuándo llamar cada herramienta, así que
redactarla bien importa tanto como el código.

`MAX_VUELTAS` es un tope: un agente sin tope puede girar indefinidamente.

## Paso 3: el mismo agente con el Agent SDK

```bash
AGENTE_SEMANAS=8 python3 agentes/paso3_agente_sdk.py                        # humo
nohup python3 agentes/paso3_agente_sdk.py "tu tarea" > agente.log 2>&1 &    # real
```

Con el SDK ya no escribes el loop; es el mismo motor de Claude Code. Tú te
concentras en tres cosas:

1. **Herramientas.** Las mismas de `herramientas.py`, envueltas con `@tool` y
   `create_sdk_mcp_server`. El modelo las ve como `mcp__des__<nombre>`.
2. **Instrucciones.** El `system_prompt` es un protocolo de investigación:
   planificar el presupuesto, lanzar y esperar, confirmar con comparaciones
   pareadas, no declarar mejoras con medias sueltas y cerrar con tabla y nivel
   de evidencia. Es lo que le enseñarías a un tesista.
3. **Controles:**
   - `tools=[]` quita Bash, Read y Edit, así que el agente solo puede usar tus
     herramientas y no toca archivos ni la terminal.
   - `allowed_tools` permite que tus herramientas se ejecuten sin pedir
     confirmación.
   - `max_turns` y `max_budget_usd` ponen tope a los turnos y al gasto.
   - Un **hook** `PreToolUse` registra cada llamada en `trabajos/bitacora.jsonl`
     y bloquea algoritmos con `n_trials` sobre `AGENTE_MAX_TRIALS`. Un hook es
     código tuyo que corre antes de cada herramienta, y el modelo no puede
     saltárselo.
   - `MCP_TOOL_TIMEOUT` se sube a 65 min porque `esperar` puede bloquear hasta 60.

La tarea por defecto corre dos algoritmos con `n_trials=40` y confirma sus
incumbentes con 10 réplicas pareadas. Revisa `bitacora.jsonl` después: es la
traza de cada decisión del agente.

## Paso 4: orquestador y subagentes

```bash
nohup python3 agentes/paso4_multiagente.py > multiagente.log 2>&1 &
```

El problema del paso 3 es que un solo agente esperando un algoritmo de 6 horas
llena su contexto con llamadas a `esperar` y lo procesa todo en un hilo. Este
paso separa roles:

- **Orquestador**: planifica, delega con la herramienta `Agent`, compara y concluye.
- **`corredor`**: un subagente por algoritmo. Lanza, espera horas si hace falta y
  devuelve solo el incumbente y su re-evaluación.
- **`evaluador`**: confirma los candidatos contra el baseline con comparaciones
  pareadas.

Cada subagente tiene su propio contexto y le devuelve al orquestador solo su
conclusión. Es la lógica de tu `ProcessPoolExecutor` (workers aislados y un
proceso que agrega), con la diferencia de que los workers razonan.

Hay un detalle de seguridad que ilustra bien la diferencia entre pedir y
garantizar. `allowed_tools` da permiso a **toda la sesión**: los subagentes lo
heredan, pero el orquestador también, así que podría lanzar simulaciones por su
cuenta aunque el prompt se lo prohíba. Por eso `hook_solo_subagentes_lanzan`
mira quién hace la llamada (`agent_id` solo aparece cuando viene de un
subagente) y bloquea los `lanzar_*` del orquestador. Los límites de
`herramientas.py` (presupuesto y 2 trabajos simultáneos) aplican a todos por
igual.

## Antes de correrlo en serio

- **Tiempos.** En la máquina donde se probó esto (2 cores), una réplica de 52
  semanas tarda unos 230 s. Mide primero en Pelluhue con
  `AGENTE_SEMANAS=52 python3 agentes/paso1_herramientas.py` y ajusta
  `n_trials`, `r_final` y `AGENTE_MAX_SIMULTANEOS` a tus 9 cores. Las
  evaluaciones del agente se reparten los cores entre trabajos simultáneos; los
  algoritmos iterativos usan todos los cores que ven (así están los runners del
  benchmark), de modo que dos algoritmos a la vez compiten por CPU.
- **El presupuesto se cuenta en réplicas** y se guarda en
  `trabajos/presupuesto.json`. Para empezar de cero, borra la carpeta `trabajos/`.
- **Semanas.** `AGENTE_SEMANAS` solo afecta a las evaluaciones del agente; los
  algoritmos siempre corren el horizonte del benchmark. Usa valores bajos solo
  para pruebas de humo y nunca mezcles horizontes en una misma comparación.
- **Supervisa.** Lee `bitacora.jsonl` y el informe final con ojo crítico. Si el
  agente toma una mala decisión estadística, la solución es corregir el protocolo
  o la herramienta, no confiar más en el modelo.
- **Estado de la prueba.** Las herramientas se probaron con el simulador real:
  las evaluaciones y comparaciones del paso 1, y una corrida completa de RS
  (`n_trials=5`, `r_final=2`, 17 réplicas de 52 semanas, 70 min en 2 cores)
  lanzada, esperada y leída con las propias herramientas; luego, tras pasar a
  `benchmark_riguroso.ejecutar` en el mismo proceso, otra corrida de RS con 4
  semanas para verificar ese camino. Los controles (job_id inválido, carrera entre
  lanzamientos simultáneos, duplicados, PID reutilizado, `max_horas`, horizontes
  distintos, reversión si el lanzamiento falla) tienen pruebas con trabajos reales.
  La prueba repetida del paso 1 da exactamente los mismos números, como
  corresponde con semillas fijas. El loop del paso 2 y la configuración de
  los pasos 3 y 4 (servidor, hooks, subagentes) se verificaron sin llamar al
  modelo, porque no había API key. La primera corrida real con modelo es tuya:
  hazla con `AGENTE_SEMANAS=8`.

## Ejercicios para afianzar

1. **Nueva herramienta.** Agrega `sensibilidad(parametro, valores, n_reps)`, que
   lance una evaluación por valor dejando el resto en baseline. Solo tocas
   `herramientas.py`; los pasos 2 a 4 la ven sin cambios.
2. **Racing.** Cambia el protocolo para que el agente evalúe varios candidatos con
   pocas réplicas, descarte los que pierden con evidencia y gaste el resto del
   presupuesto en los sobrevivientes. Ese es el tipo de decisión adaptativa en la
   que un agente aporta.
3. **Objetivo compuesto.** Pídele que corra los algoritmos con `lambda_obj` y que
   reporte el trade-off entre TTS y atenciones.
4. **Hook de aprobación.** Escribe un hook que bloquee cualquier trabajo que deje
   el presupuesto bajo el 20 % y le pida al agente resumir antes de seguir.
