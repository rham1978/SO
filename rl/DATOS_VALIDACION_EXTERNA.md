# Validación externa del DES: qué datos pedir, de dónde sacarlos y cómo validar

Objetivo: contrastar las salidas del simulador (DES) del CMI con lo observado en el CRS
Cordillera en un **período distinto del usado para calibrar**. Hoy la calibración usa 52
semanas de llegadas (3,66 pacientes/día) y el corte de lista de espera de septiembre de
2026; la única validación es la de la tesis (tiempos de espera a 1ra consulta de ese mismo
corte, Mann-Whitney p = 0,056). El informe lo declara como limitación (v): "la validación
del DES frente a los datos del CRS es de la tesis y no se repitió aquí".

Script listo: `rl/validacion_externa.py` (plantillas de los CSV en `rl/datos_crs_plantilla/`).

## 1. Qué se valida

| Indicador | Qué se compara | Salida del DES que se usa |
|---|---|---|
| Ingresos a lista | interconsultas aceptadas por mes | llegadas no rechazadas por APS |
| 1ras consultas realizadas | por mes y total | `attended` (1ra atendida) |
| Controles realizados | por mes y total | `post_control_attended` |
| Ecografías y laboratorios | realizados por mes | cupos de exámenes reservados |
| Cirugías | realizadas por mes | pacientes que pasan a postoperatorio |
| Lista de 1ra consulta | al cierre de cada mes, por prioridad | pacientes abiertos en la etapa de 1ra |
| Pacientes en control, prequirúrgico y lista quirúrgica | al cierre de cada mes | ruta en que está cada paciente |
| Espera ingreso → 1ra consulta | mediana, media, p90; P(atendida ≤ 30/60/90/180 días) | mismo cálculo y misma censura en ambos lados |

L (pacientes en el sistema, la métrica principal del proyecto) es la suma de las listas
por etapa, así que se valida a través de ellas.

**Dos modos** (el script corre ambos):

- *entradas_observadas* (principal): el DES recibe las entradas del período (tasa de
  llegadas, cupos ofrecidos por semana, lista inicial por etapa y prioridad) y se mira si
  reproduce las salidas. Valida la estructura del modelo.
- *tesis*: la configuración calibrada tal cual, sin usar nada del período. Valida la
  capacidad predictiva. Si el CRS cambió la demanda o los cupos en el período, este modo
  debe fallar y el primero no.

**Criterios, fijados antes de ver los datos** (constante `CRITERIOS` del script):

- Total del período de cada flujo dentro del intervalo de predicción 95 % del DES y con
  sesgo relativo ≤ 10 %.
- ≥ 80 % de los meses de cada flujo dentro de la banda 95 %.
- ≥ 80 % de los cortes de cada lista dentro de la banda 95 % y sesgo absoluto medio ≤ 15 %.
- Cada estadístico de espera dentro del intervalo de predicción 95 %.

Se usan bandas de predicción entre réplicas y no pruebas de hipótesis sobre pacientes
individuales: las esperas de una misma réplica no son independientes (dependen del estado
de la lista), así que una Mann-Whitney paciente a paciente rechaza de más. El script la
reporta igual, como dato informativo, para continuidad con la tesis.

## 2. Período de validación

Debe ser **disjunto del de calibración**. Recomendación: las **52 semanas anteriores** a la
ventana de llegadas usada para calibrar (por ejemplo, septiembre 2024 a agosto 2025 si la
calibración usó septiembre 2025 a septiembre 2026; ajustar a las fechas reales), partiendo
un lunes. Ventajas: los datos ya existen en los sistemas del CRS y da 12 meses completos.
Alternativa prospectiva (octubre 2026 en adelante): solo hay 1 a 2 meses antes de la
entrega del 4 de diciembre, poco para validar.

Hay que confirmar: (a) las fechas exactas de las 52 semanas de calibración, y (b) que en
el período elegido no hubo cambios grandes de operación (pandemia, cierre de pabellón,
operativos extra). Si los hubo, el modo *entradas_observadas* los absorbe si llegan en
`capacidad_semanal.csv`.

**Alcance (importante):** el corte de septiembre de 2026 tiene 2 419 pacientes de
ginecología en lista activa, pero el DES parte con 388 esperando 1ra consulta y 1 070 en
etapas posteriores (1 458 en total). La extracción tiene que usar el **mismo filtro** que
definió la población del DES (solo CMI, o las especialidades y prestaciones que la tesis
incluyó). Si no, todo lo observado sale más alto y la validación falla por definición.

## 3. Datos exactos que se piden

Cuatro archivos CSV (UTF-8, separador `,` o `;`, fechas `AAAA-MM-DD`), **anonimizados**:
sin RUT, nombre, dirección ni teléfono. El identificador es un correlativo que no permite
volver a la persona. Ninguno es obligatorio por sí solo: el script valida lo que pueda con
lo que llegue y lista lo que quedó sin validar.

### 3.1 `pacientes.csv` (lo más valioso): una fila por interconsulta CMI

Incluye todas las interconsultas **abiertas en algún momento del período**: las que
ingresaron durante el período y las que ya estaban en lista al inicio.

| Campo | Tipo | Descripción |
|---|---|---|
| `id_paciente` | texto | correlativo anónimo |
| `prioridad` | alta / media / baja | prioridad clínica asignada al ingresar |
| `fecha_ingreso_lista` | fecha | ingreso de la interconsulta a la lista (fecha de entrada en SIGTE) |
| `fecha_primera_consulta` | fecha o vacío | 1ra consulta **realizada** (no solo agendada) |
| `fecha_egreso_lista` | fecha o vacío | salida de la lista de 1ra consulta, por cualquier causa |
| `causal_egreso` | texto | causal de egreso (código SIGTE o texto: atendida, no contactable, rechazo, fallecimiento, etc.) |
| `ruta_post` (opcional) | control / prequirurgico / quirurgico | ruta tras la 1ra consulta |
| `fecha_alta_cmi` (opcional) | fecha o vacío | alta del CMI (fin del proceso) |

### 3.2 `eventos.csv`: una fila por atención agendada en el período

| Campo | Tipo | Descripción |
|---|---|---|
| `fecha` | fecha | fecha de la atención |
| `tipo` | primera / control / ecografia / laboratorio / cirugia | prestación |
| `estado` | realizada / nsp / suspendida | solo cuentan las realizadas; NSP y suspendidas sirven para revisar tasas |
| `id_paciente` (opcional) | texto | el mismo correlativo de `pacientes.csv` |

Si el CRS no puede dar filas por atención, sirve una tabla mensual con los conteos de
atenciones realizadas por tipo. En ese caso se arma `eventos.csv` con una fila por
atención y fecha = día 15 del mes (el script compara por mes).

### 3.3 `cortes_lista.csv`: lista de espera al cierre de cada mes

| Campo | Tipo | Descripción |
|---|---|---|
| `fecha_corte` | fecha | último día del mes; incluir el día anterior al inicio del período |
| `etapa` | primera / control / prequirurgico / quirurgico | etapa |
| `prioridad` | alta / media / baja / total | por prioridad en `primera`; `total` en las otras |
| `n_pacientes` | entero | pacientes en esa etapa al corte |

El corte del día anterior al inicio es **entrada** del modelo (lista inicial); los demás se
validan. Si no existe, la lista de 1ra se reconstruye desde `pacientes.csv`, pero
control, prequirúrgico y lista quirúrgica quedan sin validar.

### 3.4 `capacidad_semanal.csv`: cupos ofrecidos (entrada del modelo)

| Campo | Descripción |
|---|---|
| `semana_inicio` | lunes de la semana |
| `cupos_primera` | cupos de 1ra consulta publicados |
| `cupos_control` | cupos de control publicados |
| `cupos_eco_ugd`, `cupos_eco_matrona` | cupos de ecografía por equipo |
| `cupos_lab_matrona` | cupos de laboratorio de matrona |
| `cirugias` | pabellones o cirugías programadas para CMI |

Se piden los cupos **ofrecidos**, no los usados, porque son la decisión que el modelo toma
como dato.

**Mínimo útil**, si el CRS solo puede entregar una parte: `pacientes.csv` (con fechas de
ingreso, 1ra consulta y egreso) más los cortes mensuales de las cuatro etapas. Con eso se
validan ingresos, 1ras, lista de 1ra, esperas y las listas posteriores.

## 4. Fuentes

| Fuente | Qué tiene | ¿Sirve? |
|---|---|---|
| **Sistema de agenda del CRS** (vía la jefatura del CMI o la unidad de estadística / SOME) | cupos publicados, atenciones agendadas y realizadas, NSP | Sí: `eventos.csv` y `capacidad_semanal.csv`. Es la fuente principal. |
| **SIGTE** (Sistema de Gestión de Tiempos de Espera, MINSAL), extracto del CRS o del SSM Oriente | cada interconsulta con fecha de ingreso, egreso y causal | Sí: `pacientes.csv` y la lista de 1ra. No es público a nivel de paciente; lo tiene la unidad de listas de espera del establecimiento o del servicio. |
| **REM A07** (DEIS, datos abiertos `deis.minsal.cl`, serie REM por establecimiento, mensual) | consultas nuevas y controles de especialidad por establecimiento y especialidad | **Pública**, pero agrega toda la ginecología del CRS, no solo el CMI. Sirve como control de verosimilitud de 1ras y controles, o completa si el CRS indica qué fracción es CMI. No se pudo descargar desde este entorno (el sitio está bloqueado por la red de la nube); se baja a mano desde DEIS → Datos Abiertos → REM. |
| **Informes Glosa 06** (MINSAL, trimestrales) y **listaesperasalud.cl** | lista de espera No GES por servicio de salud, región y especialidad | No: el nivel es el servicio de salud completo (SSM Oriente), no el CRS ni el CMI. Útil solo como contexto. |
| **Ley de Transparencia** (portal de transparencia, Ley 20.285) | extracto anonimizado de SIGTE para el CRS Cordillera, ginecología | Vía alternativa si la solicitud directa no avanza; el plazo legal es de 20 días hábiles (prorrogables), demasiado para el 4 de diciembre, pero sirve para la tesis. |

No se encontró ninguna fuente pública con el detalle del CMI: los datos hay que pedirlos.

## 5. Cómo correrlo cuando lleguen los datos

```bash
cd rl
# poner los 4 CSV en rl/datos_crs/
python validacion_externa.py --datos datos_crs --inicio 2024-09-02 --fin 2025-08-31 --reps 50
```

Toma unos 2 minutos con 4 núcleos (50 réplicas × 2 modos, simulador acelerado). Escribe en
`resultados/validacion_externa/`: `validacion_externa.md` (tablas listas para el informe),
`validacion_externa.json` y las figuras `fig_flujos.png` y `fig_cortes.png`.

### Prueba con datos sintéticos

`python validacion_externa.py --demo` genera los cuatro CSV con el formato de arriba a
partir de una corrida del propio DES (semilla distinta) y los valida. Con 40 réplicas:

| Caso | entradas_observadas | tesis |
|---|---|---|
| "Realidad" = el modelo (`--demo`) | 26 de 27 chequeos | 26 de 27 |
| "Realidad" con 25 % más demanda (`--demo --demo_demanda 1.25`) | 26 de 27 | **21 de 27**: fallan ingresos (−13 %), lista de 1ra (solo 9 % de los cortes en banda) y fracción atendida |

Es decir: cuando el modelo es correcto, pasa (el chequeo que falla es el esperable por azar
con 27 chequeos al 95 %); cuando la realidad difiere, el modo *tesis* lo detecta, y el modo
*entradas_observadas* lo corrige al usar la tasa de llegadas observada.

## 6. Borrador de solicitud al CRS

> **Asunto:** Solicitud de datos anonimizados del CMI para validación de modelo de simulación (uso académico)
>
> Estimado/a [nombre, cargo]:
>
> Junto con saludar, le escribo como estudiante de doctorado de la Pontificia Universidad
> Católica de Chile. En mi tesis desarrollé un modelo de simulación del proceso de
> ginecología del CMI del CRS Cordillera, calibrado con las llegadas de [fechas de
> calibración] y el corte de lista de espera de septiembre de 2026. Para validarlo
> necesito compararlo con un período distinto, y por eso le solicito, para el período
> [fecha inicio] a [fecha fin] y con el mismo alcance (CMI) de los datos anteriores:
>
> 1. Extracto de lista de espera (SIGTE) de las interconsultas de ginecología CMI abiertas
>    en el período: prioridad, fecha de ingreso, fecha de 1ra consulta realizada, fecha y
>    causal de egreso, y si es posible ruta posterior y fecha de alta.
> 2. Atenciones agendadas del CMI en el período (1ra consulta, control, ecografía,
>    laboratorio y cirugía), con fecha y estado (realizada, NSP, suspendida).
> 3. Número de pacientes en lista al cierre de cada mes, por etapa (1ra consulta, control,
>    prequirúrgico, lista quirúrgica) y, para 1ra consulta, por prioridad.
> 4. Cupos ofrecidos por semana: 1ra consulta, control, ecografía (UGD y matrona),
>    laboratorio de matrona y cirugías.
>
> Los datos pueden venir sin RUT, nombre ni datos de contacto, con un identificador
> correlativo; no necesito ningún dato que permita identificar a las pacientes. Se usarán
> solo con fines académicos, se guardarán en un equipo protegido y los resultados se
> publicarán de forma agregada, conforme a las leyes 20.584 y 19.628. Adjunto una plantilla
> con el formato, pero acepto el formato en que el sistema los exporte. Si solo es posible
> entregar una parte, lo más importante es el punto 1 y los cortes mensuales del punto 3.
>
> Si es posible, me ayudaría mucho recibirlos antes del [fecha], ya que el informe del
> curso se entrega el 4 de diciembre. Quedo atento a cualquier requisito adicional (por
> ejemplo, aprobación de un comité de ética o convenio de confidencialidad).
>
> Saludos cordiales,
> Raúl Araneda M.
