# Relato de la presentación — IFORS 2026

Guión narrativo según el feedback del profesor (audio 23-jun). Idea central:
**contar un caso policial** — *"fue el mayordomo con el candelabro en la cocina"*.
El **asesino / protagonista es SMAC-SK**, tu caballito de batalla. La charla
construye hacia esa revelación y los demás algoritmos existen **para probar que
el triunfo de SMAC-SK no es un fluke**, no porque "tiré los tallarines a la
muralla y dejé los que pegaron".

> Cada diapo trae: **[Intención]** (qué busca el profesor) · **[Qué decir]**
> (relato en español para internalizar) · **[EN]** (frase clave para decir en sala).

---

## La columna vertebral (en 5 frases)

1. La gestión de salud importa; Chile tiene listas de espera enormes; mi caso es
   ginecología en el CRS Cordillera — fui, levanté los datos y encontré **este** problema.
2. El problema tiene 5 propiedades duras a la vez: DES estocástico,
   heterocedasticidad, mixto-entero, evaluaciones caras, sin gradiente.
3. Revisé la literatura: **nadie** cubre las 5 a la vez → ese es el **gap** (la tabla).
4. Dado ese gap, **mi opción es SMAC-SK**: combina una estructura existente y
   fácil de implementar (SMAC) con un surrogate que trabaja mejor bajo ruido
   heterocedástico/no suave (Stochastic Kriging). **Lo digo desde el principio.**
5. Para *demostrar* que SMAC-SK es la respuesta, lo comparo contra: SMAC **sin** SK
   (¿aporta el SK o fue suerte?), DFO/ASTRO-DF (algoritmo fundacional con tasa de
   convergencia) y búsqueda discreta (estándar de industria). SMAC-SK gana → es lo
   que el hospital debería usar.

---

## Cambios clave vs. la versión actual

- **Revelar al protagonista temprano** (en *Objective* / *Proposed Framework*), no
  dejar que la audiencia adivine cuál es tu método. Hoy no aparece "mi
  recomendación es SMAC-SK" en ninguna parte.
- **El bloque "Algorithm Optimizer"** del framework debe nombrarse **SMAC-SK
  (core contribution)**.
- **SMAC-SK en verde** en los resultados (ya hecho en `fig_pareto.png`): estrella
  verde + halo. Así la audiencia "sigue al protagonista".
- **Resultados = el caso resuelto**, no una tabla de pruebas. Encadena: "miren cómo
  SK agrega valor (SMAC con vs sin SK), miren que los estándares no llegan".

---

## Guión por diapositiva

### 1. Título
**[Qué decir]** Preséntate, agradece, y planta el gancho en una frase: vengo a
mostrarles cómo, en un problema de salud real y endemoniadamente difícil, un
método —SMAC-SK— logra atender más pacientes esperando menos.
**[EN]** *"Today I'll show how one method — SMAC-SK — lets a real clinic serve more patients with less waiting, in a problem that breaks the usual guarantees."*

### 2. Outline
**[Qué decir]** En 10 segundos: motivación → problema → qué hizo el mundo → mi
objetivo → mi framework → resultados → discusión. "Es un caso policial: al final
sabrán quién fue el asesino y cómo lo atrapé."

### 3. Motivation
**[Intención]** Por qué importa, a nivel mundo y Chile.
**[Qué decir]** Las esperas prolongadas matan (mortalidad +3.4%) y restan
productividad. Esto no es abstracto: es un problema de gestión con enorme espacio
para investigación de operaciones.
**[EN]** *"Prolonged waits aren't just inconvenient — they raise mortality and cut productivity."*

### 4–5. Case Study (CRSCO / métricas)
**[Intención]** Aterrizar al caso y mostrar que **fuiste tú** quien levantó los datos.
**[Qué decir]** El CRS Cordillera es un centro de referencia; 30 mil consultas/año;
ginecología es el 30% de la lista. Yo fui, me metí en los datos, y encontré que
en Cirugía Mínimamente Invasiva hay 318 casos y esperas de hasta 1.164 días. Esa
inmersión en los datos es la que destapa el problema.
**[EN]** *"I went into the clinic's own data — this is where the problem revealed itself."*

### 6. Problem Definition
**[Intención]** Nombrar el problema con el diagrama de flujo.
**[Qué decir]** Falta un marco analítico que represente y apoye, de forma
integrada, las decisiones de capacidad bajo incertidumbre. Mi pregunta:
**¿cómo optimizo este proceso para que menos pacientes esperen?**
**[EN]** *"My research question: how do I optimize this process so fewer patients wait?"*

### 7. Challenges  ← *bisagra del relato*
**[Intención]** Las 5 propiedades que vuelven único al problema. El profesor:
"estos challenges los encontré al levantar los datos".
**[Qué decir]** Al modelar el sistema aparecen cinco exigencias **simultáneas**:
demanda estocástica, espacio mixto-entero, **ruido heterocedástico** (un sistema
saturado es mucho más ruidoso que uno holgado), evaluaciones caras (cada corrida
del simulador cuesta horas) y **sin gradiente**. Recuerden estas cinco: son las
pistas del caso.
**[EN]** *"Keep these five in mind — they are the clues. No off-the-shelf method handles all five."*

### 8. Literature Review
**[Intención]** Qué ha hecho el mundo.
**[Qué decir]** DES para representar flujos; Simulation Optimization para
objetivos de caja negra; DFO cuando no hay gradiente; metaheurísticas (SA, GA,
OptQuest) populares pero **sin garantías formales** bajo ruido heterocedástico.
Cada familia resuelve *parte* del problema.
**[EN]** *"Each family solves part of the problem — none solves all of it."*

### 9. The Methodological Gap (tabla)  ← *el "hoyo"*
**[Intención]** Mostrar que **nadie** cubre las 5 columnas. Aquí está el gap.
**[Qué decir]** Esta tabla es la escena del crimen: cada familia falla en al menos
una de las cinco propiedades. La última fila —el framework propuesto— es la única
que las marca todas. **Ese hueco es lo que mi trabajo ataca.** No estoy aplicando
el paper de alguien; encontré un espacio donde no hay nada hecho.
**[EN]** *"This empty space — no method ticks all five boxes — is exactly the gap my work targets."*

### 10. Objective  ← *revelar al protagonista*
**[Intención]** El profesor lo pidió textual: "partiría del principio: mi
caballito de batalla es SMAC-SK".
**[Qué decir]** Diseñar y evaluar un framework prescriptivo para planificación de
capacidad. Y adelanto la conclusión: **dado ese gap, mi opción es SMAC-SK.** SMAC
me da una estructura probada y fácil de implementar; la variante con Stochastic
Kriging es la que captura la heterocedasticidad que rompe a los demás. Todo lo que
sigue es cómo llegué ahí y cómo lo comprobé.
**[EN]** *"Given that gap, my answer is SMAC-SK — and the rest of this talk is how I got there and how I proved it."*

### 11. Proposed Framework
**[Intención]** El bloque central ES SMAC-SK.
**[Qué decir]** DES (evaluador de caja negra, devuelve f(x)±σ(x)) → **optimizador:
SMAC-SK** (mi contribución) → salida prescriptiva aplicada al CRS Cordillera, con
re-evaluación honesta del incumbente. Benchmark con CRN para comparación pareada.
**[EN]** *"The core block is SMAC-SK; everything is evaluated under common random numbers for a fair, paired comparison."*
**[Edición sugerida]** En el `.tex`, cambiar la etiqueta del bloque central de
`Algorithm Optimizer` / *Core contribution* a **`SMAC-SK`** / *Core contribution*.

### 12–13. Macro process / Scheduling
**[Qué decir]** Breve: así fluye el paciente y así se programan las horas de
especialista. Da realismo y muestra de dónde salen las 12 variables de decisión
(cupos, lead time, bloqueos, agentes). No te detengas; es contexto.

### 14. Results: Optimized vs. Manual  ← *ATENCIÓN ESPECIAL*
**[Intención]** Aquí se ve al protagonista. SMAC-SK en **verde** (ya en la figura).
**[Qué decir]** Cada punto es una configuración. En rojo, lo manual (lo que se
hace hoy y las propuestas de gestión); en azul, los algoritmos; **en verde, mi
recomendación, SMAC-SK**. Dos lecturas: (1) **todos** los algoritmos
Pareto-dominan a lo manual —menos espera *y* más pacientes, sin trade-off—; (2)
dentro del clúster ganador, los mejores métodos están **estadísticamente
empatados** (intervalos solapados), así que la elección no es por un par de días:
recomiendo SMAC-SK porque es **el único que captura las cinco propiedades** del
problema. SMAC-SK lleva 263 → **177 días (−33%)** y de 1.148 → **~2.180 pacientes
(+90%)**.
**[EN]** *"All algorithms Pareto-dominate the manual options; among the top, methods are statistically tied, so I recommend SMAC-SK because it's the only one that natively handles all five challenges."*
**[⚠ dato]** Tu deck dice "263 → **164** días". Tus datos re-evaluados (n=50) dan
**174–177** para el clúster top (SMAC-SK = 177, SPSA = 174). Revisa de dónde sale
164; si no lo confirmas, **usa los números de SMAC-SK (177, −33%)** como titular
—coherente con que SMAC-SK es el protagonista— en lugar del "mejor entre métodos".

### 15. Parameter Comparison (heatmap)
**[Qué decir]** *Qué hace* SMAC-SK distinto a lo manual: publica de inmediato
(lead = 1 día, no 7), bloqueo mínimo (5%), más agentes y horas post-control al
máximo — **sin agregar matronas**. Son palancas operativas, no "más recursos".
**[EN]** *"The win comes from scheduling levers — immediate publishing, minimal blocking — not from hiring more staff."*

### 16. Algorithm convergence (sample efficiency)  ← *SMAC-SK en verde*
**[Intención]** Probar que SK **aporta valor** (SMAC con vs sin SK) y no es suerte.
**[Qué decir]** Aquí cazo al mayordomo: la curva verde es SMAC-SK; compárenla con
SMAC **sin** SK. El SK baja antes y más estable bajo el ruido heterocedástico → el
beneficio del SK es real, no un golpe de suerte de un seed. *(Ojo: todas las
curvas en el mismo presupuesto de 150 evaluaciones; SK-Adaptive se trunca a 150.)*
**[EN]** *"Green is SMAC-SK; compare it to SMAC without SK — the SK term, not luck, is what buys the gain."*
**[Sugerencia]** Si puedes, resalta SMAC-SK en verde también en `conv.png` (mismo
criterio que la figura Pareto) para mantener el hilo visual.

### 17. Computational cost
**[Qué decir]** Estos métodos son caros (~17–25 h por seed). Por eso el muestreo
adaptativo de réplicas importa, y por eso la eficiencia muestral de SMAC-SK es una
ventaja práctica, no solo estética.

### 18. Discussion: Results
**[Qué decir]** (1) Resultado principal: SMAC-SK reduce TTS −33% y sube pacientes
+90%, sin trade-off (Pareto-domina lo manual). (2) **Las garantías existentes no
transfieren**: todos convergen empíricamente, pero sus resultados asintóticos
suponen ruido homocedástico y suavidad — ambos violados por el DES. (3)
**Problema abierto**: ningún método maneja nativamente heterocedasticidad +
mixto-entero + garantías formales → ese es el gap que ataca la tesis doctoral.
**[EN]** *"Existing guarantees don't transfer — they assume homoscedastic, smooth problems; ours is neither. That open problem is my doctoral work."*

### 19. Future Work
**[Qué decir]** (1) Diseñar un DFO para mixto-entero con asignación adaptativa de
réplicas guiada por σ̂²(x) local. (2) Teoría de convergencia: condiciones de
regularidad sobre la estructura de ruido heterocedástico bajo las cuales la
convergencia casi-segura a puntos estacionarios sea demostrable.
**[EN]** *"Two threads: an algorithm that natively handles this, and the convergence theory behind it."*

### 20. Cierre
**[Qué decir]** Cierra el caso: el problema era difícil por cinco razones; SMAC-SK
es el único que las abraza; lo probé contra los estándares y gana; y abre la puerta
a la teoría que falta. "Ese fue el mayordomo, con el candelabro, en la cocina."

---

## Ediciones concretas recomendadas al `.tex`

1. **Proposed Framework**: bloque central `Algorithm Optimizer` → **`SMAC-SK`**
   (y mantener *Core contribution* abajo).
2. **Objective**: añadir una línea final — *"Given the gap, the proposed instrument
   is **SMAC-SK** (SMAC with a Stochastic-Kriging surrogate)."*
3. **fig_pareto.png**: usar la versión nueva (SMAC-SK estrella verde + halo). ✅ lista.
4. **Discussion / main result**: reconciliar **164 → 177 d** (o confirmar 164).
   Recomendado titular con los números de SMAC-SK: **263 → 177 d (−33%)**,
   **1.148 → 2.180 (+90%)**.
5. (Opcional) Convergencia `conv.png`: SMAC-SK en verde para mantener el hilo.

## Nota honesta (lo que el profesor enfatizó)
No presentar "desde el principio supe que SMAC-SK ganaría". El relato honesto es:
*revisé los requisitos del problema → vi que SMAC podía → la variante SK es la
única que captura la heterocedasticidad → comparé contra SMAC sin SK, DFO y
discreto para demostrar que el beneficio del SK es real*. Esa es la fuerza
metodológica que justifica recomendarle SMAC-SK al hospital.
