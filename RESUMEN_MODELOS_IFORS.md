# Resumen detallado de modelos — Presentación IFORS

**Caso:** Optimización de la programación/capacidad de una clínica (CRS) mediante
simulación de eventos discretos (DES) + optimización vía simulación (SO).
**Indicador objetivo:** `TTS = tiempo medio total del paciente en el sistema [días]` (minimizar).
**Fecha de consolidación:** batch de 15 macro-seeds por método.

---

## 1. El problema de optimización

### 1.1 Simulador (caja negra estocástica)
- Modelo DES de 52 semanas de la ruta del paciente: ingreso → primera consulta →
  post-consulta (controles, pre-quirúrgico, cirugía) → alta.
- Cada corrida (`run_once`) es **ruidosa y heterocedástica**: la varianza σ²(x)
  depende del punto de operación (un sistema saturado es mucho más ruidoso que uno holgado).
- Costo computacional alto: **~17–25 h por seed** según el algoritmo.

### 1.2 Objetivo: `tts_full_days_mean`
Tiempo total desde que el paciente entra al sistema hasta que completa toda su ruta. Incluye:
- **Backlog histórico** pre-t=0 (~54 días en baseline): espera previa al inicio de la sim.
- **Proceso primera consulta** (~140 días en baseline).
- **Post-consulta** (~127 días en baseline).
- Baseline: `tts_full_days_mean ≈ 279.8 días`.

Es el KPI clínico más completo: sensible tanto a esperas iniciales como a cuellos
de botella post-consulta. El **throughput** (total de atenciones / pacientes atendidos)
**no se optimiza**: es un resultado emergente que se reporta como contexto (vista bi-objetivo / Pareto).

### 1.3 Variables de decisión (12) y sus rangos
| # | Variable | Campo SimConfig | Rango |
|---|----------|-----------------|-------|
| 1 | Cupos 1ra consulta / sem | `fixed_weekly_capacity` | [8, 30] |
| 2 | Cupos control post / sem | `fixed_post_control_capacity` | [20, 70] |
| 3 | Cupos laboratorio (UGD) | `ugd_lab_per_week` | [20, 100] |
| 4 | Cupos ecografía (matrona) | `mat_us_per_week` | [10, 50] |
| 5 | Cupos ecografía (UGD) | `ugd_us_per_week` | [10, 50] |
| 6 | Días de publicación (lead) | `publish_lead_workdays` | [1, 10] |
| 7 | % bloqueo 1ra consulta | `blocked_pct` | [0.05, 0.50] |
| 8 | % consultas vacías | `empty_control_p_ugd` | [0.05, 0.50] |
| 9 | Nº matronas | `matrona_capacity` | [1, 4] |
| 10 | Nº agentes (UGD) | `agent_capacity` | [1, 4] |
| 11 | % no contactabilidad | `not_contactable_p` | [0.05, 0.50] |
| 12 | % bloqueo post control | `blocked_pct_post_control` | [0.05, 0.50] |

Espacio mixto entero/continuo de dimensión **d = 12**.

### 1.4 Protocolo experimental (clave para la validez)
- **15 macro-seeds** independientes por algoritmo (réplicas de toda la optimización).
- **Presupuesto:** ~150 evaluaciones del simulador por corrida de optimización.
- **Re-evaluación del incumbente:** el mejor punto encontrado se re-evalúa con
  **n = 50 réplicas** (estimación insesgada de su TTS real).
- **Common Random Numbers (CRN):** misma secuencia de seeds para todos los
  escenarios → comparación **pareada**, reduce varianza.
- **Estadística robusta a heterocedasticidad:** Levene (igualdad de varianzas) →
  **Welch ANOVA** (omnibus) + **Kruskal-Wallis** (respaldo no paramétrico) +
  **Welch t por pares con corrección Holm** (post-hoc).

---

## 2. Catálogo de algoritmos de la presentación

Seis algoritmos con datos válidos. Todos comparten el **mismo simulador, mismo
objetivo, mismas seeds y mismo presupuesto** → la comparación es justa.

Tres grandes familias:
- **A. SMAC / Optimización bayesiana global** (M4, M4RF, M7) — surrogate global + función de adquisición.
- **B. Stochastic Kriging con replicación inteligente** (M8, M10) — surrogate heterocedástico + cuánto replicar.
- **C. Aproximación estocástica por gradiente** (M13) — sin surrogate, estima gradiente.

---

### A.1 — M4 · SMAC-GP+EI (Black-Box Bayesian Optimization)
- **Familia:** Optimización bayesiana global (framework SMAC3).
- **Surrogate:** Proceso Gaussiano (GP).
- **Adquisición:** Expected Improvement (EI).
- **Inicialización:** diseño Sobol (cuasi-aleatorio de baja discrepancia).
- **Tipo SMAC:** "Black-Box" (GP + EI + Sobol).
- **Idea:** modela globalmente f(x) con un GP y propone el punto que maximiza la
  mejora esperada sobre el mejor valor actual. Excelente eficiencia muestral en
  dimensiones bajas/medias; el GP puede sufrir con d alto y mucho ruido.

### A.2 — M4RF · SMAC-RF (Hyperparameter-Optimization facade)
- **Familia:** Optimización bayesiana global (SMAC3).
- **Surrogate:** **Random Forest** (RF) en vez de GP.
- **Adquisición:** EI. **Inicialización:** Sobol.
- **Tipo SMAC:** "HPO" (RF + EI + Sobol).
- **Idea:** el RF como surrogate maneja mejor variables mixtas/condicionales y
  es más barato que el GP, a costa de una cuantificación de incertidumbre menos
  suave. Es la configuración "por defecto" de SMAC3 para HPO.
- **Nota:** es el modelo de **corridas nuevas** que motivó esta actualización;
  en la versión anterior figuraba como "pendiente".

### A.3 — M7 · SMAC-SK (SMAC con Stochastic Kriging)
- **Familia:** Optimización bayesiana global (SMAC3) con surrogate especializado en ruido.
- **Surrogate:** **Stochastic Kriging (SK)** — Kriging que modela explícitamente
  la **varianza intrínseca σ²(x) variable por punto** (heterocedasticidad).
- **Adquisición:** EI. **Réplicas:** n_corridas fijo por evaluación (las réplicas
  de cada punto corren en paralelo; σ²(x) se estima con ellas).
- **Idea:** a diferencia del RF/GP estándar, el SK pondera menos los puntos muy
  ruidosos (sistema saturado) → mejor ajuste donde el simulador es inestable.
- **Nota:** **modelo que no estaba completo en la presentación anterior** y que
  ahora se incorpora con datos válidos.

---

### B.1 — M8 · SK-Adaptive (Stochastic Kriging + replicación adaptativa)
- **Familia:** Stochastic Kriging global con asignación adaptativa de réplicas.
- **Surrogate:** SK (heterocedástico, igual que M7). **Adquisición:** EI.
- **Política de réplicas (la diferencia clave vs M7):** el nº de réplicas n(x) se
  decide dinámicamente según la varianza estimada:
  `n(x) = n_min + round( rank_percentil(σ²(x)) · (n_max − n_min) )`.
  - Zonas ruidosas (σ² alto) → más réplicas (mejor estimación de la media).
  - Zonas estables (σ² bajo) → menos réplicas (ahorro de cómputo).
  - Fase de *warmup* con n_min para poblar el almacén de varianzas.
- **Idea:** reasigna el presupuesto de simulación a donde más reduce la incertidumbre.

### B.2 — M10 · SK-KGCP (Stochastic Kriging + Knowledge Gradient)
- **Familia:** Stochastic Kriging global con adquisición de **lookahead**.
- **Surrogate:** SK heterocedástico. **Adquisición:** **KGCP** (Knowledge Gradient
  for Continuous Parameters), en lugar de EI.
- **Diferencia fundamental con EI:**
  - EI (greedy): `EI(x) = E[max(μ* − f(x), 0)]` — solo mira la mejora directa en x.
  - KGCP (un paso adelante): mide cuánto mejoraría el **incumbente global** si se
    evalúa x y se actualiza el modelo, propagando la información a **todos** los x'
    vía el kernel correlacionado del SK:
    `KG(x) ≈ mean_Z[ max_{x'} (μ_n(x') + σ_kg(x,x')·Z) ] − max μ_n`,
    con `σ_kg(x,x') = K_n(x,x') / sqrt(K_n(x,x) + σ²_sim(x)/n_reps)`.
- **Referencia:** Frazier, Powell & Dayanik (2009); discretización Monte Carlo.
- **Por qué supera a EI bajo ruido:** EI sobreexplota zonas de alta varianza
  (falsos positivos); KGCP descuenta la incertidumbre y no evalúa puntos que no
  informan el óptimo global. Óptimo de horizonte 1 bajo el prior GP.

> Nota: existen además M9 (SK-REVI), que asigna réplicas balanceando ruido del
> simulador vs incertidumbre del modelo `n*(x)=sqrt(σ²_sim/σ²_mod)`; M10 extiende
> esa línea reemplazando EI por KGCP.

---

### C.1 — M13 · SPSA (Simultaneous Perturbation Stochastic Approximation)
- **Familia:** Aproximación estocástica por gradiente (sin surrogate).
- **Referencia:** Spall (1998), IEEE T-AES 34(3):817-823.
- **Idea / ventaja clave:** estima el gradiente con **solo 2 evaluaciones por
  iteración**, **independiente de la dimensión** d=12 (los métodos trust-region
  necesitan d+1=13 a 2d+1=25 evaluaciones por iteración).
- **Algoritmo (por iteración k):**
  1. Perturbación `Δk ∈ {−1,+1}^d` (Bernoulli simétrico iid).
  2. `ck = step/(k+1)^γ` (paso de perturbación decreciente).
  3. `y± = F̄(xk ± ck·Δk, n_reps)`.
  4. Gradiente: `g_{k,i} = (y⁺ − y⁻)/(2·ck·Δ_{k,i})`.
  5. Promedio de `gavg` gradientes → ĝk.
  6. `ak = α/(k+A)^{α_exp}` (paso del iterado).
  7. `x_{k+1} = clip(xk − ak·ĝk, [0,1]^12)`.
- **Parámetros (defaults SimOpt):** α=0.602, γ=0.101, step=0.1, gavg=1, n_reps=30, A=10.

---

## 3. Resultados consolidados (mejor incumbente re-evaluado, n=50)

| Algoritmo | Módulo | Familia | TTS μ±σ [d] | IC95 | Atendidos | Δ vs Current | Veredicto |
|-----------|:---:|---|:---:|:---:|:---:|:---:|:---:|
| **SPSA** | M13 | Gradiente | **174.5 ± 6.3** | [172.7, 176.2] | 2216 | −88.5 (−34%) | ★ Pareto |
| **SMAC-GP+EI** | M4 | BO global | 174.9 ± 7.0 | [173.0, 176.9] | 1895 | −88.0 (−33%) | mejor |
| **SK-Adaptive** | M8 | SK | 176.6 ± 5.9 | [174.9, 178.2] | 2206 | −86.4 (−33%) | mejor |
| **SMAC-SK** | M7 | BO+SK | 176.9 ± 4.7 | [175.6, 178.2] | 2179 | −86.0 (−33%) | mejor |
| **SMAC-RF** | M4RF | BO global | 182.1 ± 5.3 | [180.6, 183.5] | 1807 | −80.9 (−31%) | mejor |
| **SK-KGCP** | M10 | SK | 192.0 ± 6.2 | [190.3, 193.7] | 1836 | −71.0 (−27%) | mejor |
| *Mgmt+Cap v2* | manual | — | 238.2 ± 4.7 | [236.0, 240.4] | 2261 | −24.7 (−9%) | ★ Pareto |
| *Mgmt+Cap* | manual | — | 239.5 ± 5.1 | [237.1, 241.9] | 2204 | −23.5 (−9%) | — |
| *Management* | manual | — | 240.8 ± 4.8 | [238.6, 243.0] | 1703 | −22.2 (−8%) | — |
| *Current* | manual | — | 263.0 ± 7.1 | [259.7, 266.2] | 1148 | — | baseline |

> El TTS de los algoritmos proviene de la **re-evaluación del mejor incumbente (n=50)**;
> los manuales se **re-simularon (n=18, CRN)**. El throughput de SMAC-SK (M7) y SMAC-RF (M4RF)
> se recuperó re-simulando el incumbente (el batch nuevo no guardó ese KPI para esos módulos).

**Promedios entre seeds (robustez del método, no solo el mejor seed):**
M13 ≈ 207.1 · M4 ≈ 209.3 · M7 ≈ 215.2 · M4RF ≈ 216.3 · M8 ≈ 216.5 · M10 ≈ 217.8 d.

**Costo computacional (h por seed):** M7 ≈ 17.0 · M10 ≈ 17.1 · M13 ≈ 17.2 · M4 ≈ 19.1 · M8 ≈ 21.6 · M4RF ≈ 25.2.

### Configuraciones óptimas encontradas (variables de decisión del mejor incumbente)
| Variable | SMAC-GP+EI | SMAC-SK | SPSA | SK-Adaptive | SMAC-RF | SK-KGCP |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|
| Cupos 1ra/sem | 23 | 30 | 30 | 30 | 21 | 21 |
| Cupos post/sem | 70 | 70 | 70 | 70 | 58 | 58 |
| Lab (UGD) | 86 | 100 | 20 | 100 | 93 | 93 |
| Eco (matrona) | 40 | 50 | 50 | 50 | 35 | 35 |
| Eco (UGD) | 33 | 50 | 50 | 50 | 29 | 29 |
| Lead (días) | 1 | 1 | 1 | 1 | 1 | 1 |
| Nº matronas | 1 | 1 | 1 | 1 | 1 | 1 |
| Nº agentes | 2 | 4 | 4 | 4 | 2 | 2 |
| % bloqueo 1ra | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 |
| % consultas vacías | 0.28 | 0.05 | 0.50 | 0.05 | 0.28 | 0.28 |
| % no contactab. | 0.31 | 0.05 | 0.05 | 0.05 | 0.23 | 0.23 |
| % bloqueo post | 0.15 | 0.05 | 0.05 | 0.05 | 0.12 | 0.12 |

**Palancas comunes a todos los óptimos:** lead = **1 día** (publicación inmediata),
bloqueo 1ra = **5%** (mínimo) — sin agregar matronas. SMAC-RF y SK-KGCP convergen a
una región casi idéntica del espacio.

---

## 4. Modelos considerados pero sin datos en este batch
| Modelo | Módulo | Estado | Motivo |
|--------|:---:|--------|--------|
| **ASTRO-DF** | M11 | **Falló** | Trust-region derivative-free (Shashaani et al. 2016); las corridas no completaron (deadlock del solver / *futures unfinished*). |
| **Random Search** | RS | **Sin datos válidos** | 0 incumbentes guardados en el batch. |

Además, el código base contiene otros algoritmos no incluidos en esta corrida:
**SK-REVI (M9)**, **STRONG (M12)**, **ALOE (M14)** y **Simulated Annealing**
(Alrefaei & Andradóttir, 1999), disponibles para futuras comparaciones.

---

## 5. Conclusiones para IFORS
1. **Todos** los algoritmos de optimización superan significativamente a TODOS los
   ajustes manuales en TTS (Welch t + Holm, p ≪ 0.001). Recortan la espera ~31–34%
   frente a ~8–9% del mejor ajuste manual.
2. La mejora **no sacrifica throughput**: las atenciones suben (resultado emergente).
3. **Frontera de Pareto** (TTS vs atendidos): **SPSA** (menor espera) y
   **Mgmt+Cap v2** (mayor throughput). SPSA, M4 y M8 son las mejores opciones
   tiempo-céntricas.
4. **Lectura metodológica:**
   - SPSA (gradiente, 2 evals/iter) es competitivo o mejor que la BO global pese a
     su simplicidad → en este problema ruidoso de d=12, la eficiencia por iteración importa.
   - Dentro de SMAC, GP+EI ≳ SK ≳ RF en el mejor incumbente; SK aporta robustez
     frente a la heterocedasticidad.
   - SK-KGCP queda algo por detrás en el mejor incumbente pese al lookahead, lo que
     sugiere sensibilidad al presupuesto/discretización Monte Carlo.
5. **Validez:** mismo simulador, mismas seeds (CRN), 15 macro-seeds, re-evaluación
   n=50, y tests robustos a heterocedasticidad (Welch ANOVA + Kruskal-Wallis).
6. **Siguientes pasos:** completar ASTRO-DF (deadlock) y RS; co-optimizar tiempo y
   throughput con objetivo combinado `f = TTS − λ·atenciones`.

---

### Referencias
- Frazier, P., Powell, W., Dayanik, S. (2009). *The Knowledge-Gradient Policy for Correlated Normal Beliefs.* INFORMS J. on Computing.
- Spall, J.C. (1998). *Implementation of the Simultaneous Perturbation Algorithm for Stochastic Optimization.* IEEE T-AES 34(3):817-823.
- Shashaani, S., Hashemi, F.S., Pasupathy, R. (2016). *ASTRO-DF: Adaptive Sampling Trust-Region Algorithms for Derivative-Free Stochastic Optimization.* arXiv:1610.06506.
- Ankenman, B., Nelson, B., Staum, J. (2010). *Stochastic Kriging for Simulation Metamodeling.* Operations Research.
- Lindauer et al. (2022). *SMAC3: A Versatile Bayesian Optimization Package for HPO.* JMLR.
</content>
</invoke>
