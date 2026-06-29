# Detalle por método — Benchmark IFORS

Documento técnico de apoyo a la presentación. Para **cada algoritmo** se describe:
**(1)** tipo de variables y cómo las maneja, **(2)** flujo paso a paso (ver diagrama),
**(3)** la matemática clave, **(4)** referencias, y **(5)** si es método publicado o
propuesta propia.

**Marco común a todos:** simulador DES estocástico y heterocedástico; objetivo
`tts_full_days_mean` [días] (minimizar); espacio mixto de **d = 12** variables;
presupuesto **~150 evaluaciones** del simulador por corrida; **15 macro-seeds**;
**CRN** (mismas seeds entre métodos); el mejor incumbente se **re-evalúa con n = 50**
réplicas frescas. Diagramas en `reporte_figs/metodos/`.

### Las 12 variables de decisión
| # | Variable | Tipo | Rango |
|---|----------|:---:|:---:|
| 1 | Cupos 1ra consulta/sem | entero | [8, 30] |
| 2 | Cupos control post/sem | entero | [20, 70] |
| 3 | Cupos laboratorio (UGD) | entero | [20, 100] |
| 4 | Cupos ecografía (matrona) | entero | [10, 50] |
| 5 | Cupos ecografía (UGD) | entero | [10, 50] |
| 6 | Días de publicación (lead) | entero | [1, 10] |
| 7 | Nº matronas | entero | [1, 4] |
| 8 | Nº agentes (UGD) | entero | [1, 4] |
| 9 | % bloqueo 1ra | continuo | [0.05, 0.50] |
| 10 | % consultas vacías | continuo | [0.05, 0.50] |
| 11 | % no contactabilidad | continuo | [0.05, 0.50] |
| 12 | % bloqueo post control | continuo | [0.05, 0.50] |

**8 enteras + 4 continuas.** El espacio es idéntico para todos los métodos; lo que
cambia es **cómo cada algoritmo lo representa** (columna "manejo de variables" abajo).

---

## A. Familia SMAC / Optimización Bayesiana global

Esqueleto común: diseño inicial → evaluar simulador → ajustar **surrogate** →
maximizar **función de adquisición** → repetir. El surrogate aproxima `f(x)` y la
adquisición decide dónde evaluar después.

### M4 · SMAC-GP+EI
*Diagrama:* `diag_M4_SMAC_GP.png`

- **Manejo de variables:** `ConfigSpace` con `UniformIntegerHyperparameter` +
  `UniformFloatHyperparameter` **nativos**. El GP trabaja sobre el vector normalizado
  y SMAC redondea las enteras al construir cada configuración.
- **Surrogate:** Proceso Gaussiano (kernel Matérn). Covarianza **homocedástica**:
  `K_y = K(X,X) + σ²·I` (un único nivel de ruido).
- **Adquisición:** Expected Improvement `EI(x) = E[max(μ* − f(x), 0)]` — *greedy*,
  mira solo la mejora directa en `x`.
- **Diseño inicial:** Sobol. **Fachada:** `BlackBoxFacade`.
- **Fortaleza/debilidad:** excelente eficiencia muestral en d baja/media; el GP
  homocedástico se degrada con d alto y ruido fuerte.
- **Referencias:** Jones, Schonlau & Welch (1998) *Efficient Global Optimization*;
  framework: Lindauer et al. (2022) *SMAC3*, JMLR.
- **Estatus:** **método publicado** (BO con GP+EI es estándar; lo provee SMAC3).

### M4RF · SMAC-RF
*Diagrama:* `diag_M4RF_SMAC_RF.png`

- **Manejo de variables:** igual `ConfigSpace` mixto; el **Random Forest maneja
  variables mixtas/condicionales de forma nativa** (su principal ventaja).
- **Surrogate:** Random Forest; la media y la incertidumbre salen de la dispersión
  entre árboles (incertidumbre escalonada, no suave).
- **Adquisición:** EI. **Diseño inicial:** Sobol. **Fachada:** `HyperparameterOptimizationFacade`.
- **Fortaleza:** más barato que el GP, robusto en dimensión alta; es la configuración
  **por defecto de SMAC3 para HPO**.
- **Referencias:** Hutter, Hoos & Leyton-Brown (2011) *Sequential Model-Based
  Optimization (SMAC)*, LION-5; Breiman (2001) *Random Forests*.
- **Estatus:** **método publicado** (configuración estándar de SMAC).

### M7 · SMAC-SK  — *propuesta propia*
*Diagrama:* `diag_M7_SMAC_SK.png`

- **Manejo de variables:** `ConfigSpace` mixto; el SK opera sobre el vector
  normalizado [0,1]¹², enteras redondeadas.
- **Surrogate:** **Stochastic Kriging** programado a medida. Covarianza
  **heterocedástica**: `K_y = K(X,X) + diag(V)`, con `V_i = σ²(xᵢ)/n(xᵢ)`.
- **Conexión SMAC↔SK (lo no trivial):** la API de surrogate de SMAC solo pasa
  `(X, Y)` (medias) — **nunca las varianzas**. Se usa un **canal lateral**: la función
  objetivo deposita `σ²(x)` en un `_variance_store` global; el método `_train` del SK
  lo recupera para armar `diag(V)`. El kernel es Matérn 5/2, hiperparámetros por
  grid-search sobre el negative-log-likelihood (Cholesky).
- **Adquisición:** EI sobre `(μ, σ²_modelo)` que devuelve `_predict`; maximizada con
  `LocalAndSortedRandomSearch` (1000 challengers).
- **Efecto:** un punto muy ruidoso entra con `V_i` grande → el SK **confía menos** en
  él → EI no sobre-explota zonas saturadas.
- **Referencias:** SK: **Ankenman, Nelson & Staum (2010)** *Stochastic Kriging for
  Simulation Metamodeling*, Oper. Res.; framework SMAC3 (Lindauer 2022).
- **Estatus:** **propuesta propia / integración innovadora.** Cada pieza está
  publicada, pero **SMAC3 no incluye Stochastic Kriging** (solo RF y GP); el surrogate
  SK y su acople a SMAC son contribución de este trabajo.

---

## B. Familia Stochastic Kriging con replicación/adquisición inteligente

Reutilizan el surrogate SK heterocedástico de M7 y mejoran **dónde gastar el presupuesto**.

### M8 · SK-Adaptive
*Diagrama:* `diag_M8_SK_Adaptive.png`

- **Manejo de variables:** opera en [0,1]¹² normalizado; enteras redondeadas al simular.
- **Novedad vs M7:** el nº de réplicas por punto **no es fijo**. Una política decide
  `n(x)`:
  - **Warmup** (primeras `n_warmup` configs): `n(x) = n_min` para poblar el almacén de
    varianzas.
  - **Adaptativa:** `n(x) = n_min + round( pct · (n_max − n_min) )`, donde `pct` es el
    **percentil** de `σ²(x)` entre las varianzas conocidas. `σ²(x)` se estima por
    **k-vecinos más cercanos** en el espacio normalizado.
- **Idea:** más réplicas donde el simulador es ruidoso (mejor media), menos donde es
  estable (ahorro) → reasigna el presupuesto a donde más reduce incertidumbre.
- **Surrogate/adquisición:** SK heterocedástico + EI (idénticos a M7).
- **Referencias:** SK: Ankenman et al. (2010); asignación de réplicas inspirada en
  Chen, Ankenman & Nelson (2012) *The effects of common random numbers and simulation
  replications*.
- **Estatus:** **propuesta propia** (la política de réplicas por percentil-k-NN es una
  heurística de este trabajo sobre el SK publicado).

### M10 · SK-KGCP
*Diagrama:* `diag_M10_SK_KGCP.png`

- **Manejo de variables:** [0,1]¹² normalizado; enteras redondeadas.
- **Surrogate:** SK heterocedástico (= M7).
- **Adquisición — la diferencia clave:** **Knowledge Gradient for Continuous
  Parameters (KGCP)** en vez de EI. Es **lookahead de 1 paso**: mide cuánto mejoraría
  el **incumbente global** si se evaluara `x` y se actualizara el modelo, propagando la
  información a todo `x'` vía el kernel correlacionado:
  - `σ_kg(x,x') = K_n(x,x') / √(K_n(x,x) + σ²_sim(x)/n_reps)`
  - `KG(x) ≈ (1/n_mc) Σ_Z [ max_{x'∈X_cand} (μ_n(x') + σ_kg(x,x')·Z) ] − max μ_n`
  - `Z ~ N(0,1)` con `n_mc = 64` muestras Monte Carlo; `X_cand` = puntos evaluados +
    `n_cand = 500` candidatos aleatorios.
- **Por qué supera a EI bajo ruido (en teoría):** EI sobre-explota zonas de alta
  varianza (falsos positivos); KGCP descuenta la incertidumbre y no evalúa puntos que
  no informan el óptimo global. Es el óptimo de horizonte 1 bajo el prior.
- **Referencias:** **Frazier, Powell & Dayanik (2009)** *The Knowledge-Gradient Policy
  for Correlated Normal Beliefs*, INFORMS J. Computing; Scott, Frazier & Powell (2011)
  *The Correlated Knowledge Gradient (KGCP)*, SIAM J. Optim.
- **Estatus:** **propuesta propia** (KG está publicado; acoplarlo a un surrogate SK
  heterocedástico y al ciclo de SMAC es la contribución).

> Nota: el código incluye también **M9 (SK-REVI)**, que asigna réplicas balanceando
> ruido del simulador vs incertidumbre del modelo `n*(x)=√(σ²_sim/σ²_mod)`; M10
> extiende esa línea reemplazando EI por KGCP.

---

## C. Aproximación estocástica por gradiente (sin surrogate)

### M13 · SPSA
*Diagrama:* `diag_M13_SPSA.png`

- **Manejo de variables:** itera en [0,1]¹² continuo; perturba **todas** las dims a la
  vez con Bernoulli ±1; enteras redondeadas al simular.
- **Idea / ventaja:** estima el gradiente con **solo 2 evaluaciones por iteración**,
  **independiente de d** (los trust-region necesitan d+1 a 2d+1). Por iteración `k`:
  1. `Δk ∈ {−1,+1}¹²` (Bernoulli simétrico iid).
  2. `ck = step/(k+1)^γ` (perturbación decreciente).
  3. `y⁺ = F̄(xk + ck·Δk, n_reps)`, `y⁻ = F̄(xk − ck·Δk, n_reps)`.
  4. Gradiente: `ĝ_{k,i} = (y⁺ − y⁻)/(2·ck·Δ_{k,i})`.
  5. `ak = α/(k+A)^α_exp`; `x_{k+1} = clip(xk − ak·ĝk, [0,1]¹²)`.
- **Parámetros (defaults SimOpt):** α=0.602, γ=0.101, step=0.1, n_reps=30, A=10.
- **Referencias:** **Spall (1992)** IEEE T-AC 37(3); **Spall (1998)** IEEE T-AES
  34(3):817-823.
- **Estatus:** **método publicado** (implementación estándar de SPSA).

---

## D. Considerado pero sin datos válidos

### M11 · ASTRO-DF  (no completó)
*Diagrama:* `diag_M11_ASTRODF.png`

- **Tipo:** trust-region **derivative-free** local con muestreo adaptativo.
  Construye un modelo lineal estocástico en la bola `B(xk; Δk)`, da un paso de Cauchy,
  evalúa el candidato y acepta/rechaza según el cociente `ρ̂` (descenso observado /
  descenso predicho), expandiendo o contrayendo `Δ`. Convergencia wp1 garantizada.
- **Manejo de variables:** [0,1]¹² normalizado; muestreo adaptativo `Ñ(x) ∝ σ̂/Δ²`
  (más réplicas cerca del óptimo).
- **Referencia:** **Shashaani, Hashemi & Pasupathy (2018)** *ASTRO-DF*, SIAM J. Optim.
  (arXiv:1610.06506).
- **Estatus en este batch:** **falló** (deadlock del solver — *futures unfinished*).
  Pendiente de re-correr con el runner `correr_astrodf_rs.py` (start-method `spawn`,
  `n_workers=1`).

---

## Cuadro resumen

| Método | Variables (manejo) | Surrogate | Adquisición / paso | Referencia base | ¿Publicado o propio? |
|--------|--------------------|-----------|--------------------|-----------------|----------------------|
| **M4 SMAC-GP+EI** | ConfigSpace mixto nativo | Gaussian Process | EI (greedy) | Jones+1998 / SMAC3 | Publicado |
| **M4RF SMAC-RF** | ConfigSpace mixto nativo | Random Forest | EI | Hutter+2011 | Publicado |
| **M7 SMAC-SK** | normalizado + redondeo | **Stochastic Kriging** | EI | Ankenman+2010 + SMAC3 | **Propio** (integración SK↔SMAC) |
| **M8 SK-Adaptive** | normalizado + redondeo | SK heterocedástico | EI + réplicas adaptativas | Ankenman+2010; Chen+2012 | **Propio** (política de réplicas) |
| **M10 SK-KGCP** | normalizado + redondeo | SK heterocedástico | **KGCP** (lookahead MC) | Frazier+2009; Scott+2011 | **Propio** (KGCP↔SK) |
| **M13 SPSA** | normalizado + redondeo | — (sin surrogate) | gradiente SPSA (2 evals/iter) | Spall 1992/1998 | Publicado |
| **M11 ASTRO-DF** | normalizado + redondeo | modelo lineal local | trust-region + Cauchy | Shashaani+2018 | Publicado (falló) |

**Mensaje para el jurado:** los tres modelos de la familia SK (M7/M8/M10) **no son
métodos de estantería**: combinan ingredientes publicados (SMAC, Stochastic Kriging,
Knowledge Gradient) en una integración que SMAC3 no ofrece de fábrica, motivada por la
fuerte heterocedasticidad del simulador clínico.
