# Correr la corrida completa en tu PC

Estos pasos reproducen en tu computador la misma corrida que se está ejecutando en la nube,
con el mismo código, las mismas semillas y el mismo presupuesto.

## 1. Requisitos

- Python 3.11, 3.12 o 3.13 (64 bits). En Windows: <https://www.python.org/downloads/>, marcando "Add python.exe to PATH".
- Git.
- Unas 6 h con 4 núcleos (con 8 núcleos, unas 3 h). Unos 2 GB de disco.

## 2. Bajar el código (rama del PR #13)

```bash
git clone https://github.com/rham1978/SO.git
cd SO
git checkout claude/project-thread-ql04u0
cd rl
```

Si ya tienes el repo: `git fetch origin claude/project-thread-ql04u0` y luego
`git checkout claude/project-thread-ql04u0 && git pull`.

## 3. Entorno e instalación

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**SMAC en Windows:** SMAC3 no tiene soporte oficial en Windows y a veces no instala
(falla al compilar `pyrfr`). Si `pip install` falla solo en `smac`, instala el resto con
`pip install simpy==4.1.2 numpy==2.5.3 scipy==1.18.1 matplotlib==3.11.2 torch==2.15.0 scikit-learn==1.9.1`
y corre con `--sin_smac` (paso 5): SMAC-SK se toma de la corrida de la nube.

## 4. Prueba rápida (≈ 35 min)

```bash
python validar_equivalencia.py 8          # simulador original vs acelerado en tu PC (≈ 3 min)
bash corrida_humo.sh                      # Windows sin bash: ver nota al final
```

## 5. Corrida completa

Elige cuántos núcleos usar con la variable `RL_NUCLEOS` (por defecto, todos):

Windows (PowerShell):

```powershell
$env:RL_NUCLEOS = 6
python corrida_completa.py --out resultados/completa_pc
```

macOS / Linux:

```bash
RL_NUCLEOS=6 python corrida_completa.py --out resultados/completa_pc
```

Opciones:

- `--sin_smac` omite SMAC-SK.
- `--inits 3,4` entrena solo esas inicializaciones (para repartir el trabajo con la nube).
- Si se corta, vuelve a lanzar el mismo comando: cada etapa que ya terminó se salta.

Etapas, en orden: referencias (≈ 25 min) → SMAC-SK (≈ 25 min) → ajuste de hiperparámetros
(12 corridas de 1 000 episodios) → 3 agentes × 5 inicializaciones → evaluación en prueba.
La bitácora queda en `resultados/completa_pc/corrida.log`.

## 6. Dónde quedan los resultados

```
resultados/completa_pc/
├── corrida.log
├── referencias.json      mejor fija, operación actual, aleatoria, SMAC-SK tesis
├── smac_sk.json          SMAC-SK con el mismo presupuesto
├── hp/  hp_elegidos.json ajuste de hiperparámetros
├── <método>_s<k>/        curva.json y mejor.pt de cada agente e inicialización
└── prueba/               final.json, por_semilla.csv, final.log
```

## 7. Comparar con la corrida de la nube

Copia la carpeta de la nube (`iee3031/resultados_completa` del proyecto) a, por ejemplo,
`resultados/completa_nube`, y luego:

```bash
python comparar_corridas.py resultados/completa_nube resultados/completa_pc
```

Las referencias (mejor fija, base, aleatoria) deberían coincidir, salvo diferencias mínimas de punto flotante: son las mismas
semillas y el mismo código. Los agentes pueden diferir un poco entre computadores (PyTorch no
garantiza resultados idénticos en distinto hardware); lo que debe coincidir es la conclusión,
dentro de los IC95.

**Nota (Windows sin bash):** en vez de `bash corrida_humo.sh`, ejecuta los comandos del archivo
uno por uno en PowerShell, o usa Git Bash, que viene con Git para Windows.
