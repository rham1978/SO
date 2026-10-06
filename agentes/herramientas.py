"""
Herramientas del agente: funciones Python normales, sin LLM.

Un agente es (1) un modelo, (2) herramientas y (3) un loop. Este archivo es
la parte (2) y es la más importante: el modelo solo decide QUÉ llamar; lo
que pasa de verdad (qué se simula, con qué semillas, cuánto cuesta) lo
define este código. Todas las garantías duras viven aquí, no en el prompt:

  - validación del espacio canónico de 12 parámetros (rangos e enteros),
  - números aleatorios comunes en todas las evaluaciones (offsets fijos),
  - presupuesto máximo de réplicas del simulador, persistido en disco,
  - máximo de trabajos simultáneos,
  - salidas cortas y resumidas (el modelo no necesita ver 30 números crudos).

Cada herramienta es un dict {name, description, input_schema, fn}. Los pasos
2, 3 y 4 del tutorial usan exactamente esta misma lista; solo cambia el loop.

Se pueden probar sin API key:  python3 agentes/paso1_herramientas.py
"""

import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from modulo_comparativa_caja_negra import PARAM_NAMES, PARAM_RANGES, PARAM_BASELINE  # noqa: E402

DIR_TRABAJOS = Path(os.environ.get("AGENTE_DIR_TRABAJOS", REPO / "agentes" / "trabajos"))
PRESUPUESTO_REPLICAS = int(os.environ.get("AGENTE_PRESUPUESTO_REPLICAS", 300))
MAX_SIMULTANEOS = int(os.environ.get("AGENTE_MAX_SIMULTANEOS", 2))
SEMANAS = int(os.environ.get("AGENTE_SEMANAS", 52))   # bajar SOLO para pruebas de humo

ENTEROS = {"horas_especialista_1ra", "horas_control_post", "cupos_laboratorio_ugd",
           "cupos_ecografia_matrona", "cupos_ecografia_ugd", "dias_publicacion",
           "num_matronas", "num_agentes_ugd"}

# Módulos que el agente puede lanzar (ver benchmark_riguroso._params_para_runner).
MODULOS = {
    "M4":  "SMAC-GP+EI (Bayesiana global)",
    "M7":  "SMAC + Stochastic Kriging (EI)",
    "M8":  "Stochastic Kriging adaptativo",
    "M10": "SK con KG por costo (KGCP)",
    "M11": "ASTRO-DF (región de confianza, muestreo adaptativo)",
    "M13": "SPSA (aproximación estocástica)",
    "SA":  "Recocido simulado Alrefaei & Andradóttir (1999)",
    "RS":  "Búsqueda aleatoria (línea base)",
}
# Costo previo estimado de un algoritmo, en réplicas: max(n_trials, mínimo) * factor.
# Es una ESTIMACIÓN: al terminar se concilia con las evaluaciones que reporta el
# benchmark, y el límite duro de un algoritmo es su tiempo (max_horas).
_MIN_EVALS = {"M4": 15, "M7": 15, "M8": 15, "M10": 15, "M11": 30, "M13": 15, "SA": 30}
_FACTOR = {"RS": 3, "M8": 2}   # RS: 3 réplicas/punto; M8: hasta 6 réplicas/config en vez de 3

_ID_VALIDO = re.compile(r"^(eva|alg)-\d{4}-\d{6}-[0-9a-f]{4}$")
_PROCESOS: dict = {}   # job_id -> Popen, para recoger procesos terminados (evita zombis)


class ErrorHerramienta(Exception):
    """Error que se le muestra al modelo para que corrija su llamada."""


# ─── utilidades internas ────────────────────────────────────────────────────

@contextlib.contextmanager
def _bloqueo():
    """Exclusión mutua entre llamadas simultáneas (p. ej. dos subagentes lanzando a la vez)."""
    DIR_TRABAJOS.mkdir(parents=True, exist_ok=True)
    with open(DIR_TRABAJOS / ".lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _escribir_json(ruta: Path, datos: dict) -> None:
    """Escritura atómica: un corte a mitad nunca deja un JSON a medias."""
    tmp = ruta.with_suffix(".tmp")
    tmp.write_text(json.dumps(datos, indent=2, default=str))
    os.replace(tmp, ruta)


def _carpeta(job_id: str) -> Path:
    # El job_id lo escribe el modelo: se valida el formato para que no pueda
    # apuntar fuera de DIR_TRABAJOS (p. ej. "../../otra_carpeta").
    if not isinstance(job_id, str) or not _ID_VALIDO.match(job_id):
        raise ErrorHerramienta(f"job_id inválido: '{job_id}'. Usa estado_trabajos para ver los ids.")
    c = DIR_TRABAJOS / job_id
    if not (c / "spec.json").exists():
        raise ErrorHerramienta(f"No existe el trabajo '{job_id}'. Usa estado_trabajos para ver los ids.")
    return c


def _inicio_proceso(pid: int):
    """Instante de arranque del proceso según /proc (None si no existe o es zombi).

    Junto con el PID identifica al proceso sin ambigüedad: si el PID se reutiliza
    para otro programa, su instante de arranque es distinto.
    """
    try:
        campos = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if campos[0] == "Z" else campos[19]
    except (OSError, IndexError):
        return None


def _proceso_vivo(c: Path) -> bool:
    proc = _PROCESOS.get(c.name)
    if proc is not None:
        proc.poll()                      # recoge el proceso si ya terminó
    try:
        pid, inicio = (c / "pid").read_text().split()
    except (OSError, ValueError):
        return False
    return _inicio_proceso(int(pid)) == inicio


def _estado(c: Path) -> str:
    if (c / "resultado.json").exists():
        return "terminado"
    if (c / "error.txt").exists():
        return "error"
    if (c / "cancelado").exists():
        return "cancelado"
    if not _proceso_vivo(c):
        return "muerto"     # terminó sin resultado ni error.txt (p. ej. lo mató el sistema)
    spec = json.loads((c / "spec.json").read_text())
    limite = spec.get("max_horas")
    if limite and time.time() - spec.get("inicio_epoch", time.time()) > limite * 3600:
        _detener(c, f"superó max_horas={limite}")
        return "cancelado"
    return "corriendo"


def _detener(c: Path, motivo: str) -> None:
    if _proceso_vivo(c):
        os.killpg(int((c / "pid").read_text().split()[0]), signal.SIGTERM)
    (c / "cancelado").write_text(motivo)


def _presupuesto() -> dict:
    f = DIR_TRABAJOS / "presupuesto.json"
    usado = json.loads(f.read_text())["usado"] if f.exists() else 0
    return {"limite": PRESUPUESTO_REPLICAS, "usado": usado, "disponible": PRESUPUESTO_REPLICAS - usado}


def _sumar_presupuesto(delta: int) -> None:
    _escribir_json(DIR_TRABAJOS / "presupuesto.json", {"usado": _presupuesto()["usado"] + delta})


def _conciliar(c: Path) -> None:
    """Ajusta el presupuesto de un algoritmo terminado a las evaluaciones que reportó."""
    spec = json.loads((c / "spec.json").read_text())
    if spec["tipo"] != "algoritmo" or spec.get("conciliado") or _estado(c) != "terminado":
        return
    r = json.loads((c / "resultado.json").read_text())
    real = int(r.get("n_eval_usadas") or 0) + int(spec["r_final"])
    if real > spec["costo_reservado"]:
        _sumar_presupuesto(real - spec["costo_reservado"])
    _escribir_json(c / "spec.json", {**spec, "conciliado": True, "costo_reportado": real})


def _trabajos() -> list:
    return sorted(p for p in DIR_TRABAJOS.iterdir() if (p / "spec.json").exists()) \
        if DIR_TRABAJOS.exists() else []


def _lanzar(spec: dict, costo: int) -> tuple:
    """Lanza un trabajo en segundo plano. Devuelve (job_id, reutilizado)."""
    # Huella del experimento: misma configuración + mismas semillas = mismo resultado.
    experimento = {k: v for k, v in spec.items() if k != "etiqueta"}
    spec = {**spec, "clave": hashlib.sha1(json.dumps(experimento, sort_keys=True).encode()).hexdigest()[:12]}
    with _bloqueo():   # chequeo de cupos + reserva + arranque como UNA operación
        for c in _trabajos():
            _conciliar(c)
            otro = json.loads((c / "spec.json").read_text())
            if otro.get("clave") == spec["clave"] and _estado(c) in ("corriendo", "terminado"):
                return c.name, True
        activos = [c for c in _trabajos() if _estado(c) == "corriendo"]
        if len(activos) >= MAX_SIMULTANEOS:
            raise ErrorHerramienta(
                f"Ya hay {len(activos)} trabajos corriendo (máximo {MAX_SIMULTANEOS}). "
                "Espera a que alguno termine con la herramienta esperar.")
        p = _presupuesto()
        if costo > p["disponible"]:
            raise ErrorHerramienta(
                f"Presupuesto insuficiente: este trabajo cuesta ~{costo} réplicas y quedan "
                f"{p['disponible']} de {p['limite']}. Reduce el tamaño o termina con lo que tienes.")

        job_id = f"{spec['tipo'][:3]}-{time.strftime('%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
        c = DIR_TRABAJOS / job_id
        _sumar_presupuesto(costo)
        try:
            c.mkdir(parents=True)
            _escribir_json(c / "spec.json", {**spec, "costo_reservado": costo,
                                            "creado": time.strftime("%Y-%m-%d %H:%M:%S"),
                                            "inicio_epoch": time.time()})
            # Lista de argumentos sin shell: ningún valor se interpreta como comando.
            # El único dato variable es la carpeta del trabajo, generada aquí.
            proc = subprocess.Popen([sys.executable, str(REPO / "agentes" / "ejecutor.py"), str(c)],
                                    cwd=REPO, stdout=open(c / "ejecutor.log", "w"),
                                    stderr=subprocess.STDOUT, start_new_session=True)
        except Exception:
            _sumar_presupuesto(-costo)             # no se cobra un trabajo que no arrancó
            for f in c.glob("*"):
                f.unlink()
            if c.exists():
                c.rmdir()
            raise
        inicio = None
        for _ in range(50):                        # /proc tarda un instante en aparecer
            inicio = _inicio_proceso(proc.pid)
            if inicio:
                break
            time.sleep(0.02)
        (c / "pid").write_text(f"{proc.pid} {inicio}")
        _PROCESOS[job_id] = proc
        return job_id, False


def _validar_config(config: dict) -> dict:
    desconocidos = set(config) - set(PARAM_NAMES)
    if desconocidos:
        raise ErrorHerramienta(f"Parámetros desconocidos: {sorted(desconocidos)}. Válidos: {PARAM_NAMES}")
    completa = {**PARAM_BASELINE, **config}
    errores = []
    for k in PARAM_NAMES:
        lo, hi = PARAM_RANGES[k]
        v = completa[k]
        if not isinstance(v, (int, float)) or not lo <= v <= hi:
            errores.append(f"{k}={v} fuera de [{lo}, {hi}]")
        elif k in ENTEROS:
            completa[k] = int(round(v))
    if errores:
        raise ErrorHerramienta("Configuración inválida: " + "; ".join(errores))
    return completa


def _resumen(xs: list) -> dict:
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else float("nan")
    from scipy import stats
    h = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n) if n > 1 else float("nan")
    return {"media": round(m, 3), "sd": round(sd, 3), "ic95": [round(m - h, 3), round(m + h, 3)], "n": n}


# ─── herramientas ───────────────────────────────────────────────────────────

def ver_espacio(args: dict) -> dict:
    return {"parametros": {k: {"rango": PARAM_RANGES[k], "baseline": PARAM_BASELINE[k],
                               "entero": k in ENTEROS} for k in PARAM_NAMES},
            "modulos": MODULOS,
            "presupuesto_replicas": _presupuesto(),
            "max_trabajos_simultaneos": MAX_SIMULTANEOS}


def lanzar_evaluacion(args: dict) -> dict:
    config = _validar_config(args.get("config", {}))
    n_reps = int(args.get("n_reps", 10))
    if not 2 <= n_reps <= 30:
        raise ErrorHerramienta("n_reps debe estar entre 2 y 30.")
    # Reparte los cores entre los trabajos simultáneos para no sobrecargar la máquina.
    n_workers = max(1, (os.cpu_count() or 1) // MAX_SIMULTANEOS)
    job_id, reutilizado = _lanzar({"tipo": "evaluacion", "config": config, "n_reps": n_reps,
                                   "semanas": SEMANAS, "n_workers": n_workers,
                                   "etiqueta": args.get("etiqueta", "")}, costo=n_reps)
    if reutilizado:
        return {"job_id": job_id, "reutilizado": True, "presupuesto": _presupuesto(),
                "nota": "Ya existe una evaluación idéntica (misma configuración y semillas): "
                        "su resultado sería el mismo, así que se reutiliza sin costo."}
    return {"job_id": job_id, "costo_reservado": n_reps, "presupuesto": _presupuesto(),
            "nota": "Corre en segundo plano. Usa esperar y luego resultado_trabajo."}


def lanzar_algoritmo(args: dict) -> dict:
    modulo = args.get("modulo")
    if modulo not in MODULOS:
        raise ErrorHerramienta(f"Módulo '{modulo}' no disponible. Opciones: {list(MODULOS)}")
    n_trials = int(args.get("n_trials", 30))
    r_final = int(args.get("r_final", 10))
    if not 5 <= n_trials <= 300 or not 2 <= r_final <= 30:
        raise ErrorHerramienta("n_trials debe estar en [5, 300] y r_final en [2, 30].")
    max_horas = float(args.get("max_horas", 12))
    if not 0.25 <= max_horas <= 48:
        raise ErrorHerramienta("max_horas debe estar en [0.25, 48].")
    lambda_obj = args.get("lambda_obj")
    lambda_obj = None if lambda_obj is None else float(lambda_obj)
    costo = max(n_trials, _MIN_EVALS.get(modulo, 0)) * _FACTOR.get(modulo, 1) + r_final
    job_id, reutilizado = _lanzar({"tipo": "algoritmo", "modulo": modulo, "n_trials": n_trials,
                                   "r_final": r_final, "lambda_obj": lambda_obj,
                                   "max_horas": max_horas, "etiqueta": args.get("etiqueta", "")},
                                  costo=costo)
    if reutilizado:
        return {"job_id": job_id, "reutilizado": True, "presupuesto": _presupuesto(),
                "nota": "Ya existe una corrida idéntica de este algoritmo; se reutiliza sin costo."}
    return {"job_id": job_id, "costo_estimado": costo, "max_horas": max_horas,
            "presupuesto": _presupuesto(),
            "nota": "Puede tardar horas. El costo es una estimación que se ajusta al terminar; "
                    "si supera max_horas se detiene. Usa esperar y luego resultado_trabajo."}


def estado_trabajos(args: dict) -> dict:
    trabajos = []
    for c in _trabajos():
        s = json.loads((c / "spec.json").read_text())
        trabajos.append({"job_id": c.name, "tipo": s["tipo"], "estado": _estado(c),
                         "modulo": s.get("modulo"), "etiqueta": s.get("etiqueta", ""),
                         "creado": s.get("creado")})
    return {"trabajos": trabajos, "presupuesto": _presupuesto()}


def esperar(args: dict) -> dict:
    minutos = min(float(args.get("minutos", 10)), 60.0)
    ids = args.get("job_ids") or [t["job_id"] for t in estado_trabajos({})["trabajos"]
                                  if t["estado"] == "corriendo"]
    if not ids:
        return {"mensaje": "No hay trabajos corriendo."}
    fin = time.time() + minutos * 60
    while time.time() < fin:
        estados = {j: _estado(_carpeta(j)) for j in ids}
        if any(e != "corriendo" for e in estados.values()):
            break
        time.sleep(min(15, max(1, fin - time.time())))
    estados = {j: _estado(_carpeta(j)) for j in ids}
    return {"estados": estados, "espera_minutos": minutos,
            "nota": "Termina en cuanto alguno de los trabajos deja de correr."}


def resultado_trabajo(args: dict) -> dict:
    c = _carpeta(args.get("job_id", ""))
    estado = _estado(c)
    if estado == "error":
        return {"estado": "error", "detalle": (c / "error.txt").read_text()[-1500:]}
    if estado == "cancelado":
        return {"estado": "cancelado", "motivo": (c / "cancelado").read_text() or "a pedido"}
    if estado != "terminado":
        return {"estado": estado}
    with _bloqueo():
        _conciliar(c)
    r = json.loads((c / "resultado.json").read_text())
    if r["tipo"] == "evaluacion":
        reps = r["replicas"]
        if len(reps) < 2:
            return {"estado": "terminado", "advertencia": "menos de 2 réplicas válidas", "descartadas": r["descartadas"]}
        return {"estado": "terminado", "tipo": "evaluacion", "config": r["config"],
                "tts_full_days": _resumen([x["tts_full_days_mean"] for x in reps]),
                "total_atenciones": _resumen([x["total_atenciones"] for x in reps]),
                "replicas_descartadas": len(r["descartadas"]), "duracion_seg": r["duracion_seg"]}
    return {"estado": "terminado", **{k: v for k, v in r.items()}}


def comparar_evaluaciones(args: dict) -> dict:
    """Comparación pareada A vs B usando las réplicas con el mismo offset (CRN)."""
    from scipy import stats

    def cargar(job_id):
        c = _carpeta(job_id)
        if _estado(c) != "terminado":
            raise ErrorHerramienta(f"El trabajo {job_id} no ha terminado (estado: {_estado(c)}).")
        return json.loads((c / "resultado.json").read_text())

    ra, rb = cargar(args.get("job_a", "")), cargar(args.get("job_b", ""))
    if ra["tipo"] != "evaluacion" or rb["tipo"] != "evaluacion":
        raise ErrorHerramienta("Solo se comparan trabajos de tipo evaluacion. "
                               "Para un algoritmo, evalúa su incumbente con lanzar_evaluacion.")
    if ra.get("semanas") != rb.get("semanas"):
        raise ErrorHerramienta(f"Horizontes distintos ({ra.get('semanas')} vs {rb.get('semanas')} "
                               "semanas): no son comparables.")
    a = {x["offset"]: x for x in ra["replicas"]}
    b = {x["offset"]: x for x in rb["replicas"]}
    comunes = sorted(set(a) & set(b))
    if len(comunes) < 3:
        raise ErrorHerramienta("Hay menos de 3 réplicas pareadas; evalúa ambas con más réplicas.")
    out = {"replicas_pareadas": len(comunes)}
    for kpi in ("tts_full_days_mean", "total_atenciones"):
        d = [b[o][kpi] - a[o][kpi] for o in comunes]
        res = _resumen(d)
        p = stats.wilcoxon(d).pvalue if any(d) else 1.0
        out[f"diferencia_{kpi}_(B-A)"] = {**res, "p_wilcoxon": round(float(p), 4)}
    out["lectura"] = ("TTS: negativo = B reduce el tiempo en sistema. "
                      "Atenciones: positivo = B atiende más. Significativo si el IC95 no contiene 0.")
    if len(comunes) < 6:
        out["advertencia"] = ("Con menos de 6 pares Wilcoxon no puede bajar de p=0.05; "
                              "usa el IC95 o evalúa con más réplicas para confirmar.")
    return out


def cancelar_trabajo(args: dict) -> dict:
    c = _carpeta(args.get("job_id", ""))
    if _estado(c) != "corriendo":
        return {"mensaje": f"El trabajo no está corriendo (estado: {_estado(c)})."}
    _detener(c, "cancelado por el agente")
    return {"mensaje": "Cancelado. El presupuesto reservado no se devuelve."}


# ─── catálogo: esquema + función, lo único que ve el modelo ─────────────────

_CONFIG_SCHEMA = {
    "type": "object",
    "description": "Valores de los parámetros canónicos. Los omitidos toman su valor baseline.",
    "properties": {k: {"type": "integer" if k in ENTEROS else "number",
                       "minimum": PARAM_RANGES[k][0], "maximum": PARAM_RANGES[k][1]}
                   for k in PARAM_NAMES},
    "additionalProperties": False,
}

HERRAMIENTAS = [
    {"name": "ver_espacio",
     "description": "Muestra los 12 parámetros de decisión (rangos, baseline), los algoritmos "
                    "disponibles y el presupuesto de réplicas restante. Úsala primero.",
     "input_schema": {"type": "object", "properties": {}},
     "fn": ver_espacio},
    {"name": "lanzar_evaluacion",
     "description": "Simula UNA configuración con n_reps réplicas en segundo plano. Todas las "
                    "evaluaciones usan las mismas semillas, así que dos evaluaciones se pueden "
                    "comparar de forma pareada. Cuesta n_reps réplicas del presupuesto.",
     "input_schema": {"type": "object", "properties": {
         "config": _CONFIG_SCHEMA,
         "n_reps": {"type": "integer", "minimum": 2, "maximum": 30,
                    "description": "Réplicas (10 para explorar, 30 para confirmar)."},
         "etiqueta": {"type": "string", "description": "Nombre corto para reconocerla."}},
         "required": ["config"]},
     "fn": lanzar_evaluacion},
    {"name": "lanzar_algoritmo",
     "description": "Corre un algoritmo de optimización del benchmark (una macro-semilla) en "
                    "segundo plano y devuelve su mejor configuración (incumbente) re-evaluada. "
                    "Cuesta aproximadamente n_trials + r_final réplicas (RS: 3*n_trials, M8: 2*n_trials); "
                    "el costo real se ajusta al terminar. Puede tardar horas.",
     "input_schema": {"type": "object", "properties": {
         "modulo": {"type": "string", "enum": list(MODULOS)},
         "n_trials": {"type": "integer", "minimum": 5, "maximum": 300,
                      "description": "Presupuesto de evaluaciones del simulador."},
         "r_final": {"type": "integer", "minimum": 2, "maximum": 30,
                     "description": "Réplicas para re-evaluar el incumbente."},
         "lambda_obj": {"type": "number",
                        "description": "Opcional: f = TTS - lambda*atenciones. Omitir = solo TTS."},
         "max_horas": {"type": "number", "minimum": 0.25, "maximum": 48,
                       "description": "Tiempo máximo; si lo supera, se detiene (por defecto 12)."},
         "etiqueta": {"type": "string"}},
         "required": ["modulo"]},
     "fn": lanzar_algoritmo},
    {"name": "estado_trabajos",
     "description": "Lista todos los trabajos (corriendo, terminado, error) y el presupuesto.",
     "input_schema": {"type": "object", "properties": {}},
     "fn": estado_trabajos},
    {"name": "esperar",
     "description": "Bloquea hasta que alguno de los trabajos indicados termine o pasen 'minutos' "
                    "(máx. 60). Úsala en vez de consultar el estado una y otra vez.",
     "input_schema": {"type": "object", "properties": {
         "minutos": {"type": "number", "minimum": 1, "maximum": 60},
         "job_ids": {"type": "array", "items": {"type": "string"},
                     "description": "Omitir = todos los que están corriendo."}}},
     "fn": esperar},
    {"name": "resultado_trabajo",
     "description": "Resultado resumido de un trabajo terminado: media, sd e IC95 de TTS y "
                    "atenciones (evaluación) o incumbente y su re-evaluación (algoritmo).",
     "input_schema": {"type": "object", "properties": {"job_id": {"type": "string"}},
                      "required": ["job_id"]},
     "fn": resultado_trabajo},
    {"name": "comparar_evaluaciones",
     "description": "Compara dos evaluaciones terminadas de forma pareada (mismas semillas): "
                    "diferencia B-A en TTS y atenciones con IC95 y Wilcoxon. Es la forma correcta "
                    "de decidir si una configuración es mejor que otra.",
     "input_schema": {"type": "object", "properties": {
         "job_a": {"type": "string", "description": "Referencia (p. ej. baseline)."},
         "job_b": {"type": "string", "description": "Candidata."}},
         "required": ["job_a", "job_b"]},
     "fn": comparar_evaluaciones},
    {"name": "cancelar_trabajo",
     "description": "Detiene un trabajo que está corriendo.",
     "input_schema": {"type": "object", "properties": {"job_id": {"type": "string"}},
                      "required": ["job_id"]},
     "fn": cancelar_trabajo},
]


def ejecutar(nombre: str, args: dict) -> tuple[str, bool]:
    """Ejecuta una herramienta por nombre. Devuelve (texto_json, es_error)."""
    fn = {h["name"]: h["fn"] for h in HERRAMIENTAS}.get(nombre)
    if fn is None:
        return json.dumps({"error": f"Herramienta desconocida: {nombre}"}), True
    try:
        return json.dumps(fn(args or {}), ensure_ascii=False, default=str), False
    except ErrorHerramienta as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False), True
    except Exception as e:   # un bug nuestro: que el modelo lo vea y no se caiga el loop
        return json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False), True
