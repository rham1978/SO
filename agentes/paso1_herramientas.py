"""
Paso 1 — Las herramientas, probadas SIN modelo ni API key.

Hace a mano lo que después hará el agente: evalúa el baseline y una
candidata con las mismas semillas y las compara de forma pareada.

Por defecto simula solo 8 semanas para que termine en minutos; eso sirve
para comprobar que todo funciona, NO para sacar conclusiones.

    python3 agentes/paso1_herramientas.py            # 8 semanas, 3 réplicas
    AGENTE_SEMANAS=52 python3 agentes/paso1_herramientas.py   # horizonte real
"""

import json
import os
import sys
import tempfile

os.environ.setdefault("AGENTE_SEMANAS", "8")
os.environ.setdefault("AGENTE_DIR_TRABAJOS", tempfile.mkdtemp(prefix="trabajos_prueba_"))
os.environ.setdefault("AGENTE_PRESUPUESTO_REPLICAS", "10")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from herramientas import ejecutar  # noqa: E402


def llamar(nombre, args=None):
    texto, es_error = ejecutar(nombre, args or {})
    print(f"\n>>> {nombre}({json.dumps(args or {}, ensure_ascii=False)})"
          f"{'  [ERROR]' if es_error else ''}\n{texto[:700]}")
    return json.loads(texto), es_error


if __name__ == "__main__":
    print("Trabajos en:", os.environ["AGENTE_DIR_TRABAJOS"], "| semanas:", os.environ["AGENTE_SEMANAS"])

    llamar("ver_espacio")

    # Errores esperados: el modelo los recibe como mensaje y corrige su llamada.
    _, err1 = llamar("lanzar_evaluacion", {"config": {"num_matronas": 9}})
    _, err2 = llamar("lanzar_evaluacion", {"config": {"parametro_inventado": 1}})
    assert err1 and err2, "la validación debería rechazar estas configuraciones"

    a, _ = llamar("lanzar_evaluacion", {"config": {}, "n_reps": 3, "etiqueta": "baseline"})
    b, _ = llamar("lanzar_evaluacion", {"config": {"horas_especialista_1ra": 22}, "n_reps": 3,
                                        "etiqueta": "+6 slots 1ra"})

    # Tercer trabajo: debe chocar con el máximo de simultáneos o con el presupuesto (10).
    _, err3 = llamar("lanzar_evaluacion", {"config": {}, "n_reps": 5})
    assert err3, "debería rechazar por simultáneos o presupuesto"

    pendientes = {a["job_id"], b["job_id"]}
    while pendientes:
        r, _ = llamar("esperar", {"minutos": 30, "job_ids": sorted(pendientes)})
        pendientes = {j for j, e in r["estados"].items() if e == "corriendo"}

    ra, _ = llamar("resultado_trabajo", {"job_id": a["job_id"]})
    rb, _ = llamar("resultado_trabajo", {"job_id": b["job_id"]})
    assert ra["estado"] == rb["estado"] == "terminado", "las evaluaciones deberían terminar"
    comp, err4 = llamar("comparar_evaluaciones", {"job_a": a["job_id"], "job_b": b["job_id"]})
    assert not err4
    llamar("estado_trabajos")
    print("\nOK: herramientas funcionando.")
