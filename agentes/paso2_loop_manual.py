"""
Paso 2 — El loop de un agente, escrito a mano (API de Anthropic, sin frameworks).

Todo agente es este ciclo:

    1. Le mandas al modelo: instrucciones + historial + catálogo de herramientas.
    2. El modelo responde con texto y/o pedidos de herramienta (tool_use).
    3. Tu código ejecuta cada herramienta y devuelve el resultado (tool_result).
    4. Repites hasta que el modelo responde sin pedir herramientas (stop_reason != "tool_use").

El modelo nunca ejecuta nada: solo pide. Quien ejecuta es tu código
(herramientas.ejecutar), y por eso los límites duros viven ahí.

Requisitos:
    pip install anthropic
    export ANTHROPIC_API_KEY=...        # de console.anthropic.com (se cobra aparte)

Uso (prueba de humo, 8 semanas):
    AGENTE_SEMANAS=8 python3 agentes/paso2_loop_manual.py
"""

import json
import os
import sys

import anthropic

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from herramientas import HERRAMIENTAS, ejecutar  # noqa: E402

MODELO = os.environ.get("AGENTE_MODELO", "claude-sonnet-5-5")
MAX_VUELTAS = 30   # tope del loop: un agente sin tope puede girar indefinidamente

SISTEMA = """Eres un asistente de experimentación para un modelo de simulación de eventos
discretos (DES) de una unidad de ginecología. Trabajas solo con tus herramientas.
Las simulaciones corren en segundo plano: lanza, usa esperar, y luego lee el resultado.
Para decidir si una configuración es mejor que otra usa comparar_evaluaciones
(pareada, mismas semillas), nunca compares medias sueltas.
Responde en español, breve, con números."""

TAREA = """Evalúa el baseline y una configuración con 20 slots semanales de primera
consulta (horas_especialista_1ra=20), 6 réplicas cada una, compáralas y dime si
el cambio reduce el tiempo en sistema y si atiende más pacientes."""


def main():
    cliente = anthropic.Anthropic()
    # El catálogo que ve el modelo: solo nombre, descripción y esquema (no la función).
    catalogo = [{k: h[k] for k in ("name", "description", "input_schema")} for h in HERRAMIENTAS]
    mensajes = [{"role": "user", "content": TAREA}]

    for vuelta in range(1, MAX_VUELTAS + 1):
        resp = cliente.messages.create(model=MODELO, max_tokens=2000, system=SISTEMA,
                                       tools=catalogo, messages=mensajes)
        # 1) Guardar la respuesta completa del modelo en el historial.
        mensajes.append({"role": "assistant", "content": resp.content})
        for bloque in resp.content:
            if bloque.type == "text" and bloque.text.strip():
                print(f"\n[modelo] {bloque.text}")

        # 2) ¿Terminó? Si no pidió herramientas, esa es su respuesta final.
        if resp.stop_reason != "tool_use":
            print(f"\n— fin en {vuelta} vueltas —")
            return

        # 3) Ejecutar cada herramienta pedida y devolver TODOS los resultados juntos.
        resultados = []
        for bloque in resp.content:
            if bloque.type == "tool_use":
                print(f"\n[herramienta] {bloque.name}({json.dumps(bloque.input, ensure_ascii=False)})")
                texto, es_error = ejecutar(bloque.name, bloque.input)
                print(f"   -> {texto[:300]}")
                resultados.append({"type": "tool_result", "tool_use_id": bloque.id,
                                   "content": texto, "is_error": es_error})
        mensajes.append({"role": "user", "content": resultados})

    print(f"\n— se alcanzó el máximo de {MAX_VUELTAS} vueltas sin respuesta final —")


if __name__ == "__main__":
    main()
