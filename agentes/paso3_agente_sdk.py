"""
Paso 3 — El mismo agente con el Claude Agent SDK.

El paso 2 te mostró el loop. Aquí el SDK lo maneja por ti (es el mismo motor
de Claude Code) y tú te concentras en tres cosas:

  1. Herramientas: las MISMAS de herramientas.py, envueltas como servidor MCP en proceso.
  2. Instrucciones: un protocolo de investigación en el system prompt.
  3. Controles: sin herramientas de sistema (no toca archivos ni terminal),
     tope de turnos y de costo, y un hook que registra y filtra cada llamada.

Requisitos:
    pip install claude-agent-sdk      # trae su propio Claude Code
    export ANTHROPIC_API_KEY=...       # recomendado; consumo aparte de tu plan

Uso:
    AGENTE_SEMANAS=8 python3 agentes/paso3_agente_sdk.py           # prueba de humo
    nohup python3 agentes/paso3_agente_sdk.py "tu tarea" > agente.log 2>&1 &   # real, en Pelluhue
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from claude_agent_sdk import (AssistantMessage, ClaudeAgentOptions, ClaudeSDKClient,
                              HookMatcher, ResultMessage, TextBlock, ToolUseBlock,
                              create_sdk_mcp_server, tool)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from herramientas import DIR_TRABAJOS, HERRAMIENTAS, ejecutar  # noqa: E402

MODELO = os.environ.get("AGENTE_MODELO", "claude-sonnet-5-5")
SERVIDOR = "des"                                   # las herramientas se llaman mcp__des__<nombre>
NOMBRES_MCP = [f"mcp__{SERVIDOR}__{h['name']}" for h in HERRAMIENTAS]
MAX_TRIALS_SIN_PERMISO = int(os.environ.get("AGENTE_MAX_TRIALS", 150))

SISTEMA = """Eres el asistente de experimentación de un doctorando en simulación-optimización.
El sistema es un DES (SimPy) de una unidad de ginecología mínimamente invasiva; las
decisiones son 12 parámetros de capacidad y operación; el KPI principal es el tiempo
total en sistema (tts_full_days, menor es mejor) y el secundario las atenciones totales
(mayor es mejor).

Protocolo:
1. Empieza con ver_espacio. Planifica cuánto presupuesto de réplicas usarás en cada fase
   y dilo antes de lanzar nada.
2. Las corridas son lentas y corren en segundo plano: lanza, luego usa esperar (no
   consultes el estado en bucle), luego resultado_trabajo.
3. Un algoritmo devuelve un incumbente ya re-evaluado, pero con semillas distintas a las
   tuyas. Para comparar candidatos entre sí o contra el baseline, evalúa cada uno con
   lanzar_evaluacion (mismas semillas) y usa comparar_evaluaciones. Nunca declares una
   mejora comparando medias sueltas.
4. Una mejora es real solo si el IC95 de la diferencia pareada excluye 0. Con pocas
   réplicas, dilo como indicio, no como conclusión.
5. Si una herramienta devuelve error, léelo y corrige; si el presupuesto no alcanza,
   termina con lo que tienes y dilo.
6. Termina con: tabla de configuraciones evaluadas (TTS y atenciones con IC95),
   recomendación, nivel de evidencia y qué correrías después con más presupuesto.
Escribe en español, sin relleno."""

TAREA_DEFECTO = """Busca una configuración mejor que el baseline. Corre dos algoritmos
distintos (uno bayesiano y uno de región de confianza) con n_trials=40 y r_final=6,
luego confirma sus incumbentes contra el baseline con evaluaciones pareadas de 10 réplicas."""


# ─── 1. Herramientas: envolver las funciones existentes ─────────────────────

def _envolver(h: dict):
    @tool(h["name"], h["description"], h["input_schema"])
    async def _fn(args: dict) -> dict:
        # esperar() bloquea hasta 60 min: se ejecuta en un hilo para no congelar el loop async.
        texto, es_error = await asyncio.to_thread(ejecutar, h["name"], args)
        salida = {"content": [{"type": "text", "text": texto}]}
        if es_error:
            salida["is_error"] = True
        return salida
    return _fn


def crear_servidor():
    return create_sdk_mcp_server(name=SERVIDOR, version="1.0.0",
                                 tools=[_envolver(h) for h in HERRAMIENTAS])


# ─── 3. Controles: hook que corre ANTES de cada herramienta ─────────────────

async def hook_bitacora(entrada, tool_use_id, contexto):
    """Registra cada llamada en bitacora.jsonl y bloquea algoritmos demasiado grandes.

    Un hook es código tuyo, determinista: el modelo no puede saltárselo.
    """
    DIR_TRABAJOS.mkdir(parents=True, exist_ok=True)
    with open(DIR_TRABAJOS / "bitacora.jsonl", "a") as f:
        f.write(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "agente": entrada.get("agent_type", "principal"),
                            "herramienta": entrada["tool_name"],
                            "entrada": entrada["tool_input"]}, ensure_ascii=False) + "\n")

    if entrada["tool_name"].endswith("__lanzar_algoritmo"):
        n = int(entrada["tool_input"].get("n_trials", 30))
        if n > MAX_TRIALS_SIN_PERMISO:
            return {"hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": (
                    f"n_trials={n} supera el máximo autorizado ({MAX_TRIALS_SIN_PERMISO}). "
                    "Usa un presupuesto menor."),
            }}
    return {}


def opciones_base(**extra) -> ClaudeAgentOptions:
    base = dict(
        model=MODELO,
        system_prompt=SISTEMA,
        mcp_servers={SERVIDOR: crear_servidor()},
        tools=[],                       # sin Bash/Read/Edit: solo puede usar TUS herramientas
        allowed_tools=NOMBRES_MCP,      # se ejecutan sin pedir confirmación
        max_turns=80,
        max_budget_usd=float(os.environ.get("AGENTE_MAX_USD", 5)),
        hooks={"PreToolUse": [HookMatcher(matcher=None, hooks=[hook_bitacora])]},
        # esperar() puede durar 60 min; subir el límite por llamada a herramienta (ms).
        env={"MCP_TOOL_TIMEOUT": str(65 * 60 * 1000)},
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    base.update(extra)
    return ClaudeAgentOptions(**base)


# ─── 2. Correr: el SDK hace el loop; tú solo miras los mensajes ─────────────

async def correr(tarea: str, opciones: ClaudeAgentOptions) -> None:
    async with ClaudeSDKClient(options=opciones) as cliente:
        await cliente.query(tarea)
        async for msg in cliente.receive_response():
            if isinstance(msg, AssistantMessage):
                for b in msg.content:
                    if isinstance(b, TextBlock) and b.text.strip():
                        print(f"\n[agente] {b.text}", flush=True)
                    elif isinstance(b, ToolUseBlock):
                        print(f"[→ {b.name.split('__')[-1]}] {json.dumps(b.input, ensure_ascii=False)[:200]}",
                              flush=True)
            elif isinstance(msg, ResultMessage):
                costo = f"US$ {msg.total_cost_usd:.2f}" if msg.total_cost_usd is not None else "n/d"
                print(f"\n— fin: {msg.subtype} | turnos {msg.num_turns} | costo {costo} —", flush=True)


if __name__ == "__main__":
    tarea = sys.argv[1] if len(sys.argv) > 1 else TAREA_DEFECTO
    asyncio.run(correr(tarea, opciones_base()))
