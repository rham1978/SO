"""
Paso 4 — Orquestador + subagentes.

Problema del paso 3: esperar un algoritmo que tarda horas llena el contexto del
agente con llamadas a esperar/estado, y todo pasa por un solo hilo de razonamiento.

Solución: separar roles.
  - Orquestador: planifica, reparte trabajo, compara y concluye. No corre nada él mismo.
  - "corredor": un subagente por algoritmo. Lanza, espera, y devuelve un resumen corto.
  - "evaluador": confirma candidatos contra el baseline con evaluaciones pareadas.

Cada subagente trabaja con su propio contexto y solo le devuelve al orquestador
su conclusión. Es la misma idea que tu ProcessPoolExecutor: workers aislados y un
proceso que agrega, solo que los workers razonan.

Los límites (presupuesto, máximo de trabajos simultáneos, validación) siguen en
herramientas.py y aplican a todos los agentes por igual.

Uso:
    nohup python3 agentes/paso4_multiagente.py > multiagente.log 2>&1 &
"""

import asyncio
import os
import sys

from claude_agent_sdk import AgentDefinition, HookMatcher

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paso3_agente_sdk import (MODELO, SERVIDOR, SISTEMA, correr, hook_bitacora,  # noqa: E402
                              opciones_base)


def mcp(*nombres):
    return [f"mcp__{SERVIDOR}__{n}" for n in nombres]


SUBAGENTES = {
    "corredor": AgentDefinition(
        description=("Corre UN algoritmo de optimización del benchmark (M4, M7, M8, M10, M11, M13, "
                     "SA o RS), espera a que termine aunque tarde horas y devuelve su incumbente. "
                     "Lanza un corredor por algoritmo."),
        prompt=("Recibes un módulo y su presupuesto (n_trials, r_final). Lanza el algoritmo con "
                "lanzar_algoritmo. Si responde que hay demasiados trabajos simultáneos, usa esperar "
                "y reintenta. Luego llama esperar con minutos=60 hasta que termine. Lee "
                "resultado_trabajo y devuelve SOLO: módulo, job_id, incumbente (los 12 parámetros), "
                "TTS y atenciones re-evaluados con IC95, evaluaciones usadas y duración. "
                "Si falla, devuelve el error y no reintentes con más presupuesto."),
        tools=mcp("lanzar_algoritmo", "esperar", "resultado_trabajo", "estado_trabajos"),
        model=MODELO,
    ),
    "evaluador": AgentDefinition(
        description=("Confirma configuraciones candidatas contra el baseline con evaluaciones "
                     "pareadas (mismas semillas). Pásale las configuraciones y las réplicas."),
        prompt=("Evalúa el baseline (config vacía) y cada candidata con lanzar_evaluacion usando "
                "el mismo n_reps. Respeta el máximo de trabajos simultáneos usando esperar. "
                "Compara cada candidata contra el baseline con comparar_evaluaciones. Devuelve "
                "una tabla: candidata, diferencia de TTS y de atenciones (media e IC95), p de "
                "Wilcoxon, y si la mejora es significativa (IC95 excluye 0)."),
        tools=mcp("lanzar_evaluacion", "esperar", "resultado_trabajo",
                  "comparar_evaluaciones", "estado_trabajos"),
        model=MODELO,
    ),
}

SISTEMA_ORQUESTADOR = SISTEMA + """

Eres el ORQUESTADOR. No lanzas simulaciones tú mismo: delegas con la herramienta Agent.
- Para cada algoritmo, lanza un subagente 'corredor'. Puedes lanzar varios a la vez,
  pero solo hay 2 trabajos simultáneos disponibles: los corredores esperarán su turno.
- Cuando tengas los incumbentes, pásalos juntos a un subagente 'evaluador'.
- Tú solo consultas ver_espacio, estado_trabajos y comparar_evaluaciones, y escribes
  la conclusión final. Si intentas lanzar algo directamente, un hook lo bloqueará."""

TAREA = """Compara tres algoritmos para mejorar la configuración del baseline:
M4 (SMAC-GP+EI), M11 (ASTRO-DF) y SA, cada uno con n_trials=40 y r_final=6.
Después confirma los tres incumbentes contra el baseline con 10 réplicas pareadas
y recomienda una configuración."""


async def hook_solo_subagentes_lanzan(entrada, tool_use_id, contexto):
    """El prompt le pide al orquestador no lanzar simulaciones; este hook lo GARANTIZA.

    allowed_tools da permiso a toda la sesión (orquestador incluido), porque los
    subagentes heredan esos permisos. Para separar roles de verdad se mira quién
    llama: 'agent_id' solo viene cuando la llamada sale de un subagente.
    """
    if entrada["tool_name"].split("__")[-1].startswith("lanzar_") and "agent_id" not in entrada:
        return {"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": "El orquestador no lanza simulaciones: delega en 'corredor' o 'evaluador'.",
        }}
    return {}


if __name__ == "__main__":
    tarea = sys.argv[1] if len(sys.argv) > 1 else TAREA
    opciones = opciones_base(
        hooks={"PreToolUse": [HookMatcher(matcher=None, hooks=[hook_bitacora]),
                              HookMatcher(matcher=None, hooks=[hook_solo_subagentes_lanzan])]},
        system_prompt=SISTEMA_ORQUESTADOR,
        agents=SUBAGENTES,
        tools=["Agent"],                                   # única herramienta de sistema: delegar
        allowed_tools=["Agent"] + mcp("ver_espacio", "lanzar_evaluacion", "lanzar_algoritmo",
                                      "estado_trabajos", "esperar", "resultado_trabajo",
                                      "comparar_evaluaciones", "cancelar_trabajo"),
        max_budget_usd=float(os.environ.get("AGENTE_MAX_USD", 10)),
    )
    asyncio.run(correr(tarea, opciones))
