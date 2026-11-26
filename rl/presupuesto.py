"""
Presupuesto de simulación común a todos los métodos (comentario del profesor).

La unidad de costo es un AÑO SIMULADO (un episodio de 52 semanas). Cada método,
en cada inicialización, recibe a lo sumo B años simulados entre entrenamiento y
selección en validación. La evaluación final en prueba (30 semillas) no cuenta,
porque es idéntica para todos.

  RL (REINFORCE, PPO, Double DQN):
      entrenamiento   B_TRAIN = B - N_CHECK * N_EVAL
      selección       N_CHECK puntos de control × N_EVAL semillas de validación
  Mejor configuración fija (búsqueda exhaustiva en el catálogo):
      las 28 acciones × las 100 semillas de validación = 2 800 ≤ B
  SMAC-SK (smac_sk_presupuesto.py): N_TRIALS × N_REPS + TOP_K × (100 − N_REPS) ≤ B
"""
from catalogo import N_ACCIONES

B = 3000                      # años simulados por método e inicialización
N_EVAL = 20                   # semillas de validación por punto de control
N_CHECK = 12                  # puntos de control
B_TRAIN = B - N_CHECK * N_EVAL  # 2760 episodios de entrenamiento
EVAL_CADA = B_TRAIN // N_CHECK  # 230

N_TAMIZ = 100                 # todas las semillas de validación (28 × 100 = 2 800)
TOP_K = 0
B_FIJA = N_ACCIONES * N_TAMIZ + TOP_K * (100 - N_TAMIZ)

assert B_FIJA <= B, B_FIJA

# Corrida pequeña (humo): mismo código, presupuesto reducido ~1/30
HUMO = dict(B=100, N_EVAL=4, N_CHECK=2, N_TAMIZ=2, TOP_K=3, N_CONF=4)


def nucleos() -> int:
    """Procesos en paralelo: variable de entorno RL_NUCLEOS o todos los núcleos."""
    import os
    return int(os.environ.get("RL_NUCLEOS", 0)) or os.cpu_count()


def guardar_json(obj, path, **kw):
    """Escritura atómica: si el proceso se corta, el archivo queda completo o no existe."""
    import json, os
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, **kw)
    os.replace(tmp, path)
