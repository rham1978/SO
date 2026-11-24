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
      tamizaje        42 acciones × N_TAMIZ semillas de validación
      confirmación    TOP_K finalistas × (100 - N_TAMIZ) semillas restantes
      total           42·N_TAMIZ + TOP_K·(100 - N_TAMIZ) ≤ B
"""
from catalogo import N_ACCIONES

B = 3000                      # años simulados por método e inicialización
N_EVAL = 20                   # semillas de validación por punto de control
N_CHECK = 12                  # puntos de control
B_TRAIN = B - N_CHECK * N_EVAL  # 2760 episodios de entrenamiento
EVAL_CADA = B_TRAIN // N_CHECK  # 230

N_TAMIZ = 60
TOP_K = 5
B_FIJA = N_ACCIONES * N_TAMIZ + TOP_K * (100 - N_TAMIZ)   # 2720

assert B_FIJA <= B, B_FIJA

# Corrida pequeña (humo): mismo código, presupuesto reducido ~1/30
HUMO = dict(B=100, N_EVAL=4, N_CHECK=2, N_TAMIZ=2, TOP_K=3, N_CONF=4)
