"""
Catálogo de acciones: configuraciones semanales de cupos con presupuesto en cupos.

El CRS gestiona cupos, no horas. Por eso el presupuesto se expresa en cupos por
semana y se mantiene constante dentro de cada equipo; el agente solo decide cómo
repartirlos:

  Especialista (56 cupos/sem):  1ra consulta  (fixed_weekly_capacity)
                                + control post (fixed_post_control_capacity)
  Exámenes    (75 cupos/sem):  ecografía UGD + matrona (ugd_us_per_week, mat_us_per_week)
                                + laboratorio matrona (mat_lab_per_week)

Nota: el laboratorio UGD (ugd_lab_per_week) se genera en el simulador pero ningún
paciente lo reserva; por eso la palanca de laboratorio controla mat_lab_per_week.

La cirugía queda FUERA de la acción: el CRS la resuelve como una caja negra con
~2,5 meses de demora (crs_lead_days 60–90 en el simulador), con su capacidad
base fija (surg_throughput_per_week = 20).

Acción = (d, e): se mueven d cupos de control a 1ra consulta y e cupos de
laboratorio a ecografía. Todas las acciones gastan exactamente el mismo
presupuesto de cupos que la operación base.

Regla clínica (Raúl, 24/11/2026): por cada cupo de 1ra consulta debe haber AL MENOS
3 cupos de control (control ≥ 3·1ra). Con 56 cupos de especialista, la 1ra queda
entre 8 y 14 (pasos de 2) → 4 × 7 = 28 acciones. La operación actual (16/40) NO
cumple la regla, así que se evalúa como referencia externa (OPERACION_ACTUAL).
"""
import itertools

BASE = {"first": 16, "control": 40, "eco": 50, "lab": 25}
CUPOS_ESPECIALISTA = BASE["first"] + BASE["control"]  # 56
CUPOS_EXAMENES = BASE["eco"] + BASE["lab"]            # 75
SURG_FIJO = 20

DESPLAZ_ESPECIALISTA = (-8, -6, -4, -2)              # 1ra consulta 8..14, control 48..42
RATIO_MIN_CONTROL = 3                                 # control ≥ 3 × 1ra
DESPLAZ_EXAMENES = (-9, -6, -3, 0, 3, 6, 9)           # ecografía 41..59, laboratorio 34..16
PALANCAS = ("first", "control", "eco", "lab")


def _construir():
    cat = []
    for d, e in itertools.product(DESPLAZ_ESPECIALISTA, DESPLAZ_EXAMENES):
        cat.append({"first": BASE["first"] + d, "control": BASE["control"] - d,
                    "eco": BASE["eco"] + e, "lab": BASE["lab"] - e})
    assert all(c["control"] >= RATIO_MIN_CONTROL * c["first"] for c in cat)
    return cat


CATALOGO = _construir()
# Acción factible más cercana a la operación actual (14/42/50/25); solo como acción "neutra"
ACCION_BASE = CATALOGO.index({"first": 14, "control": 42, "eco": BASE["eco"], "lab": BASE["lab"]})
# Operación actual del CRS (16/40/50/25, cirugía 20) como configuración fija del simulador
OPERACION_ACTUAL = {"fixed_weekly_capacity": BASE["first"], "fixed_post_control_capacity": BASE["control"],
                    "ugd_us_per_week": BASE["eco"] // 2, "mat_us_per_week": BASE["eco"] - BASE["eco"] // 2,
                    "mat_lab_per_week": BASE["lab"], "surg_throughput_per_week": SURG_FIJO}
N_ACCIONES = len(CATALOGO)


def aplicar_accion(cfg, a: int) -> None:
    c = CATALOGO[a]
    cfg.fixed_weekly_capacity = c["first"]
    cfg.fixed_post_control_capacity = c["control"]
    cfg.ugd_us_per_week = c["eco"] // 2
    cfg.mat_us_per_week = c["eco"] - c["eco"] // 2
    cfg.mat_lab_per_week = c["lab"]
    cfg.surg_throughput_per_week = SURG_FIJO


if __name__ == "__main__":
    print(f"{N_ACCIONES} acciones; base = {ACCION_BASE}; "
          f"presupuesto especialista {CUPOS_ESPECIALISTA}, exámenes {CUPOS_EXAMENES} cupos/sem")
    for i, c in enumerate(CATALOGO):
        print(i, c)
