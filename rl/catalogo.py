"""
Catálogo de acciones: configuraciones semanales de cupos con presupuesto en cupos.

El CRS gestiona cupos, no horas. Por eso el presupuesto se expresa en cupos por
semana y se mantiene constante dentro de cada equipo; el agente solo decide cómo
repartirlos:

  Especialista (56 cupos/sem):  1ra consulta  (fixed_weekly_capacity)
                                + control post (fixed_post_control_capacity)
  Exámenes   (104 cupos/sem):  ecografía UGD + matrona (ugd_us_per_week, mat_us_per_week)
                                + laboratorio UGD (ugd_lab_per_week)

La cirugía queda FUERA de la acción: el CRS la resuelve como una caja negra con
~2,5 meses de demora (crs_lead_days 60–90 en el simulador), con su capacidad
base fija (surg_throughput_per_week = 20).

Acción = (d, e): se mueven d cupos de control a 1ra consulta y e cupos de
laboratorio a ecografía. Todas las acciones gastan exactamente el mismo
presupuesto de cupos que la operación base.
"""
import itertools

BASE = {"first": 16, "control": 40, "eco": 50, "lab": 54}
CUPOS_ESPECIALISTA = BASE["first"] + BASE["control"]  # 56
CUPOS_EXAMENES = BASE["eco"] + BASE["lab"]            # 104
SURG_FIJO = 20

DESPLAZ_ESPECIALISTA = (-8, -4, 0, 4, 8, 12)          # 1ra consulta 8..28, control 48..28
DESPLAZ_EXAMENES = (-18, -12, -6, 0, 6, 12, 18)       # ecografía 32..68, laboratorio 72..36
PALANCAS = ("first", "control", "eco", "lab")


def _construir():
    cat = []
    for d, e in itertools.product(DESPLAZ_ESPECIALISTA, DESPLAZ_EXAMENES):
        cat.append({"first": BASE["first"] + d, "control": BASE["control"] - d,
                    "eco": BASE["eco"] + e, "lab": BASE["lab"] - e})
    return cat


CATALOGO = _construir()
ACCION_BASE = CATALOGO.index(dict(BASE))
N_ACCIONES = len(CATALOGO)


def aplicar_accion(cfg, a: int) -> None:
    c = CATALOGO[a]
    cfg.fixed_weekly_capacity = c["first"]
    cfg.fixed_post_control_capacity = c["control"]
    cfg.ugd_us_per_week = c["eco"] // 2
    cfg.mat_us_per_week = c["eco"] - c["eco"] // 2
    cfg.ugd_lab_per_week = c["lab"]
    cfg.surg_throughput_per_week = SURG_FIJO


if __name__ == "__main__":
    print(f"{N_ACCIONES} acciones; base = {ACCION_BASE}; "
          f"presupuesto especialista {CUPOS_ESPECIALISTA}, exámenes {CUPOS_EXAMENES} cupos/sem")
    for i, c in enumerate(CATALOGO):
        print(i, c)
