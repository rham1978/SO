"""
Catálogo de acciones: configuraciones semanales de cupos que respetan un
presupuesto de horas.

Palancas (cupos por semana) y su duración en horas por cupo:
    first   : 1ra consulta especialista        (fixed_weekly_capacity)       20 min
    control : control post-consulta            (fixed_post_control_capacity) 20 min
    eco     : ecografías UGD + matrona (50/50) (ugd_us_per_week, mat_us_per_week) 60 min
    lab     : laboratorio UGD                  (ugd_lab_per_week)            60 min
    surg    : cirugías por semana              (surg_throughput_per_week)    HORAS_CIRUGIA

Cada palanca toma un nivel en NIVELES (multiplicador sobre la operación base).
Se conservan solo las combinaciones cuyo costo en horas no supera el presupuesto
de la operación base (PRESUPUESTO_H) y que usan al menos USO_MIN de él
(descarta configuraciones que dejan horas sin usar). Así todas las políticas, fijas o
adaptativas, gastan como máximo las mismas horas por semana.

SUPUESTO A VALIDAR: HORAS_CIRUGIA (horas de pabellón + equipo por cirugía).
"""
import itertools

HORAS_CIRUGIA = 2.0

BASE = {"first": 16, "control": 40, "eco": 50, "lab": 54, "surg": 20}
HORAS_POR_CUPO = {"first": 20 / 60, "control": 20 / 60, "eco": 1.0, "lab": 1.0, "surg": HORAS_CIRUGIA}
NIVELES = (0.75, 1.0, 1.25)
USO_MIN = 0.95
PALANCAS = tuple(BASE)


def horas(cfg_cupos: dict) -> float:
    return sum(cfg_cupos[k] * HORAS_POR_CUPO[k] for k in PALANCAS)


PRESUPUESTO_H = horas(BASE)


def _construir():
    cat = []
    for mults in itertools.product(NIVELES, repeat=len(PALANCAS)):
        c = {k: int(round(BASE[k] * m)) for k, m in zip(PALANCAS, mults)}
        if USO_MIN * PRESUPUESTO_H - 1e-9 <= horas(c) <= PRESUPUESTO_H + 1e-9:
            cat.append(c)
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
    cfg.surg_throughput_per_week = c["surg"]


if __name__ == "__main__":
    print(f"Presupuesto: {PRESUPUESTO_H:.1f} h/semana; {N_ACCIONES} acciones; base = {ACCION_BASE}")
    for i, c in enumerate(CATALOGO):
        print(i, c, f"{horas(c):.1f} h")
