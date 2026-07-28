#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
ORQUESTADOR — correr todo en tu PC personal (Linux / macOS / Windows)
================================================================================
Ejecuta, en orden, el flujo completo sobre el indicador de los módulos
(TTS = tiempo medio en sistema, en días):

  PASO 1  Benchmark de los módulos nuevos:  M4RF (SMAC+RF) y M11 (ASTRO-DF)
          → escribe en la MISMA carpeta de resultados (conserva los previos).
  PASO 2  Análisis de los módulos: tablas + gráficas de CONVERGENCIA y tiempo.
  PASO 3  Comparación MANUAL vs ÓPTIMO: μ±σ, IC95, Welch ANOVA / Kruskal /
          Levene + Welch pareado(Holm), DECISIÓN de negocio, variables de
          decisión por opción y Pareto TTS–atenciones.
  PASO 4  (opcional) MÓDULO 1: nº de corridas recomendado + IC del TTS.

REQUISITOS
  - Python 3.10+  y  pip install -r requirements_analisis.txt
    (scikit-learn==1.6.1 es OBLIGATORIO para el Random Forest de M4RF)
  - Tener tus resultados previos en la carpeta OUT_RES (M4/M7/M8/M10/M13).

USO
  python correr_en_mi_pc.py                 # TODO: benchmark M4RF+M11 → comparar → PPT
  python correr_en_mi_pc.py --pasos 3 5     # solo comparación + PPT (usa resultados ya hechos)
  python correr_en_mi_pc.py --pasos 1       # solo generar M4RF+M11 (optimización, horas)
  python correr_en_mi_pc.py --reeval --r 50 # comparación rápida (modelos del JSON, sin re-simular)
  python correr_en_mi_pc.py --con-modulo1   # incluye también el PASO 4 (nº de corridas)
================================================================================
"""
import argparse
import os
import subprocess
import sys
import time

PY = sys.executable
DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(DIR)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN (ajústala a tu PC)
# ──────────────────────────────────────────────────────────────────────────────
OUT_RES   = "pipeline_out/resultados"        # carpeta con TUS resultados previos
OUT_CMP   = "comparacion_manual"             # salida de la comparación
N_SEEDS   = 15                               # macro-réplicas (igual que el lote previo)
N_TRIALS  = 150                              # presupuesto de evaluaciones por seed
R_FINAL   = 50                               # réplicas de reevaluación del incumbente
R_COMPARA = 50                               # réplicas (CRN) en la comparación manual
TIMEOUT_S = 900                              # timeout por réplica [s] (anti-cuelgue)
MODULOS_NUEVOS = ["M4RF", "M11"]             # SMAC+RF y ASTRO-DF
N_WORKERS = max(1, (os.cpu_count() or 2) - 1)  # núcleos − 1


# ──────────────────────────────────────────────────────────────────────────────
def run(desc, cmd):
    print("\n" + "═" * 78)
    print(f"▶ {desc}")
    print("  " + " ".join(str(c) for c in cmd))
    print("═" * 78, flush=True)
    t0 = time.time()
    r = subprocess.run(cmd)
    dt = time.time() - t0
    if r.returncode != 0:
        print(f"✗ FALLÓ (código {r.returncode}) tras {dt/60:.1f} min: {desc}")
        sys.exit(r.returncode)
    print(f"✓ OK ({dt/60:.1f} min): {desc}")


def chequear_deps():
    faltan = []
    for m in ("numpy", "scipy", "matplotlib", "simpy"):
        try:
            __import__(m)
        except Exception:
            faltan.append(m)
    if faltan:
        print(f"✗ Faltan dependencias: {', '.join(faltan)}")
        print("  → pip install -r requirements_analisis.txt")
        sys.exit(3)
    # RF de SMAC solo es necesario para el PASO 1 (M4RF)
    try:
        import sklearn
        from sklearn.tree._tree import DTYPE            # falla en sklearn ≥1.7
        from sklearn.utils.validation import validate_data  # falla en sklearn <1.6
        print(f"✓ deps OK (scikit-learn {sklearn.__version__} compatible con RF de SMAC)")
    except Exception as e:
        print(f"⚠ scikit-learn incompatible con el RF de SMAC ({e}).")
        print("  El PASO 1 (M4RF) fallará. → pip install 'scikit-learn==1.6.1'")


def paso1_benchmark():
    run("PASO 1 — Benchmark M4RF (SMAC+RF) + M11 (ASTRO-DF)",
        [PY, "benchmark_riguroso.py",
         "--modulos", *MODULOS_NUEVOS,
         "--n_seeds", str(N_SEEDS),
         "--n_trials", str(N_TRIALS),
         "--r_final", str(R_FINAL),
         "--n_cores", str(N_WORKERS),
         "--resume",
         "--out", OUT_RES])


def paso2_analisis():
    run("PASO 2 — Análisis de módulos (tablas + convergencia)",
        [PY, "analisis_salidas.py", "--res", OUT_RES, "--out", "analisis_out"])


def paso3_comparar(r, reeval=False):
    cmd = [PY, "comparar_manual_vs_optimo.py",
           "--res", OUT_RES,
           "--r", str(r),
           "--timeout", str(TIMEOUT_S),
           "--n_workers", str(N_WORKERS),
           "--out", OUT_CMP]
    if reeval:
        cmd.append("--usar-reeval-modulos")
    modo = "reeval (rápido)" if reeval else "CRN (re-simula todo)"
    run(f"PASO 3 — Comparación manual vs óptimo [{modo}]", cmd)


def paso5_ppt():
    run("PASO 5 — Reporte PPTX (inglés)",
        [PY, "generar_reporte_ppt.py",
         "--json", os.path.join(OUT_CMP, "comparacion_manual.json"),
         "--out", "report_comparison.pptx"])


def paso4_modulo1():
    run("PASO 4 — MÓDULO 1: nº de corridas + IC del TTS",
        [PY, "modulo1_num_corridas.py",
         "--kpi", "tts_full_days_mean", "--n_piloto", "10", "--eps_rel", "0.05"])


def main():
    ap = argparse.ArgumentParser(description="Orquestador para correr en PC personal.")
    ap.add_argument("--pasos", nargs="*", type=int, default=[1, 2, 3, 5],
                    help="pasos a ejecutar (def: 1 2 3 5). "
                         "1=benchmark M4RF+M11, 2=análisis, 3=comparación, 4=MÓDULO 1, 5=PPT.")
    ap.add_argument("--con-modulo1", action="store_true",
                    help="ejecuta también el PASO 4 (MÓDULO 1).")
    ap.add_argument("--reeval", action="store_true",
                    help="comparación rápida: usa el TTS guardado de los módulos "
                         "(no re-simula). Para modelos NUEVOS usa CRN (sin este flag).")
    ap.add_argument("--r", type=int, default=R_COMPARA,
                    help=f"réplicas en la comparación (def {R_COMPARA}; usa 10 para preview).")
    args = ap.parse_args()

    print("=" * 78)
    print("ORQUESTADOR — PC personal")
    print(f"  núcleos detectados: {os.cpu_count()}  →  n_workers={N_WORKERS}")
    print(f"  resultados en: {OUT_RES}")
    print(f"  pasos: {args.pasos}{'  + MÓDULO 1' if args.con_modulo1 else ''}"
          f"  | comparación: {'reeval' if args.reeval else 'CRN'}")
    print("=" * 78)

    chequear_deps()
    if 1 in args.pasos:
        paso1_benchmark()
    if 2 in args.pasos:
        paso2_analisis()
    if 3 in args.pasos:
        paso3_comparar(args.r, reeval=args.reeval)
    if args.con_modulo1 or 4 in args.pasos:
        paso4_modulo1()
    if 5 in args.pasos:
        paso5_ppt()

    print("\n" + "=" * 78)
    print("LISTO. Salidas:")
    print(f"  {OUT_RES}/                 ← JSON por módulo/seed (incluye M4RF, M11)")
    print( "  analisis_out/             ← tablas + convergencia + movimiento de variables")
    print(f"  {OUT_CMP}/       ← manual vs óptimo (tabla, ANOVA, decisión, Pareto)")
    print( "  report_comparison.pptx    ← reporte PPTX (inglés)")
    print("=" * 78)


if __name__ == "__main__":
    main()
