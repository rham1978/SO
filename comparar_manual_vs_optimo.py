#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compara escenarios MANUALES (Current / Management / Mgmt+Cap / Mgmt+Cap v2)
contra los incumbentes ÓPTIMOS del benchmark, sobre el MISMO simulador, las
MISMAS réplicas y las MISMAS seeds (CRN → comparación pareada).

Objetivo (confirmado): TTS = tiempo medio total en sistema [días] (minimizar).
Se reporta además 'total_atenciones' para la vista bi-objetivo (Pareto).

Uso:
    python comparar_manual_vs_optimo.py --res pipeline_out/resultados \
        --r 50 --out comparacion_manual

Salidas:
    - tabla por consola: TTS media±IC, atenciones, Δ vs Current, test pareado
    - <out>/pareto_tts_atenciones.png
    - <out>/comparacion_manual.json
"""
import argparse, json, os, glob, dataclasses
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ──────────────────────────────────────────────────────────────────────────────
# Mapeo variable de decisión → campo de SimConfig  (idéntico a _re_evaluar)
# ──────────────────────────────────────────────────────────────────────────────
def aplicar_incumbente(cfg, inc):
    cfg.fixed_weekly_capacity        = int(round(inc["horas_especialista_1ra"])); cfg.use_fixed_weekly_capacity = True
    cfg.fixed_post_control_capacity  = int(round(inc["horas_control_post"]));     cfg.use_fixed_post_control_hours = True
    cfg.ugd_lab_per_week             = int(round(inc["cupos_laboratorio_ugd"]))
    cfg.mat_us_per_week              = int(round(inc["cupos_ecografia_matrona"]))
    cfg.ugd_us_per_week              = int(round(inc["cupos_ecografia_ugd"]))
    cfg.publish_lead_workdays        = int(round(inc["dias_publicacion"]))
    cfg.blocked_pct                  = float(inc["pct_bloqueo_1ra"])
    cfg.empty_control_p_ugd          = float(inc["pct_consultas_vacias"])
    cfg.matrona_capacity             = int(round(inc["num_matronas"]))
    cfg.agent_capacity               = int(round(inc["num_agentes_ugd"]))
    cfg.not_contactable_p            = float(inc["pct_no_contactabilidad"])
    cfg.blocked_pct_post_control     = float(inc["pct_bloqueo_post_control"])
    return cfg

# ──────────────────────────────────────────────────────────────────────────────
# Escenarios MANUALES (de la tabla/imagen del usuario).
#
# Mapeo tomado del código (no inventado):
#   · variable→campo SimConfig: EscenarioConfig en modulo3_comparacion.py
#     (mat_us_per_week=cupos eco matrona, ugd_us_per_week=cupos eco UGD,
#      ugd_lab_per_week=cupos lab UGD, matrona_capacity=nº matronas).
#   · valores 'Base': PARAM_BASELINE en modulo_comparativa_caja_negra.py.
#   · OJO: el simulador consume 'fixed_post_control_capacity' (línea 2091),
#     NO 'fixed_post_control_hours' (que modulo3 setea pero el sim ignora).
#
#   "Diagnostic resources" — US = ECOGRAFÍA, slots = LABORATORIO (confirmado).
#   Valores ABSOLUTOS (confirmado por el usuario):
#     · "2 mid."       → matrona_capacity = 2
#     · Mgmt+Cap  US   → cupos ecografía = 25  (mat_us_per_week = 25)
#     · Mgmt+Cap v2    → 50 cupos ecografía y 50 cupos laboratorio
#                        (mat_us_per_week = 50, ugd_lab_per_week = 50)
#   'Base' = PARAM_BASELINE (lab 54, eco 25/25, matronas 1).
#   La ecografía se aplica a la capacidad de matrona (mat_us_per_week); si se
#   quisiera también la ecografía UGD, ajustar ugd_us_per_week.
#   pct_no_contactabilidad no está en la tabla → queda en base 0.15 (PARAM_BASELINE).
# ──────────────────────────────────────────────────────────────────────────────

def escenarios_manuales(CFG):
    """Devuelve {nombre: dict de overrides de campos SimConfig}."""
    return {
        # Current = baseline real: capacidad VARIABLE (no fija), defaults
        "Current": dict(use_fixed_weekly_capacity=False, use_fixed_post_control_hours=False,
                        blocked_pct=0.32, publish_lead_workdays=5, agent_capacity=1,
                        blocked_pct_post_control=0.34, empty_control_p_ugd=0.30),
        # Management = mismas capacidades base pero FIJAS + menos bloqueo + lead 7
        "Management": dict(use_fixed_weekly_capacity=True,  fixed_weekly_capacity=16,
                           use_fixed_post_control_hours=True, fixed_post_control_capacity=40,
                           blocked_pct=0.10, publish_lead_workdays=7, agent_capacity=1,
                           blocked_pct_post_control=0.10, empty_control_p_ugd=0.10),
        # Mgmt+Cap = cap 1ra(30) + 2 agentes + 2 matronas + eco 25 (absoluto)
        "Mgmt+Cap": dict(use_fixed_weekly_capacity=True,  fixed_weekly_capacity=30,
                         use_fixed_post_control_hours=True, fixed_post_control_capacity=40,
                         blocked_pct=0.10, publish_lead_workdays=7, agent_capacity=2,
                         blocked_pct_post_control=0.10, empty_control_p_ugd=0.10,
                         matrona_capacity=2, mat_us_per_week=25),
        # Mgmt+Cap v2 = cap 1ra(40, escenario real; excede el rango del optimizador
        # [8,30]) + 3 agentes + postFC 50 + 2 matronas + eco 50 + lab 50 (absoluto)
        "Mgmt+Cap v2": dict(use_fixed_weekly_capacity=True,  fixed_weekly_capacity=40,
                            use_fixed_post_control_hours=True, fixed_post_control_capacity=50,
                            blocked_pct=0.10, publish_lead_workdays=7, agent_capacity=3,
                            blocked_pct_post_control=0.10, empty_control_p_ugd=0.10,
                            matrona_capacity=2, mat_us_per_week=50, ugd_lab_per_week=50),
    }

# ──────────────────────────────────────────────────────────────────────────────
# Evaluación a prueba de cuelgues: cada réplica corre en su propio proceso con
# timeout. Si una simulación se queda pegada (deadlock del DES), se TERMINA el
# proceso y la réplica se marca NaN — nunca cuelga el script.
# ──────────────────────────────────────────────────────────────────────────────
def _worker_eval(q, idx, seed, cfg_dict):
    try:
        from simulador_clinica_baseline import run_once, SimConfig
        cfg = SimConfig(**cfg_dict)
        res = run_once(seed_offset=seed, cfg=cfg)
        q.put((idx, float(res.get("tts_full_days_mean", float("nan"))),
                    float(res.get("total_atenciones", float("nan")))))
    except Exception:
        q.put((idx, float("nan"), float("nan")))


def evaluar_cfg(cfg, r, seed_base, timeout_s=900, n_workers=4):
    """Corre r réplicas (en paralelo, hasta n_workers) con timeout por réplica.
    Devuelve (tts[], at[], n_timeout). Nunca se queda pegado."""
    import dataclasses, multiprocessing as mp, time
    cfg_dict = dataclasses.asdict(cfg)
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    results, running, n_to = {}, {}, 0
    pending = list(range(r))

    def launch(idx):
        p = ctx.Process(target=_worker_eval, args=(q, idx, seed_base + idx, cfg_dict))
        p.daemon = True; p.start(); running[idx] = (p, time.time())

    while pending and len(running) < max(1, n_workers):
        launch(pending.pop(0))

    while running:
        try:
            while True:
                idx, t, a = q.get_nowait(); results[idx] = (t, a)
        except Exception:
            pass
        now = time.time()
        for idx, (p, t0) in list(running.items()):
            timed_out = (now - t0) > timeout_s
            if not p.is_alive() or timed_out:
                if timed_out and p.is_alive():
                    p.terminate(); n_to += 1
                p.join()
                results.setdefault(idx, (float("nan"), float("nan")))
                del running[idx]
                if pending:
                    launch(pending.pop(0))
        time.sleep(0.3)
    # drenar cola final
    try:
        while True:
            idx, t, a = q.get_nowait(); results[idx] = (t, a)
    except Exception:
        pass
    tts = np.array([results.get(i, (np.nan, np.nan))[0] for i in range(r)], float)
    at  = np.array([results.get(i, (np.nan, np.nan))[1] for i in range(r)], float)
    return tts, at, n_to

def _ic(a):
    a = a[np.isfinite(a)]; n = len(a)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    m = float(a.mean())
    if n < 2:
        return m, 0.0, m, m
    s = float(a.std(ddof=1))
    return m, s, m - 1.96*s/np.sqrt(n), m + 1.96*s/np.sqrt(n)

def mejor_incumbente_por_metodo(res_dir):
    """Mejor incumbente (menor reeval/costo) de cada método con datos válidos."""
    best = {}
    for f in glob.glob(os.path.join(res_dir, "resultado_*.json")):
        j = json.load(open(f))
        if not j.get("incumbente"): continue
        c = (j.get("reeval") or {}).get("media", j.get("costo_opt"))
        if c is None or not np.isfinite(c): continue
        m = j["modulo"]
        if m not in best or c < best[m][0]:
            best[m] = (c, j["incumbente"])
    return {m: inc for m, (c, inc) in best.items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="pipeline_out/resultados")
    ap.add_argument("--r", type=int, default=50, help="réplicas (CRN) por escenario")
    ap.add_argument("--seed_base", type=int, default=500_000)
    ap.add_argument("--out", default="comparacion_manual")
    ap.add_argument("--solo_metodos", nargs="*", default=None,
                    help="limitar a estos módulos óptimos (def: todos)")
    ap.add_argument("--timeout", type=float, default=900.0,
                    help="timeout por réplica [s] (anti-deadlock; def 900)")
    ap.add_argument("--n_workers", type=int, default=4,
                    help="procesos paralelos para las réplicas (def 4)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from simulador_clinica_baseline import CFG
    import dataclasses as dc

    filas = []   # (nombre, tipo, tts_arr, at_arr)

    def _eval(cfg):
        return evaluar_cfg(cfg, args.r, args.seed_base,
                           timeout_s=args.timeout, n_workers=args.n_workers)

    # 1) escenarios manuales
    for nombre, ov in escenarios_manuales(CFG).items():
        cfg = dc.replace(CFG); cfg.benchmark_mode = True
        for k, v in ov.items(): setattr(cfg, k, v)
        tts, at, n_to = _eval(cfg)
        filas.append((nombre, "manual", tts, at))
        aviso = f"  ⚠ {n_to} timeouts" if n_to else ""
        print(f"[manual] {nombre:14} TTS={np.nanmean(tts):7.1f}d  "
              f"atenciones={np.nanmean(at):7.0f}{aviso}", flush=True)

    # 2) incumbentes óptimos (mismos seeds → CRN/pareado)
    bestinc = mejor_incumbente_por_metodo(args.res)
    for m, inc in bestinc.items():
        if args.solo_metodos and m not in args.solo_metodos: continue
        cfg = dc.replace(CFG); cfg.benchmark_mode = True
        aplicar_incumbente(cfg, inc)
        tts, at, n_to = _eval(cfg)
        filas.append((f"ÓPTIMO {m}", "optimo", tts, at))
        aviso = f"  ⚠ {n_to} timeouts" if n_to else ""
        print(f"[optimo] {m:14} TTS={np.nanmean(tts):7.1f}d  "
              f"atenciones={np.nanmean(at):7.0f}{aviso}", flush=True)

    # 3) tabla + test pareado vs Current
    cur = next(f for f in filas if f[0] == "Current")[2]
    print("\n" + "="*92)
    print(f"{'Escenario':16}{'TTS μ [IC95]':26}{'atenciones μ':14}{'ΔTTS vs Current':18}{'p (Wilcoxon)':12}")
    print("-"*92)
    salida = []
    for nombre, tipo, tts, at in filas:
        m, s, lo, hi = _ic(tts)
        am = float(np.nanmean(at))
        d = m - float(np.nanmean(cur))
        if nombre == "Current":
            p = float("nan")
        else:
            # pareado solo sobre réplicas válidas en AMBOS (CRN)
            mask = np.isfinite(tts) & np.isfinite(cur)
            try: p = stats.wilcoxon(tts[mask], cur[mask]).pvalue
            except Exception: p = float("nan")
        print(f"{nombre:16}{f'{m:.1f} [{lo:.1f},{hi:.1f}]':26}{am:<14.0f}"
              f"{d:+.1f}{'':10}{p:.4f}")
        salida.append({"escenario": nombre, "tipo": tipo, "tts_media": m,
                       "tts_ic95": [lo, hi], "atenciones_media": am,
                       "delta_tts_vs_current": d, "p_wilcoxon_vs_current": p})
    print("="*92)

    # 4) Pareto TTS vs atenciones
    plt.figure(figsize=(9, 6))
    for nombre, tipo, tts, at in filas:
        mk = "s" if tipo == "manual" else "o"
        c = "#d62728" if tipo == "manual" else "#1f77b4"
        xa, yt = float(np.nanmean(at)), float(np.nanmean(tts))
        plt.scatter(xa, yt, s=90, marker=mk, color=c, zorder=3,
                    edgecolor="k", linewidth=0.6)
        plt.annotate(nombre, (xa, yt), fontsize=8,
                     xytext=(5, 4), textcoords="offset points")
    plt.scatter([], [], marker="s", color="#d62728", label="manual")
    plt.scatter([], [], marker="o", color="#1f77b4", label="óptimo")
    plt.xlabel("Pacientes atendidos (total_atenciones)")
    plt.ylabel("TTS — tiempo medio en sistema [días]  (menor = mejor)")
    plt.title("Frontera tiempo vs atenciones — manual vs óptimo")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    png = os.path.join(args.out, "pareto_tts_atenciones.png")
    plt.savefig(png, dpi=140); plt.close()

    json.dump(salida, open(os.path.join(args.out, "comparacion_manual.json"), "w"),
              indent=2, ensure_ascii=False)
    print(f"\nGráfica: {png}")
    print(f"JSON   : {os.path.join(args.out, 'comparacion_manual.json')}")

if __name__ == "__main__":
    main()
