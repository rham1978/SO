#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análisis de las salidas del benchmark riguroso (pipeline_out/resultados).

Genera:
  1.  Tabla resumen por método: objetivo alcanzado (costo_opt + reevaluación
      con IC95), tiempo de ejecución, nº de evaluaciones.
  2.  Análisis de qué variables de decisión se mueven (incumbente vs baseline,
      posición normalizada dentro del rango admisible y dispersión entre seeds).
  3.  Gráficas:
        - convergencia por evaluaciones  (media ± banda entre seeds)
        - convergencia por tiempo         (media entre seeds)
        - boxplot de tiempo de ejecución por método
        - boxplot del objetivo reevaluado por método
        - mapa de movimiento de variables (incumbente normalizado vs baseline)

Función objetivo (minimizar):
        TTS_full_days_mean(x) = tiempo medio TOTAL EN SISTEMA (días),
        de entrada a alta, sobre pacientes que completan la ruta. Se MINIMIZA.

Uso:
    python analisis_salidas.py --res pipeline_out/resultados --out analisis_out
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────────────
# Metadatos
# ──────────────────────────────────────────────────────────────────────────────
ESTILO = {
    "M4":  {"label": "SMAC-BO (GP+EI)","color": "#1f77b4", "ls": "-",  "marker": "o"},
    "M4RF":{"label": "SMAC-RF (RF+EI)","color": "#d62728", "ls": "--", "marker": "*"},
    "M7":  {"label": "SMAC+SK (EI)", "color": "#ff7f0e", "ls": "-",  "marker": "s"},
    "M8":  {"label": "SK Adaptativo","color": "#2ca02c", "ls": "--", "marker": "D"},
    "M10": {"label": "SK-KGCP",      "color": "#9467bd", "ls": "-.", "marker": "^"},
    "M11": {"label": "ASTRO-DF",     "color": "#8c564b", "ls": "-.", "marker": "p"},
    "M13": {"label": "SPSA",         "color": "#7f7f7f", "ls": ":",  "marker": "x"},
    "RS":  {"label": "Random Search","color": "#17becf", "ls": "--", "marker": "P"},
}
ORDEN = ["M4", "M4RF", "M7", "M8", "M10", "M11", "M13", "RS"]

# Variables de decisión: (default/baseline, lower, upper, tipo)
VARIABLES = {
    "horas_especialista_1ra":     (16,   8,   30,   "int"),
    "horas_control_post":         (40,   20,  70,   "int"),
    "cupos_laboratorio_ugd":      (54,   20,  100,  "int"),
    "cupos_ecografia_matrona":    (25,   10,  50,   "int"),
    "cupos_ecografia_ugd":        (25,   10,  50,   "int"),
    "dias_publicacion":           (5,    1,   10,   "int"),
    "num_matronas":               (1,    1,   4,    "int"),
    "num_agentes_ugd":            (1,    1,   4,    "int"),
    "pct_bloqueo_1ra":            (0.32, 0.05, 0.50, "float"),
    "pct_consultas_vacias":       (0.30, 0.05, 0.50, "float"),
    "pct_no_contactabilidad":     (0.30, 0.05, 0.50, "float"),
    "pct_bloqueo_post_control":   (0.34, 0.05, 0.50, "float"),
}
OBJETIVO = "tts_full_days_mean (tiempo medio en sistema, días)"


# ──────────────────────────────────────────────────────────────────────────────
# Carga
# ──────────────────────────────────────────────────────────────────────────────
def _valido(d):
    """Un registro es válido si tiene objetivo finito e incumbente."""
    c = d.get("costo_opt", float("inf"))
    try:
        c = float(c)
    except (TypeError, ValueError):
        return False
    return np.isfinite(c) and bool(d.get("incumbente"))


def cargar(res_dir):
    """Devuelve ({modulo: [registro válido, ...]}, {modulo: n_fallidos})."""
    data = defaultdict(list)
    fallidos = defaultdict(int)
    for fn in sorted(os.listdir(res_dir)):
        if not fn.startswith("resultado_") or not fn.endswith(".json"):
            continue
        with open(os.path.join(res_dir, fn), encoding="utf-8") as f:
            d = json.load(f)
        mod = d.get("modulo")
        if not mod:
            continue
        if _valido(d):
            data[mod].append(d)
        else:
            fallidos[mod] += 1
    return data, fallidos


def label(mod):
    return ESTILO.get(mod, {}).get("label", mod)


def color(mod):
    return ESTILO.get(mod, {}).get("color", "#333333")


# ──────────────────────────────────────────────────────────────────────────────
# 1. Tabla resumen
# ──────────────────────────────────────────────────────────────────────────────
def tabla_resumen(data):
    filas = []
    for mod in ORDEN:
        regs = data.get(mod)
        if not regs:
            continue
        costo = np.array([r["costo_opt"] for r in regs], float)
        # reevaluación honesta (media e IC95 promedio sobre seeds)
        re_media = np.array([r["reeval"]["media"] for r in regs
                             if r.get("reeval")], float)
        re_sd = np.array([r["reeval"]["sd"] for r in regs
                          if r.get("reeval")], float)
        tiempo = np.array([r["tiempo_seg"] for r in regs], float)
        nev = np.array([r.get("n_eval_usadas", np.nan) for r in regs], float)
        filas.append({
            "mod": mod,
            "label": label(mod),
            "n_seeds": len(regs),
            "costo_med": costo.mean(),
            "costo_sd": costo.std(ddof=1) if len(costo) > 1 else 0.0,
            "costo_min": costo.min(),
            "re_med": re_media.mean() if len(re_media) else np.nan,
            "re_sd": re_sd.mean() if len(re_sd) else np.nan,
            "re_best": re_media.min() if len(re_media) else np.nan,
            "tiempo_h": tiempo.mean() / 3600.0,
            "nev": np.nanmean(nev),
        })
    # ordenar por reevaluación (objetivo honesto)
    filas.sort(key=lambda x: (np.isnan(x["re_med"]), x["re_med"]))
    return filas


def imprimir_tabla(filas):
    print("\n" + "=" * 104)
    print("RESUMEN POR MÉTODO  —  objetivo: TTS = tiempo medio en sistema [días] (minimizar)")
    print("=" * 104)
    hdr = (f"{'Método':<16}{'seeds':>6}{'costo_opt μ±σ':>20}"
           f"{'reeval μ (IC honesto)':>24}{'mejor':>9}{'tiempo[h]':>11}{'n_eval':>9}")
    print(hdr)
    print("-" * 104)
    for f in filas:
        costo = f"{f['costo_med']:.2f}±{f['costo_sd']:.2f}"
        if not np.isnan(f["re_med"]):
            reev = f"{f['re_med']:.2f} (±{f['re_sd']:.2f})"
            best = f"{f['re_best']:.2f}"
        else:
            reev, best = "—", "—"
        print(f"{f['label']:<16}{f['n_seeds']:>6}{costo:>20}{reev:>24}"
              f"{best:>9}{f['tiempo_h']:>11.2f}{f['nev']:>9.0f}")
    print("-" * 104)
    mejor = filas[0]
    print(f"➤ Mejor objetivo (reevaluación honesta): {mejor['label']} "
          f"con f={mejor['re_med']:.2f}  (mejor seed: {mejor['re_best']:.2f})")
    print("=" * 104 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# 2. Movimiento de variables
# ──────────────────────────────────────────────────────────────────────────────
def analisis_variables(data):
    """Para cada método: media del incumbente por variable y posición normalizada."""
    tabla = {}        # mod -> {var: (media, sd, pos_norm)}
    for mod in ORDEN:
        regs = data.get(mod)
        if not regs:
            continue
        porvar = {}
        for var, (base, lo, hi, _t) in VARIABLES.items():
            vals = np.array([r["incumbente"].get(var, np.nan)
                             for r in regs if "incumbente" in r], float)
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                continue
            pos = (vals.mean() - lo) / (hi - lo) if hi > lo else 0.0
            porvar[var] = (vals.mean(), vals.std(ddof=1) if len(vals) > 1 else 0.0, pos)
        tabla[mod] = porvar
    return tabla


def imprimir_variables(tabla):
    print("=" * 104)
    print("MOVIMIENTO DE VARIABLES DE DECISIÓN  (media del incumbente entre seeds)")
    print("baseline = valor por defecto del config space   |   Δ% respecto al rango admisible")
    print("=" * 104)
    mods = [m for m in ORDEN if m in tabla]
    hdr = f"{'variable':<26}{'baseline':>10}{'rango':>14}"
    for m in mods:
        hdr += f"{label(m)[:10]:>12}"
    print(hdr)
    print("-" * 104)
    for var, (base, lo, hi, t) in VARIABLES.items():
        fila = f"{var:<26}{base:>10}{f'[{lo},{hi}]':>14}"
        for m in mods:
            v = tabla[m].get(var)
            if v is None:
                fila += f"{'—':>12}"
            else:
                med, sd, pos = v
                fila += f"{med:>12.2f}" if t == "float" else f"{med:>12.1f}"
        print(fila)
    print("-" * 104)

    # ¿Qué variables se mueven más respecto al baseline? (promedio entre métodos)
    print("\nVariables ordenadas por desplazamiento medio respecto al baseline "
          "(|Δ| normalizado al rango):")
    desplaz = []
    for var, (base, lo, hi, t) in VARIABLES.items():
        difs = []
        for m in mods:
            v = tabla[m].get(var)
            if v:
                difs.append((v[0] - base) / (hi - lo))
        if difs:
            desplaz.append((var, float(np.mean(difs)), float(np.mean(np.abs(difs)))))
    desplaz.sort(key=lambda x: -x[2])
    for var, dsigned, dabs in desplaz:
        flecha = "↑" if dsigned > 0 else "↓"
        barra = "█" * int(round(dabs * 40))
        print(f"  {var:<26} {flecha} |Δ|={dabs*100:5.1f}% del rango  {barra}")
    print("=" * 104 + "\n")
    return desplaz


# ──────────────────────────────────────────────────────────────────────────────
# 3. Gráficas
# ──────────────────────────────────────────────────────────────────────────────
def _curva_media(curvas, n_grid=60):
    """Interpola varias curvas (x,y) a una grilla común y devuelve x, media, sd."""
    xs_all = [np.array([p[0] for p in c], float) for c in curvas if len(c) > 1]
    ys_all = [np.array([p[1] for p in c], float) for c in curvas if len(c) > 1]
    if not xs_all:
        return None
    xmin = max(x.min() for x in xs_all)
    xmax = min(x.max() for x in xs_all)
    if xmax <= xmin:
        return None
    grid = np.linspace(xmin, xmax, n_grid)
    Y = []
    for x, y in zip(xs_all, ys_all):
        order = np.argsort(x)
        Y.append(np.interp(grid, x[order], y[order]))
    Y = np.array(Y)
    return grid, Y.mean(axis=0), Y.std(axis=0)


def graf_convergencia_eval(data, out):
    plt.figure(figsize=(10, 6))
    for mod in ORDEN:
        regs = data.get(mod)
        if not regs:
            continue
        curvas = [r["conv_eval"] for r in regs if r.get("conv_eval")]
        m = _curva_media(curvas)
        if m is None:
            continue
        grid, mu, sd = m
        c = color(mod)
        plt.plot(grid, mu, color=c, lw=2, label=label(mod),
                 ls=ESTILO[mod]["ls"])
        plt.fill_between(grid, mu - sd, mu + sd, color=c, alpha=0.12)
    plt.xlabel("Evaluaciones del simulador")
    plt.ylabel("Mejor TTS — tiempo medio en sistema [días]  (menor = mejor)")
    plt.title("Convergencia por evaluaciones (media ± σ entre seeds)")
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    p = os.path.join(out, "convergencia_evaluaciones.png")
    plt.savefig(p, dpi=140)
    plt.close()
    return p


def graf_convergencia_tiempo(data, out):
    plt.figure(figsize=(10, 6))
    for mod in ORDEN:
        regs = data.get(mod)
        if not regs:
            continue
        curvas = [r["conv_time"] for r in regs if r.get("conv_time")]
        # pasar tiempo a horas
        curvas = [[[p[0] / 3600.0, p[1]] for p in c] for c in curvas]
        m = _curva_media(curvas)
        if m is None:
            continue
        grid, mu, sd = m
        c = color(mod)
        plt.plot(grid, mu, color=c, lw=2, label=label(mod), ls=ESTILO[mod]["ls"])
        plt.fill_between(grid, mu - sd, mu + sd, color=c, alpha=0.12)
    plt.xlabel("Tiempo de ejecución [horas]")
    plt.ylabel("Mejor TTS — tiempo medio en sistema [días]  (menor = mejor)")
    plt.title("Convergencia por tiempo de cómputo (media ± σ entre seeds)")
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    p = os.path.join(out, "convergencia_tiempo.png")
    plt.savefig(p, dpi=140)
    plt.close()
    return p


def graf_box_tiempo(data, out):
    mods = [m for m in ORDEN if data.get(m)]
    vals = [[r["tiempo_seg"] / 3600.0 for r in data[m]] for m in mods]
    plt.figure(figsize=(9, 5.5))
    bp = plt.boxplot(vals, tick_labels=[label(m) for m in mods],
                     patch_artist=True, showmeans=True)
    for patch, m in zip(bp["boxes"], mods):
        patch.set_facecolor(color(m))
        patch.set_alpha(0.55)
    plt.ylabel("Tiempo de ejecución por seed [horas]")
    plt.title("Tiempo de ejecución por método")
    plt.xticks(rotation=20)
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    p = os.path.join(out, "tiempo_ejecucion_box.png")
    plt.savefig(p, dpi=140)
    plt.close()
    return p


def graf_box_objetivo(data, out):
    mods = [m for m in ORDEN if data.get(m)]
    vals = [[r["reeval"]["media"] for r in data[m] if r.get("reeval")]
            for m in mods]
    plt.figure(figsize=(9, 5.5))
    bp = plt.boxplot(vals, tick_labels=[label(m) for m in mods],
                     patch_artist=True, showmeans=True)
    for patch, m in zip(bp["boxes"], mods):
        patch.set_facecolor(color(m))
        patch.set_alpha(0.55)
    plt.ylabel("TTS reevaluado — tiempo medio en sistema [días]  (menor = mejor)")
    plt.title("Objetivo alcanzado por método (reevaluación honesta, r=50)")
    plt.xticks(rotation=20)
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    p = os.path.join(out, "objetivo_box.png")
    plt.savefig(p, dpi=140)
    plt.close()
    return p


def graf_variables(tabla, out):
    """Heatmap de posición normalizada del incumbente por variable y método."""
    mods = [m for m in ORDEN if m in tabla]
    varnames = list(VARIABLES.keys())
    M = np.full((len(varnames), len(mods)), np.nan)
    for j, m in enumerate(mods):
        for i, var in enumerate(varnames):
            v = tabla[m].get(var)
            if v:
                M[i, j] = v[2]  # posición normalizada 0..1
    fig, ax = plt.subplots(figsize=(1.4 * len(mods) + 4, 0.55 * len(varnames) + 2))
    im = ax.imshow(M, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=1)
    ax.set_xticks(range(len(mods)))
    ax.set_xticklabels([label(m) for m in mods], rotation=25, ha="right")
    ax.set_yticks(range(len(varnames)))
    ax.set_yticklabels(varnames)
    # marcar baseline con texto del valor
    for i, var in enumerate(varnames):
        base, lo, hi, t = VARIABLES[var]
        for j, m in enumerate(mods):
            v = tabla[m].get(var)
            if v:
                txt = f"{v[0]:.2f}" if t == "float" else f"{v[0]:.0f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                        color="black")
    cb = fig.colorbar(im, ax=ax, fraction=0.04)
    cb.set_label("posición en el rango admisible (0=mín, 1=máx)")
    ax.set_title("Incumbente medio por variable — qué se mueve y hacia dónde")
    fig.tight_layout()
    p = os.path.join(out, "movimiento_variables.png")
    fig.savefig(p, dpi=140)
    plt.close()
    return p


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="pipeline_out/resultados")
    ap.add_argument("--out", default="analisis_out")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    data, fallidos = cargar(args.res)
    if not data:
        print(f"No se encontraron resultados válidos en {args.res}")
        return

    print(f"\nMétodos con datos válidos: "
          f"{', '.join(f'{m}({len(v)})' for m, v in sorted(data.items()))}")
    if fallidos:
        print("Corridas sin datos válidos (costo=inf / incumbente vacío): "
              f"{', '.join(f'{m}×{n}' for m, n in sorted(fallidos.items()))}")

    filas = tabla_resumen(data)
    imprimir_tabla(filas)

    tabla_vars = analisis_variables(data)
    desplaz = imprimir_variables(tabla_vars)

    p1 = graf_convergencia_eval(data, args.out)
    p2 = graf_convergencia_tiempo(data, args.out)
    p3 = graf_box_tiempo(data, args.out)
    p4 = graf_box_objetivo(data, args.out)
    p5 = graf_variables(tabla_vars, args.out)

    # exportar resumen a JSON
    resumen = {
        "objetivo": "tts_full_days_mean = tiempo medio en sistema (dias)",
        "ranking_reeval": [
            {"modulo": f["mod"], "label": f["label"], "reeval_media": f["re_med"],
             "reeval_mejor": f["re_best"], "costo_opt_media": f["costo_med"],
             "tiempo_h": f["tiempo_h"], "n_seeds": f["n_seeds"]}
            for f in filas],
        "desplazamiento_variables": [
            {"variable": v, "delta_signed_norm": ds, "delta_abs_norm": da}
            for v, ds, da in desplaz],
    }
    with open(os.path.join(args.out, "resumen_analisis.json"), "w",
              encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    print("Gráficas generadas:")
    for p in (p1, p2, p3, p4, p5):
        print(f"  - {p}")
    print(f"  - {os.path.join(args.out, 'resumen_analisis.json')}")


if __name__ == "__main__":
    main()
