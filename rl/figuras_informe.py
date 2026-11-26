"""
Regenera las figuras del informe a partir de los resultados guardados (no simula nada).
Uso: python figuras_informe.py --dir resultados/completa --out figuras
"""
import argparse, glob, json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from catalogo import CATALOGO

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})
AZUL, GRIS = "#2b6cb0", "#8a8f98"
NOMBRES = {"reinforce": "REINFORCE", "ppo": "PPO", "ddqn": "DDQN"}


def fig_pareada(d, final, out):
    t = final["tabla"]
    filas = [("SMAC-SK (mismo presup.)", t["smac_sk"], None, False)]
    for m in ("reinforce", "ddqn", "ppo"):
        k = f"{m} (prom. 5 inic.)"
        filas.append((f"{NOMBRES[m]} (prom. 5 inic.)", t[k], t[k]["dif_pareada_L_vs_mejor_fija"]["entre_inic_media"], True))
    filas += [("Aleatoria", t["aleatoria"], None, False), ("Operación actual", t["base"], None, False)]
    fig, ax = plt.subplots(figsize=(6.3, 2.6))
    for i, (n, f, ini, rl) in enumerate(filas):
        y = len(filas) - 1 - i
        dp = f["dif_pareada_L_vs_mejor_fija"]; c = AZUL if rl else GRIS
        ax.errorbar(dp["media"], y, xerr=dp["ic95"], fmt="o", color=c, capsize=3, ms=5)
        if ini:
            ax.scatter(ini, [y - 0.22] * len(ini), s=12, facecolors="none", edgecolors=c, lw=0.8)
        p = dp["wilcoxon_p"]
        ax.text(57, y, f"{dp['media']:+.1f} ± {dp['ic95']:.1f}   p{'<0,001' if p < 0.001 else '='+f'{p:.2f}'}".replace(".", ","),
                va="center", fontsize=8)
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_yticks(range(len(filas))); ax.set_yticklabels([f[0] for f in filas][::-1])
    ax.set_xlim(-32, 55); ax.set_xlabel("Δ L pareado vs mejor fija (pacientes); < 0 = mejor")
    fig.savefig(os.path.join(out, "fig_pareada.png")); plt.close(fig)


def fig_curvas(d, out):
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.2), sharey=True)
    fija = np.mean([x["L"] for x in json.load(open(os.path.join(d, "referencias.json")))["tamizaje"][
        str(json.load(open(os.path.join(d, "referencias.json")))["mejor_fija"])]])
    for ax, m in zip(axs, ("reinforce", "ppo", "ddqn")):
        C = []
        for p in sorted(glob.glob(os.path.join(d, f"{m}_s*", "curva.json"))):
            c = json.load(open(p))["curva"]; ep = [x["episodios"] for x in c]; C.append([x["L_val"] for x in c])
        C = np.array(C)
        ax.fill_between(ep, C.min(0), C.max(0), color=AZUL, alpha=0.18, lw=0)
        ax.plot(ep, C.mean(0), color=AZUL, lw=1.6)
        ax.axhline(fija, color="k", ls="--", lw=0.8)
        ax.set_title(NOMBRES[m]); ax.set_xlabel("Episodios")
    axs[0].set_ylabel("L validación (pacientes)")
    axs[2].text(2760, fija - 3, "mejor fija", ha="right", va="top", fontsize=7)
    fig.savefig(os.path.join(out, "fig_curvas.png")); plt.close(fig)


def fig_mapa(d, final, out):
    refs = json.load(open(os.path.join(d, "referencias.json")))
    firsts = sorted({c["first"] for c in CATALOGO}); ecos = sorted({c["eco"] for c in CATALOGO})
    L = np.full((len(firsts), len(ecos)), np.nan); U = np.zeros_like(L)
    for k, v in refs["tamizaje"].items():
        c = CATALOGO[int(k)]; L[firsts.index(c["first"]), ecos.index(c["eco"])] = np.mean([x["L"] for x in v])
    for n, it in final["interpretacion"].items():
        for a, cnt in it["acciones_usadas"].items():
            c = CATALOGO[int(a)]; U[firsts.index(c["first"]), ecos.index(c["eco"])] += cnt
    U = 100 * U / U.sum()
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.3))
    for ax, M, tit, fmt, cmap in ((axs[0], L, "L en validación, asignación fija", "{:.0f}", "Blues"),
                                  (axs[1], U, "Uso por los 15 agentes (% semanas)", "{:.0f}", "Greys")):
        im = ax.imshow(M, cmap=cmap if M is U else "Blues_r", aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if (M is L and M[i, j] < np.nanpercentile(L, 25)) or (M is U and M[i, j] > 10) else "black")
        ax.set_xticks(range(len(ecos))); ax.set_xticklabels([f"{e}/{75-e}" for e in ecos], fontsize=7)
        ax.set_yticks(range(len(firsts))); ax.set_yticklabels([f"{f}/{56-f}" for f in firsts])
        ax.set_xlabel("Ecografía / lab. matrona (cupos/sem)"); ax.set_title(tit, fontsize=9)
    axs[0].set_ylabel("1ra / control (cupos/sem)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_mapa.png")); plt.close(fig)


def fig_sensibilidad(d, out):
    p = os.path.join(d, "sensibilidad", "sensibilidad.json")
    if not os.path.exists(p):
        return
    S = json.load(open(p)); fin = json.load(open(os.path.join(d, "prueba", "final.json")))["tabla"]
    escen = [("×0,5", S["backlog x0.5"]["tabla"]), ("×1 (base)", fin), ("×1,5", S["backlog x1.5"]["tabla"])]
    metodos = [("smac_sk", "SMAC-SK", GRIS)] + [(f"{m} (prom. 5 inic.)", NOMBRES[m], AZUL) for m in ("reinforce", "ddqn", "ppo")] + [("base", "Op. actual", "#c05621")]
    fig, ax = plt.subplots(figsize=(6.3, 2.4))
    w = 0.15
    for j, (k, n, c) in enumerate(metodos):
        xs = np.arange(3) + (j - 2) * w
        m = [e[1][k]["dif_pareada_L_vs_mejor_fija"]["media"] for e in escen]
        ic = [e[1][k]["dif_pareada_L_vs_mejor_fija"]["ic95"] for e in escen]
        ax.errorbar(xs, m, yerr=ic, fmt="osD^v"[j], color=c, capsize=2, ms=4, label=n)
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set_xticks(range(3)); ax.set_xticklabels([f"Backlog inicial {e[0]}" for e in escen])
    ax.set_ylabel("Δ L vs mejor fija (pacientes)"); ax.legend(fontsize=7, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.savefig(os.path.join(out, "fig_sensibilidad.png")); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="resultados/completa"); ap.add_argument("--out", default="figuras")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    final = json.load(open(os.path.join(a.dir, "prueba", "final.json")))
    fig_pareada(a.dir, final, a.out); fig_curvas(a.dir, a.out); fig_mapa(a.dir, final, a.out); fig_sensibilidad(a.dir, a.out)
    print(sorted(os.listdir(a.out)))
