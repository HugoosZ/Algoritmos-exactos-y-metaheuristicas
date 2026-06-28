#!/usr/bin/env python3
# =============================================================================
#  Tarea 2 - CIT3352  |  Set-Union Knapsack Problem (SUKP)
#  Generacion de graficos de analisis a partir de los CSV producidos por sukp.cpp
# =============================================================================
import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

RES = "results"
FIG = os.path.join(RES, "figs")
os.makedirs(FIG, exist_ok=True)

INSTANCES = ["easy", "medium1", "medium2", "hard"]
INST_LABEL = {"easy": "Easy (85x100)", "medium1": "Medium1 (185x200)",
              "medium2": "Medium2 (200x200)", "hard": "Hard (500x500)"}

# Paleta consistente por algoritmo
COL = {
    "greedy_det":   "#444444",
    "greedy_stoch": "#1f77b4",
    "sa_det":       "#2ca02c",
    "sa_stoch":     "#17becf",
    "ga":           "#d62728",
}
LBL = {
    "greedy_det":   "Greedy determinista",
    "greedy_stoch": "Greedy estocastico",
    "sa_det":       "SA (inicio greedy det.)",
    "sa_stoch":     "SA (inicio greedy estoc.)",
    "ga":           "Algoritmo Genetico",
}

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
})

# -----------------------------------------------------------------------------
#  Cargar todos los summary
# -----------------------------------------------------------------------------
def load_summary():
    frames = [pd.read_csv(f) for f in sorted(glob.glob(f"{RES}/summary_*.csv"))]
    df = pd.concat(frames, ignore_index=True)
    # etiqueta unificada algoritmo+inicio
    def key(r):
        if r.algorithm == "sa":
            return "sa_" + ("det" if r.start == "det" else "stoch")
        return r.algorithm
    df["key"] = df.apply(key, axis=1)
    return df

DF = load_summary()

def savefig(fig, name):
    path = os.path.join(FIG, name)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print("  ->", path)

# =============================================================================
#  1. CONVERGENCIA SA  (FO vs iteraciones)  -- algoritmo de trayectoria
# =============================================================================
print("[1] Convergencia SA (FO vs iteraciones)")
for inst in INSTANCES:
    fig, ax = plt.subplots(figsize=(7, 4.3))
    for tag, col in [("det", COL["sa_det"]), ("stoch", COL["sa_stoch"])]:
        f = f"{RES}/conv_sa_{tag}_{inst}.csv"
        if not os.path.exists(f): continue
        d = pd.read_csv(f)
        ax.plot(d["iter"], d["best_FO"], color=col, lw=2,
                label=LBL["sa_" + tag])
    # linea de referencia: mejor greedy
    gdet = DF[(DF.instance == inst) & (DF.key == "greedy_det")].FO.max()
    ax.axhline(gdet, ls="--", color=COL["greedy_det"], lw=1.3,
               label=f"Greedy det. = {gdet}")
    ax.set_xlabel("Iteraciones (movimientos)"); ax.set_ylabel("Mejor FO encontrada")
    ax.set_title(f"Convergencia Simulated Annealing — {INST_LABEL[inst]}")
    ax.legend(loc="lower right", fontsize=9)
    savefig(fig, f"fig_conv_sa_{inst}.png")

# =============================================================================
#  2. CONVERGENCIA GA  (mejor y promedio vs generaciones)  -- algoritmo poblacional
# =============================================================================
print("[2] Convergencia GA (FO vs generaciones)")
for inst in INSTANCES:
    f = f"{RES}/conv_ga_{inst}.csv"
    if not os.path.exists(f): continue
    d = pd.read_csv(f)
    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.plot(d["gen"], d["best_FO"], color=COL["ga"], lw=2.2, label="Mejor individuo")
    ax.plot(d["gen"], d["avg_FO"], color="#ff9896", lw=1.6, ls="-",
            label="Promedio poblacion")
    ax.fill_between(d["gen"], d["avg_FO"], d["best_FO"], color="#ff9896", alpha=0.15)
    gdet = DF[(DF.instance == inst) & (DF.key == "greedy_det")].FO.max()
    ax.axhline(gdet, ls="--", color=COL["greedy_det"], lw=1.3,
               label=f"Greedy det. = {gdet}")
    ax.set_xlabel("Generacion"); ax.set_ylabel("FO")
    ax.set_title(f"Convergencia Algoritmo Genetico — {INST_LABEL[inst]}")
    ax.legend(loc="lower right", fontsize=9)
    savefig(fig, f"fig_conv_ga_{inst}.png")

# =============================================================================
#  3. COMPARACION GREEDY  (determinista vs promedio de 10 estocasticos)  [rubrica 7]
# =============================================================================
print("[3] Comparacion greedy determinista vs estocastico")
fig, ax = plt.subplots(figsize=(8, 4.6))
x = np.arange(len(INSTANCES)); width = 0.38
det_vals, stoch_mean, stoch_std, stoch_best = [], [], [], []
for inst in INSTANCES:
    det_vals.append(DF[(DF.instance == inst) & (DF.key == "greedy_det")].FO.max())
    s = DF[(DF.instance == inst) & (DF.key == "greedy_stoch")].FO
    stoch_mean.append(s.mean()); stoch_std.append(s.std(ddof=1)); stoch_best.append(s.max())
ax.bar(x - width/2, det_vals, width, color=COL["greedy_det"], label="Greedy determinista")
ax.bar(x + width/2, stoch_mean, width, yerr=stoch_std, capsize=4,
       color=COL["greedy_stoch"], label="Greedy estocastico (media ± std, 10 ejec.)")
ax.scatter(x + width/2, stoch_best, marker="D", color="black", zorder=5, s=28,
           label="Mejor estocastico")
ax.set_xticks(x); ax.set_xticklabels([INST_LABEL[i] for i in INSTANCES], rotation=12, ha="right")
ax.set_ylabel("FO"); ax.set_title("Greedy determinista vs estocastico (10 ejecuciones)")
ax.legend(fontsize=9)
savefig(fig, "fig_greedy_comparison.png")

# =============================================================================
#  4. COMPARACION GLOBAL DE ALGORITMOS  (mejor FO y media±std)  [rubrica 8 y 9]
# =============================================================================
print("[4] Comparacion global de algoritmos")
keys = ["greedy_stoch", "sa_det", "sa_stoch", "ga"]
# 4a) mejor FO
fig, ax = plt.subplots(figsize=(9, 4.8))
x = np.arange(len(INSTANCES)); width = 0.2
for k_i, k in enumerate(keys):
    vals = [DF[(DF.instance == i) & (DF.key == k)].FO.max() for i in INSTANCES]
    ax.bar(x + (k_i - 1.5)*width, vals, width, color=COL[k], label=LBL[k])
# greedy det como marcador
gdet = [DF[(DF.instance == i) & (DF.key == "greedy_det")].FO.max() for i in INSTANCES]
ax.scatter(x, gdet, marker="_", s=600, color="black", zorder=6, label="Greedy det.")
ax.set_xticks(x); ax.set_xticklabels([INST_LABEL[i] for i in INSTANCES], rotation=12, ha="right")
ax.set_ylabel("Mejor FO"); ax.set_title("Mejor solucion encontrada por algoritmo")
ax.legend(fontsize=8, ncol=3, loc="upper right")
savefig(fig, "fig_algo_best.png")

# 4b) media ± std
fig, ax = plt.subplots(figsize=(9, 4.8))
for k_i, k in enumerate(keys):
    means = [DF[(DF.instance == i) & (DF.key == k)].FO.mean() for i in INSTANCES]
    stds  = [DF[(DF.instance == i) & (DF.key == k)].FO.std(ddof=1) for i in INSTANCES]
    ax.bar(x + (k_i - 1.5)*width, means, width, yerr=stds, capsize=3,
           color=COL[k], label=LBL[k])
ax.set_xticks(x); ax.set_xticklabels([INST_LABEL[i] for i in INSTANCES], rotation=12, ha="right")
ax.set_ylabel("FO media ± std (10 ejec.)")
ax.set_title("Calidad media por algoritmo (10 ejecuciones)")
ax.legend(fontsize=8, ncol=2, loc="upper right")
savefig(fig, "fig_algo_mean.png")

# =============================================================================
#  5. BOXPLOTS de distribucion de FO por instancia (variabilidad)
# =============================================================================
print("[5] Boxplots de variabilidad por instancia")
for inst in INSTANCES:
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    data, labels, colors = [], [], []
    for k in keys:
        v = DF[(DF.instance == inst) & (DF.key == k)].FO.values
        if len(v) == 0: continue
        data.append(v); labels.append(LBL[k].replace(" (", "\n(")); colors.append(COL[k])
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, showmeans=True,
                    meanprops=dict(marker="o", markerfacecolor="white",
                                   markeredgecolor="black", markersize=6))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.55)
    for med in bp["medians"]: med.set_color("black")
    gdet = DF[(DF.instance == inst) & (DF.key == "greedy_det")].FO.max()
    ax.axhline(gdet, ls="--", color=COL["greedy_det"], lw=1.3, label=f"Greedy det. = {gdet}")
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("FO"); ax.set_title(f"Distribucion de FO (10 ejec.) — {INST_LABEL[inst]}")
    ax.legend(fontsize=9)
    savefig(fig, f"fig_box_{inst}.png")

# =============================================================================
#  6. TIEMPO de ejecucion por algoritmo (escala log)
# =============================================================================
print("[6] Tiempo de ejecucion por algoritmo")
fig, ax = plt.subplots(figsize=(9, 4.6))
allkeys = ["greedy_det", "greedy_stoch", "sa_det", "sa_stoch", "ga"]
width = 0.16
for k_i, k in enumerate(allkeys):
    vals = [DF[(DF.instance == i) & (DF.key == k)].time_ms.mean() for i in INSTANCES]
    ax.bar(x + (k_i - 2)*width, vals, width, color=COL[k], label=LBL[k])
ax.set_yscale("log")
ax.set_xticks(x); ax.set_xticklabels([INST_LABEL[i] for i in INSTANCES], rotation=12, ha="right")
ax.set_ylabel("Tiempo medio [ms] (log)")
ax.set_title("Tiempo de ejecucion promedio por algoritmo")
ax.legend(fontsize=8, ncol=3)
savefig(fig, "fig_time.png")

# =============================================================================
#  7. SWEEPS de parametros (justificacion)  [rubrica 4 y 6]
# =============================================================================
print("[7] Barridos de parametros")
def sweep_plot(fname, xcol, xlabel, title, out, chosen=None, logx=False):
    f = f"{RES}/{fname}"
    if not os.path.exists(f): return
    d = pd.read_csv(f)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.errorbar(d[xcol], d["mean_FO"], yerr=d["std_FO"], marker="o", capsize=4,
                color="#1f77b4", lw=2, label="FO media ± std")
    if chosen is not None:
        ax.axvline(chosen, ls="--", color="#d62728", lw=1.5,
                   label=f"Valor elegido = {chosen}")
    if logx: ax.set_xscale("log")
    ax.set_xlabel(xlabel); ax.set_ylabel("FO media (10 ejec.)")
    ax.set_title(title); ax.legend(fontsize=9)
    ax.xaxis.set_major_locator(MaxNLocator(integer=False))
    savefig(fig, out)

sweep_plot("sweep_sa_alpha_medium1.csv", "alpha",
           "Tasa de enfriamiento  α", "SA: sensibilidad a α  (Medium1)",
           "fig_sweep_sa_alpha.png", chosen=0.95)
sweep_plot("sweep_sa_chi0_medium1.csv", "chi0",
           "Prob. de aceptacion inicial  χ₀", "SA: sensibilidad a χ₀ / T₀  (Medium1)",
           "fig_sweep_sa_chi0.png", chosen=0.85)
sweep_plot("sweep_ga_pop_medium1.csv", "pop",
           "Tamaño de poblacion", "GA: sensibilidad al tamaño de poblacion  (Medium1)",
           "fig_sweep_ga_pop.png", chosen=50)
sweep_plot("sweep_ga_pm_medium1.csv", "pm_x_m",
           "Tasa de mutacion  (× 1/m)", "GA: sensibilidad a la tasa de mutacion  (Medium1)",
           "fig_sweep_ga_pm.png", chosen=1.0)

# =============================================================================
#  8. Tabla resumen consolidada (CSV) para el informe
# =============================================================================
print("[8] Tabla resumen consolidada")
rows = []
for inst in INSTANCES:
    for k in ["greedy_det", "greedy_stoch", "sa_det", "sa_stoch", "ga"]:
        s = DF[(DF.instance == inst) & (DF.key == k)]
        if len(s) == 0: continue
        rows.append({
            "instancia": inst, "algoritmo": LBL[k], "n_ejec": len(s),
            "FO_mejor": int(s.FO.max()), "FO_media": round(s.FO.mean(), 1),
            "FO_std": round(s.FO.std(ddof=1), 1) if len(s) > 1 else 0.0,
            "FO_peor": int(s.FO.min()),
            "tiempo_medio_ms": round(s.time_ms.mean(), 2),
        })
tabla = pd.DataFrame(rows)
tabla.to_csv(f"{RES}/tabla_resumen.csv", index=False)
print(tabla.to_string(index=False))

# gap respecto al mejor conocido por instancia
print("\n[8b] GAP (%) respecto a la mejor FO conocida por instancia")
best_known = {i: DF[DF.instance == i].FO.max() for i in INSTANCES}
gap_rows = []
for inst in INSTANCES:
    bk = best_known[inst]
    for k in ["greedy_det", "greedy_stoch", "sa_det", "sa_stoch", "ga"]:
        s = DF[(DF.instance == inst) & (DF.key == k)]
        if len(s) == 0: continue
        gap_rows.append({"instancia": inst, "algoritmo": LBL[k],
                         "mejor_conocida": int(bk),
                         "gap_mejor_%": round(100*(bk - s.FO.max())/bk, 2),
                         "gap_media_%": round(100*(bk - s.FO.mean())/bk, 2)})
gap = pd.DataFrame(gap_rows)
gap.to_csv(f"{RES}/tabla_gap.csv", index=False)
print(gap.to_string(index=False))

print("\nListo. Figuras en", FIG)
