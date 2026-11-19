"""Análisis y figuras de la narrativa. Uso: python analisis.py <carpeta_out>"""
import sys, os, json
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT = sys.argv[1]; FIG = os.path.join(OUT, "fig"); os.makedirs(FIG, exist_ok=True)
x = pd.ExcelFile(os.path.join(OUT, "modelo_vida_integral.xlsx"))
f = x.parse("fact_atenciones"); pac = x.parse("dim_pacientes"); hos = x.parse("dim_hospitales")
ser = x.parse("dim_servicios"); dia = x.parse("dim_diagnosticos"); ase = x.parse("dim_aseguradoras")
m = (f.merge(pac, on="paciente_id").merge(hos, on="hospital_id").merge(ser, on="servicio_id")
      .merge(dia, on="diagnostico_id").merge(ase, on="aseguradora_id"))
A = {}

# ---------------- estilo
BG = "#f7f5f0"; INK = "#1d1d1b"; INK2 = "#55534e"; GRID = "#e4e1d9"
BLUE = "#2a78d6"; ORANGE = "#eb6834"; GRAY = "#b9b6ad"; RED = "#d03b3b"; AQUA = "#1baf7a"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
plt.rcParams.update({"font.family": "Inter", "font.size": 15, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.facecolor": BG, "figure.facecolor": BG,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.titlesize": 16, "axes.titleweight": "semibold",
                     "axes.titlecolor": INK, "axes.titlelocation": "left", "legend.frameon": False})
CORTO = {"Hospital Central": "Central", "Clínica del Norte": "Norte", "Hospital San Gabriel": "San Gabriel",
         "Centro Médico Oriente": "Oriente", "Hospital Regional": "Regional", "Centro Asistencial Sur": "Sur"}
COLH = {"Regional": ORANGE, "Oriente": ORANGE, "Norte": BLUE, "Sur": BLUE, "Central": GRAY, "San Gabriel": GRAY}
m["h"] = m.institucion.map(CORTO)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def ols(df, y, num=[], cat=[]):
    d = df.dropna(subset=[y] + num).copy()
    X = [np.ones(len(d))]; names = ["const"]
    for c in num: X.append(d[c].values.astype(float)); names.append(c)
    for c in cat:
        for l in sorted(d[c].unique())[1:]: X.append((d[c] == l).values.astype(float)); names.append(f"{c}[{l}]")
    X = np.column_stack(X); Y = d[y].values.astype(float)
    b, *_ = np.linalg.lstsq(X, Y, rcond=None); res = Y - X @ b; n, k = X.shape
    se = np.sqrt(np.diag(res @ res / (n - k) * np.linalg.inv(X.T @ X)))
    r2 = 1 - res @ res / ((Y - Y.mean()) @ (Y - Y.mean()))
    return pd.DataFrame({"coef": b, "se": se, "p": 2 * stats.t.sf(np.abs(b / se), n - k)}, index=names), r2, n


# ================= KPIs globales
A["kpi"] = {"atenciones": len(m), "pacientes_unicos": int(m.paciente_id.nunique()),
            "espera_prom": round(m.tiempo_espera_min.mean(), 1), "espera_mediana": float(m.tiempo_espera_min.median()),
            "tasa_hosp": round(m.es_hospitalizado.mean() * 100, 1), "tasa_reing": round(m.reingreso_30_dias.mean() * 100, 2),
            "costo_prom": round(m.costo_atencion_cop.mean()), "costo_total": float(m.costo_atencion_cop.sum()),
            "satisf_prom": round(m.satisfaccion_1_a_5.mean(), 2), "estancia_prom": round(m.estancia_dias.mean(), 2),
            "pct_espera_120": round((m.tiempo_espera_min.dropna() > 120).mean() * 100, 1),
            "pct_insatisfechos": round((m.satisfaccion_1_a_5.dropna() <= 3).mean() * 100, 1)}

# ================= H1: estabilidad temporal
m["mi"] = (m.fecha_hora_ingreso.dt.year - 2025) * 12 + m.fecha_hora_ingreso.dt.month - 1
tend = {}
for c in ["tiempo_espera_min", "satisfaccion_1_a_5", "es_hospitalizado", "reingreso_30_dias", "costo_atencion_cop", "estancia_dias"]:
    r, _, _ = ols(m, c, ["mi"]); tend[c] = {"pendiente_mes": float(r.loc["mi", "coef"]), "p": float(r.loc["mi", "p"])}
A["tendencia"] = tend
j = m[m.mes_num <= 10]
comp = j.groupby("anio").agg(atenciones=("atencion_id", "size"), espera=("tiempo_espera_min", "mean"), hosp=("es_hospitalizado", "mean"),
                              reing=("reingreso_30_dias", "mean"), sat=("satisfaccion_1_a_5", "mean"), costo=("costo_atencion_cop", "mean"),
                              estancia=("estancia_dias", "mean"))
A["ene_oct_2025_vs_2026"] = comp.round(4).reset_index().to_dict("records")
A["p_ene_oct"] = {"espera_mw": float(stats.mannwhitneyu(*[g.dropna() for _, g in j.groupby("anio").tiempo_espera_min]).pvalue),
                  "reing_chi2": float(stats.chi2_contingency(pd.crosstab(j.anio, j.reingreso_30_dias))[1]),
                  "hosp_chi2": float(stats.chi2_contingency(pd.crosstab(j.anio, j.es_hospitalizado))[1]),
                  "sat_mw": float(stats.mannwhitneyu(*[g.dropna() for _, g in j.groupby("anio").satisfaccion_1_a_5]).pvalue)}
mes = m.groupby("mes").agg(espera=("tiempo_espera_min", "mean"), sat=("satisfaccion_1_a_5", "mean"), hosp=("es_hospitalizado", "mean"),
                           reing=("reingreso_30_dias", "mean"), n=("atencion_id", "size")).reset_index()
fig, axs = plt.subplots(2, 2, figsize=(14, 8)); axs = axs.ravel()
panels = [("espera", "Espera promedio (min)", 1, (70, 120)), ("sat", "Satisfacción promedio (1 a 5)", 1, (3.4, 4.3)),
          ("hosp", "Tasa de hospitalización (%)", 100, (10, 35)), ("reing", "Tasa de reingreso (%)", 100, (-3, 15))]
xs = np.arange(len(mes))
for ax, (c, t, k, yl) in zip(axs, panels):
    v = mes[c] * k
    ax.plot(xs, v, color=BLUE, lw=2.2, marker="o", ms=4)
    for yr, sl in [(2025, slice(0, 12)), (2026, slice(12, 22))]:
        mu = v.iloc[sl].mean(); ax.hlines(mu, xs[sl][0], xs[sl][-1], color=INK2, lw=1.2)
        txt = (f"{yr}: {mu:.1f} min" if c == "espera" else f"{yr}: {mu:.2f}" if c == "sat" else f"{yr}: {mu:.1f} %").replace(".", ",")
        ax.text((xs[sl][0] + xs[sl][-1]) / 2, yl[0] + (yl[1] - yl[0]) * 0.06, txt, fontsize=13, color=INK, ha="center",
                bbox=dict(boxstyle="round,pad=0.25", fc=BG, ec="none"))
    ax.set_title(t, fontsize=15); ax.set_ylim(*yl)
    ax.set_xticks([0, 6, 12, 18, 21]); ax.set_xticklabels(["ene 2025", "jul 2025", "ene 2026", "jul 2026", "oct 2026"], fontsize=12)
    ax.axvline(11.5, color=GRID, lw=1.5)
fig.tight_layout(w_pad=3, h_pad=2.5); save(fig, "f1_tendencia.png")

# ================= H2: triage vs norma
tri = m.groupby("triage").tiempo_espera_min.describe(percentiles=[.25, .5, .75])
A["triage"] = {int(t): {"n": int(r["count"]), "media": round(r["mean"], 1), "mediana": float(r["50%"]), "p25": float(r["25%"]), "p75": float(r["75%"]),
                        "satisf": round(m[m.triage == t].satisfaccion_1_a_5.mean(), 2), "hosp": round(m[m.triage == t].es_hospitalizado.mean() * 100, 1),
                        "costo": round(m[m.triage == t].costo_atencion_cop.mean())} for t, r in tri.iterrows()}
t1 = m[m.triage == 1].tiempo_espera_min.dropna(); t2 = m[m.triage == 2].tiempo_espera_min.dropna()
A["norma"] = {"t1_mediana": float(t1.median()), "t1_pct_5min": round((t1 <= 5).mean() * 100, 1), "t1_pct_mas30": round((t1 > 30).mean() * 100, 1),
              "t2_pct_en_meta": round((t2 <= 30).mean() * 100, 1), "t2_media": round(t2.mean(), 1), "t2_n": int(len(t2)),
              "t2_fuera_meta_n": int((t2 > 30).sum()), "t2_fuera_meta_por_anio": round((t2 > 30).sum() / 22 * 12),
              "rho_triage_espera": round(stats.spearmanr(m.triage, m.tiempo_espera_min, nan_policy="omit").statistic, 3)}
porh_t2 = m[m.triage == 2].groupby("h").tiempo_espera_min.apply(lambda s: (s.dropna() <= 30).mean() * 100).round(1)
porh_t1 = m[m.triage == 1].groupby("h").tiempo_espera_min.median()
A["t2_en_meta_por_hospital"] = porh_t2.to_dict(); A["t1_mediana_por_hospital"] = porh_t1.to_dict()
urg = m[(m.servicio_id == "S01") & (m.triage == 2)].tiempo_espera_min.dropna()
A["norma"]["t2_urgencias_en_meta"] = round((urg <= 30).mean() * 100, 1)

fig, ax = plt.subplots(figsize=(10, 5.4))
lv = ["I", "II", "III", "IV", "V"]
for i, t in enumerate(range(1, 6)):
    r = A["triage"][t]
    col = RED if t <= 2 else BLUE
    ax.plot([r["p25"], r["p75"]], [i, i], color=col, lw=7, solid_capstyle="round", alpha=.35)
    ax.plot(r["mediana"], i, "o", color=col, ms=13, mec=BG, mew=2)
    ax.text(r["p75"] + 6, i, f"mediana {r['mediana']:.0f} min", va="center", fontsize=13, color=INK)
ax.axvline(30, color=RED, lw=1.5); ax.text(32, 3.9, "Norma triage II:\n30 minutos", color=RED, fontsize=12.5, va="center")
ax.axvline(0, color=RED, lw=1.5); ax.text(2, 3.35, "Triage I:\natención\ninmediata", color=RED, fontsize=12.5, va="center")
ax.set_yticks(range(5)); ax.set_yticklabels([f"Triage {l}" for l in lv]); ax.invert_yaxis()
ax.set_xlim(-2, 230); ax.set_xlabel("Minutos de espera (punto = mediana, barra = 50 % central de los casos)")
ax.grid(axis="y", visible=False); fig.tight_layout(); save(fig, "f2_triage_norma.png")

fig, ax = plt.subplots(figsize=(8.6, 5.0))
o = porh_t2.sort_values()
ax.barh(o.index, o.values, color=[COLH[i] for i in o.index], height=0.6)
for i, v in enumerate(o.values): ax.text(v + 1, i, f"{v:.0f} %", va="center", fontsize=14, color=INK)
ax.axvline(A["norma"]["t2_pct_en_meta"], color=INK2, lw=1.2)
ax.text(A["norma"]["t2_pct_en_meta"] + 0.5, 5.45, f"Promedio red: {A['norma']['t2_pct_en_meta']:.0f} %".replace(".", ","), fontsize=12.5, color=INK2)
ax.set_ylim(-0.6, 5.8); ax.set_xlim(0, 40); ax.set_xlabel("% de pacientes triage II atendidos en 30 minutos o menos")
ax.grid(axis="y", visible=False); fig.tight_layout(); save(fig, "f3b_t2_meta_hospital.png")

# ================= H3: exceso de espera ajustado (estandarización indirecta por triage y servicio)
d = m.dropna(subset=["tiempo_espera_min"]).copy()
d["esperado"] = d.groupby(["triage", "servicio_id"]).tiempo_espera_min.transform("mean")
d["exceso"] = d.tiempo_espera_min - d.esperado
ex = d.groupby("h").agg(obs=("tiempo_espera_min", "mean"), esp=("esperado", "mean"), exceso=("exceso", "mean"), se=("exceso", "sem"), n=("exceso", "size"))
ex["lo"] = ex.exceso - 1.96 * ex.se; ex["hi"] = ex.exceso + 1.96 * ex.se
A["exceso_espera"] = ex.round(2).reset_index().to_dict("records")
r, r2, _ = ols(m, "tiempo_espera_min", [], ["hospital_id", "triage", "servicio_id"])
A["ols_espera"] = {"r2": round(r2, 3), "coef": r.round(3).reset_index().to_dict("records")}
A["kw_espera_hosp_p"] = float(stats.kruskal(*[g.dropna() for _, g in m.groupby("hospital_id").tiempo_espera_min]).pvalue)
A["volumen_vs_exceso"] = m.groupby("h").size().to_dict()
piv = pd.pivot_table(m, values="tiempo_espera_min", index="h", columns="triage", aggfunc="mean").round(1)
A["espera_hosp_triage"] = piv.to_dict()
A["pct120_hosp"] = m.groupby("h").tiempo_espera_min.apply(lambda s: round((s.dropna() > 120).mean() * 100, 1)).to_dict()
# hora / noche / fin de semana
A["hora"] = {k: float(v.loc["hora", "p"]) for k, v in [("p_hora", ols(m, "tiempo_espera_min", ["hora"], ["triage"])[0])]}
m["fds"] = (m.dia_semana_num >= 6).astype(int)
A["fin_semana_p"] = float(ols(m, "tiempo_espera_min", ["fds"], ["triage"])[0].loc["fds", "p"])
A["franja"] = m.groupby("franja").tiempo_espera_min.mean().round(1).to_dict()
A["mix_triage_hosp"] = m.groupby("h").triage.apply(lambda s: round((s <= 2).mean() * 100, 1)).to_dict()

fig, ax = plt.subplots(figsize=(10, 5.0))
o = ex.sort_values("exceso")
for i, (h, r_) in enumerate(o.iterrows()):
    ax.plot([r_.lo, r_.hi], [i, i], color=COLH[h], lw=3, solid_capstyle="round")
    ax.plot(r_.exceso, i, "o", color=COLH[h], ms=12, mec=BG, mew=2)
    ax.text(r_.hi + 0.6, i, f"{r_.exceso:+.1f} min".replace(".", ","), va="center", fontsize=14, color=INK)
ax.axvline(0, color=INK2, lw=1.2)
ax.set_yticks(range(len(o))); ax.set_yticklabels(o.index)
ax.set_xlim(-12, 15); ax.set_xlabel("Minutos de espera por encima (+) o por debajo (−) de lo esperado\npara el mismo nivel de triage y el mismo servicio (intervalo de confianza 95 %)")
ax.grid(axis="y", visible=False); fig.tight_layout(); save(fig, "f3_exceso_espera.png")

# heatmap hospital x triage
fig, ax = plt.subplots(figsize=(9, 4.6))
ordh = ex.sort_values("exceso", ascending=False).index
P = piv.loc[ordh]
from matplotlib.colors import LinearSegmentedColormap
cm = LinearSegmentedColormap.from_list("b", ["#e6eefa", "#86b6ef", "#2a78d6", "#104281"])
ax.imshow(P.values, cmap=cm, aspect="auto", vmin=20, vmax=185)
for i in range(P.shape[0]):
    for k in range(P.shape[1]):
        v = P.values[i, k]; ax.text(k, i, f"{v:.0f}", ha="center", va="center", fontsize=14, color="white" if v > 110 else INK)
ax.set_xticks(range(5)); ax.set_xticklabels([f"Triage {l}" for l in lv]); ax.set_yticks(range(len(P))); ax.set_yticklabels(P.index)
ax.grid(False); [s.set_visible(False) for s in ax.spines.values()]
ax.set_title("Espera promedio en minutos", fontsize=14); fig.tight_layout(); save(fig, "f3c_heatmap.png")

# ================= H4: espera y satisfacción
w = m.dropna(subset=["tiempo_espera_min", "satisfaccion_1_a_5"]).copy()
A["rel_espera_sat"] = {"spearman": round(stats.spearmanr(w.tiempo_espera_min, w.satisfaccion_1_a_5).statistic, 3),
                       "pearson": round(stats.pearsonr(w.tiempo_espera_min, w.satisfaccion_1_a_5).statistic, 3),
                       "por_triage": {int(t): round(stats.spearmanr(g.tiempo_espera_min, g.satisfaccion_1_a_5).statistic, 3) for t, g in w.groupby("triage")}}
r, r2, _ = ols(w, "satisfaccion_1_a_5", ["tiempo_espera_min"], ["triage"])
A["rel_espera_sat"]["coef_min_ajust_triage"] = round(float(r.loc["tiempo_espera_min", "coef"]), 5)
A["rel_espera_sat"]["coef_triage_p"] = {k: float(v) for k, v in r.p.items() if "triage" in k}
A["rel_espera_sat"]["r2_espera_triage"] = round(r2, 3)
A["rel_espera_sat"]["r2_solo_triage"] = round(ols(w, "satisfaccion_1_a_5", [], ["triage"])[1], 3)
A["rel_espera_sat"]["r2_solo_espera"] = round(ols(w, "satisfaccion_1_a_5", ["tiempo_espera_min"])[1], 3)
A["rel_espera_sat"]["insatisf_espera_le60"] = round((w[w.tiempo_espera_min <= 60].satisfaccion_1_a_5 <= 3).mean() * 100, 1)
A["rel_espera_sat"]["insatisf_espera_gt120"] = round((w[w.tiempo_espera_min > 120].satisfaccion_1_a_5 <= 3).mean() * 100, 1)
w["wb"] = pd.cut(w.tiempo_espera_min, [-1, 30, 60, 90, 120, 150, 180, 400], labels=["0-30", "31-60", "61-90", "91-120", "121-150", "151-180", "más de 180"])
A["sat_por_rango_espera"] = w.groupby("wb", observed=True).satisfaccion_1_a_5.agg(["size", "mean"]).round(3).reset_index().astype({"wb": str}).to_dict("records")
A["sat_por_resultado"] = m.groupby("resultado_atencion").satisfaccion_1_a_5.mean().round(2).to_dict()
A["espera_por_resultado"] = m.groupby("resultado_atencion").tiempo_espera_min.mean().round(1).to_dict()
sb = ols(m, "satisfaccion_1_a_5", [], ["hospital_id"])[0]; sa = ols(m, "satisfaccion_1_a_5", ["tiempo_espera_min"], ["hospital_id"])[0]
A["sat_hosp_bruta_p"] = sb.p.round(4).to_dict(); A["sat_hosp_ajustada_p"] = sa.p.round(4).to_dict()
A["kw_sat_hosp_p"] = float(stats.kruskal(*[g.dropna() for _, g in m.groupby("hospital_id").satisfaccion_1_a_5]).pvalue)
w["sat_esp"] = w.satisfaccion_1_a_5 - (sa.loc["const", "coef"] + sa.loc["tiempo_espera_min", "coef"] * w.tiempo_espera_min)
satb = w.groupby("h").satisfaccion_1_a_5.mean(); sata = w.groupby("h").sat_esp.mean() + w.satisfaccion_1_a_5.mean() - w.sat_esp.mean()
A["sat_hosp"] = pd.DataFrame({"bruta": satb, "ajustada_por_espera": sata}).round(3).reset_index().to_dict("records")

fig, ax = plt.subplots(figsize=(10, 5.4))
for t, g in w.groupby("triage"):
    s = g.groupby("wb", observed=True).satisfaccion_1_a_5.agg(["mean", "size"]); s = s[s["size"] >= 25]
    idx = [list(w.wb.cat.categories).index(c) for c in s.index]
    ax.plot(idx, s["mean"], color=RAMP[t - 1], lw=2.2, marker="o", ms=6, label=f"Triage {lv[t-1]}")
ax.set_xticks(range(7)); ax.set_xticklabels(w.wb.cat.categories)
ax.set_xlabel("Minutos de espera"); ax.set_ylabel("Satisfacción promedio (1 a 5)")
ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.12), fontsize=13); ax.set_ylim(2.6, 4.9)
fig.tight_layout(); save(fig, "f4_espera_satisfaccion.png")

fig, ax = plt.subplots(figsize=(9, 5.0))
sh = pd.DataFrame({"b": satb, "a": sata}).sort_values("b")
for i, (h, r_) in enumerate(sh.iterrows()):
    ax.plot([r_.b, r_.a], [i, i], color=GRAY, lw=2)
    ax.plot(r_.b, i, "o", color=COLH[h] if COLH[h] != GRAY else INK2, ms=11, mec=BG, mew=2)
    ax.plot(r_.a, i, "D", color=AQUA, ms=9, mec=BG, mew=1.5)
ax.set_yticks(range(len(sh))); ax.set_yticklabels(sh.index); ax.set_xlim(3.82, 4.06)
ax.plot([], [], "o", color=INK2, label="Satisfacción observada"); ax.plot([], [], "D", color=AQUA, label="Si todos esperaran lo mismo")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=2, fontsize=12.5); ax.set_xlabel("Satisfacción promedio (1 a 5)")
ax.grid(axis="y", visible=False); fig.tight_layout(); save(fig, "f5_sat_hospital_ajustada.png")

# ================= H5: costos y resultados no dependen del hospital
pv = {}
for c in ["estancia_dias", "costo_atencion_cop", "satisfaccion_1_a_5", "tiempo_espera_min"]:
    pv[c] = float(stats.kruskal(*[g.dropna() for _, g in m.groupby("hospital_id")[c]]).pvalue)
for c in ["es_hospitalizado", "reingreso_30_dias"]:
    pv[c] = float(stats.chi2_contingency(pd.crosstab(m.hospital_id, m[c]))[1])
A["p_hospital"] = pv
A["r2_costo"] = {"hospital": round(ols(m, "costo_atencion_cop", [], ["hospital_id"])[1], 4), "estancia": round(ols(m, "costo_atencion_cop", ["estancia_dias"])[1], 3),
                 "servicio": round(ols(m, "costo_atencion_cop", [], ["servicio_id"])[1], 3), "servicio_estancia": round(ols(m, "costo_atencion_cop", ["estancia_dias"], ["servicio_id"])[1], 3),
                 "servicio_estancia_hospital": round(ols(m, "costo_atencion_cop", ["estancia_dias"], ["servicio_id", "hospital_id"])[1], 3)}
rc = ols(m, "costo_atencion_cop", ["estancia_dias"], ["servicio_id"])[0]
A["costo_por_dia_ajustado"] = round(float(rc.loc["estancia_dias", "coef"]))
A["rho_costo_estancia"] = round(stats.spearmanr(m.costo_atencion_cop, m.estancia_dias, nan_policy="omit").statistic, 3)
A["costo_hosp_vs_no"] = m.groupby("es_hospitalizado").agg(costo=("costo_atencion_cop", "mean"), estancia=("estancia_dias", "mean")).round(2).to_dict()
cs = m.groupby("servicio").agg(n=("atencion_id", "size"), costo=("costo_atencion_cop", "mean"), total=("costo_atencion_cop", "sum"), estancia=("estancia_dias", "mean"),
                               espera=("tiempo_espera_min", "mean"), sat=("satisfaccion_1_a_5", "mean"), hosp=("es_hospitalizado", "mean"), reing=("reingreso_30_dias", "mean"))
cs["pct_costo"] = cs.total / cs.total.sum() * 100
A["servicios"] = cs.round(3).reset_index().to_dict("records")
hs = m.groupby("h").agg(n=("atencion_id", "size"), pac=("paciente_id", "nunique"), espera=("tiempo_espera_min", "mean"), sat=("satisfaccion_1_a_5", "mean"),
                        hosp=("es_hospitalizado", "mean"), reing=("reingreso_30_dias", "mean"), costo=("costo_atencion_cop", "mean"), estancia=("estancia_dias", "mean"))
A["hospitales"] = hs.round(4).reset_index().to_dict("records")
A["ejemplo_pdf_espera_baja_estancia_alta"] = {"estancia_por_hospital": hs.estancia.round(2).to_dict(), "p": pv["estancia_dias"]}

fig, ax = plt.subplots(figsize=(10, 4.4))
lab = ["Solo el hospital", "Solo la estancia", "Solo el servicio", "Servicio + estancia"]
val = [A["r2_costo"]["hospital"], A["r2_costo"]["estancia"], A["r2_costo"]["servicio"], A["r2_costo"]["servicio_estancia"]]
cols = [ORANGE, BLUE, BLUE, BLUE]
ax.barh(lab, [v * 100 for v in val], color=cols, height=0.6)
for i, v in enumerate(val): ax.text(v * 100 + 1, i, (f"{v*100:.1f} %" if v >= 0.01 else f"{v*100:.2f} %").replace(".", ","), va="center", fontsize=14, color=INK)
ax.set_xlim(0, 110); ax.invert_yaxis(); ax.set_xlabel("% de la variación del costo por atención que explica cada factor (R²)")
ax.grid(axis="y", visible=False); fig.tight_layout(); save(fig, "f6_costo_r2.png")

fig, axs = plt.subplots(2, 2, figsize=(13, 8)); axs = axs.ravel()
order = list(ex.sort_values("exceso", ascending=False).index)
specs = [("estancia", "Estancia promedio (días)", 1, "{:.2f}", pv["estancia_dias"]), ("costo", "Costo promedio (miles COP)", 1e-3, "{:.0f}", pv["costo_atencion_cop"]),
         ("hosp", "Hospitalización (%)", 100, "{:.1f}", pv["es_hospitalizado"]), ("reing", "Reingreso (%)", 100, "{:.1f}", pv["reingreso_30_dias"])]
for ax, (c, t, k, fm, p) in zip(axs, specs):
    v = hs.loc[order, c] * k
    ax.barh(order, v, color=[COLH[i] for i in order], height=0.6)
    for i, val_ in enumerate(v): ax.text(val_ * 1.01, i, fm.format(val_).replace(".", ","), va="center", fontsize=13, color=INK)
    ax.set_title(f"{t}\nDiferencia entre hospitales: p = {p:.2f}".replace("0.", "0,"), fontsize=14); ax.invert_yaxis(); ax.set_xlim(0, v.max() * 1.25); ax.grid(axis="y", visible=False)
    if ax in (axs[1], axs[3]): ax.set_yticklabels([])
fig.tight_layout(w_pad=1.5, h_pad=2); save(fig, "f7_hospitales_resultados.png")

# ================= H6: reingresos
m["cron"] = np.where(m.curso == "Crónica", "Crónica", "No crónica"); m["e60"] = np.where(m.edad >= 60, "60 años o más", "Menos de 60")
rg = m.groupby(["cron", "e60"]).agg(n=("atencion_id", "size"), reing=("reingreso_30_dias", "mean"), r_n=("reingreso_30_dias", "sum")).reset_index()
A["reing_cron_edad"] = rg.round(4).to_dict("records")
A["reing_curso"] = m.groupby("curso").reingreso_30_dias.mean().round(4).to_dict()
A["reing_diag"] = m.groupby("diagnostico").reingreso_30_dias.mean().round(4).sort_values().to_dict()
A["reing_grupo_edad"] = m.groupby("grupo_edad").reingreso_30_dias.mean().round(4).to_dict()
A["reing_regimen"] = m.groupby("regimen").reingreso_30_dias.mean().round(4).to_dict()
A["reing_sexo"] = m.groupby("sexo").reingreso_30_dias.mean().round(4).to_dict()
A["p_reing"] = {k: float(stats.chi2_contingency(pd.crosstab(m[k], m.reingreso_30_dias))[1]) for k in ["curso", "grupo_edad", "regimen", "sexo", "hospital_id", "triage", "servicio_id"]}
m["cr"] = (m.curso == "Crónica").astype(int); m["m60"] = (m.edad >= 60).astype(int)
rl = ols(m, "reingreso_30_dias", ["cr", "m60"], ["hospital_id", "triage", "servicio_id"])[0]
A["lpm_reing"] = {"cronico_pp": round(rl.loc["cr", "coef"] * 100, 1), "mayor60_pp": round(rl.loc["m60", "coef"] * 100, 1), "p_cr": float(rl.loc["cr", "p"]), "p_60": float(rl.loc["m60", "p"])}
A["cronicos"] = {"pct_atenciones": round(m.cr.mean() * 100, 1), "reingresos_cronicos": int(m[m.cr == 1].reingreso_30_dias.sum()), "reingresos_total": int(m.reingreso_30_dias.sum()),
                 "pct_reingresos": round(m[m.cr == 1].reingreso_30_dias.sum() / m.reingreso_30_dias.sum() * 100, 1),
                 "pct_costo": round(m[m.cr == 1].costo_atencion_cop.sum() / m.costo_atencion_cop.sum() * 100, 1),
                 "costo_prom_reingreso": round(m[m.reingreso_30_dias == 1].costo_atencion_cop.mean()),
                 "costo_prom_no_reingreso": round(m[m.reingreso_30_dias == 0].costo_atencion_cop.mean()),
                 "reing_cronicos_por_anio": round(m[m.cr == 1].reingreso_30_dias.sum() / 22 * 12)}
re_ = ols(m, "estancia_dias", [], ["curso", "servicio_id", "triage"])[0]
A["estancia_cronico_extra_dias"] = round(float(re_.loc["curso[Crónica]", "coef"]), 2)
A["estancia_curso"] = m.groupby("curso").estancia_dias.mean().round(2).to_dict()
A["kw_estancia_diag_p"] = float(stats.kruskal(*[g for _, g in m.groupby("diagnostico_id").estancia_dias]).pvalue)
A["rho_edad_reing"] = round(stats.spearmanr(m.edad, m.reingreso_30_dias).statistic, 3)

fig, ax = plt.subplots(figsize=(9.5, 5.0))
cats = [("No crónica", "Menos de 60"), ("No crónica", "60 años o más"), ("Crónica", "Menos de 60"), ("Crónica", "60 años o más")]
labs = ["No crónico\nmenor de 60", "No crónico\n60 o más", "Crónico\nmenor de 60", "Crónico\n60 o más"]
ns0 = [int(rg[(rg.cron == a) & (rg.e60 == b)].n.iloc[0]) for a, b in cats]
labs = [f"{l}\n({n:,} atenciones)".replace(",", ".") for l, n in zip(labs, ns0)]
vals = [rg[(rg.cron == a) & (rg.e60 == b)].reing.iloc[0] * 100 for a, b in cats]
ns = [int(rg[(rg.cron == a) & (rg.e60 == b)].n.iloc[0]) for a, b in cats]
ax.bar(labs, vals, color=[GRAY, GRAY, BLUE, ORANGE], width=0.6)
for i, (v, n) in enumerate(zip(vals, ns)): ax.text(i, v + 0.5, f"{v:.1f} %".replace(".", ","), ha="center", fontsize=16, fontweight="semibold", color=INK)
ax.set_ylim(0, 26); ax.set_ylabel("Tasa de reingreso a 30 días"); ax.grid(axis="x", visible=False)
fig.tight_layout(); save(fig, "f8_reingreso_cronico_edad.png")

fig, ax = plt.subplots(figsize=(9.5, 5.4))
rd = pd.Series(A["reing_diag"]) * 100
cronset = set(dia[dia.curso == "Crónica"].diagnostico)
ax.barh(rd.index, rd.values, color=[ORANGE if i in cronset else GRAY for i in rd.index], height=0.62)
for i, v in enumerate(rd.values): ax.text(v + 0.3, i, f"{v:.1f} %".replace(".", ","), va="center", fontsize=13, color=INK)
ax.set_xlim(0, 21); ax.set_xlabel("Tasa de reingreso a 30 días"); ax.grid(axis="y", visible=False)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=ORANGE, label="Diagnóstico crónico"), Patch(color=GRAY, label="Agudo o preventivo")], loc="lower right", fontsize=12.5)
fig.tight_layout(); save(fig, "f9_reingreso_diagnostico.png")

# estancia por curso dentro de servicio
pe = pd.pivot_table(m, values="estancia_dias", index="servicio", columns="curso", aggfunc="mean")
A["estancia_servicio_curso"] = pe.round(2).to_dict()
fig, ax = plt.subplots(figsize=(10, 5.2))
pe = pe.sort_values("Crónica")
yy = np.arange(len(pe))
for i, (s, r_) in enumerate(pe.iterrows()):
    ax.plot([r_["Aguda"], r_["Crónica"]], [i, i], color=GRAY, lw=2)
    ax.plot(r_["Aguda"], i, "o", color=BLUE, ms=11, mec=BG, mew=2); ax.plot(r_["Crónica"], i, "o", color=ORANGE, ms=11, mec=BG, mew=2)
ax.set_yticks(yy); ax.set_yticklabels(pe.index); ax.set_xlabel("Estancia promedio (días)")
ax.plot([], [], "o", color=BLUE, label="Diagnóstico agudo"); ax.plot([], [], "o", color=ORANGE, label="Diagnóstico crónico"); ax.legend(loc="lower right", fontsize=12.5)
ax.grid(axis="y", visible=False); fig.tight_layout(); save(fig, "f10_estancia_cronico.png")

# relaciones resumen
A["spearman"] = m[["tiempo_espera_min", "satisfaccion_1_a_5", "estancia_dias", "costo_atencion_cop", "triage", "edad", "reingreso_30_dias", "es_hospitalizado"]].corr(method="spearman").round(3).to_dict()

json.dump(A, open(os.path.join(OUT, "analisis.json"), "w"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: A[k] for k in ["kpi", "norma", "t2_en_meta_por_hospital", "r2_costo", "cronicos", "lpm_reing", "rel_espera_sat", "p_hospital", "p_ene_oct", "hora", "fin_semana_p", "mix_triage_hosp", "sat_hosp", "estancia_cronico_extra_dias", "costo_por_dia_ajustado", "pct120_hosp"]}, ensure_ascii=False, indent=1, default=float))
