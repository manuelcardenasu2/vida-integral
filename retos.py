"""Retos de la Fase 1: pacientes únicos y duplicación por relaciones incorrectas.
Uso: python retos.py <carpeta_out>"""
import sys, json, os
import pandas as pd
OUT = sys.argv[1]
x = pd.ExcelFile(os.path.join(OUT, "modelo_vida_integral.xlsx"))
f = x.parse("fact_atenciones"); pac = x.parse("dim_pacientes"); dia = x.parse("dim_diagnosticos")
ase = x.parse("dim_aseguradoras"); hos = x.parse("dim_hospitales"); mun = x.parse("dim_municipios")
R = {}
# --- Reto 1: pacientes únicos
R["atenciones"] = len(f)
R["pacientes_unicos_red"] = int(f.paciente_id.nunique())
R["pacientes_en_dim"] = len(pac)
R["atenciones_por_paciente"] = round(len(f) / f.paciente_id.nunique(), 2)
R["dist_atenciones_por_paciente"] = {int(k): int(v) for k, v in f.paciente_id.value_counts().value_counts().sort_index().items()}
porh = f.groupby("hospital_id").paciente_id.nunique()
R["suma_unicos_por_hospital"] = int(porh.sum())
R["pacientes_en_mas_de_un_hospital"] = int((f.groupby("paciente_id").hospital_id.nunique() > 1).sum())
R["tasa_reingreso_atenciones"] = round(f.reingreso_30_dias.mean() * 100, 2)
R["pacientes_con_algun_reingreso"] = int(f.groupby("paciente_id").reingreso_30_dias.max().sum())
R["tasa_reingreso_pacientes"] = round(f.groupby("paciente_id").reingreso_30_dias.max().mean() * 100, 2)
m = f.merge(pac, on="paciente_id")
R["edad_prom_atenciones"] = round(m.edad.mean(), 2)
R["edad_prom_pacientes"] = round(pac[pac.paciente_id.isin(f.paciente_id)].edad.mean(), 2)
R["pct_60_atenciones"] = round((m.edad >= 60).mean() * 100, 2)
R["pct_60_pacientes"] = round((pac[pac.paciente_id.isin(f.paciente_id)].edad >= 60).mean() * 100, 2)

base = {"filas": len(f), "costo_total": float(f.costo_atencion_cop.sum()), "hospitalizaciones": int(f.es_hospitalizado.sum()), "reingresos": int(f.reingreso_30_dias.sum())}
R["base_correcta"] = base


def demo(nombre, df, explicacion):
    r = {"filas": len(df), "costo_total": float(df.costo_atencion_cop.sum()), "hospitalizaciones": int(df.es_hospitalizado.sum()),
         "reingresos": int(df.reingreso_30_dias.sum())}
    r["inflacion_costo_pct"] = round((r["costo_total"] / base["costo_total"] - 1) * 100, 1)
    r["explicacion"] = explicacion
    R[nombre] = r


# Error A: unir el hecho con diagnósticos por grupo_diagnostico (no es único: Cardiovascular tiene D02 y D06)
fa = f.merge(dia[["diagnostico_id", "grupo_diagnostico"]], on="diagnostico_id")
ea = fa.drop(columns="diagnostico_id").merge(dia, on="grupo_diagnostico")
demo("error_A_grupo_diagnostico", ea, "Relacionar por grupo_diagnostico en vez de diagnostico_id. El grupo Cardiovascular agrupa Hipertensión y Enfermedad cardiovascular, así que cada atención de ese grupo se repite dos veces.")
R["error_A_detalle"] = {"atenciones_cardiovasculares": int(fa.grupo_diagnostico.eq("Cardiovascular").sum())}

# Error B: unir pacientes con aseguradoras por régimen (Contributivo y Subsidiado tienen dos EPS cada uno)
pb = pac.merge(ase[["aseguradora_id", "regimen"]], on="aseguradora_id").drop(columns="aseguradora_id").merge(ase, on="regimen")
eb = f.merge(pb, on="paciente_id")
demo("error_B_regimen", eb, "Relacionar pacientes y aseguradoras por régimen. Contributivo y Subsidiado tienen dos EPS cada uno, así que cada atención de esos regímenes se duplica.")

# Error C: unir hecho y municipios por dos caminos a la vez (municipio del paciente y municipio del hospital) usando el nombre del municipio
fc = f.merge(pac[["paciente_id", "municipio_id"]], on="paciente_id").merge(mun[["municipio_id", "clasificacion"]], on="municipio_id")
ec = fc.merge(hos[["zona"]].assign(clasificacion=hos.zona), on="clasificacion")
demo("error_C_zona_clasificacion", ec, "Unir el municipio del paciente con los hospitales por la zona (Urbana o Intermedia) para 'asignar' hospital por zona. Hay 4 hospitales urbanos y 2 intermedios, así que cada atención se repite 2 o 4 veces y las de Lebrija (rural) desaparecen.")

# Error D: auto-unión del hecho por paciente (típico al querer calcular 'visitas del paciente' con un merge)
ed = f.merge(f[["paciente_id"]], on="paciente_id")
demo("error_D_hecho_con_hecho", ed, "Unir fact_atenciones consigo misma por paciente_id (relación muchos a muchos) para contar visitas. Un paciente con 5 atenciones aparece 25 veces.")

json.dump(R, open(os.path.join(OUT, "retos.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False, indent=1))
