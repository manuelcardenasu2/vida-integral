"""
Proyecto integrador: Del dato a la decisión en el sector salud
Red Hospitalaria Vida Integral - ETL y modelo de datos

Equipo: Manuel Cardenas (2251663), Santiago Saavedra, Santiago Cortes (2251513)

Qué hace este script, paso a paso:
  1. Lee las 7 tablas del Excel original y la tabla principal del CSV.
  2. Comprueba que el CSV y la hoja fact_atenciones tengan los mismos datos.
  3. Revisa llaves, duplicados, faltantes, rangos y consistencia entre campos.
  4. Homologa formatos (nombres con tilde, días en español, municipios).
  5. Agrega columnas derivadas y banderas de calidad (sin borrar registros).
  6. Construye dim_fecha y dim_triage, que no venían en el archivo.
  7. Guarda el modelo limpio en Excel y CSV, más una bitácora de cada cambio.

Uso:  python etl.py <ruta_excel> <ruta_csv> <carpeta_salida>
"""
import sys, json, os
import numpy as np
import pandas as pd

XLSX, CSV, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
log = []          # bitácora de transformaciones
calidad = {}      # métricas de calidad para el documento


def paso(tabla, accion, detalle, filas=None):
    log.append({"orden": len(log) + 1, "tabla": tabla, "accion": accion,
                "detalle": detalle, "registros_afectados": filas})


# ---------------------------------------------------------------- 1. lectura
x = pd.ExcelFile(XLSX)
f = x.parse("fact_atenciones")
pac = x.parse("dim_pacientes")
hos = x.parse("dim_hospitales")
ser = x.parse("dim_servicios")
dia = x.parse("dim_diagnosticos")
mun = x.parse("dim_municipios")
ase = x.parse("dim_aseguradoras")
csv = pd.read_csv(CSV)
paso("todas", "Lectura", f"Se leyeron {len(x.sheet_names)} hojas del Excel y el CSV de la tabla principal")

# ------------------------------------------------- 2. CSV contra hoja de Excel
csv["fecha_hora_ingreso"] = pd.to_datetime(csv["fecha_hora_ingreso"])
csv["fecha"] = pd.to_datetime(csv["fecha"])
dif = 0
for c in f.columns:
    a, b = f[c], csv[c]
    iguales = (a == b) | (a.isna() & b.isna())
    dif += int((~iguales).sum())
calidad["csv_vs_excel_celdas_distintas"] = dif
paso("fact_atenciones", "Comparación de fuentes",
     f"El CSV y la hoja fact_atenciones tienen {len(csv)} y {len(f)} filas y {dif} celdas distintas. "
     "Se trabaja con la hoja del Excel porque trae tipos de dato (fecha-hora) ya reconocidos.")

# ------------------------------------------------------- 3. limpieza de texto
for df in [f, pac, hos, ser, dia, mun, ase]:
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].astype("string").str.strip()
paso("todas", "Recorte de espacios", "Se quitaron espacios al inicio y al final de todos los campos de texto (no se encontró ninguno con espacios)")

# ------------------------------------------------------------ 4. llaves
claves = {"fact_atenciones": (f, "atencion_id"), "dim_pacientes": (pac, "paciente_id"),
          "dim_hospitales": (hos, "hospital_id"), "dim_servicios": (ser, "servicio_id"),
          "dim_diagnosticos": (dia, "diagnostico_id"), "dim_municipios": (mun, "municipio_id"),
          "dim_aseguradoras": (ase, "aseguradora_id")}
calidad["pk_duplicadas"] = {t: int(df[k].duplicated().sum()) for t, (df, k) in claves.items()}
calidad["pk_nulas"] = {t: int(df[k].isna().sum()) for t, (df, k) in claves.items()}
paso("todas", "Validación de claves primarias",
     "Ninguna clave primaria está repetida ni vacía: " + json.dumps(calidad["pk_duplicadas"]))

fks = [("paciente_id", pac), ("hospital_id", hos), ("servicio_id", ser), ("diagnostico_id", dia)]
calidad["fk_huerfanas"] = {c: int((~f[c].isin(d[c])).sum()) for c, d in fks}
calidad["fk_huerfanas"]["pacientes.municipio_id"] = int((~pac.municipio_id.isin(mun.municipio_id)).sum())
calidad["fk_huerfanas"]["pacientes.aseguradora_id"] = int((~pac.aseguradora_id.isin(ase.aseguradora_id)).sum())
paso("fact_atenciones", "Integridad referencial",
     "Todas las claves foráneas encuentran su registro en la dimensión: " + json.dumps(calidad["fk_huerfanas"]))

# ------------------------------------------------------------ 5. duplicados
calidad["dup_filas_completas"] = int(f.duplicated().sum())
calidad["dup_sin_id"] = int(f.drop(columns="atencion_id").duplicated().sum())
calidad["dup_paciente_fechahora"] = int(f.duplicated(["paciente_id", "fecha_hora_ingreso"]).sum())
mismo_dia = f[f.duplicated(["paciente_id", "fecha"], keep=False)]
calidad["pares_mismo_paciente_mismo_dia"] = int(len(mismo_dia) // 2)
calidad["mismo_dia_distinto_hospital"] = int(mismo_dia.groupby(["paciente_id", "fecha"]).hospital_id.nunique().gt(1).sum())
paso("fact_atenciones", "Detección de duplicados",
     f"0 filas idénticas, 0 repetidas al ignorar atencion_id y 0 con el mismo paciente y la misma fecha-hora. "
     f"Hay {calidad['pares_mismo_paciente_mismo_dia']} pares de atenciones del mismo paciente el mismo día, todas en hospitales "
     "distintos, a horas distintas y con servicio o diagnóstico distinto. Se interpretan como atenciones reales "
     "(por ejemplo, una remisión) y se conservan con la bandera flag_mismo_dia.", calidad["pares_mismo_paciente_mismo_dia"] * 2)
f["flag_mismo_dia"] = f.duplicated(["paciente_id", "fecha"], keep=False).astype(int)

# ------------------------------------------------------------ 6. faltantes
falt = f.isna().sum()
calidad["faltantes"] = {c: int(v) for c, v in falt.items() if v > 0}
calidad["faltantes_pct"] = {c: round(v / len(f) * 100, 2) for c, v in falt.items() if v > 0}
calidad["filas_con_algun_faltante"] = int(f.isna().any(axis=1).sum())
for c in ["tiempo_espera_min", "costo_atencion_cop", "satisfaccion_1_a_5"]:
    f["falta_" + c.split("_")[0]] = f[c].isna().astype(int)
# sensibilidad: qué pasa si se imputara con la mediana del triage
sens = {}
for c in ["tiempo_espera_min", "costo_atencion_cop", "satisfaccion_1_a_5"]:
    imp = f[c].fillna(f.groupby("triage")[c].transform("median"))
    sens[c] = {"promedio_sin_imputar": round(float(f[c].mean()), 3), "promedio_imputado": round(float(imp.mean()), 3)}
calidad["sensibilidad_imputacion"] = sens
paso("fact_atenciones", "Tratamiento de faltantes",
     "Faltan 35 tiempos de espera (0,58 %), 12 costos (0,20 %) y 55 satisfacciones (0,92 %). Están repartidos entre los 6 hospitales "
     "y los dos años, y solo una fila tiene dos faltantes. No se imputan: cada indicador excluye del numerador y del denominador "
     "las filas sin dato. Se probó imputar con la mediana del triage y los promedios casi no cambian (ver hoja calidad_datos). "
     "Se agregan las banderas falta_tiempo, falta_costo y falta_satisfaccion para poder contarlos en el tablero.",
     calidad["filas_con_algun_faltante"])

# ------------------------------------------------------------ 7. rangos y atípicos
num = ["tiempo_espera_min", "estancia_dias", "costo_atencion_cop", "satisfaccion_1_a_5", "triage"]
calidad["rangos"] = {c: {"min": float(f[c].min()), "max": float(f[c].max())} for c in num}
atip = {}
for c in ["tiempo_espera_min", "estancia_dias", "costo_atencion_cop"]:
    q1, q3 = f[c].quantile([.25, .75]); lim = q3 + 1.5 * (q3 - q1)
    atip[c] = {"limite_superior_iqr": float(lim), "n_sobre_limite": int((f[c] > lim).sum())}
calidad["atipicos_iqr"] = atip
paso("fact_atenciones", "Rangos y atípicos",
     "No hay negativos, el triage está entre 1 y 5 y la satisfacción entre 1 y 5. Los valores altos por la regla del rango "
     "intercuartílico son plausibles (espera máxima 341 min, estancia máxima 11,8 días, costo máximo 4,7 millones en UCI y cirugía). "
     "Se conservan porque describen casos reales de la operación y borrarlos bajaría artificialmente los promedios.")

# ------------------------------------------------------------ 8. consistencia
calidad["fecha_vs_fechahora"] = int((f.fecha_hora_ingreso.dt.normalize() != f.fecha).sum())
calidad["mes_vs_fechahora"] = int((f.fecha_hora_ingreso.dt.strftime("%Y-%m") != f.mes).sum())
calidad["dia_vs_fechahora"] = int((f.fecha_hora_ingreso.dt.day_name() != f.dia_semana).sum())
ct = pd.crosstab(f.resultado_atencion, f.es_hospitalizado)
calidad["resultado_vs_hospitalizado"] = {r: {str(k): int(v) for k, v in row.items()} for r, row in ct.iterrows()}
paso("fact_atenciones", "Consistencia de campos derivados",
     "fecha, mes y dia_semana coinciden con fecha_hora_ingreso en las 6.000 filas. es_hospitalizado es coherente con el resultado: "
     "vale 1 en todas las hospitalizaciones y UCI y 0 en todas las altas y observaciones. Las remisiones tienen ambos valores "
     "(241 sin hospitalizar y 159 hospitalizadas), lo que es posible porque un paciente puede remitirse después de estar internado.")

bins = [-1, 17, 29, 44, 59, 200]; lab = ["0-17", "18-29", "30-44", "45-59", "60+"]
calidad["grupo_edad_inconsistente"] = int((pd.cut(pac.edad, bins, labels=lab).astype(str) != pac.grupo_edad).sum())

m = f.merge(pac, on="paciente_id", how="left")
r1 = (m.diagnostico_id == "D07") & (m.sexo == "M")
r2 = (m.servicio_id == "S06") & (m.sexo == "M")
r3 = (m.servicio_id == "S04") & (m.edad > 17)
r4 = (m.diagnostico_id == "D07") & ((m.edad < 12) | (m.edad > 55))
calidad["inconsistencias_clinicas"] = {
    "embarazo_en_hombres": int(r1.sum()), "ginecologia_en_hombres": int(r2.sum()),
    "pediatria_mayores_17": int(r3.sum()), "embarazo_edad_menor_12_o_mayor_55": int(r4.sum())}
f["flag_inconsistencia_clinica"] = (r1 | r2 | r3 | r4).astype(int).values
calidad["filas_con_inconsistencia_clinica"] = int(f.flag_inconsistencia_clinica.sum())
paso("fact_atenciones", "Consistencia clínica",
     f"Al cruzar con dim_pacientes aparecen combinaciones imposibles: {int(r1.sum())} complicaciones del embarazo en hombres, "
     f"{int(r2.sum())} atenciones de ginecología en hombres, {int(r3.sum())} atenciones de pediatría en mayores de 17 años y "
     f"{int(r4.sum())} complicaciones del embarazo en menores de 12 o mayores de 55. Se marcan con flag_inconsistencia_clinica "
     "y no se borran, porque los tiempos, costos y resultados de esas atenciones siguen siendo válidos para los indicadores "
     "operativos. Por esta razón no se sacan conclusiones que crucen sexo o edad con servicio o diagnóstico.",
     calidad["filas_con_inconsistencia_clinica"])

# validación del indicador de reingreso con las fechas
g = f.sort_values(["paciente_id", "fecha_hora_ingreso"]).copy()
g["dias_sig"] = (g.groupby("paciente_id").fecha_hora_ingreso.shift(-1) - g.fecha_hora_ingreso).dt.total_seconds() / 86400
g["dias_ant"] = (g.fecha_hora_ingreso - g.groupby("paciente_id").fecha_hora_ingreso.shift(1)).dt.total_seconds() / 86400
g["vuelve_30"] = (g.dias_sig <= 30).astype(int)
g["venia_30"] = (g.dias_ant <= 30).astype(int)
calidad["reingreso_validacion"] = {
    "marcados_reingreso": int(g.reingreso_30_dias.sum()),
    "marcados_con_atencion_siguiente_30d": int(((g.reingreso_30_dias == 1) & (g.vuelve_30 == 1)).sum()),
    "marcados_con_atencion_anterior_30d": int(((g.reingreso_30_dias == 1) & (g.venia_30 == 1)).sum()),
    "no_marcados_con_atencion_siguiente_30d": int(((g.reingreso_30_dias == 0) & (g.vuelve_30 == 1)).sum())}
f = f.merge(g[["atencion_id", "vuelve_30"]].rename(columns={"vuelve_30": "regresa_red_30d"}), on="atencion_id")
paso("fact_atenciones", "Validación del reingreso",
     "Se comparó reingreso_30_dias con las fechas reales de cada paciente en la red. De 489 atenciones marcadas, solo 31 tienen otra "
     "atención en los 30 días siguientes y 37 en los 30 días anteriores; 370 atenciones sin marca sí tienen regreso en 30 días. "
     "El campo no se puede reconstruir con los registros, así que se usa tal como viene (es el dato oficial) y se reporta como "
     "limitación. Se agrega regresa_red_30d, calculado con fechas, solo como referencia.")

# ------------------------------------------------------------ 9. homologación
dias = {"Monday": "Lunes", "Tuesday": "Martes", "Wednesday": "Miércoles", "Thursday": "Jueves",
        "Friday": "Viernes", "Saturday": "Sábado", "Sunday": "Domingo"}
f["dia_semana"] = f.dia_semana.map(dias)
f["dia_semana_num"] = f.fecha_hora_ingreso.dt.dayofweek + 1
paso("fact_atenciones", "Homologación", "dia_semana pasó de inglés a español (Monday a Lunes) y se agregó dia_semana_num para ordenar", len(f))
res = {"Alta": "Alta", "Hospitalizacion": "Hospitalización", "Observacion": "Observación", "Remision": "Remisión", "UCI": "UCI"}
f["resultado_atencion"] = f.resultado_atencion.map(res)
paso("fact_atenciones", "Homologación", "resultado_atencion con tildes: Hospitalizacion, Observacion y Remision", int(f.resultado_atencion.isin(["Hospitalización", "Observación", "Remisión"]).sum()))
ser["servicio"] = ser.servicio.replace({"Pediatria": "Pediatría", "Cirugia": "Cirugía", "Ginecologia": "Ginecología"})
ser["tipo_atencion"] = ser.tipo_atencion.replace({"Hospitalizacion": "Hospitalización"})
dia["diagnostico"] = dia.diagnostico.replace({"Hipertension": "Hipertensión", "Dolor musculoesqueletico": "Dolor musculoesquelético"})
dia["grupo_diagnostico"] = dia.grupo_diagnostico.replace({"Metabolica": "Metabólica"})
dia["curso"] = dia.curso.replace({"Cronica": "Crónica"})
hos["institucion"] = hos.institucion.replace({"Clinica del Norte": "Clínica del Norte", "Centro Medico Oriente": "Centro Médico Oriente"})
mun["municipio"] = mun.municipio.replace({"Giron": "Girón"})
mun["clasificacion"] = mun.clasificacion.replace({"Rural-periferica": "Rural-periférica"})
ase["regimen"] = ase.regimen.replace({"Regimen especial": "Régimen especial"})
paso("dimensiones", "Homologación", "Tildes en nombres de servicios, diagnósticos, hospitales, municipios y regímenes (Pediatria a Pediatría, Giron a Girón, etc.)")
# dim_hospitales traía el municipio como texto: se reemplaza por la clave
hos = hos.merge(mun[["municipio_id", "municipio"]], on="municipio", how="left")
calidad["hospitales_sin_municipio_id"] = int(hos.municipio_id.isna().sum())
hos = hos[["hospital_id", "institucion", "municipio_id", "municipio", "zona", "complejidad"]]
paso("dim_hospitales", "Homologación de clave", "El municipio venía como texto y con una grafía distinta a dim_municipios (Girón y Giron). "
     "Después de homologar se agregó municipio_id para que el hospital y el paciente usen el mismo código de municipio", 6)
pac["sexo"] = pac.sexo.map({"F": "Femenino", "M": "Masculino"})
paso("dim_pacientes", "Homologación", "sexo de F/M a Femenino/Masculino para que el tablero sea legible", len(pac))

# ------------------------------------------------------------ 10. derivadas
f["anio"] = f.fecha_hora_ingreso.dt.year
f["mes_num"] = f.fecha_hora_ingreso.dt.month
f["trimestre"] = f.fecha_hora_ingreso.dt.to_period("Q").astype(str).str.replace("Q", "-T")
f["hora"] = f.fecha_hora_ingreso.dt.hour
f["franja"] = pd.cut(f.hora, [-1, 5, 11, 17, 23], labels=["Madrugada (0-5)", "Mañana (6-11)", "Tarde (12-17)", "Noche (18-23)"]).astype(str)
f["triage_ii_en_meta"] = np.where(f.triage == 2, np.where(f.tiempo_espera_min.isna(), np.nan, (f.tiempo_espera_min <= 30).astype(float)), np.nan)
f["espera_mayor_120"] = np.where(f.tiempo_espera_min.isna(), np.nan, (f.tiempo_espera_min > 120).astype(float))
paso("fact_atenciones", "Columnas derivadas", "anio, mes_num, trimestre, hora, franja horaria, triage_ii_en_meta (1 si un triage II esperó 30 min o menos) y espera_mayor_120", len(f))

# ------------------------------------------------------------ 11. dimensiones nuevas
fechas = pd.date_range("2025-01-01", "2026-10-31", freq="D")
meses_es = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
dfe = pd.DataFrame({"fecha": fechas})
dfe["anio"] = dfe.fecha.dt.year; dfe["mes_num"] = dfe.fecha.dt.month
dfe["mes_nombre"] = dfe.mes_num.map(lambda i: meses_es[i - 1])
dfe["anio_mes"] = dfe.fecha.dt.strftime("%Y-%m")
dfe["trimestre"] = dfe.fecha.dt.to_period("Q").astype(str).str.replace("Q", "-T")
dfe["dia_semana"] = dfe.fecha.dt.day_name().map(dias); dfe["dia_semana_num"] = dfe.fecha.dt.dayofweek + 1
dfe["fin_de_semana"] = (dfe.dia_semana_num >= 6).astype(int)
paso("dim_fecha", "Creación", f"Calendario continuo de {len(dfe)} días (2025-01-01 a 2026-10-31) para filtrar por periodo sin huecos")
tri = pd.DataFrame({"triage": [1, 2, 3, 4, 5], "nivel": ["I", "II", "III", "IV", "V"],
                    "descripcion": ["Atención inmediata", "Atención en máximo 30 minutos", "Urgencia, requiere medidas diagnósticas",
                                    "Sin riesgo evidente para la vida", "Condición no urgente"],
                    "grupo_prioridad": ["Alta prioridad (I-II)", "Alta prioridad (I-II)", "Prioridad media y baja (III-V)",
                                        "Prioridad media y baja (III-V)", "Prioridad media y baja (III-V)"],
                    "meta_minutos": [0, 30, None, None, None]})
paso("dim_triage", "Creación", "Tabla de los 5 niveles con la descripción y los tiempos de la Resolución 5596 de 2015 (I inmediata, II hasta 30 minutos)")
calidad["pacientes_dim"] = int(len(pac)); calidad["pacientes_con_atencion"] = int(f.paciente_id.nunique())
calidad["pacientes_sin_atencion"] = int(len(pac) - f.paciente_id.nunique())
paso("dim_pacientes", "Revisión de cobertura", f"dim_pacientes tiene {len(pac)} pacientes, pero solo {f.paciente_id.nunique()} aparecen en fact_atenciones. "
     f"Los {calidad['pacientes_sin_atencion']} restantes se conservan en la dimensión (no estorban) pero no cuentan como pacientes atendidos.")

# ------------------------------------------------------------ 12. orden y guardado
f["fecha"] = f.fecha.dt.date
dfe["fecha"] = dfe.fecha.dt.date
cols = ["atencion_id", "fecha_hora_ingreso", "fecha", "anio", "mes", "mes_num", "trimestre", "dia_semana", "dia_semana_num", "hora", "franja",
        "paciente_id", "hospital_id", "servicio_id", "diagnostico_id", "triage", "tiempo_espera_min", "estancia_dias",
        "resultado_atencion", "es_hospitalizado", "reingreso_30_dias", "costo_atencion_cop", "satisfaccion_1_a_5",
        "triage_ii_en_meta", "espera_mayor_120", "falta_tiempo", "falta_costo", "falta_satisfaccion",
        "flag_mismo_dia", "flag_inconsistencia_clinica", "regresa_red_30d"]
f = f[cols].sort_values("atencion_id")

cal = pd.DataFrame([{"chequeo": k, "resultado": json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v} for k, v in calidad.items()])
with pd.ExcelWriter(os.path.join(OUT, "modelo_vida_integral.xlsx"), engine="openpyxl") as w:
    f.to_excel(w, "fact_atenciones", index=False)
    pac.to_excel(w, "dim_pacientes", index=False); hos.to_excel(w, "dim_hospitales", index=False)
    ser.to_excel(w, "dim_servicios", index=False); dia.to_excel(w, "dim_diagnosticos", index=False)
    mun.to_excel(w, "dim_municipios", index=False); ase.to_excel(w, "dim_aseguradoras", index=False)
    dfe.to_excel(w, "dim_fecha", index=False); tri.to_excel(w, "dim_triage", index=False)
    pd.DataFrame(log).to_excel(w, "bitacora_transformaciones", index=False)
    cal.to_excel(w, "calidad_datos", index=False)
f.to_csv(os.path.join(OUT, "fact_atenciones_limpia.csv"), index=False, encoding="utf-8-sig")
json.dump({"calidad": calidad, "bitacora": log}, open(os.path.join(OUT, "etl_log.json"), "w"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(calidad, ensure_ascii=False, indent=1, default=str))
print(len(log), "pasos en la bitácora")
