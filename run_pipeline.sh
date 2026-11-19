#!/usr/bin/env bash
# Corre todo el análisis en orden. Ejecutar desde la carpeta scripts/.
set -e
python etl.py ../datos/Hospital_360_dataset_6000reg.xlsx ../datos/Hospital_360_fact_atenciones_6000.csv salida
python retos.py salida
python analisis.py salida
echo "Listo: resultados en scripts/salida/"
