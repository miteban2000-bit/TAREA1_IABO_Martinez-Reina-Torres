"""
02_parsear_log_arduino.py
---------------------------
Reemplaza al script genérico de extracción de características para
el caso real: el Arduino Nano 33 BLE Sense manda todo por el puerto
serial en un único archivo de texto (log), usando el protocolo:

    Listo. Formato de comando: <etiqueta> <numero_de_ventanas>
    INICIO_VENTANA <etiqueta>
    ax,ay,az,gx,gy,gz          <- header (a veces se repite, a veces no)
    0.0417,-1.0219,...          <- filas de datos (acelerómetro+giroscopio)
    ...
    FIN_VENTANA
    Progreso: n/N
    Lote completado: <etiqueta>
    ...

Este script:
  1. Recorre el .txt línea por línea con una máquina de estados simple.
  2. Ignora líneas de control ("Listo...", "Progreso...", "Lote completado...").
  3. Agrupa las filas numéricas entre cada INICIO_VENTANA/FIN_VENTANA.
  4. Descarta ventanas que quedaron incompletas (p. ej. si llega un
     nuevo INICIO_VENTANA antes de que llegue el FIN_VENTANA anterior
     -> se perdió la transmisión de esa ventana).
  5. Calcula las 6 características (media, varianza, curtosis,
     simetría, entropía, energía) por eje (ax,ay,az,gx,gy,gz) para
     cada ventana.
  6. Guarda 'caracteristicas.csv'.
"""

import re
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew, entropy

RUTA_LOG = "/mnt/user-data/uploads/DATASET_TAREA_CLASIFICADOR.txt"
EJES = ["ax", "ay", "az", "gx", "gy", "gz"]
PATRON_FILA = re.compile(r'^-?\d+\.\d+,-?\d+\.\d+,-?\d+\.\d+,-?\d+\.\d+,-?\d+\.\d+,-?\d+\.\d+$')

# ---------- 1. Parsear el log a ventanas crudas ----------
def parsear_log(ruta):
    ventanas = []   # lista de (etiqueta, DataFrame)
    etiqueta_actual = None
    filas_actuales = []
    en_ventana = False
    descartadas = 0

    with open(ruta, encoding="utf-8", errors="replace") as f:
        for linea in f:
            linea = linea.strip()
            if not linea:
                continue

            if linea.startswith("INICIO_VENTANA"):
                # Si había una ventana abierta sin FIN_VENTANA, se perdió -> descartar
                if en_ventana:
                    descartadas += 1
                partes = linea.split()
                etiqueta_actual = partes[1] if len(partes) > 1 else None
                filas_actuales = []
                en_ventana = True

            elif linea.startswith("FIN_VENTANA"):
                if en_ventana and etiqueta_actual is not None and len(filas_actuales) > 0:
                    df = pd.DataFrame(filas_actuales, columns=EJES)
                    ventanas.append((etiqueta_actual, df))
                en_ventana = False
                etiqueta_actual = None
                filas_actuales = []

            elif en_ventana and PATRON_FILA.match(linea):
                valores = [float(x) for x in linea.split(",")]
                filas_actuales.append(valores)

            # cualquier otra línea (header "ax,ay,az...", "Listo...",
            # "Progreso...", "Lote completado...") se ignora

    if descartadas:
        print(f"Aviso: se descartaron {descartadas} ventana(s) incompleta(s) "
              f"(transmisión serial cortada a mitad de captura).")
    return ventanas

# ---------- 2. Características por ventana ----------
def calcular_entropia(señal, bins=10):
    hist, _ = np.histogram(señal, bins=bins, density=True)
    hist = hist[hist > 0]
    return entropy(hist)

def calcular_energia(señal):
    return np.sum(señal ** 2) / len(señal)

def extraer_features_ventana(df_ventana):
    fila = {}
    for eje in EJES:
        s = df_ventana[eje].values
        fila[f"{eje}_media"] = np.mean(s)
        fila[f"{eje}_varianza"] = np.var(s)
        fila[f"{eje}_curtosis"] = kurtosis(s)
        fila[f"{eje}_simetria"] = skew(s)
        fila[f"{eje}_entropia"] = calcular_entropia(s)
        fila[f"{eje}_energia"] = calcular_energia(s)
    return fila

def main():
    ventanas = parsear_log(RUTA_LOG)
    filas = []
    for idx, (etiqueta, df_ventana) in enumerate(ventanas):
        fila = extraer_features_ventana(df_ventana)
        fila["clase"] = etiqueta
        fila["ventana_id"] = idx
        filas.append(fila)

    df_features = pd.DataFrame(filas)
    df_features.to_csv("caracteristicas.csv", index=False)

    print(f"\nVentanas válidas procesadas: {len(df_features)}")
    print(f"Características por ventana: {df_features.shape[1] - 2}")
    print("\nConteo de ventanas por clase:")
    print(df_features["clase"].value_counts())
    print("\nGuardado en 'caracteristicas.csv'")

if __name__ == "__main__":
    main()
