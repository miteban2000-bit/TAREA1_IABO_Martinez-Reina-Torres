"""
03_visualizacion_reduccion.py
-------------------------------
Punto 2 del enunciado: reducción de dimensionalidad + gráfica de
dispersión que muestre la mejor separación entre clases.

Probamos DOS técnicas y nos quedamos con la que separe mejor:
  - PCA (no supervisada): busca las direcciones de mayor varianza.
  - LDA (supervisada): busca las direcciones que MAXIMIZAN la
    separación entre clases -- casi siempre da mejor separación
    visual que PCA cuando el objetivo es clasificar.

Genera:
  - dispersión_pca_2d.png / dispersión_lda_2d.png
  - dispersión_pca_3d.png / dispersión_lda_3d.png (si hay >=3 clases)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

df = pd.read_csv("caracteristicas.csv")
X = df.drop(columns=["clase", "ventana_id"]).values
y = df["clase"].values
clases_unicas = np.unique(y)

# Estandarizar (crítico: media, varianza, energía, etc. tienen escalas muy distintas)
X_std = StandardScaler().fit_transform(X)

def graficar_2d(X_proy, y, titulo, nombre_archivo, ejes_lbl):
    plt.figure(figsize=(7, 6))
    for c in clases_unicas:
        mask = y == c
        plt.scatter(X_proy[mask, 0], X_proy[mask, 1], label=c, alpha=0.7, s=50)
    plt.xlabel(ejes_lbl[0]); plt.ylabel(ejes_lbl[1])
    plt.title(titulo); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(nombre_archivo, dpi=150)
    plt.close()
    print(f"Guardado: {nombre_archivo}")

def graficar_3d(X_proy, y, titulo, nombre_archivo, ejes_lbl):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    for c in clases_unicas:
        mask = y == c
        ax.scatter(X_proy[mask, 0], X_proy[mask, 1], X_proy[mask, 2],
                   label=c, alpha=0.7, s=50)
    ax.set_xlabel(ejes_lbl[0]); ax.set_ylabel(ejes_lbl[1]); ax.set_zlabel(ejes_lbl[2])
    ax.set_title(titulo); ax.legend()
    plt.tight_layout()
    plt.savefig(nombre_archivo, dpi=150)
    plt.close()
    print(f"Guardado: {nombre_archivo}")

# --- PCA ---
pca_2d = PCA(n_components=2).fit_transform(X_std)
graficar_2d(pca_2d, y, "PCA 2D", "dispersion_pca_2d.png", ["PC1", "PC2"])

if len(clases_unicas) >= 3 and X.shape[1] >= 3:
    pca_3d = PCA(n_components=3).fit_transform(X_std)
    graficar_3d(pca_3d, y, "PCA 3D", "dispersion_pca_3d.png", ["PC1", "PC2", "PC3"])

# --- LDA (supervisado, suele separar mejor las clases) ---
n_componentes_lda = min(len(clases_unicas) - 1, X.shape[1])
lda_2d = LDA(n_components=min(2, n_componentes_lda)).fit_transform(X_std, y)
if lda_2d.shape[1] == 1:
    # Si solo hay 2 clases, LDA da 1 sola dimensión: añadimos un jitter en Y para graficar
    lda_2d = np.column_stack([lda_2d, np.random.normal(0, 0.05, len(lda_2d))])
graficar_2d(lda_2d, y, "LDA 2D (supervisado)", "dispersion_lda_2d.png", ["LD1", "LD2"])

if n_componentes_lda >= 3:
    lda_3d = LDA(n_components=3).fit_transform(X_std, y)
    graficar_3d(lda_3d, y, "LDA 3D (supervisado)", "dispersion_lda_3d.png",
                ["LD1", "LD2", "LD3"])

print("\nRecomendación: para el informe usa la proyección LDA -- al ser "
      "supervisada, maximiza la separación entre clases y normalmente "
      "se ve mejor separada que PCA.")
