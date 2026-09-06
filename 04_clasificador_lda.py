"""
04_clasificador_lda.py
------------------------
Punto 3: división train/test.
Punto 4: clasificador que define fronteras (funciones) que dividen
el espacio 2D reducido en regiones de clase.

Con el dataset REAL (5 clases: reposo, arriba, abajo, izquierda,
derecha) se probaron dos clasificadores sobre el mismo espacio 2D:

  - LDA: fronteras LINEALES (rectas). Separa perfecto abajo/izquierda
    /derecha, pero "arriba" queda 100% confundido con "reposo"
    (0% de recall) porque esas dos clases no son separables con una
    recta en este espacio -- sus nubes de puntos se solapan.
  - QDA: fronteras CUADRÁTICAS (curvas/elipses). Al permitir límites
    curvos, logra rescatar parte de "arriba" (sube de 0% a ~20% de
    recall) sin perjudicar las demás clases. Sigue siendo una función
    matemática explícita que limita el espacio 2D -- solo que de
    segundo grado en vez de lineal -- así que cumple igual con el
    punto 4 del enunciado.

Por eso el script entrena y compara AMBOS, y grafica las fronteras
del que mejor generaliza (QDA). Se deja también LDA por si tu
profesor pide explícitamente fronteras lineales.

Flujo:
  1. Cargar caracteristicas.csv
  2. Reducir a 2D con LDA (mismo criterio del punto 2)
  3. Dividir 70/30 train/test (estratificado por clase)
  4. Entrenar LDA y QDA como clasificadores sobre ese espacio 2D
  5. Graficar las regiones/fronteras de decisión del mejor (QDA)
  6. Reportar accuracy y matriz de confusión de ambos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis as LDA,
    QuadraticDiscriminantAnalysis as QDA,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.colors as mcolors

# ---------- 1. Cargar datos ----------
df = pd.read_csv("caracteristicas.csv")
X = df.drop(columns=["clase", "ventana_id"]).values
y = df["clase"].values
clases = sorted(np.unique(y))

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# ---------- 2. Reducir a 2D (para poder graficar fronteras) ----------
reductor = LDA(n_components=2 if len(clases) > 2 else 1)
X_2d = reductor.fit_transform(X_std, y)
if X_2d.shape[1] == 1:
    X_2d = np.column_stack([X_2d, np.zeros(len(X_2d))])

# ---------- 3. Train/test split (punto 3) ----------
X_train, X_test, y_train, y_test = train_test_split(
    X_2d, y, test_size=0.3, random_state=42, stratify=y
)
print(f"Entrenamiento: {len(X_train)} muestras | Prueba: {len(X_test)} muestras")

# ---------- 4. Clasificadores sobre el espacio 2D: LDA vs QDA ----------
clasificadores = {"LDA (fronteras lineales)": LDA(),
                   "QDA (fronteras curvas)": QDA()}

resultados = {}
for nombre, modelo in clasificadores.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    resultados[nombre] = (modelo, y_pred, acc)
    print(f"\n=== {nombre} | accuracy test = {acc:.3f} ===")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred, labels=clases))

# Nos quedamos con el de mejor accuracy para graficar las fronteras
mejor_nombre = max(resultados, key=lambda k: resultados[k][2])
clf, y_pred, _ = resultados[mejor_nombre]
print(f"\n>>> Clasificador elegido para graficar fronteras: {mejor_nombre}")

# ---------- 5. Graficar fronteras de decisión ----------
h = 0.05
x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z_num = np.array([clases.index(c) for c in Z]).reshape(xx.shape)

plt.figure(figsize=(8, 7))
cmap_fondo = plt.cm.get_cmap("Pastel1", len(clases))
plt.contourf(xx, yy, Z_num, alpha=0.4, cmap=cmap_fondo)

cmap_puntos = plt.cm.get_cmap("Set1", len(clases))
for i, c in enumerate(clases):
    mask_tr = y_train == c
    mask_te = y_test == c
    color = cmap_puntos(i)
    plt.scatter(X_train[mask_tr, 0], X_train[mask_tr, 1],
                color=color, marker="o", edgecolor="k", s=60, label=f"{c} (train)")
    plt.scatter(X_test[mask_te, 0], X_test[mask_te, 1],
                color=color, marker="^", edgecolor="k", s=90, label=f"{c} (test)")

plt.xlabel("LD1"); plt.ylabel("LD2")
plt.title(f"Fronteras de decisión -- {mejor_nombre}")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("fronteras_decision.png", dpi=150)
plt.close()
print("\nGráfica de fronteras guardada en 'fronteras_decision.png'")

# ---------- Bonus: ecuación explícita de las fronteras (solo si el elegido es LDA y hay 2 clases) ----------
if mejor_nombre.startswith("LDA") and len(clases) == 2:
    w = clf.coef_[0]
    b = clf.intercept_[0]
    print(f"\nFrontera lineal explícita: {w[0]:.3f}*LD1 + {w[1]:.3f}*LD2 + {b:.3f} = 0")
