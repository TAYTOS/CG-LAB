"""
EJERCICIO 3 - Conversión a negativo y escala de grises
Lab 7 - Procesamiento de Imágenes
Computación Gráfica, Visión Computacional y Multimedia
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "resultados")
os.makedirs(output_dir, exist_ok=True)

# ── Cargar imagen combinada del ejercicio anterior ───────────────────────────
img_comb = cv2.imread(os.path.join(output_dir, "combinada.jpg"))
if img_comb is None:
    raise FileNotFoundError(f"No se encontró {os.path.join(output_dir, 'combinada.jpg')}. Ejecutar ejercicio 2 primero.")
print(f"Imagen combinada cargada — forma: {img_comb.shape}, dtype: {img_comb.dtype}")

# ── Paso 1: Convertir a negativo ─────────────────────────────────────────────
# Negativo: invertir cada valor de píxel (0→255, 255→0)
# Operación: negativo = 255 - imagen
# NumPy opera elemento a elemento, muy eficiente
negativo = 255 - img_comb

negativo_path = os.path.join(output_dir, "negativo.jpg")
cv2.imwrite(negativo_path, negativo)
print(f"Negativo guardado en {negativo_path.replace('\\', '/')}" )

# Verificación: el negativo del negativo debe ser igual al original
doble_negativo = 255 - negativo
diferencia = np.sum(np.abs(img_comb.astype(int) - doble_negativo.astype(int)))
print(f"Verificación (diferencia negativo doble vs original): {diferencia} (debe ser 0)")

# ── Paso 2: Convertir negativo a escala de grises ────────────────────────────
# Cargar el negativo guardado en modo GRAYSCALE
# cv2.IMREAD_GRAYSCALE aplica la fórmula: Gray = 0.299*R + 0.587*G + 0.114*B
gris = cv2.imread(os.path.join(output_dir, "negativo.jpg"), cv2.IMREAD_GRAYSCALE)
if gris is None:
    raise FileNotFoundError(f"No se pudo cargar el negativo desde {os.path.join(output_dir, 'negativo.jpg')}")

cv2.imwrite(os.path.join(output_dir, "gris.jpg"), gris)
print(f"Imagen en grises guardada en {output_dir.replace('\\', '/')}/gris.jpg")
print(f"Forma imagen gris: {gris.shape}  (sin canal de color)")

# ── Visualización comparativa ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
fig.suptitle("Ejercicio 3 — Negativo y Escala de Grises", fontsize=14, fontweight="bold")

axes[0].imshow(cv2.cvtColor(img_comb, cv2.COLOR_BGR2RGB))
axes[0].set_title("Imagen original\n(canales combinados)")
axes[0].axis("off")

axes[1].imshow(cv2.cvtColor(negativo, cv2.COLOR_BGR2RGB))
axes[1].set_title("Negativo\n(255 - valor pixel)")
axes[1].axis("off")

axes[2].imshow(gris, cmap="gray")
axes[2].set_title("Escala de grises\n(del negativo)")
axes[2].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "ej3_negativo_grises.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Resultado guardado en {output_dir.replace('\\', '/')}/ej3_negativo_grises.png")

# ── Histogramas ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("Histogramas — Original vs Negativo", fontsize=13, fontweight="bold")

for canal, color, label in zip(range(3), ("blue","green","red"), ("B","G","R")):
    h_orig = cv2.calcHist([img_comb],  [canal], None, [256], [0, 256])
    h_neg  = cv2.calcHist([negativo],  [canal], None, [256], [0, 256])
    axes[0].plot(h_orig, color=color, alpha=0.7, label=label)
    axes[1].plot(h_neg,  color=color, alpha=0.7, label=label)

for ax, titulo in zip(axes, ("Original", "Negativo")):
    ax.set_title(titulo)
    ax.set_xlim([0, 256])
    ax.set_xlabel("Intensidad")
    ax.set_ylabel("Frecuencia")
    ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "ej3_histogramas.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Histogramas guardados en {output_dir.replace('\\', '/')}/ej3_histogramas.png")
