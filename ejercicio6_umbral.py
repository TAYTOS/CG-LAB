"""
EJERCICIO 6 - Aplicación de umbral binario (Binary Threshold)
Lab 7 - Procesamiento de Imágenes
Computación Gráfica, Visión Computacional y Multimedia
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(base_dir, "img1.jpg")
output_dir = os.path.join(base_dir, "resultados")
os.makedirs(output_dir, exist_ok=True)

# ── Cargar imagen en escala de grises ────────────────────────────────────────
img_color = cv2.imread(input_path)
if img_color is None:
    raise FileNotFoundError(f"No se encontró {input_path}")
img_gris  = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# ── Aplicar distintos umbrales para comparación ──────────────────────────────
umbrales = [80, 127, 180]
resultados_thresh = []

for umbral_val in umbrales:
    # cv2.threshold devuelve (valor_umbral_usado, imagen_resultado)
    # THRESH_BINARY: píxel > umbral → maxVal (255), sino → 0
    ret, thresh = cv2.threshold(img_gris, umbral_val, 255, cv2.THRESH_BINARY)
    resultados_thresh.append((umbral_val, thresh))
    cv2.imwrite(os.path.join(output_dir, f"umbral_{umbral_val}.jpg"), thresh)
    pct_blanco = (thresh == 255).sum() / thresh.size * 100
    print(f"Umbral {umbral_val}: {pct_blanco:.1f}% píxeles blancos")

# ── Umbral adaptativo (bonus) ────────────────────────────────────────────────
thresh_adapt = cv2.adaptiveThreshold(
    img_gris, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    blockSize=11,
    C=2
)
cv2.imwrite(os.path.join(output_dir, "umbral_adaptativo.jpg"), thresh_adapt)

# ── Visualización ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
fig.suptitle("Ejercicio 6 — Umbral Binario (Threshold)", fontsize=14, fontweight="bold")

axes[0, 0].imshow(img_color[:, :, ::-1])
axes[0, 0].set_title("Imagen original a color")
axes[0, 0].axis("off")

axes[0, 1].imshow(img_gris, cmap="gray")
axes[0, 1].set_title("Imagen en escala de grises")
axes[0, 1].axis("off")

# Histograma con líneas de umbral
axes[0, 2].hist(img_gris.ravel(), bins=256, range=(0, 256),
                color="gray", alpha=0.7, label="Histograma")
for val, color in zip(umbrales, ["red", "orange", "purple"]):
    axes[0, 2].axvline(val, color=color, linestyle="--", linewidth=1.5, label=f"Umbral {val}")
axes[0, 2].set_title("Histograma + líneas de umbral")
axes[0, 2].set_xlabel("Intensidad")
axes[0, 2].set_ylabel("Frecuencia")
axes[0, 2].legend(fontsize=8)

for i, (val, thresh_img) in enumerate(resultados_thresh):
    axes[1, i].imshow(thresh_img, cmap="gray")
    pct = (thresh_img == 255).sum() / thresh_img.size * 100
    axes[1, i].set_title(f"Umbral = {val}\n({pct:.0f}% blanco)")
    axes[1, i].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "ej6_umbral_binario.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"\nResultado guardado en {output_dir.replace('\\', '/')}/ej6_umbral_binario.png")

print("\nFunciones de OpenCV utilizadas:")
print("  cv2.threshold(img, umbral, maxVal, tipo) → (ret, imagen)")
print("    THRESH_BINARY:     pixel > umbral ? maxVal : 0")
print("    THRESH_BINARY_INV: pixel > umbral ? 0 : maxVal")
print("    THRESH_TRUNC:      pixel > umbral ? umbral : pixel")
print("  cv2.adaptiveThreshold() → umbral local por bloques")
