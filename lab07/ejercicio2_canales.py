"""
EJERCICIO 2 - Crear imagen con canales de color combinados
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

# ── Cargar imágenes redimensionadas del ejercicio anterior ───────────────────
img1_r = cv2.imread(os.path.join(output_dir, "img1_redim.jpg"))
img2_r = cv2.imread(os.path.join(output_dir, "img2_redim.jpg"))
img3_r = cv2.imread(os.path.join(output_dir, "img3_redim.jpg"))

for idx, img in enumerate((img1_r, img2_r, img3_r), start=1):
    if img is None:
        raise FileNotFoundError(f"No se encontró {os.path.join(output_dir, f'img{idx}_redim.jpg')}")

# Verificar que tengan el mismo tamaño
assert img1_r.shape == img2_r.shape == img3_r.shape, "Las imágenes deben tener el mismo tamaño"

# ── Extraer canales individuales ─────────────────────────────────────────────
# OpenCV maneja el orden BGR (Blue, Green, Red), no RGB
# img[:, :, 0] = Canal Azul  (B)
# img[:, :, 1] = Canal Verde (G)
# img[:, :, 2] = Canal Rojo  (R)

canal_r = img1_r[:, :, 2]   # Canal Rojo   de imagen 1
canal_g = img2_r[:, :, 1]   # Canal Verde  de imagen 2
canal_b = img3_r[:, :, 0]   # Canal Azul   de imagen 3

print("Canales extraídos:")
print(f"  Canal R (img1) — min: {canal_r.min()}, max: {canal_r.max()}, media: {canal_r.mean():.1f}")
print(f"  Canal G (img2) — min: {canal_g.min()}, max: {canal_g.max()}, media: {canal_g.mean():.1f}")
print(f"  Canal B (img3) — min: {canal_b.min()}, max: {canal_b.max()}, media: {canal_b.mean():.1f}")

# ── Combinar canales en nueva imagen ─────────────────────────────────────────
# cv2.merge espera orden [B, G, R]
imagen_combinada = cv2.merge([canal_b, canal_g, canal_r])

cv2.imwrite(os.path.join(output_dir, "combinada.jpg"), imagen_combinada)
print(f"\nImagen combinada guardada — tamaño: {imagen_combinada.shape}")

# ── Visualización ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
fig.suptitle("Ejercicio 2 — Combinación de Canales de Color", fontsize=14, fontweight="bold")

# Fila 1: fuentes
axes[0, 0].imshow(cv2.cvtColor(img1_r, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Img 1 (fuente canal R)")
axes[0, 1].imshow(cv2.cvtColor(img2_r, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title("Img 2 (fuente canal G)")
axes[0, 2].imshow(cv2.cvtColor(img3_r, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title("Img 3 (fuente canal B)")
axes[0, 3].imshow(cv2.cvtColor(imagen_combinada, cv2.COLOR_BGR2RGB))
axes[0, 3].set_title("Imagen Combinada\n(R+G+B de distintas imágenes)")

# Fila 2: canales aislados
axes[1, 0].imshow(canal_r, cmap="Reds_r")
axes[1, 0].set_title("Canal R extraído")
axes[1, 1].imshow(canal_g, cmap="Greens_r")
axes[1, 1].set_title("Canal G extraído")
axes[1, 2].imshow(canal_b, cmap="Blues_r")
axes[1, 2].set_title("Canal B extraído")
# Histograma de la imagen combinada
axes[1, 3].set_title("Histograma combinada")
colores = ("red", "green", "blue")
for i, color in enumerate(colores):
    hist = cv2.calcHist([imagen_combinada], [i], None, [256], [0, 256])
    axes[1, 3].plot(hist, color=color, alpha=0.7)
axes[1, 3].set_xlim([0, 256])
axes[1, 3].set_xlabel("Intensidad")
axes[1, 3].set_ylabel("Frecuencia")

for ax in axes.flat:
    ax.axis("off") if ax != axes[1, 3] else None

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "ej2_canales_combinados.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Resultado guardado en {output_dir.replace('\\', '/')}/ej2_canales_combinados.png")
