"""
EJERCICIO 1 - Redimensionar imágenes
Lab 7 - Procesamiento de Imágenes
Computación Gráfica, Visión Computacional y Multimedia
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ── Carga de imágenes ────────────────────────────────────────────────────────
base_dir = os.path.dirname(os.path.abspath(__file__))
image_paths = [os.path.join(base_dir, f"img{i}.jpg") for i in range(1, 4)]

img1 = cv2.imread(image_paths[0])
img2 = cv2.imread(image_paths[1])
img3 = cv2.imread(image_paths[2])

for idx, img in enumerate((img1, img2, img3), start=1):
    if img is None:
        raise FileNotFoundError(f"No se pudo leer img{idx}.jpg en {image_paths[idx-1]}")

print("Dimensiones originales (alto x ancho x canales):")
print(f"  img1: {img1.shape}")
print(f"  img2: {img2.shape}")
print(f"  img3: {img3.shape}")

# ── Encontrar la imagen más grande ──────────────────────────────────────────
# shape devuelve (alto, ancho, canales)
max_h = max(img1.shape[0], img2.shape[0], img3.shape[0])
max_w = max(img1.shape[1], img2.shape[1], img3.shape[1])

print(f"\nDimensiones objetivo: {max_w} x {max_h} (ancho x alto)")

# ── Redimensionar todas al tamaño mayor ─────────────────────────────────────
# cv2.resize recibe (ancho, alto) — orden inverso al shape
img1_r = cv2.resize(img1, (max_w, max_h), interpolation=cv2.INTER_LINEAR)
img2_r = cv2.resize(img2, (max_w, max_h), interpolation=cv2.INTER_LINEAR)
img3_r = cv2.resize(img3, (max_w, max_h), interpolation=cv2.INTER_LINEAR)

print(f"\nDimensiones finales:")
print(f"  img1_r: {img1_r.shape}")
print(f"  img2_r: {img2_r.shape}")
print(f"  img3_r: {img3_r.shape}")

# ── Guardar resultados ───────────────────────────────────────────────────────
output_dir = os.path.join(base_dir, "resultados")
os.makedirs(output_dir, exist_ok=True)
cv2.imwrite(os.path.join(output_dir, "img1_redim.jpg"), img1_r)
cv2.imwrite(os.path.join(output_dir, "img2_redim.jpg"), img2_r)
cv2.imwrite(os.path.join(output_dir, "img3_redim.jpg"), img3_r)

# ── Visualización comparativa ────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
fig.suptitle("Ejercicio 1 — Redimensionado de Imágenes", fontsize=14, fontweight="bold")

imagenes_orig  = [img1, img2, img3]
imagenes_redim = [img1_r, img2_r, img3_r]
titulos = ["Imagen 1 (Persona)", "Imagen 2 (Perro)", "Imagen 3 (Gato)"]

for i in range(3):
    axes[0, i].imshow(cv2.cvtColor(imagenes_orig[i],  cv2.COLOR_BGR2RGB))
    axes[0, i].set_title(f"Original {titulos[i]}\n{imagenes_orig[i].shape[1]}x{imagenes_orig[i].shape[0]}")
    axes[0, i].axis("off")

    axes[1, i].imshow(cv2.cvtColor(imagenes_redim[i], cv2.COLOR_BGR2RGB))
    axes[1, i].set_title(f"Redimensionada\n{imagenes_redim[i].shape[1]}x{imagenes_redim[i].shape[0]}")
    axes[1, i].axis("off")

axes[0, 0].set_ylabel("Original", fontsize=12, labelpad=10)
axes[1, 0].set_ylabel("Redimensionada", fontsize=12, labelpad=10)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "ej1_redimensionar.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"\nResultado guardado en {output_dir.replace('\\', '/')}/ej1_redimensionar.png")
