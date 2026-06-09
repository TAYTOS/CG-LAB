"""
EJERCICIO 5 - Dibujo de figuras y texto en imágenes
Lab 7 - Procesamiento de Imágenes
Computación Gráfica, Visión Computacional y Multimedia
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(base_dir, "img1.jpg")
output_dir = os.path.join(base_dir, "resultados")
os.makedirs(output_dir, exist_ok=True)

# ── Cargar imagen 1 (persona) ────────────────────────────────────────────────
img_orig = cv2.imread(image_path)
if img_orig is None:
    raise FileNotFoundError(f"No se encontró {image_path}")
img = img_orig.copy()

h, w = img.shape[:2]
print(f"Imagen cargada — tamaño: {w}x{h}")

# ── Detección aproximada de cara ─────────────────────────────────────────────
# Para imágenes generadas: la cabeza está aprox. en (320, 170) con radio 60
# Con imágenes reales se usaría un clasificador Haar; aquí calculamos el centro
centro_x = w // 2
centro_y = int(h * 0.35)
radio     = int(min(w, h) * 0.13)

print(f"Centro estimado de la cara: ({centro_x}, {centro_y}), radio: {radio}")

# ── Dibujar círculo sobre la cara ────────────────────────────────────────────
# cv2.circle(imagen, centro, radio, color_BGR, grosor)
# grosor=-1 rellena; grosor>0 dibuja contorno
cv2.circle(img, (centro_x, centro_y), radio + 10, (0, 255, 0), 3)

# ── Agregar texto descriptivo ────────────────────────────────────────────────
etiqueta   = "Persona"
pos_texto  = (centro_x - 40, centro_y - radio - 20)

# Fondo del texto para mejor legibilidad
(tw, th), baseline = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
cv2.rectangle(img,
              (pos_texto[0] - 4,       pos_texto[1] - th - 4),
              (pos_texto[0] + tw + 4,  pos_texto[1] + baseline + 2),
              (0, 200, 0), -1)

cv2.putText(img, etiqueta, pos_texto,
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

# ── Dibujar también líneas cruzadas para marcar el centro ────────────────────
cv2.line(img, (centro_x - 15, centro_y), (centro_x + 15, centro_y), (0, 255, 0), 2)
cv2.line(img, (centro_x, centro_y - 15), (centro_x, centro_y + 15), (0, 255, 0), 2)

# ── Guardar resultado ────────────────────────────────────────────────────────
cv2.imwrite(os.path.join(output_dir, "figura_etiquetada.jpg"), img)
print(f"Imagen con anotaciones guardada en {output_dir.replace('\\', '/')}/figura_etiquetada.jpg")

# ── Visualización ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle("Ejercicio 5 — Dibujo de Figuras y Texto", fontsize=14, fontweight="bold")

axes[0].imshow(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))
axes[0].set_title("Imagen original")
axes[0].axis("off")

axes[1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[1].set_title("Con círculo y etiqueta de detección")
axes[1].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "ej5_figuras_texto.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Resultado guardado en {output_dir.replace('\\', '/')}/ej5_figuras_texto.png")

# ── Funciones usadas (resumen) ───────────────────────────────────────────────
print("\nFunciones de OpenCV utilizadas:")
print("  cv2.circle(img, centro, radio, color, grosor)")
print("  cv2.putText(img, texto, posicion, fuente, escala, color, grosor)")
print("  cv2.rectangle(img, pt1, pt2, color, grosor)")
print("  cv2.getTextSize(texto, fuente, escala, grosor) → tamaño para fondo")
print("  cv2.line(img, pt1, pt2, color, grosor)")
