"""
EJERCICIO 4 - Aplicación interactiva de visualización de canales de color
Lab 7 - Procesamiento de Imágenes
Computación Gráfica, Visión Computacional y Multimedia

Controles:
  R  → activar/desactivar canal Rojo
  G  → activar/desactivar canal Verde
  B  → activar/desactivar canal Azul
  ESC → salir
"""
import cv2
import numpy as np
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "resultados")

# ── Cargar imagen ────────────────────────────────────────────────────────────
img = cv2.imread(os.path.join(output_dir, "combinada.jpg"))
if img is None:
    raise FileNotFoundError(f"No se encontró {os.path.join(output_dir, 'combinada.jpg')}. Ejecutar ejercicio 2 primero.")

# Evitar imágenes excesivamente grandes: redimensionar si es necesario
MAX_DIM = 900
h, w = img.shape[:2]
if max(h, w) > MAX_DIM:
    scale = MAX_DIM / float(max(h, w))
    new_w, new_h = int(w * scale), int(h * scale)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    print(f"Imagen muy grande; redimensionada a {new_w}x{new_h}.")

show_r = True
show_g = True
show_b = True

print("Aplicación iniciada.")
print("  Teclas: R = Rojo  |  G = Verde  |  B = Azul  |  ESC = Salir")

while True:
    # Canales activos o cero si están desactivados
    b_out = img[:, :, 0] if show_b else np.zeros(img.shape[:2], dtype=np.uint8)
    g_out = img[:, :, 1] if show_g else np.zeros(img.shape[:2], dtype=np.uint8)
    r_out = img[:, :, 2] if show_r else np.zeros(img.shape[:2], dtype=np.uint8)

    display = cv2.merge([b_out, g_out, r_out])

    # Panel de estado en la parte inferior
    panel = np.zeros((50, display.shape[1], 3), dtype=np.uint8)

    def icono(activo, color_on, x):
        color = color_on if activo else (60, 60, 60)
        estado = "ON" if activo else "OFF"
        return color, estado, x

    for texto, activo, color_on, x in [
        ("R", show_r, (0, 0, 220),   20),
        ("G", show_g, (0, 200, 0),  120),
        ("B", show_b, (220, 0, 0),  220),
    ]:
        color = color_on if activo else (60, 60, 60)
        estado = "ON " if activo else "OFF"
        cv2.putText(panel, f"[{texto}] {estado}", (x, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    cv2.putText(panel, "ESC=salir", (display.shape[1] - 130, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    final = np.vstack([display, panel])
    cv2.imshow("Visualizacion de Canales (R/G/B para toggle)", final)

    key = cv2.waitKey(30) & 0xFF
    if key == 27:       # ESC
        break
    elif key == ord("r") or key == ord("R"):
        show_r = not show_r
        print(f"Canal Rojo:  {'ON' if show_r else 'OFF'}")
    elif key == ord("g") or key == ord("G"):
        show_g = not show_g
        print(f"Canal Verde: {'ON' if show_g else 'OFF'}")
    elif key == ord("b") or key == ord("B"):
        show_b = not show_b
        print(f"Canal Azul:  {'ON' if show_b else 'OFF'}")

cv2.destroyAllWindows()
print("Aplicación cerrada.")
