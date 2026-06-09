"""
EJERCICIO 7 - Programa de dibujo interactivo con eventos de mouse y teclado
Lab 7 - Procesamiento de Imágenes
Computación Gráfica, Visión Computacional y Multimedia

CONTROLES:
  Mouse arrastrar  → dibuja la figura seleccionada
  C                → modo círculo
  R                → modo rectángulo
  L                → modo línea
  Ctrl+Z (key 26)  → deshacer último trazo
  S                → guardar imagen actual
  ESC              → salir
"""
import cv2
import numpy as np
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "resultados")
os.makedirs(output_dir, exist_ok=True)

# ── Estado global ────────────────────────────────────────────────────────────
ANCHO, ALTO = 800, 560
canvas     = np.full((ALTO, ANCHO, 3), 245, dtype=np.uint8)  # fondo gris claro
historial  = [canvas.copy()]   # pila para deshacer
modo       = "circulo"
dibujando  = False
ix, iy     = -1, -1
color_actual = (0, 0, 180)     # rojo
colores    = [(0, 0, 180), (0, 150, 0), (180, 0, 0), (0, 140, 180), (120, 0, 120)]
idx_color  = 0
grosor     = 2

# Mapa de modos y colores para el panel
MODOS = {"circulo": "C", "rectangulo": "R", "linea": "L"}

def dibujar_figura(imagen, x0, y0, x1, y1, m, col, gr):
    """Dibuja la figura 'm' desde (x0,y0) hasta (x1,y1)."""
    if m == "circulo":
        radio = int(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5)
        cv2.circle(imagen, (x0, y0), radio, col, gr)
    elif m == "rectangulo":
        cv2.rectangle(imagen, (x0, y0), (x1, y1), col, gr)
    elif m == "linea":
        cv2.line(imagen, (x0, y0), (x1, y1), col, gr)

def evento_mouse(event, x, y, flags, param):
    global canvas, dibujando, ix, iy, historial

    if event == cv2.EVENT_LBUTTONDOWN:
        dibujando = True
        ix, iy    = x, y

    elif event == cv2.EVENT_MOUSEMOVE and dibujando:
        # Vista previa en tiempo real (no se agrega al historial)
        preview = historial[-1].copy()
        dibujar_figura(preview, ix, iy, x, y, modo, color_actual, grosor)
        canvas = preview

    elif event == cv2.EVENT_LBUTTONUP:
        dibujando = False
        nuevo = historial[-1].copy()
        dibujar_figura(nuevo, ix, iy, x, y, modo, color_actual, grosor)
        canvas = nuevo
        historial.append(canvas.copy())
        print(f"  Trazo añadido ({modo}) — historial: {len(historial)} estados")

def render_panel(frame):
    """Superpone el panel de controles en la parte inferior."""
    panel_h = 45
    h, w    = frame.shape[:2]
    panel   = np.full((panel_h, w, 3), 40, dtype=np.uint8)

    # Modo actual
    txt_modo = f"Modo: {modo.upper()}  |  "
    cv2.putText(panel, txt_modo, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # Indicador de color
    xc = 210
    cv2.rectangle(panel, (xc, 10), (xc + 24, 35), color_actual, -1)
    cv2.putText(panel, "Color", (xc + 30, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # Atajos
    atajos = "C=circulo  R=rect  L=linea  X=color  Z=deshacer  S=guardar  ESC=salir"
    cv2.putText(panel, atajos, (290, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1)

    return np.vstack([frame, panel])

# ── Ventana y callback ───────────────────────────────────────────────────────
cv2.namedWindow("Dibujo Interactivo")
cv2.setMouseCallback("Dibujo Interactivo", evento_mouse)

print("Programa de dibujo iniciado.")
print("  C = círculo  |  R = rectángulo  |  L = línea")
print("  X = cambiar color  |  Z = deshacer  |  S = guardar  |  ESC = salir")

while True:
    mostrar = render_panel(canvas.copy())
    cv2.imshow("Dibujo Interactivo", mostrar)
    key = cv2.waitKey(15) & 0xFF

    if key == 27:            # ESC
        break
    elif key == ord("c") or key == ord("C"):
        modo = "circulo";    print("Modo: círculo")
    elif key == ord("r") or key == ord("R"):
        modo = "rectangulo"; print("Modo: rectángulo")
    elif key == ord("l") or key == ord("L"):
        modo = "linea";      print("Modo: línea")
    elif key == ord("x") or key == ord("X"):
        idx_color    = (idx_color + 1) % len(colores)
        color_actual = colores[idx_color]
        print(f"Color cambiado a: {color_actual}")
    elif key == 26:          # Ctrl+Z
        if len(historial) > 1:
            historial.pop()
            canvas = historial[-1].copy()
            print(f"  Deshacer — historial: {len(historial)} estados")
    elif key == ord("s") or key == ord("S"):
        ruta = os.path.join(output_dir, "dibujo_final.jpg")
        cv2.imwrite(ruta, canvas)
        print(f"  Dibujo guardado en {ruta.replace('\\', '/')}" )

cv2.destroyAllWindows()
print("Programa de dibujo cerrado.")
