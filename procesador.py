"""
Pipeline de procesamiento de fotos para reconocimiento facial  v8
─────────────────────────────────────────────────────────────────
DETECCIÓN DE ROSTROS EN CADENA (3 intentos):

  Intento 1 — Haarcascade (rápido)
    Detector clásico de OpenCV. Funciona bien con rostros despejados
    y bien iluminados. Se valida que el rostro detectado tenga un
    tamaño mínimo razonable (≥ 8% del ancho de la imagen) para
    descartar falsos positivos como el caso del ojo/lente.

  Intento 2 — DNN ResNet-10 (robusto)
    Red neuronal SSD entrenada por OpenCV. Detecta rostros con gafas,
    parcialmente tapados, en distintos ángulos e iluminaciones.
    Se activa automáticamente si el intento 1 falla o detecta algo
    sospechosamente pequeño.
    Requiere ejecutar setup_modelos.py una vez para descargar los pesos.
    Aplica DNN_ZOOM_OUT para compensar que el bounding box del DNN
    es más amplio que el de haarcascade → encuadre más alejado.

  Intento 3 — Fallback centrado (último recurso)
    Si ningún detector encuentra el rostro, se asume que está en el
    centro de la imagen (válido para la mayoría de fotos de perfil).
    La imagen se procesa pero se marca como [REVISAR] en el log y en
    la lista de salida de procesar_carpeta().

DOS MODOS DE ENCUADRE:
  MODO A — Instructivo oficial: ratio 4:5, 640×800px, hombros arriba
  MODO B — Cuadrado: ratio 1:1, 750×750px, rostro centrado

ZERO PADDING — el recorte nunca se sale de los bordes originales.

Prerequisito (solo para el Intento 2):
  python3 setup_modelos.py

Librerías:
  pip install opencv-python Pillow
"""

import cv2
import numpy as np
from PIL import Image
import os
import sys
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ══════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════

MODO = 'B'   # 'A' = instructivo oficial 4:5  |  'B' = cuadrado 1:1

# ── Parámetros MODO A ────────────────────────────────────────────
A_ANCHO      = 640
A_ALTO       = 800
A_POSICION_V = 0.50   # rostro al 50% del alto → centrado verticalmente

# ── Parámetros MODO B ────────────────────────────────────────────
B_ANCHO      = 750
B_ALTO       = 750
B_POSICION_V = 0.50   # rostro centrado verticalmente

# ── Parámetros comunes ───────────────────────────────────────────
KB_MIN            = 50
KB_MAX            = 300
ABRIR_RESULTADO   = True

# Tamaño mínimo del rostro como fracción del ancho de la imagen.
# Detecciones por debajo de este umbral se tratan como falsos positivos.
# 0.08 = el rostro debe medir al menos el 8% del ancho de la imagen.
MIN_ROSTRO_RATIO  = 0.08

# Ruta a los modelos DNN (generados por setup_modelos.py)
MODELO_PROTOTXT   = './modelos/deploy.prototxt'
MODELO_CAFFEMODEL = './modelos/res10_300x300_ssd_iter_140000.caffemodel'

# Confianza mínima para el detector DNN (0.0 - 1.0)
DNN_CONFIANZA_MIN = 0.50

# Factor de alejamiento adicional cuando el detector es DNN.
# El DNN tiende a devolver un bounding box más amplio que haarcascade,
# lo que produce recortes más cerrados con el mismo factor de encuadre.
# Este multiplicador reduce el factor resultante → más alejado → más contexto.
#
# Ejemplos:
#   0.80 → factor reducido al 80%  (más alejado — valor por defecto)
#   0.70 → reducción mayor         (aún más alejado)
#   1.00 → sin cambio              (mismo comportamiento que haarcascade)
DNN_ZOOM_OUT = 0.80

# Asignar según modo activo
ANCHO_FINAL       = A_ANCHO       if MODO == 'A' else B_ANCHO
ALTO_FINAL        = A_ALTO        if MODO == 'A' else B_ALTO
POSICION_VERTICAL = A_POSICION_V  if MODO == 'A' else B_POSICION_V


# ══════════════════════════════════════════════════════════════════
# CARGA Y NORMALIZACIÓN DE FORMATO
# ══════════════════════════════════════════════════════════════════

def cargar_imagen_normalizada(ruta):
    """
    Carga una imagen desde disco y la normaliza a BGR equivalente a JPEG.

    Pasos:
      1. Lee con IMREAD_UNCHANGED para preservar el canal alpha si existe.
      2. Si tiene 4 canales (BGRA/PNG con transparencia): aplana el alpha
         sobre fondo blanco. Fórmula: pixel = alpha*pixel + (1-alpha)*255
         Así los píxeles transparentes quedan blancos, no negros.
      3. Si tiene 1 canal (escala de grises): convierte a BGR.
      4. Re-codifica como JPEG en memoria y decodifica de vuelta.
         Normaliza el formato: elimina metadatos de PNG/BMP/WEBP y garantiza
         que el array resultante es uniforme como si fuera un JPG.
         Resuelve el problema de PNG con fondo blanco asimétrico que hace
         que el pipeline incluya zonas vacías en el encuadre.

    Devuelve: array BGR normalizado, o None si no se pudo cargar.
    """
    # ── 1. Leer preservando alpha ─────────────────────────────────
    img = cv2.imread(ruta, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    # ── 2. Aplanar canal alpha sobre fondo blanco ─────────────────
    if img.ndim == 3 and img.shape[2] == 4:
        bgr   = img[:, :, :3].astype(np.float32)
        alpha = img[:, :, 3:].astype(np.float32) / 255.0
        fondo = np.ones_like(bgr) * 255.0
        img   = (alpha * bgr + (1.0 - alpha) * fondo).clip(0, 255).astype(np.uint8)

    # ── 3. Escala de grises a BGR ─────────────────────────────────
    elif img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # ── 4. Normalizar formato: re-codificar como JPEG en memoria ──
    #       Calidad 95 para no degradar en este paso interno.
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        return img
    img_norm = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img_norm if img_norm is not None else img


# ══════════════════════════════════════════════════════════════════
# UTILIDAD
# ══════════════════════════════════════════════════════════════════

def abrir_imagen(ruta):
    try:
        if sys.platform == 'darwin':
            subprocess.Popen(['open', ruta])
        elif sys.platform == 'win32':
            os.startfile(ruta)
        else:
            subprocess.Popen(['xdg-open', ruta])
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════
# DETECCIÓN — Intento 1: Haarcascade con filtro de tamaño
# ══════════════════════════════════════════════════════════════════

def detectar_haarcascade(imagen_bgr):
    """
    Detector clásico. Devuelve (x,y,w,h) solo si el rostro detectado
    es suficientemente grande (≥ MIN_ROSTRO_RATIO del ancho).
    Descarta falsos positivos pequeños (ojos, lentes, reflejos).
    """
    img_w  = imagen_bgr.shape[1]
    min_px = int(img_w * MIN_ROSTRO_RATIO)

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)

    rostros = cascade.detectMultiScale(
        gris, scaleFactor=1.1, minNeighbors=5,
        minSize=(max(70, min_px), max(70, min_px))
    )

    if len(rostros) == 0:
        return None, 'haarcascade_sin_deteccion'

    rostros = sorted(rostros, key=lambda r: r[2] * r[3], reverse=True)
    x, y, w, h = rostros[0]

    if w < min_px:
        return None, f'haarcascade_falso_positivo (rostro={w}px < min={min_px}px)'

    return (x, y, w, h), 'haarcascade'


# ══════════════════════════════════════════════════════════════════
# DETECCIÓN — Intento 2: DNN ResNet-10
# ══════════════════════════════════════════════════════════════════

def detectar_dnn(imagen_bgr):
    """
    Detector basado en red neuronal. Más robusto para gafas, oclusiones
    parciales y variaciones de iluminación.
    Devuelve (x,y,w,h) del rostro con mayor confianza, o None si no
    encuentra ninguno por encima de DNN_CONFIANZA_MIN.
    """
    if not os.path.exists(MODELO_PROTOTXT) or not os.path.exists(MODELO_CAFFEMODEL):
        #ejecutar setup_modelos.py para descargar los modelos necesarios
        import setup_modelos
        setup_modelos.setup_modelos() #intenta descargar los modelos si no se encuentran, pero si falla devuelve error
        return None, 'dnn_modelos_no_encontrados (ejecuta setup_modelos.py)'

    try:
        net = cv2.dnn.readNetFromCaffe(MODELO_PROTOTXT, MODELO_CAFFEMODEL)
    except Exception as e:
        return None, f'dnn_error_carga ({e})'

    img_h, img_w = imagen_bgr.shape[:2]
    min_px = int(img_w * MIN_ROSTRO_RATIO)

    blob = cv2.dnn.blobFromImage(
        cv2.resize(imagen_bgr, (300, 300)),
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    detecciones = net.forward()

    mejor      = None
    mejor_conf = 0.0

    for i in range(detecciones.shape[2]):
        confianza = float(detecciones[0, 0, i, 2])
        if confianza < DNN_CONFIANZA_MIN:
            continue

        x1 = int(detecciones[0, 0, i, 3] * img_w)
        y1 = int(detecciones[0, 0, i, 4] * img_h)
        x2 = int(detecciones[0, 0, i, 5] * img_w)
        y2 = int(detecciones[0, 0, i, 6] * img_h)

        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(img_w, x2); y2 = min(img_h, y2)

        w = x2 - x1
        h = y2 - y1

        if w < min_px or h < min_px:
            continue

        if confianza > mejor_conf:
            mejor_conf = confianza
            mejor      = (x1, y1, w, h)

    if mejor is None:
        return None, f'dnn_sin_deteccion (conf_min={DNN_CONFIANZA_MIN})'

    return mejor, f'dnn (conf={mejor_conf:.2f})'


# ══════════════════════════════════════════════════════════════════
# DETECCIÓN — Intento 3: Fallback centrado
# ══════════════════════════════════════════════════════════════════

def detectar_fallback_centrado(imagen_bgr):
    """
    Último recurso: asume que el rostro está en el centro superior.
    La imagen queda marcada como [REVISAR].
    """
    img_h, img_w = imagen_bgr.shape[:2]
    w_est = int(img_w * 0.30)
    x_est = (img_w - w_est) // 2
    y_est = int(img_h * 0.10)
    return (x_est, y_est, w_est, w_est), 'fallback_centrado [REVISAR]'


# ══════════════════════════════════════════════════════════════════
# ORQUESTADOR — cadena de 3 intentos
# ══════════════════════════════════════════════════════════════════

def detectar_rostro_robusto(imagen_bgr):
    """
    Intenta detectar el rostro con tres métodos en cascada.
    Devuelve (rostro, metodo, necesita_revision, es_dnn).
    """
    # Intento 1: Haarcascade
    rostro, metodo = detectar_haarcascade(imagen_bgr)
    if rostro is not None:
        return rostro, metodo, False, False

    print(f"       Haarcascade: {metodo}")
    print(f"       Intentando con DNN...")

    # Intento 2: DNN
    rostro, metodo = detectar_dnn(imagen_bgr)
    if rostro is not None:
        return rostro, metodo, False, True   # <-- es_dnn = True

    print(f"       DNN: {metodo}")
    print(f"       Usando fallback centrado — marcar para revision manual")

    # Intento 3: Fallback
    rostro, metodo = detectar_fallback_centrado(imagen_bgr)
    return rostro, metodo, True, False


# ══════════════════════════════════════════════════════════════════
# FACTOR DE ENCUADRE DINÁMICO
# ══════════════════════════════════════════════════════════════════

def calcular_factor_encuadre(w_rostro, w_imagen, es_dnn=False):
    """
    Ajusta el factor según qué tan grande es el rostro relativo a la imagen.

    Rostro pequeño (<25%)   → 0.38  (mucho cuerpo visible)
    Rostro mediano (25-55%) → interpolación lineal
    Rostro grande  (>55%)   → 0.65  (solo rostro y cuello)

    Si es_dnn=True, aplica DNN_ZOOM_OUT para compensar que el bounding
    box del DNN es más amplio que el de haarcascade. Un factor más
    pequeño → recorte más grande → rostro más alejado en el resultado.

    Ejemplo con factor=0.54 y DNN_ZOOM_OUT=0.80:
      factor_final = 0.54 * 0.80 = 0.43
      ancho_recorte = w_rostro / 0.43  →  más grande que / 0.54
      → el rostro ocupa menos espacio en el encuadre → más alejado
    """
    ratio = w_rostro / w_imagen
    if ratio < 0.25:
        factor = 0.38
    elif ratio > 0.55:
        factor = 0.65
    else:
        t      = (ratio - 0.25) / (0.55 - 0.25)
        factor = 0.38 + t * (0.65 - 0.38)

    if es_dnn:
        factor *= DNN_ZOOM_OUT

    return factor


# ══════════════════════════════════════════════════════════════════
# RECORTE — zero padding, nunca se sale de la imagen
# ══════════════════════════════════════════════════════════════════

def recortar_sin_deformar(imagen_bgr, rostro, es_dnn=False):
    RATIO        = ANCHO_FINAL / ALTO_FINAL
    x, y, wf, hf = rostro
    cx           = x + wf // 2
    cy           = y + hf // 2
    img_h, img_w = imagen_bgr.shape[:2]

    factor      = calcular_factor_encuadre(wf, img_w, es_dnn)
    ancho_ideal = int(wf / factor)
    alto_ideal  = int(ancho_ideal / RATIO)

    ajuste = 'ideal'

    # Reducir si supera el tamaño de la imagen
    if ancho_ideal > img_w or alto_ideal > img_h:
        escala      = min(img_w / ancho_ideal, img_h / alto_ideal)
        ancho_ideal = int(ancho_ideal * escala)
        alto_ideal  = int(ancho_ideal / RATIO)
        ajuste      = 'reducido'

    ancho_r = ancho_ideal
    alto_r  = alto_ideal

    # Posicionar centrado en el rostro
    x1 = cx - ancho_r // 2
    y1 = cy - int(alto_r * POSICION_VERTICAL)
    x2 = x1 + ancho_r
    y2 = y1 + alto_r

    # Deslizar si se sale de algún borde
    if x1 < 0:
        x1, x2 = 0, ancho_r
        ajuste = 'deslizado'
    elif x2 > img_w:
        x2, x1 = img_w, img_w - ancho_r
        ajuste = 'deslizado'

    if y1 < 0:
        y1, y2 = 0, alto_r
        ajuste = 'deslizado'
    elif y2 > img_h:
        y2, y1 = img_h, img_h - alto_r
        ajuste = 'deslizado'

    # Seguridad final
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(img_w, x2); y2 = min(img_h, y2)

    return imagen_bgr[y1:y2, x1:x2], factor, ajuste, (x1, y1, x2-x1, y2-y1)


# ══════════════════════════════════════════════════════════════════
# REDIMENSIONAR
# ══════════════════════════════════════════════════════════════════

def redimensionar(imagen_bgr):
    h_a, w_a = imagen_bgr.shape[:2]
    metodo   = cv2.INTER_AREA if w_a > ANCHO_FINAL else cv2.INTER_CUBIC
    return cv2.resize(imagen_bgr, (ANCHO_FINAL, ALTO_FINAL), interpolation=metodo)


# ══════════════════════════════════════════════════════════════════
# COMPRIMIR
# ══════════════════════════════════════════════════════════════════

def comprimir_en_rango(imagen_bgr, ruta_salida):
    pil     = Image.fromarray(cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB))
    calidad = 85
    for _ in range(25):
        pil.save(ruta_salida, 'JPEG', quality=calidad, optimize=True)
        peso_kb = os.path.getsize(ruta_salida) / 1024
        if KB_MIN <= peso_kb <= KB_MAX:
            break
        calidad += -5 if peso_kb > KB_MAX else 5
        calidad = max(20, min(95, calidad))
    return calidad, os.path.getsize(ruta_salida) / 1024


# ══════════════════════════════════════════════════════════════════
# VALIDACIONES
# ══════════════════════════════════════════════════════════════════

def validar_resultado(ruta_jpg, w_rostro, ancho_recorte):
    peso_kb         = os.path.getsize(ruta_jpg) / 1024
    img             = cv2.imread(ruta_jpg)
    h, w            = img.shape[:2]
    rostro_px_final = int(w_rostro * (ANCHO_FINAL / ancho_recorte))

    ancho_ok = (512 <= w <= 800)  if MODO == 'A' else (w == B_ANCHO)
    alto_ok  = (640 <= h <= 1024) if MODO == 'A' else (h == B_ALTO)

    return {
        'formato_jpg':     ruta_jpg.lower().endswith('.jpg'),
        'peso_en_rango':   KB_MIN <= peso_kb <= KB_MAX,
        'ancho_en_rango':  ancho_ok,
        'alto_en_rango':   alto_ok,
        'rostro_en_rango': 80 <= rostro_px_final <= 512,
    }, peso_kb, rostro_px_final


# ══════════════════════════════════════════════════════════════════
# VISUALIZACIÓN
# ══════════════════════════════════════════════════════════════════

def guardar_visualizacion(imagen_orig, recorte, ruta_resultado,
                           rostro, recorte_coords, factor, ajuste,
                           metodo_deteccion, necesita_revision,
                           peso_final, calidad, validaciones, ruta_vis):
    x, y, wf, hf   = rostro
    rx, ry, rw, rh = recorte_coords
    res            = cv2.imread(ruta_resultado)
    todo_ok        = all(validaciones.values()) and not necesita_revision
    h_orig, w_orig = imagen_orig.shape[:2]
    RATIO          = ANCHO_FINAL / ALTO_FINAL

    fig, axes = plt.subplots(1, 3, figsize=(14, 7))
    fig.patch.set_facecolor('#f5f6fa')

    color_det = '#e74c3c' if 'fallback' in metodo_deteccion else \
                '#f39c12' if 'dnn' in metodo_deteccion else '#e74c3c'

    ax = axes[0]
    ax.imshow(cv2.cvtColor(imagen_orig, cv2.COLOR_BGR2RGB))
    rect_f = patches.Rectangle((x, y), wf, hf, lw=2,
                                edgecolor=color_det, facecolor='none',
                                label=f'rostro [{metodo_deteccion}]')
    rect_c = patches.Rectangle((rx, ry), rw, rh, lw=2,
                                edgecolor='#27ae60', facecolor='none',
                                linestyle='--', label=f'recorte [{ajuste}]')
    ax.add_patch(rect_f)
    ax.add_patch(rect_c)
    ax.legend(loc='lower right', fontsize=7, handles=[rect_f, rect_c])
    ax.set_title(
        f'Original  {w_orig}x{h_orig}px  |  MODO {MODO}\n'
        f'detector: {metodo_deteccion}  |  factor={factor:.2f}  ajuste={ajuste}',
        fontsize=8)
    ax.axis('off')

    axes[1].imshow(cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB))
    axes[1].set_title(
        f'Recorte  {recorte.shape[1]}x{recorte.shape[0]}px\n'
        f'Ratio {recorte.shape[1]/recorte.shape[0]:.3f}  (destino: {RATIO:.3f})',
        fontsize=9)
    axes[1].axis('off')

    estado = 'OK' if todo_ok else ('REVISAR - fallback' if necesita_revision else 'REVISAR')
    axes[2].imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
    axes[2].set_title(
        f'Final  {ANCHO_FINAL}x{ALTO_FINAL}px\n'
        f'{peso_final:.1f} KB  JPEG q={calidad}  |  {estado}',
        fontsize=9)
    axes[2].axis('off')

    plt.suptitle(
        f'Pipeline reconocimiento facial  v8  —  MODO {MODO}  —  deteccion en cadena',
        fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(ruta_vis, dpi=130, bbox_inches='tight', facecolor='#f5f6fa')
    plt.close()

    if ABRIR_RESULTADO:
        abrir_imagen(ruta_vis)


# ══════════════════════════════════════════════════════════════════
# PIPELINE COMPLETO
# ══════════════════════════════════════════════════════════════════

def procesar_foto_usuario(ruta_entrada, ruta_salida, mostrar=True):
    """
    Devuelve: (ok, necesita_revision)
      ok               : True si el procesamiento fue exitoso
      necesita_revision: True si se usó el fallback centrado
    """
    print(f"\n{'─'*55}")
    print(f"  {os.path.basename(ruta_entrada)}  [MODO {MODO}]")
    print(f"{'─'*55}")

    imagen = cargar_imagen_normalizada(ruta_entrada)
    if imagen is None:
        print("  [ERROR] No se pudo cargar la imagen.")
        return False, False

    h_orig, w_orig = imagen.shape[:2]
    print(f"  Original : {w_orig}x{h_orig}px  |  "
          f"{os.path.getsize(ruta_entrada)/1024:.1f} KB")

    # Detección en cadena
    print("  [1] Detectando rostro...")
    rostro, metodo_det, necesita_revision, es_dnn = detectar_rostro_robusto(imagen)
    x, y, wf, hf = rostro
    zoom_info = f'  DNN_ZOOM_OUT={DNN_ZOOM_OUT}' if es_dnn else ''
    print(f"       [{metodo_det}]  ({x},{y})  {wf}x{hf}px  "
          f"({wf/w_orig*100:.0f}% ancho){zoom_info}")

    # Recortar — pasa es_dnn para aplicar zoom out si corresponde
    print(f"  [2] Recortando ratio {'4:5' if MODO == 'A' else '1:1'}...")
    recorte, factor, ajuste, recorte_coords = recortar_sin_deformar(
        imagen, rostro, es_dnn)
    ancho_recorte = recorte.shape[1]
    print(f"       {recorte.shape[1]}x{recorte.shape[0]}px  "
          f"ratio={recorte.shape[1]/recorte.shape[0]:.4f}  "
          f"factor={factor:.2f}  ajuste={ajuste}")

    # Redimensionar
    print(f"  [3] Redimensionando a {ANCHO_FINAL}x{ALTO_FINAL}px...")
    final = redimensionar(recorte)

    # Comprimir
    print(f"  [4] Comprimiendo ({KB_MIN}-{KB_MAX} KB)...")
    calidad, peso_final = comprimir_en_rango(final, ruta_salida)
    print(f"       JPEG q={calidad}  ->  {peso_final:.1f} KB")

    # Validar
    validaciones, peso_kb, rostro_px = validar_resultado(
        ruta_salida, wf, ancho_recorte)

    print(f"\n  Validaciones:")
    for nombre_v, ok in [
        ('Formato JPG',                                       validaciones['formato_jpg']),
        (f'Peso {peso_kb:.1f} KB  ({KB_MIN}-{KB_MAX} KB)',   validaciones['peso_en_rango']),
        (f'Ancho {ANCHO_FINAL}px',                            validaciones['ancho_en_rango']),
        (f'Alto  {ALTO_FINAL}px',                             validaciones['alto_en_rango']),
        (f'Rostro ~{rostro_px}px  (80-512 px)',               validaciones['rostro_en_rango']),
    ]:
        print(f"    {'[OK]' if ok else '[--]'}  {nombre_v}")

    if necesita_revision:
        print(f"\n  [!!] Procesado con fallback — REQUIERE REVISION MANUAL")
    else:
        todo_ok = all(validaciones.values())
        print(f"\n  {'[OK] Listo.' if todo_ok else '[!!] Revisar validaciones.'}")

    if mostrar:
        ruta_vis = ruta_salida.replace('.jpg', '_proceso.png')
        guardar_visualizacion(
            imagen, recorte, ruta_salida,
            rostro, recorte_coords, factor, ajuste,
            metodo_det, necesita_revision,
            peso_final, calidad, validaciones, ruta_vis)
        print(f"  Proceso guardado: {os.path.basename(ruta_vis)}")

    return True, necesita_revision


# ══════════════════════════════════════════════════════════════════
# PROCESAR CARPETA COMPLETA
# ══════════════════════════════════════════════════════════════════

def procesar_carpeta(carpeta_entrada, carpeta_salida):
    extensiones = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    os.makedirs(carpeta_salida, exist_ok=True)

    archivos = [f for f in os.listdir(carpeta_entrada)
                if f.lower().endswith(extensiones)]

    print(f"\nProcesando {len(archivos)} imagenes  [MODO {MODO}]")
    exitosas   = 0
    fallidas   = 0
    revisiones = []
    errores    = []

    for archivo in archivos:
        nombre  = os.path.splitext(archivo)[0]
        entrada = os.path.join(carpeta_entrada, archivo)
        salida  = os.path.join(carpeta_salida,  nombre + '.jpg')

        ok, revision = procesar_foto_usuario(entrada, salida, mostrar=False)
        if ok:
            exitosas += 1
            if revision:
                revisiones.append(archivo)
        else:
            fallidas += 1
            errores.append(archivo)

    print(f"\n{'─'*50}")
    print(f"  Exitosas           : {exitosas}")
    print(f"  Con error          : {fallidas}")
    print(f"  Requieren revision : {len(revisiones)}")
    if errores:
        print(f"\n  Errores (no procesadas):")
        for f in errores:
            print(f"    - {f}")
    if revisiones:
        print(f"\n  Para revision manual (usaron fallback):")
        for f in revisiones:
            print(f"    ~ {f}")
    print(f"{'─'*50}")


# ══════════════════════════════════════════════════════════════════
# USO
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    # ── Una sola imagen ─────────────────────────────────────────
    #foto = "391401.png"  # cambiar por la foto que quieras procesar
    #procesar_foto_usuario(
    #    ruta_entrada = f'./imagenes-entrada/{foto}',
    #    ruta_salida  = f'./resultado/391401.jpg',
    #    mostrar      = True
    #)

    # ── Carpeta completa ─────────────────────────────────────────
    procesar_carpeta(
        carpeta_entrada = './imagenes-entrada/',
        carpeta_salida  = './resultado/'
    )
