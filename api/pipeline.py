"""
pipeline.py — Módulo de procesamiento de imágenes para reconocimiento facial
─────────────────────────────────────────────────────────────────────────────
Diseñado para ser importado por FastAPI u otro servidor.
NO usa variables globales — todo se pasa por parámetros.
Trabaja con bytes en memoria (sin archivos temporales), listo para uso en API.

FUNCIONES PÚBLICAS:
  procesar_pipeline_completo(...)  → pipeline completo: detectar + recortar + comprimir
  solo_comprimir(...)              → solo reducir peso, sin tocar el encuadre
  procesar_carpeta(...)            → procesa todas las imágenes de una carpeta

MODOS:
  'A' — Instructivo oficial: ratio 4:5, por defecto 640×800px
  'B' — Cuadrado: ratio 1:1, por defecto 750×750px

USO DESDE CONSOLA:
  # Una imagen — pipeline completo
  python3 pipeline.py foto.jpg B
  python3 pipeline.py foto.jpg A 640 800

  # Una imagen — solo comprimir
  python3 pipeline.py foto.jpg compresion

  # Carpeta completa — pipeline completo
  python3 pipeline.py ./fotos/ B ./resultado/
  python3 pipeline.py ./fotos/ A ./resultado/ 640 800

  # Carpeta completa — solo comprimir
  python3 pipeline.py ./fotos/ compresion ./resultado/

Instalar dependencias:
  pip install opencv-python Pillow

Para el detector DNN (opcional, más robusto):
  python3 setup_modelos.py
"""

import cv2
import io
import os
import numpy as np
from PIL import Image, ImageOps
from dataclasses import dataclass, field
from typing import Optional, Tuple, List


# ══════════════════════════════════════════════════════════════════
# CONSTANTES — valores por defecto y límites fijos
# ══════════════════════════════════════════════════════════════════

DEFAULTS = {
    'A': {'ancho': 640, 'alto': 800, 'posicion_v': 0.50},
    'B': {'ancho': 750, 'alto': 750, 'posicion_v': 0.50},
}

KB_MIN_DEFAULT = 50
KB_MAX_DEFAULT = 300

MIN_ROSTRO_RATIO  = 0.08
DNN_CONFIANZA_MIN = 0.50
DNN_ZOOM_OUT      = 0.80

_PROJECT_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELO_PROTOTXT   = os.path.join(_PROJECT_ROOT, 'modelos', 'deploy.prototxt')
MODELO_CAFFEMODEL = os.path.join(_PROJECT_ROOT, 'modelos',
                                  'res10_300x300_ssd_iter_140000.caffemodel')

EXTENSIONES_IMAGEN = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


# ══════════════════════════════════════════════════════════════════
# DATACLASSES DE RESULTADO
# ══════════════════════════════════════════════════════════════════

@dataclass
class ResultadoProcesamiento:
    """
    Resultado de procesar una sola imagen.

    imagen_bytes  : bytes JPEG listos para guardar o enviar por HTTP
    ok            : True si el procesamiento fue exitoso
    necesita_rev  : True si se usó el fallback centrado (revisar manualmente)
    modo          : 'A', 'B' o 'solo_comprimir'
    ancho_final   : ancho real de la imagen resultante en píxeles
    alto_final    : alto real de la imagen resultante en píxeles
    peso_kb       : peso del JPEG resultante en KB
    calidad_jpeg  : calidad JPEG usada (ajustada automáticamente)
    metodo_detec  : qué detector encontró el rostro
    ajuste_recorte: 'ideal' | 'reducido' | 'deslizado'
    validaciones  : dict con resultado de cada validación del instructivo
    error         : mensaje de error si ok=False
    """
    imagen_bytes   : bytes = field(default=b'')
    ok             : bool  = False
    necesita_rev   : bool  = False
    modo           : str   = ''
    ancho_final    : int   = 0
    alto_final     : int   = 0
    peso_kb        : float = 0.0
    calidad_jpeg   : int   = 0
    metodo_detec   : str   = ''
    ajuste_recorte : str   = ''
    validaciones   : dict  = field(default_factory=dict)
    error          : str   = ''


@dataclass
class ResultadoCarpeta:
    """
    Resultado de procesar una carpeta completa.

    carpeta_entrada  : ruta de la carpeta de entrada
    carpeta_salida   : ruta de la carpeta donde se guardaron los resultados
    total            : total de imágenes encontradas
    exitosas         : imágenes procesadas correctamente
    con_error        : imágenes que fallaron (no se pudieron procesar)
    necesitan_rev    : imágenes procesadas pero con fallback (revisar manualmente)
    archivos_error   : lista de nombres de archivos que fallaron
    archivos_rev     : lista de nombres de archivos que necesitan revisión
    """
    carpeta_entrada : str       = ''
    carpeta_salida  : str       = ''
    total           : int       = 0
    exitosas        : int       = 0
    con_error       : int       = 0
    necesitan_rev   : int       = 0
    archivos_error  : List[str] = field(default_factory=list)
    archivos_rev    : List[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════
# UTILIDADES INTERNAS
# ══════════════════════════════════════════════════════════════════

def _bytes_a_bgr(imagen_bytes: bytes) -> Optional[np.ndarray]:
    """
    Convierte bytes de cualquier formato de imagen a un array BGR de OpenCV
    normalizado como si fuera JPEG.

    Pasos:
      1. Abre con Pillow y aplica ImageOps.exif_transpose() para corregir la
         orientación EXIF antes de cualquier procesamiento.
         OpenCV ignora el tag de orientación EXIF, lo que causa que fotos tomadas
         con teléfonos aparezcan rotadas 90° o 180°. Pillow sí lo respeta.
      2. Convierte a array BGR de OpenCV según el modo de color.
      3. Si tiene canal alpha (RGBA): aplana sobre fondo blanco.
         Fórmula: pixel = alpha * pixel + (1 - alpha) * 255
      4. Re-codifica como JPEG en memoria y decodifica de vuelta.
         Normaliza el formato: elimina metadatos, aplana transparencias,
         garantiza un array BGR puro equivalente a un JPG estándar.
    """
    # ── 1. Abrir con Pillow y corregir orientación EXIF ──────────────────────
    # ImageOps.exif_transpose lee el tag Orientation (0x0112) y rota/voltea
    # los píxeles para que queden en la orientación visual correcta.
    # Cubre los 8 casos del estándar EXIF: normal, 90°CW, 180°, 90°CCW y sus
    # variantes con espejo horizontal.
    try:
        pil = Image.open(io.BytesIO(imagen_bytes))
        pil = ImageOps.exif_transpose(pil)   # ← corrige rotación de teléfonos
    except Exception:
        pil = None

    # ── 2. Convertir Pillow → array BGR para OpenCV ──────────────────────────
    if pil is not None:
        mode = pil.mode
        if mode == 'RGBA':
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGBA2BGRA)
        elif mode == 'RGB':
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        elif mode == 'L':
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_GRAY2BGR)
        else:
            # Modos poco comunes (P, CMYK, etc.) → convertir a RGB primero
            img = cv2.cvtColor(np.array(pil.convert('RGB')), cv2.COLOR_RGB2BGR)
    else:
        # Fallback: decodificación directa con OpenCV (sin corrección EXIF)
        arr = np.frombuffer(imagen_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None

    # ── 3. Aplanar canal alpha sobre fondo blanco ────────────────────────────
    if img.ndim == 3 and img.shape[2] == 4:
        bgr   = img[:, :, :3].astype(np.float32)
        alpha = img[:, :, 3:].astype(np.float32) / 255.0
        fondo = np.ones_like(bgr) * 255.0
        img   = (alpha * bgr + (1.0 - alpha) * fondo).clip(0, 255).astype(np.uint8)

    # ── 4. Normalizar formato: re-codificar como JPEG y decodificar de vuelta ─
    #       Elimina metadatos residuales, garantiza 3 canales BGR, fondo blanco
    #       uniforme. Calidad 95 para no degradar en este paso interno.
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        return img   # si falla la re-codificación, devolver el array tal cual

    img_normalizada = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img_normalizada if img_normalizada is not None else img


def _bgr_a_bytes_jpeg(imagen_bgr: np.ndarray, calidad: int) -> bytes:
    rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil.save(buf, 'JPEG', quality=calidad, optimize=True)
    return buf.getvalue()


def _peso_kb(imagen_bytes: bytes) -> float:
    return len(imagen_bytes) / 1024


# ══════════════════════════════════════════════════════════════════
# DETECCIÓN — Intento 1: Haarcascade
# ══════════════════════════════════════════════════════════════════

def _detectar_haarcascade(imagen_bgr: np.ndarray) -> Tuple[Optional[tuple], str]:
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
        return None, f'haarcascade_falso_positivo ({w}px < min {min_px}px)'

    return (x, y, w, h), 'haarcascade'


# ══════════════════════════════════════════════════════════════════
# DETECCIÓN — Intento 2: DNN ResNet-10
# ══════════════════════════════════════════════════════════════════

def _detectar_dnn(imagen_bgr: np.ndarray) -> Tuple[Optional[tuple], str]:
    if not os.path.exists(MODELO_PROTOTXT) or not os.path.exists(MODELO_CAFFEMODEL):
        return None, 'dnn_modelos_no_encontrados'

    try:
        net = cv2.dnn.readNetFromCaffe(MODELO_PROTOTXT, MODELO_CAFFEMODEL)
    except Exception as e:
        return None, f'dnn_error_carga ({e})'

    img_h, img_w = imagen_bgr.shape[:2]
    min_px = int(img_w * MIN_ROSTRO_RATIO)

    blob = cv2.dnn.blobFromImage(
        cv2.resize(imagen_bgr, (300, 300)),
        scalefactor=1.0, size=(300, 300),
        mean=(104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    dets = net.forward()

    mejor = None
    mejor_conf = 0.0

    for i in range(dets.shape[2]):
        conf = float(dets[0, 0, i, 2])
        if conf < DNN_CONFIANZA_MIN:
            continue
        x1 = max(0, int(dets[0, 0, i, 3] * img_w))
        y1 = max(0, int(dets[0, 0, i, 4] * img_h))
        x2 = min(img_w, int(dets[0, 0, i, 5] * img_w))
        y2 = min(img_h, int(dets[0, 0, i, 6] * img_h))
        w, h = x2 - x1, y2 - y1
        if w < min_px or h < min_px:
            continue
        if conf > mejor_conf:
            mejor_conf = conf
            mejor = (x1, y1, w, h)

    if mejor is None:
        return None, f'dnn_sin_deteccion (conf_min={DNN_CONFIANZA_MIN})'

    return mejor, f'dnn (conf={mejor_conf:.2f})'


# ══════════════════════════════════════════════════════════════════
# DETECCIÓN — Intento 3: Fallback centrado
# ══════════════════════════════════════════════════════════════════

def _detectar_fallback_centrado(imagen_bgr: np.ndarray) -> Tuple[tuple, str]:
    img_h, img_w = imagen_bgr.shape[:2]
    w_est = int(img_w * 0.30)
    x_est = (img_w - w_est) // 2
    y_est = int(img_h * 0.10)
    return (x_est, y_est, w_est, w_est), 'fallback_centrado'


# ══════════════════════════════════════════════════════════════════
# ORQUESTADOR — cadena de 3 intentos
# ══════════════════════════════════════════════════════════════════

def _detectar_rostro(imagen_bgr: np.ndarray) -> Tuple[tuple, str, bool, bool]:
    # modelo Haarcascade (rápido pero menos robusto) — recomendado para fotos tipo selfie
    #rostro, metodo = _detectar_haarcascade(imagen_bgr)
    #if rostro is not None:
    #    return rostro, metodo, False, False

    rostro, metodo = _detectar_dnn(imagen_bgr)
    if rostro is not None:
        return rostro, metodo, False, True

    rostro, metodo = _detectar_fallback_centrado(imagen_bgr)
    return rostro, metodo, True, False


# ══════════════════════════════════════════════════════════════════
# FACTOR DE ENCUADRE DINÁMICO
# ══════════════════════════════════════════════════════════════════

def _calcular_factor(w_rostro: int, w_imagen: int, es_dnn: bool) -> float:
    ratio = w_rostro / w_imagen
    if ratio < 0.25:
        factor = 0.38
    elif ratio > 0.55:
        factor = 0.65
    else:
        t      = (ratio - 0.25) / (0.55 - 0.25)
        factor = 0.38 + t * (0.65 - 0.38)
    return factor * DNN_ZOOM_OUT if es_dnn else factor


# ══════════════════════════════════════════════════════════════════
# RECORTE — zero padding, nunca se sale de la imagen
# ══════════════════════════════════════════════════════════════════

def _recortar(imagen_bgr: np.ndarray, rostro: tuple, es_dnn: bool,
              ancho_final: int, alto_final: int,
              posicion_v: float) -> Tuple[np.ndarray, float, str]:
    RATIO        = ancho_final / alto_final
    x, y, wf, hf = rostro
    cx, cy       = x + wf // 2, y + hf // 2
    img_h, img_w = imagen_bgr.shape[:2]

    factor      = _calcular_factor(wf, img_w, es_dnn)
    ancho_ideal = int(wf / factor)
    alto_ideal  = int(ancho_ideal / RATIO)
    ajuste      = 'ideal'

    if ancho_ideal > img_w or alto_ideal > img_h:
        escala      = min(img_w / ancho_ideal, img_h / alto_ideal)
        ancho_ideal = int(ancho_ideal * escala)
        alto_ideal  = int(ancho_ideal / RATIO)
        ajuste      = 'reducido'

    ancho_r, alto_r = ancho_ideal, alto_ideal
    x1 = cx - ancho_r // 2
    y1 = cy - int(alto_r * posicion_v)
    x2, y2 = x1 + ancho_r, y1 + alto_r

    if x1 < 0:
        x1, x2 = 0, ancho_r;                ajuste = 'deslizado'
    elif x2 > img_w:
        x2, x1 = img_w, img_w - ancho_r;    ajuste = 'deslizado'

    if y1 < 0:
        y1, y2 = 0, alto_r;                 ajuste = 'deslizado'
    elif y2 > img_h:
        y2, y1 = img_h, img_h - alto_r;     ajuste = 'deslizado'

    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(img_w, x2); y2 = min(img_h, y2)

    return imagen_bgr[y1:y2, x1:x2], factor, ajuste


# ══════════════════════════════════════════════════════════════════
# REDIMENSIONAR
# ══════════════════════════════════════════════════════════════════

def _redimensionar(imagen_bgr: np.ndarray,
                   ancho_final: int, alto_final: int) -> np.ndarray:
    h_a, w_a = imagen_bgr.shape[:2]
    metodo   = cv2.INTER_AREA if w_a > ancho_final else cv2.INTER_CUBIC
    return cv2.resize(imagen_bgr, (ancho_final, alto_final), interpolation=metodo)


# ══════════════════════════════════════════════════════════════════
# COMPRIMIR EN RANGO
# ══════════════════════════════════════════════════════════════════

def _comprimir(imagen_bgr: np.ndarray,
               kb_min: float, kb_max: float) -> Tuple[bytes, int]:
    calidad   = 85
    resultado = b''
    for _ in range(25):
        resultado = _bgr_a_bytes_jpeg(imagen_bgr, calidad)
        kb        = _peso_kb(resultado)
        if kb_min <= kb <= kb_max:
            break
        calidad += -5 if kb > kb_max else 5
        calidad = max(20, min(95, calidad))
    return resultado, calidad


# ══════════════════════════════════════════════════════════════════
# VALIDACIONES
# ══════════════════════════════════════════════════════════════════

def _validar(imagen_bgr: np.ndarray, imagen_bytes: bytes,
             modo: str, ancho_final: int, alto_final: int,
             w_rostro: int, ancho_recorte: int,
             kb_min: float, kb_max: float) -> dict:
    h, w      = imagen_bgr.shape[:2]
    peso_kb   = _peso_kb(imagen_bytes)
    rostro_px = int(w_rostro * (ancho_final / ancho_recorte))

    ancho_ok = (512 <= w <= 800)  if modo == 'A' else (w == ancho_final)
    alto_ok  = (640 <= h <= 1024) if modo == 'A' else (h == alto_final)

    return {
        'formato_jpg':     True,
        'peso_en_rango':   kb_min <= peso_kb <= kb_max,
        'ancho_en_rango':  ancho_ok,
        'alto_en_rango':   alto_ok,
        'rostro_en_rango': 80 <= rostro_px <= 512,
    }


# ══════════════════════════════════════════════════════════════════
# FUNCIÓN PÚBLICA 1 — Pipeline completo (una imagen)
# ══════════════════════════════════════════════════════════════════

def procesar_pipeline_completo(
    imagen_bytes : bytes,
    modo         : str           = 'B',
    ancho        : Optional[int] = None,
    alto         : Optional[int] = None,
    kb_min       : float         = KB_MIN_DEFAULT,
    kb_max       : float         = KB_MAX_DEFAULT,
) -> ResultadoProcesamiento:
    """
    Pipeline completo: detectar rostro → recortar → redimensionar → comprimir.

    Parámetros:
      imagen_bytes  bytes de la imagen (cualquier formato soportado por OpenCV)
      modo          'A' ratio 4:5 (instructivo oficial) | 'B' ratio 1:1 (cuadrado)
      ancho         ancho de salida en px. None = usa el default del modo
      alto          alto de salida en px.  None = usa el default del modo
      kb_min        peso mínimo en KB (default: 50)
      kb_max        peso máximo en KB (default: 300)
    """
    modo = modo.upper()
    if modo not in ('A', 'B'):
        return ResultadoProcesamiento(error=f"Modo '{modo}' invalido. Usar 'A' o 'B'.")

    ancho_final = ancho if ancho is not None else DEFAULTS[modo]['ancho']
    alto_final  = alto  if alto  is not None else DEFAULTS[modo]['alto']
    posicion_v  = DEFAULTS[modo]['posicion_v']

    imagen = _bytes_a_bgr(imagen_bytes)
    if imagen is None:
        return ResultadoProcesamiento(error='No se pudo decodificar la imagen.')

    rostro, metodo_det, necesita_rev, es_dnn = _detectar_rostro(imagen)
    x, y, wf, hf = rostro

    recorte, factor, ajuste = _recortar(
        imagen, rostro, es_dnn, ancho_final, alto_final, posicion_v)
    ancho_recorte = recorte.shape[1]

    final = _redimensionar(recorte, ancho_final, alto_final)

    resultado_bytes, calidad = _comprimir(final, kb_min, kb_max)
    peso_kb = _peso_kb(resultado_bytes)

    validaciones = _validar(
        final, resultado_bytes, modo,
        ancho_final, alto_final,
        wf, ancho_recorte, kb_min, kb_max)

    # Determinar si fue exitoso — considerar que fallĂ³ si no alcanzĂ³ el rango de peso
    peso_ok = validaciones['peso_en_rango']
    ok_final = peso_ok  # ok=False si no estĂ¡ en rango
    error_msg = None if ok_final else f'No se pudo comprimir al rango {kb_min}-{kb_max} KB. Peso final: {peso_kb:.1f} KB'

    return ResultadoProcesamiento(
        imagen_bytes   = resultado_bytes,
        ok             = ok_final,
        error          = error_msg,
        necesita_rev   = necesita_rev or not peso_ok,  # marcar para revisar si no alcanzĂ³ rango
        modo           = modo,
        ancho_final    = ancho_final,
        alto_final     = alto_final,
        peso_kb        = round(peso_kb, 1),
        calidad_jpeg   = calidad,
        metodo_detec   = metodo_det,
        ajuste_recorte = ajuste,
        validaciones   = validaciones,
    )


# ══════════════════════════════════════════════════════════════════
# FUNCIÓN PÚBLICA 2 — Solo comprimir (una imagen)
# ══════════════════════════════════════════════════════════════════

def solo_comprimir(
    imagen_bytes : bytes,
    kb_min       : float = KB_MIN_DEFAULT,
    kb_max       : float = KB_MAX_DEFAULT,
) -> ResultadoProcesamiento:
    """
    Solo ajusta el peso JPEG sin modificar el encuadre ni las dimensiones.
    Útil cuando la foto ya está recortada y centrada correctamente.

    Parámetros:
      imagen_bytes  bytes de la imagen (cualquier formato)
      kb_min        peso mínimo en KB (default: 50)
      kb_max        peso máximo en KB (default: 300)
    """
    imagen = _bytes_a_bgr(imagen_bytes)
    if imagen is None:
        return ResultadoProcesamiento(error='No se pudo decodificar la imagen.')

    h, w = imagen.shape[:2]
    resultado_bytes, calidad = _comprimir(imagen, kb_min, kb_max)
    peso_kb = _peso_kb(resultado_bytes)

    peso_en_rango = kb_min <= peso_kb <= kb_max
    return ResultadoProcesamiento(
        imagen_bytes   = resultado_bytes,
        ok             = peso_en_rango,  # ok=False si no alcanzó el rango
        error          = None if peso_en_rango else f'No se pudo comprimir al rango {kb_min}-{kb_max} KB. Peso final: {peso_kb:.1f} KB',
        necesita_rev   = False if peso_en_rango else True,  # marcar para revisar si no alcanzó rango
        modo           = 'solo_comprimir',
        ancho_final    = w,
        alto_final     = h,
        peso_kb        = round(peso_kb, 1),
        calidad_jpeg   = calidad,
        metodo_detec   = '',
        ajuste_recorte = '',
        validaciones   = {
            'formato_jpg':     True,
            'peso_en_rango':   peso_en_rango,
            'ancho_en_rango':  True,
            'alto_en_rango':   True,
            'rostro_en_rango': None,  # no aplica
        },
    )


# ══════════════════════════════════════════════════════════════════
# FUNCIÓN PÚBLICA 3 — Procesar carpeta completa
# ══════════════════════════════════════════════════════════════════

def procesar_carpeta(
    carpeta_entrada : str,
    carpeta_salida  : str,
    modo            : str           = 'B',
    ancho           : Optional[int] = None,
    alto            : Optional[int] = None,
    kb_min          : float         = KB_MIN_DEFAULT,
    kb_max          : float         = KB_MAX_DEFAULT,
    solo_comprimir_ : bool          = False,
    verbose         : bool          = True,
) -> ResultadoCarpeta:
    """
    Procesa todas las imágenes de una carpeta y guarda los resultados
    en la carpeta de salida (se crea si no existe).

    El archivo de salida conserva el mismo nombre que el original
    pero siempre con extensión .jpg.

    Parámetros:
      carpeta_entrada  ruta a la carpeta con las imágenes originales
      carpeta_salida   ruta donde guardar los resultados
                       (puede ser nueva o existente)
      modo             'A' o 'B'  (ignorado si solo_comprimir_=True)
      ancho            ancho de salida en px. None = usa default del modo
      alto             alto de salida en px.  None = usa default del modo
      kb_min           peso mínimo en KB
      kb_max           peso máximo en KB
      solo_comprimir_  True = solo ajustar peso, sin tocar el encuadre
      verbose          True = imprimir progreso en consola

    Devuelve: ResultadoCarpeta con el resumen del procesamiento
    """
    # ── Validar carpeta de entrada ────────────────────────────────
    if not os.path.isdir(carpeta_entrada):
        if verbose:
            print(f"[ERROR] La carpeta de entrada no existe: {carpeta_entrada}")
        return ResultadoCarpeta(
            carpeta_entrada = carpeta_entrada,
            carpeta_salida  = carpeta_salida,
        )

    # ── Crear carpeta de salida si no existe ─────────────────────
    os.makedirs(carpeta_salida, exist_ok=True)

    # ── Recopilar archivos de imagen ──────────────────────────────
    archivos = sorted([
        f for f in os.listdir(carpeta_entrada)
        if os.path.isfile(os.path.join(carpeta_entrada, f))
        and f.lower().endswith(EXTENSIONES_IMAGEN)
    ])

    resumen = ResultadoCarpeta(
        carpeta_entrada = os.path.abspath(carpeta_entrada),
        carpeta_salida  = os.path.abspath(carpeta_salida),
        total           = len(archivos),
    )

    if verbose:
        sep = '─' * 55
        print(f"\n{sep}")
        operacion = 'solo comprimir' if solo_comprimir_ else f'pipeline modo {modo.upper()}'
        print(f"  Carpeta entrada : {carpeta_entrada}")
        print(f"  Carpeta salida  : {carpeta_salida}")
        print(f"  Operacion       : {operacion}")
        print(f"  Total imagenes  : {len(archivos)}")
        print(sep)

    # ── Procesar cada archivo ─────────────────────────────────────
    for i, archivo in enumerate(archivos, start=1):
        ruta_entrada = os.path.join(carpeta_entrada, archivo)
        nombre_sin_ext = os.path.splitext(archivo)[0]
        ruta_salida  = os.path.join(carpeta_salida, nombre_sin_ext + '.jpg')

        if verbose:
            print(f"  [{i:>{len(str(len(archivos)))}}/{len(archivos)}] {archivo}", end=' ... ')

        try:
            with open(ruta_entrada, 'rb') as f:
                img_bytes = f.read()

            if solo_comprimir_:
                res = solo_comprimir(img_bytes, kb_min, kb_max)
            else:
                res = procesar_pipeline_completo(
                    img_bytes,
                    modo   = modo,
                    ancho  = ancho,
                    alto   = alto,
                    kb_min = kb_min,
                    kb_max = kb_max,
                )

            if not res.ok:
                resumen.con_error += 1
                resumen.archivos_error.append(archivo)
                if verbose:
                    print(f"[ERROR] {res.error}")
                continue

            # Guardar imagen resultante
            with open(ruta_salida, 'wb') as f:
                f.write(res.imagen_bytes)

            resumen.exitosas += 1

            if res.necesita_rev:
                resumen.necesitan_rev += 1
                resumen.archivos_rev.append(archivo)
                if verbose:
                    print(f"[OK - REVISAR] {res.ancho_final}x{res.alto_final}px "
                          f"{res.peso_kb}KB  detector={res.metodo_detec}")
            else:
                if verbose:
                    det = f"  detector={res.metodo_detec}" if res.metodo_detec else ''
                    print(f"[OK] {res.ancho_final}x{res.alto_final}px "
                          f"{res.peso_kb}KB{det}")

        except Exception as e:
            resumen.con_error += 1
            resumen.archivos_error.append(archivo)
            if verbose:
                print(f"[ERROR] {e}")

    # ── Resumen final ─────────────────────────────────────────────
    if verbose:
        sep = '─' * 55
        print(f"\n{sep}")
        print(f"  Exitosas          : {resumen.exitosas}")
        print(f"  Con error         : {resumen.con_error}")
        print(f"  Requieren revision: {resumen.necesitan_rev}")
        if resumen.archivos_error:
            print(f"\n  Errores:")
            for f in resumen.archivos_error:
                print(f"    [XX] {f}")
        if resumen.archivos_rev:
            print(f"\n  Para revision manual:")
            for f in resumen.archivos_rev:
                print(f"    [~~] {f}")
        print(f"\n  Resultado guardado en: {resumen.carpeta_salida}")
        print(sep)

    return resumen


# ══════════════════════════════════════════════════════════════════
# USO DIRECTO DESDE CONSOLA
# ══════════════════════════════════════════════════════════════════

def _imprimir_resultado_imagen(res: ResultadoProcesamiento, ruta_salida: str) -> None:
    """Imprime el resultado de procesar una sola imagen."""
    if not res.ok:
        print(f"\n  [ERROR] {res.error}")
        return
    print(f"\n  Modo          : {res.modo}")
    print(f"  Dimensiones   : {res.ancho_final}x{res.alto_final}px")
    print(f"  Peso          : {res.peso_kb} KB  (JPEG q={res.calidad_jpeg})")
    if res.metodo_detec:
        print(f"  Detector      : {res.metodo_detec}")
        print(f"  Ajuste recorte: {res.ajuste_recorte}")
    if res.necesita_rev:
        print(f"  [!!] Requiere revision manual")
    print(f"\n  Validaciones:")
    for k, v in res.validaciones.items():
        if v is None:
            print(f"    [--]  {k} (no aplica)")
        else:
            print(f"    {'[OK]' if v else '[XX]'}  {k}")
    print(f"\n  Guardado en: {ruta_salida}")


if __name__ == '__main__':
    import sys

    def _ayuda():
        print("""
Uso:
  Una imagen:
    python3 pipeline.py <imagen> <modo>               [ancho alto]
    python3 pipeline.py <imagen> compresion

  Carpeta completa:
    python3 pipeline.py <carpeta_entrada> <modo>      <carpeta_salida> [ancho alto]
    python3 pipeline.py <carpeta_entrada> compresion  <carpeta_salida>

Modos: A  (ratio 4:5, 640x800px por defecto)
       B  (ratio 1:1, 750x750px por defecto)
       compresion  (solo ajustar peso, sin recortar)

Ejemplos:
  python3 pipeline.py foto.jpg B
  python3 pipeline.py foto.jpg A 640 800
  python3 pipeline.py foto.jpg compresion
  python3 pipeline.py ./fotos/ B ./resultado/
  python3 pipeline.py ./fotos/ A ./resultado/ 640 800
  python3 pipeline.py ./fotos/ compresion ./resultado/
        """)

    if len(sys.argv) < 3:
        _ayuda()
        sys.exit(1)

    entrada  = sys.argv[1]
    modo_arg = sys.argv[2].lower()

    # ── Detectar si es carpeta o archivo ──────────────────────────
    es_carpeta = os.path.isdir(entrada)

    if es_carpeta:
        # ── Modo carpeta ──────────────────────────────────────────
        # python3 pipeline.py <carpeta> <modo> <salida> [ancho alto]
        if len(sys.argv) < 4:
            print("[ERROR] Para carpeta debes indicar la carpeta de salida.")
            print("  Ej: python3 pipeline.py ./fotos/ B ./resultado/")
            sys.exit(1)

        carpeta_salida = sys.argv[3]
        ancho_arg = int(sys.argv[4]) if len(sys.argv) > 4 else None
        alto_arg  = int(sys.argv[5]) if len(sys.argv) > 5 else None

        procesar_carpeta(
            carpeta_entrada = entrada,
            carpeta_salida  = carpeta_salida,
            modo            = modo_arg if modo_arg != 'compresion' else 'B',
            ancho           = ancho_arg,
            alto            = alto_arg,
            solo_comprimir_ = (modo_arg == 'compresion'),
            verbose         = True,
        )

    else:
        # ── Modo imagen única ─────────────────────────────────────
        # python3 pipeline.py <imagen> <modo> [ancho alto]
        if not os.path.isfile(entrada):
            print(f"[ERROR] No se encontro el archivo: {entrada}")
            sys.exit(1)

        ancho_arg = int(sys.argv[3]) if len(sys.argv) > 3 else None
        alto_arg  = int(sys.argv[4]) if len(sys.argv) > 4 else None

        with open(entrada, 'rb') as f:
            img_bytes = f.read()

        print(f"\n{'─'*50}")
        print(f"  Archivo : {os.path.basename(entrada)}")
        print(f"  Modo    : {modo_arg}")
        if ancho_arg:
            print(f"  Tamano  : {ancho_arg}x{alto_arg}px (personalizado)")

        if modo_arg == 'compresion':
            res = solo_comprimir(img_bytes)
        else:
            res = procesar_pipeline_completo(
                img_bytes, modo=modo_arg,
                ancho=ancho_arg, alto=alto_arg)

        nombre_base = os.path.splitext(os.path.basename(entrada))[0]
        ruta_salida = os.path.join(os.path.dirname(entrada) or '.', nombre_base + '_procesado.jpg')

        if res.ok:
            with open(ruta_salida, 'wb') as f:
                f.write(res.imagen_bytes)

        _imprimir_resultado_imagen(res, ruta_salida)
        print(f"{'─'*50}\n")
        sys.exit(0 if res.ok else 1)
