"""
setup_modelos.py — Descarga los modelos DNN para detección de rostros
──────────────────────────────────────────────────────────────────────
Ejecutar UNA SOLA VEZ antes de usar procesador6.py:

  python3 setup_modelos.py

Descarga dos archivos en la carpeta ./modelos/:
  - deploy.prototxt          (arquitectura de la red, 28 KB)
  - res10_300x300_ssd_iter_140000.caffemodel  (pesos entrenados, 10 MB)

Fuente: modelo SSD ResNet-10 de OpenCV
  Entrenado para detectar rostros frontales, de perfil, con gafas,
  parcialmente tapados y en condiciones de iluminación variada.
  Mucho más robusto que haarcascade para casos difíciles.
"""

import urllib.request
import os

CARPETA_MODELOS = './modelos'

ARCHIVOS = {
    'deploy.prototxt': (
        'https://raw.githubusercontent.com/opencv/opencv/master/'
        'samples/dnn/face_detector/deploy.prototxt'
    ),
    'res10_300x300_ssd_iter_140000.caffemodel': (
        'https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/'
        'res10_300x300_ssd_iter_140000.caffemodel'
    ),
}

os.makedirs(CARPETA_MODELOS, exist_ok=True)

for nombre, url in ARCHIVOS.items():
    ruta = os.path.join(CARPETA_MODELOS, nombre)
    if os.path.exists(ruta):
        print(f'  [YA EXISTE]  {nombre}')
        continue
    print(f'  Descargando {nombre} ...')
    try:
        urllib.request.urlretrieve(url, ruta)
        kb = os.path.getsize(ruta) / 1024
        print(f'  [OK]  {nombre}  ({kb:.0f} KB)')
    except Exception as e:
        print(f'  [ERROR]  {nombre}: {e}')

print('\nListo. Ya puedes usar procesador6.py')
