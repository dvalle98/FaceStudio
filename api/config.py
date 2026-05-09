"""
config.py — Configuración central de la API
"""
import os

# ── Directorio de almacenamiento ────────────────────────────────────────────
# Configurable con la variable de entorno STORAGE_PATH.
# En Docker: se define en docker-compose.yml y apunta al directorio montado
# en el HOST (no dentro del contenedor).
#
# Ejemplos:
#   STORAGE_PATH=/mnt/datos/procesador       → disco de datos del servidor
#   STORAGE_PATH=/var/lib/procesador         → ruta estándar Linux
#   (sin variable)                           → ./storage/ junto al código
#
_DEFAULT_STORAGE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'storage')
STORAGE_PATH = os.environ.get('STORAGE_PATH', _DEFAULT_STORAGE)

STORAGE_ROOT = os.path.join(STORAGE_PATH, 'projects')

# Subcarpeta dentro de cada carpeta de trabajo donde van las imágenes subidas
SUBDIR_ENTRADA  = 'entrada'
# Subcarpeta donde van los resultados procesados
SUBDIR_SALIDA   = 'salida'
# Subcarpeta donde se guardan los ZIPs de descarga
SUBDIR_ZIPS     = 'zips'

# ── Límites de carga ────────────────────────────────────────────────────────
# Tamaño máximo por archivo individual (20 MB)
MAX_UPLOAD_SIZE_MB  = 20
MAX_UPLOAD_SIZE_B   = MAX_UPLOAD_SIZE_MB * 1024 * 1024

# Extensiones permitidas para subir
EXTENSIONES_PERMITIDAS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# ── Configuración de jobs ───────────────────────────────────────────────────
# Ruta al archivo JSON que persiste los jobs entre reinicios del servidor
JOBS_DB_PATH = os.path.join(STORAGE_PATH, 'jobs.json')
