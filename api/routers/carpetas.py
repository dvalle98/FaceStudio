"""
routers/carpetas.py — Endpoints para gestión de carpetas de trabajo

POST   /carpetas/                   → crear carpeta nueva
GET    /carpetas/                   → listar todas las carpetas
GET    /carpetas/{nombre}           → info detallada de una carpeta
DELETE /carpetas/{nombre}           → eliminar carpeta y su contenido

POST   /carpetas/{nombre}/imagenes  → subir imágenes (múltiples archivos)
GET    /carpetas/{nombre}/imagenes  → listar imágenes en la carpeta
"""
import os
import re
import shutil
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from ..config import (
    EXTENSIONES_PERMITIDAS,
    MAX_UPLOAD_SIZE_B,
    MAX_UPLOAD_SIZE_MB,
    STORAGE_ROOT,
    SUBDIR_ENTRADA,
    SUBDIR_SALIDA,
    SUBDIR_ZIPS,
)

router = APIRouter(prefix='/carpetas', tags=['Carpetas'])

NOMBRE_RE = re.compile(r'^[a-zA-Z0-9_\-]{1,64}$')


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ruta_carpeta(nombre: str) -> str:
    return os.path.join(STORAGE_ROOT, nombre)


def _validar_nombre(nombre: str) -> None:
    if not NOMBRE_RE.match(nombre):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                'El nombre solo puede contener letras, números, guiones '
                'y guiones bajos. Máximo 64 caracteres.'
            ),
        )


def _carpeta_existe(nombre: str) -> bool:
    return os.path.isdir(_ruta_carpeta(nombre))


def _info_carpeta(nombre: str) -> dict:
    base    = _ruta_carpeta(nombre)
    entrada = os.path.join(base, SUBDIR_ENTRADA)
    salida  = os.path.join(base, SUBDIR_SALIDA)

    def _contar(directorio: str):
        if not os.path.isdir(directorio):
            return 0, 0
        archivos = [f for f in os.listdir(directorio)
                    if os.path.isfile(os.path.join(directorio, f))]
        total_bytes = sum(os.path.getsize(os.path.join(directorio, f)) for f in archivos)
        return len(archivos), total_bytes

    imgs_e, bytes_e = _contar(entrada)
    imgs_s, bytes_s = _contar(salida)

    return {
        'nombre'           : nombre,
        'imagenes_entrada' : imgs_e,
        'tamano_entrada_mb': round(bytes_e / (1024 * 1024), 2),
        'imagenes_salida'  : imgs_s,
        'tamano_salida_mb' : round(bytes_s / (1024 * 1024), 2),
    }


# ── Carpetas ──────────────────────────────────────────────────────────────────

@router.post(
    '/',
    status_code=status.HTTP_201_CREATED,
    summary='Crear carpeta de trabajo',
)
def crear_carpeta(nombre: str):
    _validar_nombre(nombre)

    if _carpeta_existe(nombre):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"La carpeta '{nombre}' ya existe.",
        )

    base = _ruta_carpeta(nombre)
    for sub in [SUBDIR_ENTRADA, SUBDIR_SALIDA, SUBDIR_ZIPS]:
        os.makedirs(os.path.join(base, sub), exist_ok=True)

    return {
        'mensaje' : f"Carpeta '{nombre}' creada correctamente.",
        'carpeta' : _info_carpeta(nombre),
    }


@router.get('/', summary='Listar carpetas de trabajo')
def listar_carpetas():
    if not os.path.isdir(STORAGE_ROOT):
        return {'total': 0, 'carpetas': []}

    carpetas = [
        _info_carpeta(d)
        for d in sorted(os.listdir(STORAGE_ROOT))
        if os.path.isdir(os.path.join(STORAGE_ROOT, d))
    ]
    return {'total': len(carpetas), 'carpetas': carpetas}


@router.get('/{nombre}', summary='Información de una carpeta')
def info_carpeta(nombre: str):
    _validar_nombre(nombre)
    if not _carpeta_existe(nombre):
        raise HTTPException(status_code=404, detail=f"Carpeta '{nombre}' no encontrada.")
    return _info_carpeta(nombre)


@router.delete('/{nombre}', summary='Eliminar carpeta de trabajo')
def eliminar_carpeta(nombre: str):
    _validar_nombre(nombre)
    if not _carpeta_existe(nombre):
        raise HTTPException(status_code=404, detail=f"Carpeta '{nombre}' no encontrada.")
    shutil.rmtree(_ruta_carpeta(nombre))
    return {'mensaje': f"Carpeta '{nombre}' eliminada correctamente."}


# ── Imágenes ──────────────────────────────────────────────────────────────────

@router.post(
    '/{nombre}/imagenes',
    status_code=status.HTTP_201_CREATED,
    summary='Subir imágenes a una carpeta',
    description=(
        f'Formatos: JPG, JPEG, PNG, BMP, WEBP. '
        f'Máximo {MAX_UPLOAD_SIZE_MB} MB por archivo. '
        'Archivos con nombre duplicado se sobreescriben.'
    ),
)
async def subir_imagenes(nombre: str, archivos: List[UploadFile] = File(...)):
    _validar_nombre(nombre)

    if not _carpeta_existe(nombre):
        raise HTTPException(
            status_code=404,
            detail=f"Carpeta '{nombre}' no existe. Créala primero con POST /carpetas/.",
        )

    entrada = os.path.join(_ruta_carpeta(nombre), SUBDIR_ENTRADA)
    os.makedirs(entrada, exist_ok=True)

    subidos  = []
    rechazos = []

    for archivo in archivos:
        filename = archivo.filename or ''
        ext      = os.path.splitext(filename)[1].lower()

        # Validar extensión
        if ext not in EXTENSIONES_PERMITIDAS:
            rechazos.append({
                'archivo': filename,
                'razon'  : f"Extensión '{ext}' no permitida.",
            })
            await archivo.close()
            continue

        # Leer todo el contenido con await read() — compatible con todas las
        # versiones de FastAPI/Starlette. El async for chunk in archivo falla
        # en versiones recientes porque UploadFile ya no es un async iterator.
        try:
            contenido = await archivo.read()
        except Exception as e:
            rechazos.append({'archivo': filename, 'razon': f"Error leyendo archivo: {e}"})
            continue
        finally:
            await archivo.close()

        # Validar tamaño después de leer
        if len(contenido) > MAX_UPLOAD_SIZE_B:
            rechazos.append({
                'archivo': filename,
                'razon'  : f"Supera el límite de {MAX_UPLOAD_SIZE_MB} MB "
                           f"({len(contenido)/(1024*1024):.1f} MB).",
            })
            continue

        # Guardar en disco
        ruta_destino = os.path.join(entrada, filename)
        with open(ruta_destino, 'wb') as f:
            f.write(contenido)

        subidos.append({
            'archivo'   : filename,
            'tamano_kb' : round(len(contenido) / 1024, 1),
        })

    return {
        'carpeta'            : nombre,
        'subidos'            : len(subidos),
        'rechazados'         : len(rechazos),
        'archivos_subidos'   : subidos,
        'archivos_rechazados': rechazos,
    }


@router.get('/{nombre}/imagenes', summary='Listar imágenes en la carpeta de entrada')
def listar_imagenes(nombre: str):
    _validar_nombre(nombre)

    if not _carpeta_existe(nombre):
        raise HTTPException(status_code=404, detail=f"Carpeta '{nombre}' no encontrada.")

    entrada = os.path.join(_ruta_carpeta(nombre), SUBDIR_ENTRADA)
    if not os.path.isdir(entrada):
        return {'carpeta': nombre, 'total': 0, 'imagenes': []}

    imagenes = [
        {
            'nombre'    : f,
            'tamano_kb' : round(os.path.getsize(os.path.join(entrada, f)) / 1024, 1),
        }
        for f in sorted(os.listdir(entrada))
        if os.path.isfile(os.path.join(entrada, f))
        and os.path.splitext(f)[1].lower() in EXTENSIONES_PERMITIDAS
    ]

    return {'carpeta': nombre, 'total': len(imagenes), 'imagenes': imagenes}
