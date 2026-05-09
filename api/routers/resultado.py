"""
routers/resultado.py — Endpoints para descarga del resultado

GET /resultado/{job_id}/informe   → JSON con resumen del procesamiento
GET /resultado/{job_id}/imagenes  → JSON con lista de imágenes y sus estados
GET /resultado/{job_id}/descarga  → ZIP con todas las imágenes procesadas

Este archivo maneja SOLO la capa HTTP de descarga.
La persistencia de jobs está en api/jobs_store.py.
"""
import io
import os
import zipfile

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse

from ..config import STORAGE_ROOT, SUBDIR_SALIDA
from .. import jobs_store as jobs_db   # ← renombrado de jobs → jobs_store

router = APIRouter(prefix='/resultado', tags=['Resultado'])


# ── Helper ────────────────────────────────────────────────────────────────────

def _obtener_job_completado(job_id: str) -> dict:
    """Valida que el job existe y está completado. Lanza HTTP error si no."""
    job = jobs_db.obtener_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' no encontrado.")
    if job['estado'] == jobs_db.EstadoJob.PENDIENTE:
        raise HTTPException(status_code=409, detail='El job aún no ha comenzado.')
    if job['estado'] == jobs_db.EstadoJob.PROCESANDO:
        total = job.get('total', 0)
        proc  = job.get('procesadas', 0)
        pct   = round(proc / total * 100, 1) if total > 0 else 0
        raise HTTPException(
            status_code=409,
            detail=f'El job está procesando ({proc}/{total} — {pct}%). Espera a que termine.',
        )
    if job['estado'] == jobs_db.EstadoJob.FALLIDO:
        raise HTTPException(
            status_code=500,
            detail=f"El job falló: {job.get('error_mensaje', 'error desconocido')}",
        )
    return job


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get(
    '/{job_id}/informe',
    summary='Informe JSON del procesamiento',
    description='Resumen completo: totales, errores y archivos que necesitan revisión manual.',
)
def informe_job(job_id: str):
    job      = _obtener_job_completado(job_id)
    total    = job.get('total', 0)
    exitosas = job.get('exitosas', 0)

    return {
        'job_id'         : job_id,
        'carpeta'        : job['carpeta_nombre'],
        'modo'           : job['modo'],
        'solo_comprimir' : job['solo_comprimir'],
        'configuracion'  : {
            'ancho'  : job['ancho'],
            'alto'   : job['alto'],
            'kb_min' : job['kb_min'],
            'kb_max' : job['kb_max'],
        },
        'resumen'        : {
            'total'          : total,
            'exitosas'       : exitosas,
            'con_error'      : job.get('con_error', 0),
            'necesitan_rev'  : job.get('necesitan_rev', 0),
            'tasa_exito_pct' : round(exitosas / total * 100, 1) if total > 0 else 0,
        },
        'archivos_error' : job.get('archivos_error', []),
        'archivos_rev'   : job.get('archivos_rev', []),
        'tiempos'        : {
            'creado_en'    : job['creado_en'],
            'iniciado_en'  : job.get('iniciado_en'),
            'completado_en': job.get('completado_en'),
        },
        'descarga_en'    : f'/resultado/{job_id}/descarga',
    }


@router.get(
    '/{job_id}/imagenes',
    summary='Listar imágenes procesadas con sus estados',
    description='Devuelve lista de imágenes con sus estados (ok, error, revisar) para previsualización. Disponible incluso mientras se está procesando.',
)
def listar_imagenes(job_id: str, request: Request):
    """Lista todas las imágenes procesadas con sus estados (incluso en progreso)."""
    job = jobs_db.obtener_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' no encontrado.")
    
    # Permitir listar imágenes aunque esté procesando
    # Solo fallar si está pendiente
    if job['estado'] == jobs_db.EstadoJob.PENDIENTE:
        raise HTTPException(
            status_code=409,
            detail='El job aún no ha comenzado. No hay imágenes disponibles.',
        )
    
    carpeta_salida = os.path.join(STORAGE_ROOT, job['carpeta_nombre'], SUBDIR_SALIDA)

    if not os.path.isdir(carpeta_salida):
        return {
            'job_id': job_id,
            'total': 0,
            'imagenes': [],
            'resumen': {
                'ok': 0,
                'revisar': 0,
                'error': 0,
            },
        }

    # Obtener todas las imágenes procesadas
    archivos = sorted([
        f for f in os.listdir(carpeta_salida)
        if os.path.isfile(os.path.join(carpeta_salida, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if not archivos:
        return {
            'job_id': job_id,
            'total': 0,
            'imagenes': [],
            'resumen': {
                'ok': 0,
                'revisar': 0,
                'error': 0,
            },
        }

    # Construir conjuntos de archivos por estado
    archivos_error_set = set(e['archivo'] for e in job.get('archivos_error', []))
    archivos_rev_set = set(job.get('archivos_rev', []))

    # Obtener la URL base desde el request (para evitar problemas de puerto)
    base_url = request.url.scheme + "://" + request.url.netloc

    # Clasificar cada imagen
    imagenes = []
    for archivo in archivos:
        if archivo in archivos_error_set:
            estado = 'error'
        elif archivo in archivos_rev_set:
            estado = 'revisar'
        else:
            estado = 'ok'

        imagenes.append({
            'nombre': archivo,
            'estado': estado,
            'url': f'{base_url}/resultado/{job_id}/imagen/{archivo}',
        })

    return {
        'job_id': job_id,
        'total': len(imagenes),
        'imagenes': imagenes,
        'resumen': {
            'ok': len([i for i in imagenes if i['estado'] == 'ok']),
            'revisar': len([i for i in imagenes if i['estado'] == 'revisar']),
            'error': len([i for i in imagenes if i['estado'] == 'error']),
        },
    }


@router.get(
    '/{job_id}/imagen/{nombre_archivo}',
    summary='Descargar imagen individual procesada',
    description='Sirve una imagen individual procesada. Disponible incluso mientras se está procesando.',
)
def obtener_imagen(job_id: str, nombre_archivo: str):
    """Sirve una imagen individual procesada (incluso en progreso)."""
    job = jobs_db.obtener_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' no encontrado.")
    
    # Permitir acceso a imágenes aunque esté procesando
    if job['estado'] == jobs_db.EstadoJob.PENDIENTE:
        raise HTTPException(status_code=409, detail='El job aún no ha comenzado.')
    
    # Validar nombre de archivo (prevenir path traversal)
    if '/' in nombre_archivo or '\\' in nombre_archivo or nombre_archivo.startswith('.'):
        raise HTTPException(status_code=400, detail='Nombre de archivo inválido.')
    
    ruta_imagen = os.path.join(STORAGE_ROOT, job['carpeta_nombre'], SUBDIR_SALIDA, nombre_archivo)
    
    # Verificar que existe y está en el directorio correcto
    if not os.path.isfile(ruta_imagen):
        raise HTTPException(status_code=404, detail='Imagen no encontrada.')
    
    try:
        return FileResponse(
            path=ruta_imagen,
            media_type='image/jpeg',
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error sirviendo imagen: {str(e)}')


@router.get(
    '/{job_id}/descarga',
    summary='Descargar resultados como ZIP',
    description=(
        'Descarga un ZIP con todas las imágenes procesadas. '
        'El archivo se construye leyendo imagen por imagen para no saturar la RAM, '
        'lo que permite manejar miles de fotos sin problema. '
        'Solo disponible cuando el job tiene estado completado.'
    ),
)
def descargar_resultado(job_id: str):
    job     = _obtener_job_completado(job_id)
    carpeta = os.path.join(STORAGE_ROOT, job['carpeta_nombre'], SUBDIR_SALIDA)

    if not os.path.isdir(carpeta):
        raise HTTPException(
            status_code=404,
            detail='Carpeta de resultados no encontrada.',
        )

    archivos = sorted([
        f for f in os.listdir(carpeta)
        if os.path.isfile(os.path.join(carpeta, f)) and f.lower().endswith('.jpg')
    ])

    if not archivos:
        raise HTTPException(
            status_code=404,
            detail='No hay imágenes procesadas disponibles para descargar.',
        )

    # Construir ZIP leyendo de a un archivo para no cargar todo en RAM
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        for nombre_archivo in archivos:
            ruta = os.path.join(carpeta, nombre_archivo)
            try:
                zf.write(ruta, arcname=nombre_archivo)
            except OSError:
                continue   # si un archivo individual falla, sigue con los demás

    zip_buffer.seek(0)
    nombre_zip = f"resultado_{job['carpeta_nombre']}_{job_id[:8]}.zip"

    return StreamingResponse(
        content    = iter([zip_buffer.read()]),
        media_type = 'application/zip',
        headers    = {
            'Content-Disposition': f'attachment; filename="{nombre_zip}"',
            'X-Total-Imagenes'   : str(len(archivos)),
            'X-Job-Id'           : job_id,
            'X-Carpeta'          : job['carpeta_nombre'],
        },
    )
