"""
routers/jobs.py — Endpoints HTTP para gestión de jobs de procesamiento

POST /jobs/          → crear y lanzar un job en background
GET  /jobs/          → listar todos los jobs
GET  /jobs/{job_id}  → estado y progreso de un job específico

Este archivo maneja SOLO la capa HTTP (validaciones, respuestas).
La persistencia de los jobs está en api/jobs_store.py.
La lógica de imagen está en pipeline.py (raíz del proyecto).
"""
import os
import threading
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from ..config import STORAGE_ROOT, SUBDIR_ENTRADA, SUBDIR_SALIDA
from .. import jobs_store as jobs_db   # ← renombrado de jobs → jobs_store

from .. import pipeline as pipeline_module

router = APIRouter(prefix='/jobs', tags=['Jobs'])


# ── Esquema de entrada ────────────────────────────────────────────────────────

class CrearJobRequest(BaseModel):
    carpeta_nombre : str = Field(
        ...,
        description='Nombre de la carpeta de trabajo con las imágenes a procesar.',
        example='lote_enero',
    )
    modo : str = Field(
        default='B',
        description=(
            '"A" = pipeline oficial ratio 4:5 (640×800px por defecto). '
            '"B" = cuadrado ratio 1:1 (750×750px por defecto). '
            '"compresion" = solo ajustar peso sin tocar el encuadre.'
        ),
        example='B',
    )
    ancho : Optional[int] = Field(
        default=None,
        description='Ancho de salida en píxeles. None = usa el default del modo.',
        ge=100, le=2000,
        example=None,
    )
    alto : Optional[int] = Field(
        default=None,
        description='Alto de salida en píxeles. None = usa el default del modo.',
        ge=100, le=2000,
        example=None,
    )
    kb_min : float = Field(default=30,  description='Peso mínimo en KB.', ge=30, le=500)
    kb_max : float = Field(default=300, description='Peso máximo en KB.', ge=300, le=5000)

    model_config = {
        'json_schema_extra': {
            'example': {
                'carpeta_nombre': 'lote_enero',
                'modo'          : 'B',
                'ancho'         : None,
                'alto'          : None,
                'kb_min'        : 30,
                'kb_max'        : 300,
            }
        }
    }


# ── Worker — corre en hilo separado, no bloquea el servidor ──────────────────

def _ejecutar_job(job_id: str) -> None:
    """
    Procesa todas las imágenes de la carpeta de entrada imagen por imagen.
    Actualiza el progreso en jobs_store después de cada imagen para que
    GET /jobs/{job_id} refleje el avance en tiempo real.
    """
    job = jobs_db.obtener_job(job_id)
    if not job:
        return

    jobs_db.actualizar_job(
        job_id,
        estado      = jobs_db.EstadoJob.PROCESANDO,
        iniciado_en = datetime.now(timezone.utc).isoformat(),
    )

    entrada = os.path.join(STORAGE_ROOT, job['carpeta_nombre'], SUBDIR_ENTRADA)
    salida  = os.path.join(STORAGE_ROOT, job['carpeta_nombre'], SUBDIR_SALIDA)
    os.makedirs(salida, exist_ok=True)

    # Listar solo archivos con extensión de imagen válida, ordenados alfabéticamente
    archivos = sorted([
        f for f in os.listdir(entrada)
        if os.path.isfile(os.path.join(entrada, f))
        and os.path.splitext(f)[1].lower() in pipeline_module.EXTENSIONES_IMAGEN
    ])

    jobs_db.actualizar_job(job_id, total=len(archivos))

    exitosas = con_error = necesitan_rev = 0
    archivos_error: list = []
    archivos_rev:   list = []

    # Procesar cada imagen secuencialmente (puede optimizarse con procesamiento paralelo si es necesario)
    for i, archivo in enumerate(archivos, start=1):
        ruta_entrada = os.path.join(entrada, archivo)
        ruta_salida  = os.path.join(salida, os.path.splitext(archivo)[0] + '.jpg')

        try:
            with open(ruta_entrada, 'rb') as f:
                img_bytes = f.read()

            # Dependiendo del modo, se llama a la función correspondiente del pipeline.
            if job['solo_comprimir']:
                res = pipeline_module.solo_comprimir(
                    img_bytes, job['kb_min'], job['kb_max'])
            else:
                res = pipeline_module.procesar_pipeline_completo(
                    img_bytes,
                    modo   = job['modo'],
                    ancho  = job['ancho'],
                    alto   = job['alto'],
                    kb_min = job['kb_min'],
                    kb_max = job['kb_max'],
                )

            if not res.ok:
                con_error += 1
                archivos_error.append({'archivo': archivo, 'razon': res.error})
            else:
                with open(ruta_salida, 'wb') as f:
                    f.write(res.imagen_bytes)
                exitosas += 1
                if res.necesita_rev:
                    necesitan_rev += 1
                    archivos_rev.append(archivo)

        except Exception as e:
            con_error += 1
            archivos_error.append({'archivo': archivo, 'razon': str(e)})

        jobs_db.actualizar_job(
            job_id,
            procesadas     = i,
            exitosas       = exitosas,
            con_error      = con_error,
            necesitan_rev  = necesitan_rev,
            archivos_error = archivos_error,
            archivos_rev   = archivos_rev,
        )

    jobs_db.actualizar_job(
        job_id,
        estado        = jobs_db.EstadoJob.COMPLETADO,
        completado_en = datetime.now(timezone.utc).isoformat(),
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    '/',
    status_code=status.HTTP_202_ACCEPTED,
    summary='Crear y lanzar un job de procesamiento',
    description=(
        'Lanza el procesamiento de todas las imágenes de la carpeta indicada. '
        'La respuesta es inmediata — el trabajo corre en segundo plano. '
        'Consulta el progreso con GET /jobs/{job_id}.'
    ),
)
def crear_job(body: CrearJobRequest):
    entrada = os.path.join(STORAGE_ROOT, body.carpeta_nombre, SUBDIR_ENTRADA)

    if not os.path.isdir(entrada):
        raise HTTPException(
            status_code=404,
            detail=(
                f"La carpeta '{body.carpeta_nombre}' no existe o no tiene imágenes. "
                "Crea la carpeta con POST /carpetas/ y sube imágenes con "
                "POST /carpetas/{nombre}/imagenes."
            ),
        )

    total = sum(
        1 for f in os.listdir(entrada)
        if os.path.isfile(os.path.join(entrada, f))
        and os.path.splitext(f)[1].lower() in pipeline_module.EXTENSIONES_IMAGEN
    )

    if total == 0:
        raise HTTPException(
            status_code=422,
            detail=f"La carpeta '{body.carpeta_nombre}' no tiene imágenes para procesar.",
        )

    modo_upper = body.modo.upper()
    if modo_upper not in ('A', 'B', 'COMPRESION'):
        raise HTTPException(
            status_code=400,
            detail="El modo debe ser 'A', 'B' o 'compresion'.",
        )

    solo_comprimir = (modo_upper == 'COMPRESION')
    modo_real      = 'B' if solo_comprimir else modo_upper

    job = jobs_db.crear_job(
        carpeta_nombre = body.carpeta_nombre,
        modo           = modo_real,
        ancho          = body.ancho,
        alto           = body.alto,
        kb_min         = body.kb_min,
        kb_max         = body.kb_max,
        solo_comprimir = solo_comprimir,
    )

    threading.Thread(target=_ejecutar_job, args=(job['job_id'],), daemon=True).start()

    return {
        'mensaje'        : 'Job lanzado. El procesamiento corre en segundo plano.',
        'job_id'         : job['job_id'],
        'total_imagenes' : total,
        'consultar_en'   : f"/jobs/{job['job_id']}",
        'descargar_en'   : f"/resultado/{job['job_id']}/descarga",
    }


@router.get(
    '/',
    summary='Listar todos los jobs',
    description='Devuelve todos los jobs del más reciente al más antiguo.',
)
def listar_jobs():
    todos = jobs_db.listar_jobs()
    return {'total': len(todos), 'jobs': [_formatear(j) for j in todos]}


@router.get(
    '/{job_id}',
    summary='Estado y progreso de un job',
    description='Estado actual: pendiente | procesando | completado | fallido.',
)
def estado_job(job_id: str):
    job = jobs_db.obtener_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' no encontrado.")
    return _formatear(job)

@router.delete(
    '/{job_id}',
    summary='Eliminar un job',
    description=(
        'Elimina un job específico. Solo se puede eliminar un job que esté en estado '
        '"completado" o "fallido". No se pueden eliminar jobs en proceso o pendientes.'
    ),
)
def eliminar_job(job_id: str):
    job = jobs_db.obtener_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' no encontrado.")
    if job['estado'] not in (jobs_db.EstadoJob.COMPLETADO, jobs_db.EstadoJob.FALLIDO):
        raise HTTPException(
            status_code=400,
            detail="Solo se pueden eliminar jobs en estado 'completado' o 'fallido'."
        )
    jobs_db.eliminar_job(job_id)
    return {'mensaje': f"Job '{job_id}' eliminado exitosamente."}

# ── Formato de respuesta ──────────────────────────────────────────────────────

def _formatear(job: dict) -> dict:
    total      = job.get('total', 0)
    procesadas = job.get('procesadas', 0)
    pct        = round(procesadas / total * 100, 1) if total > 0 else 0

    return {
        'job_id'         : job['job_id'],
        'carpeta'        : job['carpeta_nombre'],
        'modo'           : job['modo'],
        'solo_comprimir' : job['solo_comprimir'],
        'ancho'          : job['ancho'],
        'alto'           : job['alto'],
        'kb_min'         : job['kb_min'],
        'kb_max'         : job['kb_max'],
        'estado'         : job['estado'],
        'progreso_pct'   : pct,
        'total'          : total,
        'procesadas'     : procesadas,
        'exitosas'       : job.get('exitosas', 0),
        'con_error'      : job.get('con_error', 0),
        'necesitan_rev'  : job.get('necesitan_rev', 0),
        'archivos_error' : job.get('archivos_error', []),
        'archivos_rev'   : job.get('archivos_rev', []),
        'creado_en'      : job['creado_en'],
        'iniciado_en'    : job.get('iniciado_en'),
        'completado_en'  : job.get('completado_en'),
        'descargar_en'   : f"/resultado/{job['job_id']}/descarga",
        'informe_en'     : f"/resultado/{job['job_id']}/informe",
    }
