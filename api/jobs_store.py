"""
jobs_store.py — Capa de datos para jobs de procesamiento
──────────────────────────────────────────────────────────
Responsabilidad única: persistir y leer jobs del archivo JSON.
No sabe nada de FastAPI ni de HTTP — solo CRUD de jobs.

Persiste en un archivo JSON para que los jobs sobrevivan reinicios.
Usa threading.Lock para garantizar consistencia con múltiples workers.

Importado por:
  api/routers/jobs.py  → para crear, consultar y actualizar jobs desde los endpoints
  api/routers/resultado.py → para verificar el estado antes de permitir la descarga
"""
import json
import os
import shutil
import threading
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from .config import JOBS_DB_PATH, STORAGE_ROOT


class EstadoJob(str, Enum):
    PENDIENTE  = 'pendiente'   # creado, en cola, aún no inició
    PROCESANDO = 'procesando'  # corriendo en hilo background
    COMPLETADO = 'completado'  # terminó (con o sin errores parciales)
    FALLIDO    = 'fallido'     # error crítico que impidió completar


_lock = threading.Lock()


# ── Persistencia ──────────────────────────────────────────────────────────────

def _cargar_db() -> dict:
    """Lee el archivo JSON. Devuelve {} si no existe o está corrupto."""
    if not os.path.exists(JOBS_DB_PATH):
        return {}
    try:
        with open(JOBS_DB_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _guardar_db(db: dict) -> None:
    os.makedirs(os.path.dirname(JOBS_DB_PATH), exist_ok=True)
    with open(JOBS_DB_PATH, 'w', encoding='utf-8') as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


# ── CRUD ──────────────────────────────────────────────────────────────────────

def crear_job(
    carpeta_nombre : str,
    modo           : str,
    ancho          : Optional[int],
    alto           : Optional[int],
    kb_min         : float,
    kb_max         : float,
    solo_comprimir : bool,
) -> dict:
    """Crea un job nuevo, lo persiste y devuelve su dict completo."""
    job_id = str(uuid.uuid4())
    ahora  = datetime.now(timezone.utc).isoformat()

    job = {
        'job_id'         : job_id,
        'carpeta_nombre' : carpeta_nombre,
        'modo'           : modo,
        'ancho'          : ancho,
        'alto'           : alto,
        'kb_min'         : kb_min,
        'kb_max'         : kb_max,
        'solo_comprimir' : solo_comprimir,
        'estado'         : EstadoJob.PENDIENTE,
        'creado_en'      : ahora,
        'iniciado_en'    : None,
        'completado_en'  : None,
        # Progreso
        'total'          : 0,
        'procesadas'     : 0,
        'exitosas'       : 0,
        'con_error'      : 0,
        'necesitan_rev'  : 0,
        # Detalle
        'archivos_error' : [],
        'archivos_rev'   : [],
        'error_mensaje'  : None,
    }

    with _lock:
        db = _cargar_db()
        db[job_id] = job
        _guardar_db(db)

    return job


def obtener_job(job_id: str) -> Optional[dict]:
    """Devuelve el job o None si no existe."""
    with _lock:
        db = _cargar_db()
    return db.get(job_id)


def listar_jobs() -> list:
    """Todos los jobs ordenados del más reciente al más antiguo."""
    with _lock:
        db = _cargar_db()
    return sorted(db.values(), key=lambda j: j['creado_en'], reverse=True)


def actualizar_job(job_id: str, **campos) -> None:
    """Actualiza campos arbitrarios de un job de forma thread-safe."""
    with _lock:
        db = _cargar_db()
        if job_id not in db:
            return
        db[job_id].update(campos)
        _guardar_db(db)

def eliminar_job(job_id: str) -> None:
    """Elimina un job del almacenamiento y su carpeta de storage/projects."""
    with _lock:
        db = _cargar_db()
        if job_id not in db:
            return
        
        # Obtener el nombre de la carpeta antes de eliminar del JSON
        carpeta_nombre = db[job_id].get('carpeta_nombre')
        
        # Eliminar del JSON
        del db[job_id]
        _guardar_db(db)
    
    # Eliminar la carpeta de storage/projects (sin lock, fuera del critical section)
    if carpeta_nombre:
        carpeta_ruta = os.path.join(STORAGE_ROOT, carpeta_nombre)
        if os.path.exists(carpeta_ruta):
            try:
                shutil.rmtree(carpeta_ruta)
            except Exception as e:
                # Log del error pero no fallar la eliminación del job
                print(f"Advertencia: No se pudo eliminar la carpeta {carpeta_ruta}: {e}")