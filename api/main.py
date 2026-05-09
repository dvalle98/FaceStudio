"""
main.py — Punto de entrada de la API FastAPI
─────────────────────────────────────────────
Iniciar el servidor:
  cd procesador-imagenes
  uvicorn api.main:app --reload --port 8000

Acceder a la interfaz:
  http://localhost:8000

Documentación interactiva:
  http://localhost:8000/docs
  http://localhost:8000/redoc
"""
import os
import traceback

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import STORAGE_ROOT
from .routers import carpetas, jobs, resultado

# ── Directorios necesarios al arrancar ───────────────────────────────────────
os.makedirs(STORAGE_ROOT, exist_ok=True)

# ── Aplicación ────────────────────────────────────────────────────────────────
# ROOT_PATH permite que FastAPI sepa que está montado bajo un sub-path en el
# reverse proxy (ej: /facestudio). Afecta la generación del OpenAPI schema,
# las URLs de /docs y /redoc, y los redirects internos.
# En desarrollo local se deja vacío (por defecto).
app = FastAPI(
    title     = 'API de Procesamiento de Fotos — Reconocimiento Facial',
    version   = '1.0.0',
    docs_url  = '/docs',
    redoc_url = '/redoc',
    root_path = os.environ.get('ROOT_PATH', ''),
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# allow_credentials DEBE ser False cuando allow_origins=["*"].
# Si se pone True con "*" FastAPI lanza un error de configuración.
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ['*'],
    allow_credentials = False,
    allow_methods     = ['*'],
    allow_headers     = ['*'],
    expose_headers    = ['*'],
)

# ── Handler global de errores 500 ────────────────────────────────────────────
# Cuando ocurre una excepción no capturada, FastAPI devuelve un 500 sin
# los headers CORS, lo que hace que el navegador muestre "CORS error" en
# lugar del error real. Este handler garantiza que los headers siempre
# estén presentes en cualquier respuesta, incluyendo los errores.
@app.exception_handler(Exception)
async def handler_error_global(request: Request, exc: Exception):
    traceback.print_exc()   # imprime el traceback real en la consola del servidor
    return JSONResponse(
        status_code = 500,
        content     = {'detail': f'Error interno del servidor: {type(exc).__name__}: {exc}'},
        headers     = {
            'Access-Control-Allow-Origin' : '*',
            'Access-Control-Allow-Methods': '*',
            'Access-Control-Allow-Headers': '*',
        },
    )

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(carpetas.router)
app.include_router(jobs.router)
app.include_router(resultado.router)

# ── Frontend estático ─────────────────────────────────────────────────────────
_FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend')

if os.path.isdir(_FRONTEND_DIR):
    app.mount('/static', StaticFiles(directory=_FRONTEND_DIR), name='static')


# ── Rutas generales ───────────────────────────────────────────────────────────

@app.get('/health', tags=['General'], summary='Health check')
def health():
    return {'status': 'ok', 'API': 'FotoProcess'}


@app.get('/', include_in_schema=False)
def frontend():
    index_path = os.path.join(_FRONTEND_DIR, 'index.html')
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {'mensaje': 'FotoProcess API v1.0', 'docs': '/docs'}
