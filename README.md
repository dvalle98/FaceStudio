# API de Procesamiento de Fotos — Reconocimiento Facial

## Instalación

```bash
# Crear y activar entorno virtual
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# Instalar dependencias
pip install -r requirements.txt

# Descargar modelos DNN (solo una vez, ~10 MB)
python3 setup_modelos.py
```

## Iniciar el servidor

```bash
# Desde la carpeta procesador-imagenes/
uvicorn api.main:app --reload --port 8000
```

El servidor queda en: `http://localhost:8000`  
Documentación interactiva: `http://localhost:8000/docs`

---

## Flujo de trabajo completo

### 1. Crear una carpeta de trabajo
```
POST http://localhost:8000/carpetas/?nombre=lote_enero
```

### 2. Subir imágenes
```
POST http://localhost:8000/carpetas/lote_enero/imagenes
Content-Type: multipart/form-data
archivos: [foto1.jpg, foto2.jpg, ...]
```
Puedes subir cientos de archivos en una sola petición o en lotes.

### 3. Verificar las imágenes cargadas
```
GET http://localhost:8000/carpetas/lote_enero/imagenes
```

### 4. Lanzar el procesamiento
```
POST http://localhost:8000/jobs/
Content-Type: application/json

{
  "carpeta_nombre": "lote_enero",
  "modo": "B",
  "kb_min": 50,
  "kb_max": 300
}
```
La respuesta es inmediata — el procesamiento corre en segundo plano.  
Recibes un `job_id`.

### 5. Consultar el progreso
```
GET http://localhost:8000/jobs/{job_id}
```
Respuesta de ejemplo:
```json
{
  "estado": "procesando",
  "progreso_pct": 47.3,
  "procesadas": 142,
  "total": 300,
  "exitosas": 139,
  "con_error": 3
}
```

### 6. Descargar resultados
```
GET http://localhost:8000/resultado/{job_id}/descarga
```
Descarga un ZIP con todas las imágenes procesadas.

### 7. Ver informe detallado
```
GET http://localhost:8000/resultado/{job_id}/informe
```

---

## Modos de procesamiento

| Modo | Ratio | Dimensiones por defecto | Uso |
|------|-------|------------------------|-----|
| `A`  | 4:5   | 640×800px               | Instructivo oficial |
| `B`  | 1:1   | 750×750px               | Cuadrado centrado |
| `compresion` | — | Sin cambio | Solo ajustar peso |

## Dimensiones personalizadas

```json
{
  "carpeta_nombre": "lote_enero",
  "modo": "A",
  "ancho": 512,
  "alto": 640,
  "kb_min": 50,
  "kb_max": 300
}
```

---

## Estructura del proyecto

```
procesador-imagenes/
├── pipeline.py          ← Motor de procesamiento (sin FastAPI)
├── setup_modelos.py     ← Descarga los modelos DNN
├── requirements.txt
├── api/
│   ├── main.py          ← Punto de entrada FastAPI
│   ├── config.py        ← Configuración central
│   ├── jobs.py          ← Sistema de jobs persistente
│   └── routers/
│       ├── carpetas.py  ← Gestión de carpetas e imágenes
│       ├── jobs.py      ← Crear y consultar jobs
│       └── resultado.py ← Descarga de resultados
└── storage/             ← Creado automáticamente al iniciar
    ├── jobs.json        ← Base de datos de jobs
    └── {nombre_carpeta}/
        ├── entrada/     ← Imágenes originales subidas
        ├── salida/      ← Imágenes procesadas
        └── zips/        ← ZIPs de descarga (uso futuro)
```
