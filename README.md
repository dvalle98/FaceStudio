# FaceStudio — Procesador de imágenes con detección facial

FaceStudio es una aplicación web para procesar fotos de perfil y carnet en lotes. Detecta automáticamente el rostro de cada imagen, la recorta con el encuadre correcto, la redimensiona al formato requerido y ajusta su peso en KB — todo desde el navegador, sin instalar nada en el cliente.

Pensado para procesar decenas o cientos de fotos a la vez: el usuario sube las imágenes, elige el modo de procesamiento y descarga un ZIP con los resultados.

---

## Características

- **Detección facial en cadena** — intenta tres métodos en orden: Haarcascade (rápido), DNN ResNet-10 (robusto), y fallback centrado (último recurso, marcado para revisión manual)
- **Tres modos de procesamiento** — ratio 4:5 (carnet oficial), ratio 1:1 (cuadrado), o solo comprimir el peso sin tocar el encuadre
- **Dimensiones personalizables** — ancho, alto y rango de peso en KB configurables por lote
- **Procesamiento en background** — la API responde de inmediato y el progreso se actualiza en tiempo real
- **Descarga en ZIP** — todas las imágenes procesadas de un lote en un solo archivo
- **Interfaz web incluida** — no requiere frontend externo

---

## Requisitos

### Para correr con Docker (recomendado)
- [Docker](https://docs.docker.com/get-docker/) y Docker Compose
- Python 3.x instalado en el servidor (solo para el paso inicial de descarga de modelos)
- Conexión a internet la primera vez (para descargar los modelos DNN, ~10 MB)

### Para correr sin Docker
- Python 3.10 o superior
- Las dependencias listadas en `api/requirements.txt`

---

## Estructura del proyecto

```
procesador-imagenes/
├── api/
│   ├── main.py              ← Punto de entrada FastAPI
│   ├── config.py            ← Configuración central (paths, límites)
│   ├── pipeline.py          ← Motor de procesamiento de imágenes
│   ├── jobs_store.py        ← Persistencia de jobs en JSON
│   ├── requirements.txt     ← Dependencias Python
│   └── routers/
│       ├── carpetas.py      ← Subida y gestión de imágenes
│       ├── jobs.py          ← Crear y consultar jobs
│       └── resultado.py     ← Descarga de resultados
├── frontend/
│   └── index.html           ← Interfaz web (SPA vanilla JS)
├── modelos/                 ← Modelos DNN — se descargan una vez, no van en la imagen Docker
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
├── storage/                 ← Datos en tiempo de ejecución — no van en la imagen Docker
│   ├── jobs.json            ← Base de datos de jobs
│   └── projects/
│       └── {nombre_lote}/
│           ├── entrada/     ← Imágenes originales subidas
│           ├── salida/      ← Imágenes procesadas
│           └── zips/        ← ZIPs generados
├── setup_modelos.py         ← Script para descargar los modelos DNN
├── Dockerfile
└── docker-compose.yml
```

> `modelos/` y `storage/` existen en el repositorio pero **nunca se incluyen en la imagen Docker**. Se montan desde el servidor como bind mounts.

---

## Configuración inicial

Estos pasos se hacen **una sola vez** al desplegar el proyecto por primera vez.

### 1. Clonar el repositorio

```bash
git clone <url-del-repositorio> procesador-imagenes
cd procesador-imagenes
```

### 2. Descargar los modelos DNN

Los modelos de detección facial (~10 MB) no están en el repositorio. Descárgalos ejecutando:

```bash
python setup_modelos.py
```

Esto crea la carpeta `modelos/` con los dos archivos necesarios. Solo necesitas hacerlo una vez; Docker los monta directamente desde ahí.

### 3. Crear la carpeta de storage

Si no existe, créala manualmente:

```bash
mkdir -p storage/projects
```

Docker la monta desde el servidor. Todo lo que la app escriba ahí (imágenes, jobs.json) quedará en tu máquina, no dentro del contenedor.

---

## Iniciar con Docker (recomendado)

```bash
docker compose up --build
```

La primera vez construye la imagen (instala dependencias Python, libs del sistema). Las siguientes veces reutiliza la imagen cacheada:

```bash
docker compose up
```

Para correr en background:

```bash
docker compose up -d
```

Verifica que el servicio está funcionando:

```bash
curl http://localhost:8000/health
# → {"status":"ok"}
```

Accede a la interfaz en: **http://localhost:8000**

### Comandos útiles

```bash
docker compose logs -f          # ver logs en tiempo real
docker compose down             # detener el servicio
docker compose build --no-cache # reconstruir la imagen desde cero
```

---

## Iniciar sin Docker

```bash
# Crear y activar entorno virtual
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# Instalar dependencias
pip install -r api/requirements.txt

# Descargar modelos (si no lo hiciste antes)
python setup_modelos.py

# Iniciar el servidor
uvicorn api.main:app --reload --port 8000
```

Accede a la interfaz en: **http://localhost:8000**  
Documentación interactiva de la API: **http://localhost:8000/docs**

---

## Uso

### Desde la interfaz web

1. Abre `http://localhost:8000` en el navegador
2. En **Nuevo lote**, escribe un nombre para el lote (o usa el generado automáticamente)
3. Arrastra las fotos o haz clic para seleccionarlas
4. Elige el modo de procesamiento:
   - **Modo A** — ratio 4:5, 640×800px (instructivo oficial de carnet)
   - **Modo B** — ratio 1:1, 750×750px (cuadrado centrado)
   - **Solo comprimir** — ajusta el peso sin modificar el encuadre
5. Configura opciones avanzadas si lo necesitas (dimensiones personalizadas, rango de peso en KB)
6. Haz clic en **Procesar fotos** y observa el progreso en tiempo real
7. Cuando termine, descarga el ZIP o revisa las miniaturas en pantalla
8. En **Mis lotes** puedes ver el historial de todos los trabajos anteriores

### Modos de procesamiento

| Modo | Ratio | Dimensiones por defecto | Cuándo usarlo |
|------|-------|------------------------|---------------|
| `A`  | 4:5   | 640×800 px             | Fotos carnet con formato oficial |
| `B`  | 1:1   | 750×750 px             | Fotos cuadradas para perfiles |
| `compresion` | — | Sin cambio | La foto ya está encuadrada, solo reducir el peso |

### Desde la API (uso programático)

**Documentación interactiva completa:** `http://localhost:8000/docs`

Flujo básico:

```bash
# 1. Crear una carpeta de trabajo
curl -X POST "http://localhost:8000/carpetas/?nombre=lote_enero"

# 2. Subir imágenes
curl -X POST "http://localhost:8000/carpetas/lote_enero/imagenes" \
     -F "archivos=@foto1.jpg" -F "archivos=@foto2.jpg"

# 3. Lanzar el procesamiento
curl -X POST "http://localhost:8000/jobs/" \
     -H "Content-Type: application/json" \
     -d '{"carpeta_nombre": "lote_enero", "modo": "B", "kb_min": 50, "kb_max": 300}'
# → {"job_id": "abc123...", "total_imagenes": 2}

# 4. Consultar el progreso
curl "http://localhost:8000/jobs/abc123..."
# → {"estado": "procesando", "procesadas": 1, "total": 2, "progreso_pct": 50.0}

# 5. Descargar el ZIP con los resultados
curl -OJ "http://localhost:8000/resultado/abc123.../descarga"
```

**Parámetros del job:**

| Campo | Tipo | Default | Descripción |
|-------|------|---------|-------------|
| `carpeta_nombre` | string | — | Nombre del lote a procesar |
| `modo` | `"A"` / `"B"` / `"compresion"` | `"B"` | Modo de procesamiento |
| `ancho` | int | según modo | Ancho de salida en píxeles (100–2000) |
| `alto` | int | según modo | Alto de salida en píxeles (100–2000) |
| `kb_min` | float | `30` | Peso mínimo del JPEG resultante en KB |
| `kb_max` | float | `300` | Peso máximo del JPEG resultante en KB |

---

## Cambiar la ubicación del storage

Por defecto los datos se guardan en `./storage/` dentro del repositorio clonado. Si necesitas apuntarlo a otro disco o partición, edita `docker-compose.yml`:

```yaml
volumes:
  - /ruta/en/el/servidor:/app/storage   # ← cambia la parte izquierda
  - ./modelos:/app/modelos

environment:
  - STORAGE_PATH=/app/storage           # ← debe coincidir con la parte derecha
```

Ejemplos:
```yaml
- /mnt/datos/procesador:/app/storage
- /var/lib/faceStudio/storage:/app/storage
```

---

## Notas de funcionamiento

- Las imágenes se procesan en memoria (sin archivos temporales) — eficiente para la API
- El archivo `storage/jobs.json` persiste el historial de jobs entre reinicios del servidor
- Si el contenedor se reinicia con jobs en estado `procesando`, esos jobs quedarán detenidos — es necesario relanzarlos manualmente
- Los formatos soportados para subida son: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`
- Tamaño máximo por archivo: 20 MB
