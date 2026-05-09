FROM python:3.11-slim

# libgl1 requerido por cv2.dnn incluso en headless
# libglib2.0-0 y libgomp1 requeridos por OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar dependencias Python
# (capa estable: se cachea y no se reconstruye si solo cambia el código)
COPY api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Código fuente
COPY api/ ./api/
COPY frontend/ ./frontend/

EXPOSE 8001

HEALTHCHECK --interval=60s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001"]
