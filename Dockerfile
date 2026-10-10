FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    MODELS_DIR=/app/models

WORKDIR /app

# PyTorch solo CPU (la red es diminuta; no hace falta GPU ni ventana gráfica)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY game.py model.py pytest.ini ./
COPY evolution ./evolution
COPY tests ./tests
COPY scripts ./scripts

# /app/models es un volumen: los modelos NO viven en la capa efímera del contenedor
VOLUME ["/app/models"]

# Forma exec: Python es el proceso principal y recibe SIGTERM (docker stop) directamente
STOPSIGNAL SIGTERM
CMD ["python", "-m", "evolution", "run"]
