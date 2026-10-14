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
COPY models_statics ./models_statics
COPY trainPuzzles ./trainPuzzles
COPY tests ./tests
COPY scripts ./scripts

# /app/models es un volumen: los modelos NO viven en la capa efímera del contenedor
VOLUME ["/app/models"]

# Puerto de la página para jugar contra la IA (publicar con -p 127.0.0.1:8765:8765)
EXPOSE 8765

# Forma exec: Python es el proceso principal y recibe SIGTERM (docker stop) directamente
STOPSIGNAL SIGTERM
CMD ["python", "-m", "evolution", "run"]