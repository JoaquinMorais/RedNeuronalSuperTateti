FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    MODELS_DIR=/app/models

# Versión e índice de PyTorch parametrizables:
#   * CPU (default): wheels +cpu, portables y livianas -> Linux/CI.
#   * GPU (Windows con RTX): pasar --build-arg TORCH_INDEX_URL=.../cu128
#     (ver docker-compose.gpu.yml). La red es diminuta.
ARG TORCH_VERSION=2.8.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

WORKDIR /app

RUN pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX_URL}"
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