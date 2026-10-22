# syntax=docker/dockerfile:1
# Hugging Face Space image: RAG API only (Chroma + Groq), listening on the
# Space's fixed port 7860.
FROM python:3.11-slim AS wheels

ENV PIP_NO_CACHE_DIR=1

WORKDIR /build
COPY requirements.txt .
# The dependencies are installed in this stage so the chroma embedding model
# can be downloaded and cached before the runtime image is assembled.
RUN pip wheel --wheel-dir /wheels -r requirements.txt \
    && pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && python -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction; DefaultEmbeddingFunction()(['warm up the embedding cache'])" \
    && mkdir -p /root/.cache/chroma

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CHROMA_PERSIST_PATH=/app/data/chroma

WORKDIR /app

COPY --from=wheels /wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

COPY rag_assistant ./rag_assistant
COPY data ./data

# Ship the pre-warmed chroma ONNX embedding cache so the first ingest does
# not pull 79MB over a slow connection on every restart.
COPY --from=wheels /root/.cache /home/appuser/.cache

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data/chroma \
    && chown -R appuser:appuser /app /home/appuser/.cache
USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=4); sys.exit(0)"

CMD ["uvicorn", "rag_assistant.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
