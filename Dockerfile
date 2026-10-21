FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app/src HF_HOME=/app/.hf
COPY requirements.txt .
# CPU-only torch keeps the image ~1 GB smaller than the default CUDA wheel
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY scripts ./scripts
COPY app ./app
RUN python scripts/fetch_corpus.py
# Bake models + index into the image so the container starts fast and offline-capable (generation still needs the API).
ARG RAG_EMBEDDER=BAAI/bge-small-en-v1.5
ARG RAG_RERANKER=BAAI/bge-reranker-base
ENV RAG_EMBEDDER=$RAG_EMBEDDER RAG_RERANKER=$RAG_RERANKER
RUN python -c "from ragkit.pipeline import Config, build_index; build_index(Config.from_env())" \
 && python -c "from sentence_transformers import CrossEncoder; CrossEncoder('$RAG_RERANKER')"
EXPOSE 8000
CMD ["uvicorn", "ragkit.api:app", "--host", "0.0.0.0", "--port", "8000"]
