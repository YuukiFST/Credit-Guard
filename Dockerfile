# Dockerfile — API FastAPI

FROM python:3.12-slim

WORKDIR /app

# Instala uv (gerenciador de pacotes ultra-rápido)
RUN pip install uv

# Instala dependências primeiro (cache de layer do Docker)
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Copia código-fonte
COPY src/ ./src/
COPY config.yaml .

ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
