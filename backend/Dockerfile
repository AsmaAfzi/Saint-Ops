# Build from repository root:
#   docker compose build backend
FROM python:3.11-slim

WORKDIR /app

RUN adduser --disabled-password --gecos "" appuser

# Dependencies (paths relative to repo root build context)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Detection logic lives in ml/inference.py (v8 fixed val_p99 + CUSUM).
# docker-compose also bind-mounts ./ml/inference.py for live updates.
COPY backend/backend.py .
COPY ml/inference.py .
COPY ml/registry.py .

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]
