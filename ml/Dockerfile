# syntax=docker/dockerfile:1
# Build from repo root: docker compose --profile train build train
FROM python:3.11-slim

WORKDIR /app

ENV PIP_DEFAULT_TIMEOUT=1000 \
    PIP_RETRIES=10

COPY ml/requirements.txt ml/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=1000 --retries=10 -r ml/requirements.txt

COPY ml/train_model.py ml/train_model.py
COPY ml/mlflow_log.py ml/mlflow_log.py

CMD ["python", "ml/train_model.py"]
