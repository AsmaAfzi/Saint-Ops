# Build from repo root: docker compose --profile train build train
FROM python:3.11-slim

WORKDIR /app

COPY ml/requirements.txt ml/requirements.txt
RUN pip install --no-cache-dir -r ml/requirements.txt

COPY ml/train_model.py ml/train_model.py
COPY ml/mlflow_log.py ml/mlflow_log.py

CMD ["python", "ml/train_model.py"]
