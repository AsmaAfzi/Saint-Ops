FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train_model.py .

# Volumes for data and output artifacts
VOLUME ["/app/data", "/app/models", "/app/artifacts"]

CMD ["python", "train_model.py"]