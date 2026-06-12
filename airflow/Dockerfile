# syntax=docker/dockerfile:1
# Reuse ML packages from train image without overwriting Airflow's SQLAlchemy.
# Build train first:  docker compose --profile train build train
ARG TRAIN_IMAGE=miniproj-train:latest
FROM ${TRAIN_IMAGE} AS saint-ml

FROM apache/airflow:2.10.4-python3.11

USER root
RUN mkdir -p /opt/saint-ml && chown airflow:root /opt/saint-ml

USER airflow
COPY --from=saint-ml /usr/local/lib/python3.11/site-packages /opt/saint-ml/site-packages
ENV SAINT_ML_PYTHONPATH=/opt/saint-ml/site-packages
