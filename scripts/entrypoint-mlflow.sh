#!/bin/sh
set -e

ARTIFACT_ROOT="${MLFLOW_ARTIFACT_ROOT:-/mlflow/artifacts}"
BACKEND_STORE="${MLFLOW_BACKEND_STORE_URI:-sqlite:///mlflow/mlflow.db}"

echo "MLflow artifact root: ${ARTIFACT_ROOT}"
echo "MLflow backend store: ${BACKEND_STORE}"

exec mlflow server \
  --host 0.0.0.0 \
  --backend-store-uri "${BACKEND_STORE}" \
  --artifacts-destination "${ARTIFACT_ROOT}" \
  --serve-artifacts
