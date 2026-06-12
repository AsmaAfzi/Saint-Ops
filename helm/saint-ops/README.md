# SAINT-OPS Helm Chart

Kubernetes packaging for the SAINT-OPS application plane: **FastAPI backend**, **React/Nginx frontend**, and **MLflow** model registry, with optional autoscaling, scheduled jobs, and network policies.

---

## Architecture (in-cluster)

```
                    ┌─────────────────────────────────────┐
                    │  Ingress (optional, production)      │
                    └──────────────┬──────────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
     ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐
     │ Frontend Svc    │  │ Backend Svc     │  │ MLflow Svc   │
     │ ClusterIP :80   │  │ ClusterIP :8000 │  │ :5000        │
     └────────┬────────┘  └────────┬────────┘  └──────┬───────┘
              │                    │                     │
     ┌────────▼────────┐  ┌────────▼────────┐  ┌────────▼───────┐
     │ Deployment      │  │ Deployment      │  │ Deployment     │
     │ Nginx + React   │  │ FastAPI + TF    │  │ MLflow server  │
     │ /api → backend  │  │ inference.py    │  │                │
     └─────────────────┘  └────────┬────────┘  └────────────────┘
                                   │
                          ┌────────▼────────┐
                          │ PVC (ReadWriteMany│
                          │ or per-env policy)│
                          │ data/ artifacts/  │
                          │ models/ mlflow/   │
                          └───────────────────┘

     ┌─────────────────┐  ┌─────────────────┐
     │ HPA (backend)   │  │ CronJobs          │
     │ CPU / latency   │  │ drift report,     │
     └─────────────────┘  │ data freshness    │
                          └─────────────────┘
```

The frontend ConfigMap mirrors Docker Compose: Nginx serves the SPA and proxies `/api/` to the backend service.

---

## Prerequisites

- Kubernetes 1.25+
- Helm 3.12+
- `metrics-server` (for CPU-based HPA)
- Optional: `prometheus-adapter` for latency-based HPA (`values-production.yaml`)

---

## Install per environment

```bash
# Development (local images, single replica)
helm upgrade --install saint-dev ./helm/saint-ops \
  -f helm/saint-ops/values-dev.yaml \
  -n saint-dev --create-namespace

# Staging
helm upgrade --install saint-staging ./helm/saint-ops \
  -f helm/saint-ops/values-staging.yaml \
  -n saint-staging --create-namespace

# Production
helm upgrade --install saint-prod ./helm/saint-ops \
  -f helm/saint-ops/values-production.yaml \
  -n saint-production --create-namespace
```

### Seed PVC from local files

```bash
./scripts/k8s_seed_data.sh saint-dev
# or use scripts/k8s_seed_pod.yaml — see terraform output / main README
```

### Access (dev)

```bash
kubectl port-forward -n saint-dev svc/saint-dev-saint-ops-frontend 3000:80
kubectl port-forward -n saint-dev svc/saint-dev-saint-ops-backend 8000:8000
```

Dashboard: http://localhost:3000 · API docs: http://localhost:8000/docs

---

## Values files

| File | Use case |
|------|----------|
| `values.yaml` | Defaults (3 backend replicas, HPA, MLflow env) |
| `values-dev.yaml` | Local images (`imagePullPolicy: Never`), reduced resources |
| `values-staging.yaml` | Pre-production overrides |
| `values-production.yaml` | LoadBalancer, S3 artifacts, stricter HPA |

Key `backend` settings (via ConfigMap): `MLFLOW_TRACKING_URI`, `MLFLOW_MODEL_URI`, `MLFLOW_SHADOW_MODEL_URI`, `MLFLOW_SHADOW_TRAFFIC_PCT`, `STREAM_CSV`.

---

## Resources created

| Resource | Purpose |
|----------|---------|
| Deployments | `backend`, `frontend`, `mlflow` |
| Services | ClusterIP (LoadBalancer in production frontend) |
| PVC | Shared `data`, `artifacts`, `models`, MLflow store |
| HPA | Backend CPU target 70%; optional request latency metric |
| CronJobs | `k8s_drift_report.py`, `k8s_data_freshness.py` |
| ConfigMaps | Frontend Nginx, MLflow entrypoint, cron scripts |
| Secrets | AWS/S3 credentials (production) |
| ServiceAccount + RBAC | Least-privilege pod identity |
| NetworkPolicy | Restrict backend ingress to frontend + monitoring |

---

## Lint and dry-run

```bash
helm lint helm/saint-ops -f helm/saint-ops/values-dev.yaml
helm template saint-dev helm/saint-ops -f helm/saint-ops/values-dev.yaml
```

---

## S3 artifacts (production)

In `values-production.yaml` or via `--set`:

```yaml
mlflow:
  s3:
    enabled: true
  artifactRoot: s3://your-bucket/mlflow/artifacts
secrets:
  awsAccessKeyId: ...
  awsSecretAccessKey: ...
```

---

## Terraform integration

Phase 7 deploys this chart to Docker Desktop Kubernetes:

```bash
cd terraform && terraform apply
```

See [terraform/README.md](../../terraform/README.md).
