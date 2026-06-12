# SAINT-OPS Helm Chart (Phase 4)

Deploys FastAPI backend, React frontend, and MLflow on Kubernetes.

## Prerequisites

- Kubernetes 1.25+
- `helm` 3.12+
- `metrics-server` (for CPU HPA)
- Optional: `prometheus-adapter` for latency-based HPA (`values-production.yaml`)

## Install per environment

```bash
# Development
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

## Seed data on PVC

```bash
./scripts/k8s_seed_data.sh saint-dev saint-dev-saint-ops
```

## Lint & dry-run

```bash
helm lint helm/saint-ops -f helm/saint-ops/values-dev.yaml
helm template saint-dev helm/saint-ops -f helm/saint-ops/values-dev.yaml
```

## Resources created

| Resource | Purpose |
|----------|---------|
| Deployments | backend (3), frontend (2), mlflow (1) — overridden per values |
| Services | ClusterIP internal; LoadBalancer in production |
| PVC | Shared storage: data, artifacts, models, mlflow |
| HPA | Backend CPU 70%, optional latency metric |
| CronJobs | Drift report + dataset freshness |
| ConfigMaps | Nginx proxy, MLflow entrypoint, cron scripts |
| Secrets | AWS/S3 credentials (encrypted at rest by K8s) |

## S3 artifacts (production)

Set in `values-production.yaml` or via `--set`:

```yaml
mlflow:
  s3:
    enabled: true
  artifactRoot: s3://your-bucket/mlflow/artifacts
secrets:
  awsAccessKeyId: ...
  awsSecretAccessKey: ...
```
