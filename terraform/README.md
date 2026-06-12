# SAINT-OPS Terraform (Phase 7)

Infrastructure-as-code for deploying the SAINT-OPS **Helm chart** to a local Kubernetes cluster. Targets **Docker Desktop Kubernetes** — no AWS/Azure resources, **$0** cloud cost for the FYP demo.

---

## What Terraform creates

| Resource | Description |
|----------|-------------|
| `kubernetes_namespace` | e.g. `saint-dev` |
| `helm_release` | Installs `../helm/saint-ops` with `values-dev.yaml` overrides |

Terraform does **not** build Docker images or seed PVC data — those are pre-requisites (see below).

```
terraform apply
      │
      ▼
kubernetes_namespace (saint-dev)
      │
      ▼
helm_release → saint-ops chart
      ├── backend Deployment + Service + HPA
      ├── frontend Deployment + Service (Nginx → /api proxy)
      ├── mlflow Deployment + Service + PVC
      ├── ConfigMaps, RBAC, NetworkPolicy, CronJobs
      └── ...
```

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) with **Kubernetes enabled** (Settings → Kubernetes)
- [Terraform](https://developer.hashicorp.com/terraform/downloads) ≥ 1.5
- [kubectl](https://kubernetes.io/docs/tasks/tools/)
- Helm ≥ 3.14 (optional — Terraform Helm provider runs installs)

Verify cluster:

```powershell
kubectl config use-context docker-desktop
kubectl get nodes
```

---

## Before you apply

1. **Stop Docker Compose** (avoid port conflicts on 3000/8000/5000):

   ```powershell
   docker compose --profile monitoring --profile airflow down
   ```

2. **Build local images** (required by `values-dev.yaml` with `pullPolicy: Never`):

   ```powershell
   docker compose build backend frontend
   ```

---

## Deploy

```powershell
cd terraform
terraform init -upgrade
terraform apply
```

Useful outputs: `namespace`, `release_name`, `demo_commands` (port-forward hints).

`terraform destroy` removes the Helm release and namespace only — it does **not** disable Docker Desktop Kubernetes.

---

## After apply

```powershell
kubectl get pods -n saint-dev
kubectl wait --for=condition=ready pod -l app.kubernetes.io/instance=saint-dev -n saint-dev --timeout=300s
```

Seed data and open the UI:

```bash
./scripts/k8s_seed_data.sh saint-dev
kubectl port-forward -n saint-dev svc/saint-dev-saint-ops-frontend 3000:80
kubectl port-forward -n saint-dev svc/saint-dev-saint-ops-backend 8000:8000
```

- Dashboard: http://localhost:3000  
- API: http://localhost:8000/docs

Alternative seed pod: `scripts/k8s_seed_pod.yaml`

---

## Variables (see `variables.tf`)

| Variable | Typical value | Purpose |
|----------|---------------|---------|
| `kubeconfig_path` | `~/.kube/config` | Cluster credentials |
| `kubeconfig_context` | `docker-desktop` | Active context |
| `namespace` | `saint-dev` | K8s namespace |
| `release_name` | `saint-dev` | Helm release name |
| `helm_values_file` | `helm/saint-ops/values-dev.yaml` | Values file path |

---

## Migrating from kind-based Terraform

If you previously used a `kind` cluster resource:

```powershell
docker rm -f saint-ops-demo-control-plane saint-ops-demo-worker 2>$null
cd terraform
terraform init -upgrade
terraform state rm kind_cluster.saint 2>$null
terraform apply
```

---

## Relationship to Docker Compose

| Concern | Docker Compose | Terraform + Helm |
|---------|----------------|------------------|
| Use case | Local dev, full demo (monitoring, Airflow) | K8s deployment demo |
| Frontend | `localhost:3000` | `port-forward` to frontend Service |
| MLflow | Included | Included in chart |
| Airflow / Grafana | Compose profiles | Not in Helm chart (Compose only) |
| Image source | `docker compose build` | Same images tagged for K8s |

Run **one** at a time on the same machine.

---

## Production cloud note (documentation)

A production deployment would target a managed cluster (e.g. AWS `me-central-1` UAE or `me-south-1` Bahrain) with remote state, IRSA/workload identity, and S3-backed MLflow artifacts. This module intentionally stays local for zero-cost academic demonstration.
