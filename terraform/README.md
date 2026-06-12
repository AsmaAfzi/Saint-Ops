# SAINT-OPS Terraform (Phase 7) — Local / Zero Cost

Deploys the SAINT-OPS Helm chart to **Docker Desktop Kubernetes**. No kind cluster, no AWS/Azure charges.

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) with **Kubernetes enabled** (Settings → Kubernetes → Enable)
- [Terraform](https://developer.hashicorp.com/terraform/downloads) ≥ 1.5
- [kubectl](https://kubernetes.io/docs/tasks/tools/)
- [Helm](https://helm.sh/docs/intro/install/) ≥ 3.14 (optional — Terraform runs Helm for you)

Verify the cluster is up:

```powershell
kubectl config use-context docker-desktop
kubectl get nodes
```

## Before you apply

1. **Stop the Docker Compose demo** (do not run Compose and K8s together on the same machine):

   ```powershell
   docker compose --profile monitoring --profile airflow down
   ```

2. **Build local images** (used by `values-dev.yaml` with `imagePullPolicy: Never`):

   ```powershell
   docker compose build backend frontend
   ```

## Deploy

```powershell
cd terraform
terraform init -upgrade
terraform apply
```

`terraform destroy` removes the Helm release and namespace only — it does **not** disable Docker Desktop Kubernetes.

## After apply

```powershell
kubectl get pods -n saint-dev
```

Seed the PVC and port-forward — see `demo_commands` in `terraform output`, or use `scripts/k8s_seed_pod.yaml` as documented in the main README.

## Migrating from the old kind-based Terraform

If you previously ran `terraform apply` with kind:

```powershell
# Remove stale kind cluster (optional)
docker rm -f saint-ops-demo-control-plane saint-ops-demo-worker 2>$null

cd terraform
terraform init -upgrade
terraform state rm kind_cluster.saint 2>$null   # drop removed resource from state
terraform apply
```

## UAE cloud note (documentation only)

Production would target `me-central-1` (UAE) or `me-south-1` (Bahrain) on AWS. This module uses **Docker Desktop K8s** so the FYP demo runs at **$0** on your laptop.
