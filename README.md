# SAINT-OPS
### Semantic AI for Predictive Maintenance — Production DevOps Edition

![CI/CD](https://github.com/AsmaAfzi/Saint-Ops/actions/workflows/ci.yml/badge.svg)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![Python](https://img.shields.io/badge/Python-3.11-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.116-teal)
![License](https://img.shields.io/badge/License-MIT-yellow)

> Final Year Project extending the SAINT mini project into a production-grade,
> DevOps-augmented MLOps system for oil & gas sensor drift detection.
>
> **Student:** Asma Mohammad Afzal | 220101055  
> **Mentor:** Dr. Pamba Rajavarma  
> **Institution:** School of Engineering and IT — B.Tech CSE (Honors)  
> **Academic Year:** 2025–2026

---

## What is SAINT-OPS?

SAINT (Semantic AI for Predictive Maintenance) achieved **~84% F1-score** detecting sensor drift in oil & gas operations using an **LSTM autoencoder** trained on the [Equinor Volve dataset](https://www.equinor.com/energy/volve-data-sharing).

**SAINT-OPS** wraps that ML core in containerized infrastructure: a React operations dashboard, FastAPI inference API, MLflow model registry, optional Prometheus/Grafana monitoring, Airflow-based retraining, Helm/Kubernetes deployment, and Terraform for local cluster provisioning.

---

## System architecture

```
                         ┌─────────────────────────────────────────┐
                         │           Browser (operator)            │
                         └────────────────────┬────────────────────┘
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    │                         │                         │
                    ▼                         ▼                         ▼
           ┌────────────────┐        ┌────────────────┐       ┌─────────────────┐
           │ Frontend :3200 │        │ Grafana :3301  │       │ Airflow UI :8081│
           │ React + Nginx  │        │ (monitoring)   │       │ (airflow prof.) │
           └───────┬────────┘        └───────┬────────┘       └────────┬────────┘
                   │ /api/* proxy            │ scrape                  │ DAGs
                   ▼                         ▼                         ▼
           ┌───────────────────────────────────────────────────────────────────┐
           │                    saint-network (Docker Compose)               │
           │  ┌─────────────┐  ┌─────────────┐  ┌──────────┐  ┌───────────┐ │
           │  │ Backend     │  │ MLflow      │  │Prometheus│  │ Airflow   │ │
           │  │ FastAPI     │◄─┤ Registry    │  │+ Alertmgr│  │ scheduler │ │
           │  │ TensorFlow  │  │ :5000       │  └────┬─────┘  └─────┬─────┘ │
           │  │ :8000       │  └─────────────┘       │              │       │
           │  └──────┬──────┘                        │              │       │
           │         │ reads                         │              │ train │
           │         ▼                               │              ▼       │
           │  data/  models/  artifacts/  mlflow-data/  (shared volumes)     │
           └───────────────────────────────────────────────────────────────────┘

  Kubernetes path (optional):  Terraform → Docker Desktop K8s → Helm chart
                               See helm/saint-ops/ and terraform/
```

### Planes

| Plane | Components | Role |
|-------|------------|------|
| **Data** | `data/*.csv`, Volve xlsx | Scenario timelines replayed as a simulated live stream |
| **ML** | `ml/`, `models/`, `artifacts/`, MLflow | LSTM training, inference, thresholds, registry lifecycle |
| **Application** | `frontend/`, `backend/` | Dashboard + REST API for drift monitoring and MLOps |
| **Observability** | `monitoring/` (Prometheus, Grafana, Alertmanager) | Metrics, dashboards, alert routing |
| **Orchestration** | `airflow/` (DAGs, scheduler, webserver) | Drift-triggered and scheduled retraining |
| **Platform** | `helm/`, `terraform/`, `.github/workflows/` | K8s packaging, local IaC, CI/CD |

### Request flow (drift dashboard)

1. Browser loads the React app from **frontend** (Nginx on port 3000).
2. The UI polls `GET /api/drift_data?t=N` — Nginx proxies `/api/*` → **backend:8000**.
3. On first use, the backend loads the LSTM model (MLflow Production, or local fallback), precomputes a **stream timeline** from `scenario_C_annulus.csv`, and caches it in memory.
4. Each response returns reconstruction errors, drift flags, coarse/fine labels, and per-feature explanations.
5. The **MLOps & system** tab calls registry, health, and shadow-comparison endpoints (see [frontend/README.md](frontend/README.md)).

---

## Project structure

```
saint-ops/
├── .github/workflows/ci.yml     # pytest, flake8, Docker build/push, Trivy, Locust
├── airflow/
│   ├── dags/                    # saint_retrain, saint_drift_sensor
│   ├── Dockerfile               # Airflow image (bundles train env)
│   └── Dockerfile.ui-proxy      # Demo UI with auto-login on :8081
├── artifacts/                   # Scaler, thresholds, feature_meta.json (.npy)
├── backend/
│   ├── backend.py               # FastAPI — drift, registry, ops, retrain APIs
│   ├── Dockerfile
│   └── tests/test_api.py
├── data/                        # Volve-derived CSV scenarios (A/B/C)
├── frontend/
│   ├── src/                     # React dashboard (monitor + MLOps tabs)
│   ├── nginx.conf               # Static files + /api reverse proxy
│   └── Dockerfile
├── helm/saint-ops/              # Kubernetes chart (backend, frontend, MLflow, HPA, CronJobs)
├── ml/
│   ├── train_model.py           # Training + MLflow logging
│   ├── inference.py             # Stream timeline + drift classification
│   ├── registry.py              # Staging → Production lifecycle
│   └── Dockerfile               # Optional `train` compose profile
├── models/                      # lstm_autoencoder.keras (local fallback)
├── monitoring/                  # Prometheus, Grafana, Alertmanager configs
├── scripts/                     # demo_stack.ps1, k8s seed, retrain triggers
├── terraform/                   # Docker Desktop K8s + Helm via Terraform
├── tests/load/locustfile.py     # Load testing
├── docker-compose.yml           # Core + optional profiles
└── README.md
```

---

## Quick start

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (with Compose v2)

### Core stack (dashboard + API + MLflow)

```bash
git clone https://github.com/AsmaAfzi/Saint-Ops.git
cd Saint-Ops
docker compose up --build
```

| Service | URL | Notes |
|---------|-----|-------|
| Live drift dashboard | http://localhost:3200 | React UI — **Drift monitor** + **MLOps & system** tabs |
| API + Swagger | http://localhost:8000/docs | FastAPI OpenAPI |
| Health / readiness | http://localhost:8000/health · /ready | Used by Docker healthcheck |
| MLflow registry | http://localhost:5000 | Model versions and artifacts |

First backend startup may take **15–30 seconds** while TensorFlow loads the model and builds the stream cache.

### Full demo (monitoring + Airflow)

**Windows (recommended):**

```powershell
.\scripts\demo_stack.ps1
```

**Manual:**

```bash
docker compose --profile monitoring --profile airflow up -d --build
```

| Service | URL | Credentials / notes |
|---------|-----|---------------------|
| Grafana | http://localhost:3301 | Anonymous admin (demo) |
| Prometheus | http://localhost:9090 | Scrapes `/metrics` on backend |
| Alertmanager | http://localhost:9093 | Routes alert webhooks |
| Airflow UI | http://localhost:8081 | Auto-login demo proxy — **not** port 8080 |

Airflow first start can take **2–5 minutes**. If the UI fails, run `.\scripts\start_airflow.ps1` to reset the Airflow DB volume.

**Demo narration:** see [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md) for a full word-for-word viva script and port reference.

**Optional profiles:**

| Profile | Command | Purpose |
|---------|---------|---------|
| `train` | `docker compose --profile train run --rm train python ml/train_model.py` | Train in Linux container |
| `monitoring` | `--profile monitoring` | Prometheus + Grafana + Alertmanager |
| `airflow` | `--profile airflow` | Retraining DAGs |
| `s3` | `--profile s3` | MinIO for S3-compatible MLflow artifacts |

---

## Model lifecycle (MLflow)

1. **Train** → version logged to MLflow, auto-staged to **Staging**:
   ```bash
   docker compose --profile train run --rm train python ml/train_model.py
   docker compose --profile train run --rm train python ml/test_model.py
   ```
2. **Compare** staging vs production: `GET /models/compare` or the **MLOps** tab in the UI.
3. **Approve** (when `MLFLOW_REQUIRE_APPROVAL=1`): `POST /models/approve`
4. **Promote** to Production: `POST /models/promote` then `POST /models/reload`
5. **Rollback**: `POST /models/rollback` then `POST /models/reload`

**Shadow / canary:** set `MLFLOW_SHADOW_TRAFFIC_PCT` or use `?variant=shadow` on `/drift_data`. Compare at timestep: `GET /models/shadow/compare?t=50`.

**S3 artifacts:** copy `.env.example` → `.env`, set `MLFLOW_ARTIFACT_ROOT=s3://...`, optionally `docker compose --profile s3 up -d minio`.

---

## Auto-retraining (Airflow)

DAGs in `airflow/dags/`:

| DAG | Purpose |
|-----|---------|
| `saint_retrain` | Train → evaluate → F1 gate → register to MLflow Staging |
| `saint_drift_sensor` | Polls `GET /ops/drift-summary`; can trigger retrain |

Trigger from API:

```bash
curl -X POST http://localhost:8000/retrain/trigger \
  -H "Content-Type: application/json" \
  -d "{\"reason\":\"demo\"}"
```

Status: `GET /retrain/status`

---

## Kubernetes (Helm)

```bash
helm upgrade --install saint-dev ./helm/saint-ops \
  -f helm/saint-ops/values-dev.yaml \
  -n saint-dev --create-namespace

./scripts/k8s_seed_data.sh saint-dev

kubectl port-forward -n saint-dev svc/saint-dev-saint-ops-frontend 3000:80
kubectl port-forward -n saint-dev svc/saint-dev-saint-ops-backend 8000:8000
```

| Environment | Namespace | Values file |
|-------------|-----------|-------------|
| Dev | `saint-dev` | `helm/saint-ops/values-dev.yaml` |
| Staging | `saint-staging` | `helm/saint-ops/values-staging.yaml` |
| Production | `saint-production` | `helm/saint-ops/values-production.yaml` |

Details: [helm/saint-ops/README.md](helm/saint-ops/README.md)

---

## Terraform (local K8s, $0 cloud)

```bash
cd terraform && terraform init && terraform apply
```

Provisions namespace + Helm release on **Docker Desktop Kubernetes**. See [terraform/README.md](terraform/README.md).

> Stop Docker Compose before applying Terraform to avoid port conflicts.

---

## CI/CD

Every push to `main` / `develop` runs pytest, flake8, Docker image builds (main only), Trivy CVE scan, and a Locust smoke test.

Public images (main branch):

- `asmaafzi/saint-backend:latest`
- `asmaafzi/saint-frontend:latest`

---

## API endpoints (summary)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness |
| GET | `/ready` | Readiness (artifacts, stream rows, shadow status) |
| GET | `/metrics` | Prometheus metrics |
| GET | `/drift_data?t=` | Drift result at timestep (`variant`, shadow traffic header) |
| GET | `/stream/info` | Stream length, features, window size |
| GET | `/models/info` | MLflow version history |
| GET | `/models/compare` | Staging vs production metrics + promotion gate |
| GET | `/models/shadow/compare?t=` | Production vs staging at timestep |
| POST | `/models/approve` · `/promote` · `/rollback` · `/reload` | Registry lifecycle |
| GET | `/ops/drift-summary` | Rolling drift stats (Airflow sensor) |
| GET | `/ops/alerts` | Recent alert events |
| POST | `/ops/alerts/webhook` | Alertmanager webhook receiver |
| GET | `/retrain/status` | Retrain pipeline state |
| POST | `/retrain/trigger` | Enqueue retrain (writes trigger file for Airflow) |
| POST | `/retrain/complete` | Mark retrain finished (called by DAG) |

Interactive docs: http://localhost:8000/docs

---

## ML model (mini project baseline)

Evaluation uses the **SAINT headline protocol** (binary drift detection + sensor/environmental pathway F1 on injected scenarios). Coarse 3-class cause-attribution is reported separately in `ml/results/evaluation_report.txt`.

| Metric | Global | Environmental | Sensor |
|--------|--------|---------------|--------|
| F1-Score | **~0.84** | ~0.79 | ~0.84 |
| False alarm rate | 15–18% | — | — |

> **Note:** Headline F1 (~0.84) reflects drift-presence detection and sensor-pathway performance (Scenario B ≈ 0.82 F1). FAR (15–18%) is measured on validation-baseline windows without injected drift. See `ml/test_model.py` → HEADLINE METRICS block after `docker compose --profile train run --rm train python ml/test_model.py`.

### Monitored features (5-channel LSTM)

| Feature | Role |
|---------|------|
| AVG_DOWNHOLE_PRESSURE | Environmental |
| AVG_DOWNHOLE_TEMPERATURE | Environmental |
| BORE_OIL_VOL | Environmental |
| AVG_WHP_P | Environmental |
| AVG_ANNULUS_PRESS | Sensor (annulus pressure) |

Default replay stream: `data/scenario_C_annulus.csv` (annulus sensor drift scenario).

---

## Implementation phases

| Phase | Status | Description |
|-------|--------|-------------|
| 1 — Containerization | Done | Docker Compose: backend, frontend, MLflow |
| 2 — CI/CD | Done | GitHub Actions: test, lint, build, Trivy, Locust |
| 3 — MLflow registry | Done | Staging/Production, shadow serving, promote/rollback API |
| 4 — Kubernetes | Done | Helm chart, HPA, PVCs, CronJobs, NetworkPolicy |
| 5 — Monitoring | Done | Prometheus + Grafana + Alertmanager (`monitoring` profile) |
| 6 — Auto-retraining | Done | Airflow DAGs, `/retrain/*`, drift sensor |
| 7 — Terraform IaC | Done | Docker Desktop K8s + Helm (zero cloud cost) |
| 8 — Hardening | Done | Trivy CI, load tests, K8s RBAC |

---

## Dataset

**Equinor Volve Field** — open production data from the Norwegian Continental Shelf (2008–2016). Processed scenarios live in `data/` (`scenario_A_env_full.csv`, `scenario_B_env_single.csv`, `scenario_C_annulus.csv`, etc.).

---

## References

- Belay et al. (2023). Anomaly Detection in IoT Sensor Data Using LSTM-NDT. *Sensors*, 23(5), 2844.
- Equinor ASA. (2018). Volve Field Open Dataset.
- Siemens Senseye. (2024). The True Cost of Downtime 2024.
