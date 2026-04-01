# SAINT-OPS 🔍
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

## 📌 What is SAINT-OPS?

SAINT (Semantic AI for Predictive Maintenance) was built as a mini project that achieved **84% F1-score** in detecting sensor drift in oil & gas operations using an LSTM Autoencoder trained on the [Equinor Volve dataset](https://www.equinor.com/energy/volve-data-sharing).

**SAINT-OPS** takes that proven ML core and wraps it in full production DevOps infrastructure — making it containerized, automatically tested, monitored, and self-retraining.

---

## 🆚 Mini Project vs SAINT-OPS

| Feature | Mini Project | SAINT-OPS |
|---------|-------------|-----------|
| Deployment | Manual only | Docker + Kubernetes |
| CI/CD | None | GitHub Actions ✅ |
| Monitoring | None | Prometheus + Grafana |
| Model Versioning | None | MLflow Registry |
| Auto-retraining | None | Drift-triggered Airflow |
| Infrastructure | Manual | Terraform IaC |
| Testing | None | pytest + locust |
| Security | None | RBAC + Vault + CVE scanning |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      SAINT-OPS                          │
├───────────────┬─────────────────┬───────────────────────┤
│  Data Plane   │    ML Plane     │     DevOps Plane      │
│               │                 │                       │
│  Volve xlsx   │  LSTM AutoEnc   │  Docker + CI/CD       │
│  Redis Stream │  FastAPI        │  Kubernetes + Helm    │
│               │  MLflow         │  Terraform IaC        │
├───────────────┴─────────────────┴───────────────────────┤
│                  Observability Plane                     │
│            Prometheus + Grafana + ELK Stack              │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
saint-ops/
├── .github/
│   └── workflows/
│       └── ci.yml              # CI/CD pipeline (test → lint → build → push)
├── artifacts/                  # Drift thresholds, feature errors (.npy files)
├── backend/
│   ├── backend.py              # FastAPI server with /drift_data, /health, /ready
│   ├── Dockerfile              # Multi-stage production image
│   ├── requirements.txt        # Minimal, trimmed dependencies
│   └── tests/
│       └── test_api.py         # pytest test suite
├── data/
│   └── synthetic_drift_dataset_v2.xlsx
├── frontend/
│   ├── src/
│   │   ├── App.jsx             # React dashboard with Plotly charts
│   │   └── main.jsx
│   ├── Dockerfile              # Node build → Nginx serve
│   └── nginx.conf              # Reverse proxy config
├── ml/
│   └── Dockerfile              # Training container
├── models/                     # Trained LSTM Autoencoder weights + scalers
├── docker-compose.yml          # Orchestrates backend + frontend + MLflow
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed

### Run the full stack

```bash
git clone https://github.com/AsmaAfzi/Saint-Ops.git
cd Saint-Ops
docker-compose up --build
```

Then open:

| Service | URL |
|---------|-----|
| 🖥️ Live Drift Dashboard | http://localhost:3000 |
| ⚡ API + Swagger Docs | http://localhost:8000/docs |
| 🏥 Health Check | http://localhost:8000/health |
| 📊 MLflow Model Registry | http://localhost:5000 |

---

## 🤖 ML Model Performance (Mini Project Baseline)

| Metric | Global | Environmental Drift | Sensor Drift |
|--------|--------|--------------------|--------------| 
| Precision | 0.82 | 0.78 | 0.85 |
| Recall | 0.87 | 0.80 | 0.83 |
| **F1-Score** | **0.84** | **0.79** | **0.84** |
| False Alarm Rate | 15–18% | — | — |
| Detection Lead Time | 2–4 windows | 2–4 windows | 1–3 windows |

> Conventional threshold-based systems have 60–80% false alarm rates. SAINT reduces this to 15–18%.

---

## ⚙️ CI/CD Pipeline

Every `git push` to `main` or `develop` triggers:

```
Push to GitHub
      │
      ▼
┌─────────────┐     ┌─────────────┐
│  Run Tests  │     │  Lint Code  │
│  (pytest)   │     │  (flake8)   │
└──────┬──────┘     └──────┬──────┘
       │                   │
       └─────────┬─────────┘
                 ▼
      ┌──────────────────────┐
      │  Build Docker Images │
      │  Push to Docker Hub  │  ← only on main branch
      └──────────────────────┘
```

Docker images are publicly available:
- `asmaafzi/saint-backend:latest`
- `asmaafzi/saint-frontend:latest`

---

## 📦 Phases — Implementation Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 — Containerization | ✅ Complete | Docker + Docker Compose for all 3 services |
| Phase 2 — CI/CD | ✅ Complete | GitHub Actions: test → lint → build → push to Docker Hub |
| Phase 3 — MLflow Registry | 🔄 In Progress | Model versioning, experiment tracking |
| Phase 4 — Kubernetes | ⏳ Planned | Helm charts, HPA autoscaling, self-healing |
| Phase 5 — Monitoring | ⏳ Planned | Prometheus + Grafana dashboards |
| Phase 6 — Auto-retraining | ⏳ Planned | Drift-triggered Airflow DAG |
| Phase 7 — Terraform IaC | ⏳ Planned | AWS/Azure deployment (UAE region) |
| Phase 8 — Hardening | ⏳ Planned | CVE scanning, RBAC, load testing |

---

## 🛢️ Dataset

**Equinor Volve Oil Field Dataset** — real production data from the Norwegian Continental Shelf (2008–2016), released open-source in 2018.

Sensor variables used:

| Variable | Type |
|----------|------|
| AVG_DOWNHOLE_PRESSURE | Environmental |
| AVG_DOWNHOLE_TEMPERATURE | Environmental |
| BORE_OIL_VOL | Environmental |
| AVG_WHP_P | Environmental |
| DP_CHOKE_SIZE | Sensor-Specific |

---

## 🔑 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service health check |
| GET | `/ready` | Readiness probe |
| GET | `/drift_data?t={index}` | Drift detection result at time index |

### Sample Response — `/drift_data?t=900`

```json
{
  "time": 900,
  "datetime": "2014-03-15",
  "sensor_error": 0.0842,
  "env_error": 0.0341,
  "predicted_drift": true,
  "drift_type": "Environmental Drift",
  "feature_names": ["AVG_DOWNHOLE_PRESSURE", "AVG_DOWNHOLE_TEMPERATURE", "BORE_OIL_VOL", "AVG_WHP_P", "DP_CHOKE_SIZE"],
  "feature_error": [0.091, 0.034, 0.078, 0.055, 0.012],
  "explanation": [
    {
      "feature_name": "AVG_DOWNHOLE_PRESSURE",
      "error": 0.091,
      "baseline": 0.021,
      "reason": "Deviation increasing from baseline"
    }
  ],
  "done": false
}
```

---

## 🌍 UAE Career Relevance

This project is directly aligned with UAE's Vision 2030 digital transformation priorities:

- **ADNOC / Oil & Gas:** Predictive maintenance for oil & gas sensors is a live use case at ADNOC, Petrofac, and Halliburton ME
- **Cloud:** Terraform + AWS/Azure (targeting `me-south-1` Bahrain / `me-central-1` UAE region)
- **MLOps:** MLOps roles in UAE command 20–30% salary premium over pure DevOps roles

---

## 📚 References

- Belay et al. (2023). Anomaly Detection in IoT Sensor Data Using LSTM-NDT. *Sensors*, 23(5), 2844.
- Equinor ASA. (2018). Volve Field Open Dataset. Norwegian Continental Shelf.
- Siemens Senseye. (2024). The True Cost of Downtime 2024.
- McKinsey & Company. (2016). Digital Solutions in Oil & Gas.
- Kotari et al. (2024). Edge-based federated learning for anomaly detection. *Heliyon*, 10(24).