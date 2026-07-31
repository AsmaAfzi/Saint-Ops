# SAINT-OPS Frontend

React operations dashboard for live drift monitoring and MLOps controls. Built with **Vite**, **React 19**, **Plotly**, and **Axios**.

---

## Role in the stack

The frontend is a **static SPA** served by Nginx in Docker. It does not run ML inference — it visualizes data from the FastAPI backend.

```
Browser  →  Nginx (:3200)  →  /api/* proxied to backend:8000
          →  /* static React bundle
```

In development, Vite proxies `/api` to `http://localhost:8000` (see `vite.config.js`).

---

## UI structure

### Tab: Drift monitor

Simulates a live monitoring session by polling the backend timestep-by-step.

| Component | Purpose |
|-----------|---------|
| `Header` | Branding, system ready badge, stream/model metadata |
| `MetricCards` | Timestep progress, drift event count, sensor/env alert state |
| `ControlPanel` | Pause/resume, playback speed, inference variant, shadow traffic %, reset, CSV export |
| `DriftChart` ×2 | Sensor vs environmental reconstruction error (Plotly) |
| `EventLog` | Table of detected drift events (click row to inspect) |
| `ExplainabilityPanel` | Feature-level explanations + contribution bar chart |
| `NotificationStack` | Toast alerts on drift detection |

**Polling:** `GET /api/drift_data?t=N` every ~500ms (scaled by speed). Sequential requests — no overlap.

**Inference variant** (passed as query param / header):

- `auto` — production model; optional `X-Shadow-Traffic-Pct` header
- `production` — force Production registry model
- `shadow` — force Staging (shadow) model

### Tab: MLOps & system

Loaded lazily when the tab is opened (keeps initial load fast).

| Component | API endpoints |
|-----------|---------------|
| `SystemPanel` | `/health`, `/ready`, `/stream/info` |
| `ModelRegistryPanel` | `/models/info`, `/models/compare`, `POST /models/{approve,promote,rollback,reload}` |
| `ShadowComparePanel` | `/models/shadow/compare?t=` at current or selected timestep |

---

## Source layout

```
frontend/src/
├── api/client.js           # Axios instance (base URL /api)
├── utils/features.js       # Env vs sensor feature grouping from API
├── App.jsx                 # State, polling, tab routing
├── App.css / index.css     # Dark ops-center theme
├── main.jsx
└── components/
    ├── Header.jsx
    ├── TabNav.jsx
    ├── MetricCards.jsx
    ├── ControlPanel.jsx
    ├── DriftChart.jsx
    ├── EventLog.jsx
    ├── ExplainabilityPanel.jsx
    ├── SystemPanel.jsx
    ├── ModelRegistryPanel.jsx
    ├── ShadowComparePanel.jsx
    └── NotificationStack.jsx
```

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `VITE_API_URL` | `/api` | API base path (set only if not using Nginx proxy) |

---

## Local development

```bash
# Terminal 1 — backend on :8000
cd backend && pip install -r requirements.txt
# from repo root, with data/models/artifacts present:
uvicorn backend.backend:app --reload --app-dir backend

# Terminal 2 — frontend
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 (Vite dev server proxies `/api` → `:8000`).

---

## Docker build

```bash
docker compose build frontend
# or full stack:
docker compose up --build
```

Production image: multi-stage **Node build** → **Nginx alpine** (`frontend/Dockerfile`).  
Nginx config: `nginx.conf` (SPA fallback + `/api/` proxy).

---

## Export

**Export CSV** on the Drift monitor tab downloads `saint_ops_drift_logs.csv` with datetime, drift type, fine label, and top contributing features for the current session.
