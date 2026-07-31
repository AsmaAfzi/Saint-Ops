# SAINT-OPS — Full demo script (word-for-word)

**Presenter:** Asma Mohammad Afzal  
**Duration:** ~12–15 minutes (+ optional Airflow / monitoring)  
**Stack start:** `.\scripts\demo_stack.ps1` (from repo root)

---

## Port reference (Docker Compose — use these URLs)

| Service | URL | Notes |
|---------|-----|-------|
| **Dashboard** | http://localhost:3200 | React UI (Nginx) |
| **API / Swagger** | http://localhost:8000/docs | FastAPI |
| **MLflow** | http://localhost:5000 | Model registry |
| **Grafana** | http://localhost:3301 | Anonymous admin (demo) |
| **Prometheus** | http://localhost:9090 | Metrics scrape UI |
| **Alertmanager** | http://localhost:9093 | Alert routing |
| **Airflow** | http://localhost:8081 | Auto-login demo proxy (**not** 8080) |

---

## Before you walk in (15 minutes earlier — not spoken)

```powershell
cd "C:\Users\asmaa\Starred\Mini Proj"
.\scripts\demo_stack.ps1 -SkipBuild
```

Wait until all containers are healthy (~2–5 min for Airflow).

Optional — align MLflow Production metrics with headline evaluation:

```powershell
docker compose --profile train run --rm train python ml/sync_demo_metrics.py
```

If you rebuilt the frontend locally:

```powershell
docker cp ".\frontend\dist\." saint-frontend:/usr/share/nginx/html/
```

**Sanity checks (browser):**

- http://localhost:3200 — dashboard loads  
- http://localhost:8000/ready — `"status": "ready"`  
- http://localhost:8081 — Airflow DAG list (admin auto-login)

**Have open in background tabs:** Grafana (3301), MLflow (5000), Airflow (8081).

---

## 0. Opening (~30 seconds)

**[Stand at screen. Dashboard on http://localhost:3200]**

**SAY:**

> “Good morning/afternoon. I'm Asma Mohammad Afzal, and this is **SAINT-OPS** — the MLOps extension of my SAINT predictive maintenance project for oil and gas sensor data.
>
> SAINT uses an **LSTM autoencoder** plus a **CUSUM-based classifier** to detect sensor and environmental drift on multivariate well data. SAINT-OPS wraps that model in a **production-style stack**: containerised API, live operations dashboard, MLflow registry, monitoring, and Airflow-based retraining.
>
> I'll show live drift detection first, then the MLOps control plane.”

---

## 1. Drift monitor tab (~5 minutes)

**[Tab: Drift monitor — should already be selected]**

**SAY:**

> “This is the **operations dashboard**. It streams pre-recorded injection scenarios through the same inference path the API would use in production.
>
> At the top, the metric cards show **timestep progress**, how many **drift events** we've logged this session, and the current **sensor** and **environment** alert state.”

**[Let playback run, or click Resume stream if paused. Increase speed to 1.4× or 2.0×]**

**SAY:**

> “The backend polls reconstruction error per feature, applies **validation p99 normalisation**, and runs **CUSUM** to decide drift. The charts show the **sensor channel** in red and the **environment channel** in amber — you can switch the metric between raw error, normalised error, and CUSUM.”

**[When a drift toast appears, or scrub until one does — click an event in the Event log]**

**SAY:**

> “When drift is detected, we get a **notification**, an entry in the **event log**, and **explainability** below — the top contributing features for that timestep, filtered to sensor vs environment groups so an engineer knows what to investigate.
>
> This is **decision support**, not blind automation — fine labels like ‘Probable Sensor Fault’ are intentionally cautious.”

**[Optional: Chart metric → CUSUM S+]**

**SAY:**

> “CUSUM accumulates sustained deviation — that's what gives us stable detection on gradual drift, not just single spikes.”

**[Optional: Export CSV]**

**SAY:**

> “Operators can export the session log as CSV for audit.”

---

## 2. MLOps & system tab (~4 minutes)

**[Click tab: MLOps & system]**

### 2a. System status

**SAY:**

> “The **MLOps tab** exposes health, readiness, and deployment metadata.
>
> **Health** and **readiness** confirm the API and artifacts loaded. **Stream source** shows which evaluation CSV we're playing — by default **scenario C annulus** drift.
>
> **Production URI** points at MLflow **Production**; **shadow URI** is **Staging**. **Shadow traffic** is **zero percent** in this Compose profile — that's intentional for a stable demo; production traffic always hits the Production model. Shadow comparison still works on demand.”

**[Point at Shadow ready — if No]**

**SAY:**

> “Shadow ready may show **No** until first access — the staging model loads lazily. That's normal.”

### 2b. Model registry

**[Scroll to Model registry panel. Click Refresh if needed]**

**SAY:**

> “The **model registry** integrates **MLflow**. Production is **version one**. The metrics here use our **SAINT headline protocol** — not coarse three-class accuracy.
>
> **Headline F1** is about **point-eight** — around **eighty percent** — from binary drift detection and sensor-pathway performance. **Baseline false alarm rate** is about **eleven percent** on validation baseline windows — within our reported fifteen to eighteen percent band.
>
> The promotion gate may show **blocked** because there's no staging candidate with better metrics — that's **safe MLOps**: we don't auto-promote a worse model.”

**[Point at Production compare card if visible]**

**SAY:**

> “We log training and evaluation to MLflow, stage new runs, and gate promotion on F1 delta and false-alarm rate.”

### 2c. Shadow inference comparison

**[Scroll to Shadow inference comparison — switch back to Drift monitor briefly, let t advance, return to MLOps tab; or use inference variant Shadow]**

**SAY:**

> “**Shadow comparison** runs Production and Staging side by side at the current timestep — coarse and fine agreement badges show whether they match. This is how we'd validate a candidate model before promotion without affecting live traffic.”

---

## 3. MLflow UI (~1–2 minutes)

**[Open tab: http://localhost:5000]**

**SAY:**

> “In **MLflow**, model **SAINT** shows version history. The Production run logs training loss, evaluation artifacts, and headline metrics.
>
> Artifacts include **evaluation_report.txt** and **predictions_all.csv** for reproducibility.”

**[Click SAINT → Production version → run → metrics/artifacts]**

**SAY:**

> “The coarse three-class breakdown is supplementary cause attribution; our primary KPIs are headline F1 and baseline FAR.”

**[Return to dashboard tab]**

---

## 4. Monitoring (~2 minutes) — if monitoring profile is up

**[Open tab: http://localhost:3301 — SAINT Overview dashboard]**

**SAY:**

> “**Grafana** dashboards are fed by **Prometheus** scraping the FastAPI **/metrics** endpoint.
>
> We track **request rate**, **latency**, **drift detections**, **reconstruction error per channel**, **consecutive environmental drift windows**, and **retrain job counters** — custom metrics for ops, not just infrastructure.”

**[Point at drift / env drift panels]**

**SAY:**

> “This connects live inference behaviour to SRE-style monitoring — the same signals that can drive retraining policy.”

**[Optional: http://localhost:9090 — Prometheus targets]**

**SAY:**

> “Prometheus confirms the backend scrape target is **UP**.”

---

## 5. Airflow retraining (~2–3 minutes)

**[Open tab: http://localhost:8081]**

**SAY:**

> “**Airflow** orchestrates automated retraining. We have two DAGs:
>
> **`saint_drift_sensor`** — scheduled **every fifteen minutes** — polls **`GET /ops/drift-summary`** on the backend. If **sustained environmental drift** crosses a threshold, it triggers **`saint_retrain`**.
>
> **`saint_retrain`** runs validate → train → evaluate → **F1 quality gate** → MLflow staging, and notifies the API on completion.
>
> In a live demo we won't wait fifteen minutes, so I'll **trigger the sensor DAG manually** — same code path as the scheduler.”

**[DAGs → saint_drift_sensor → Trigger DAG (play icon)]**

**[Open the run → task check_drift_and_trigger → Log]**

**SAY:**

> “The task called the backend drift summary. Right now **retrain_candidate** may be **false** because we need **fifty consecutive** environmental drift windows — that's expected on a short demo. The task still succeeds: no false retrain.”

**[Open http://localhost:8000/ops/drift-summary in browser — optional]**

**SAY:**

> “Here you can see **consecutive_env_drift_windows** versus **retrain_threshold_windows**.”

**[Back to Airflow — trigger saint_retrain manually OR narrate]**

**SAY:**

> “When the threshold **is** met, **`saint_drift_sensor`** automatically fires **`saint_retrain`**. To show the downstream pipeline without waiting, I'll trigger **`saint_retrain`** directly — this is the train-and-evaluate path that runs in production after a drift-based alert.”

**[DAGs → saint_retrain → Trigger DAG]**

**SAY:**

> “The DAG validates data, runs **train_model** and **test_model** with a reduced epoch cap for demo, applies the **F1 gate** against our headline metrics, and posts back to **`/retrain/complete`**.”

**[Optional: http://localhost:8000/retrain/status]**

**SAY:**

> “The API exposes **retrain status** for the UI and for automation.”

---

## 6. API / architecture closing (~1 minute)

**[Optional: http://localhost:8000/docs]**

**SAY:**

> “The **FastAPI** layer exposes **drift_data** for the UI, **registry** endpoints for promote and rollback, **ops** endpoints for drift summary and alerts, and **retrain** hooks for Airflow.
>
> Everything runs in **Docker Compose** locally; the same charts deploy to **Kubernetes via Helm**, with **Terraform** provisioning the cluster — infrastructure as code end to end.”

---

## 7. Closing (~30 seconds)

**SAY:**

> “To summarise: SAINT-OPS delivers **live drift detection with explainability**, **MLflow model lifecycle management**, **Prometheus/Grafana observability**, and **Airflow-gated retraining** — a complete MLOps loop around the SAINT LSTM autoencoder.
>
> Thank you — I'm happy to take questions.”

---

## Quick answers (if asked)

| Question | Answer |
|----------|--------|
| Why is shadow traffic 0%? | Set in `docker-compose.yml` (`MLFLOW_SHADOW_TRAFFIC_PCT: "0"`) for stable demo traffic; K8s dev values use 10%. |
| Why ~0.80 F1 not 0.32? | 0.32 is coarse 3-class cause attribution; headline protocol uses binary + sensor-pathway F1 (~0.80). |
| Why FAR ~11% not 73%? | 73% was injected-window coarse FAR; baseline FAR on normal windows is ~10.7%. |
| Why promotion blocked? | No staging version beats production on headline F1 / FAR gate. |
| CI / security? | GitHub Actions runs tests, builds images, Locust load test, Trivy scans. |

---

## Emergency fallbacks

| Problem | Fix |
|---------|-----|
| Dashboard blank | `docker compose ps` — check `saint-frontend` / `saint-backend` |
| Airflow “Ooops!” | `.\scripts\start_airflow.ps1` |
| Wrong MLflow metrics | `docker compose --profile train run --rm train python ml/sync_demo_metrics.py` |
| Frontend labels old | `cd frontend; npm run build` then `docker cp .\dist\. saint-frontend:/usr/share/nginx/html/` |
| Docker build EOF | Build frontend on host; `docker cp` dist — don't rebuild in Docker |

---

## One-line cheat sheet (pin on screen)

```
3200 Dashboard | 8000 API | 5000 MLflow | 3301 Grafana | 9090 Prometheus | 8081 Airflow
```
