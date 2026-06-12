"""
SAINT-OPS load test (Phase 8) — run against local backend.

  pip install locust
  locust -f tests/load/locustfile.py --host http://localhost:8000

Headless smoke (CI-friendly):
  locust -f tests/load/locustfile.py --host http://localhost:8000 \
    --headless -u 20 -r 5 -t 30s --only-summary
"""

from __future__ import annotations

from locust import HttpUser, between, task


class SaintInferenceUser(HttpUser):
    wait_time = between(0.1, 0.5)
    t = 10

    @task(5)
    def drift_data(self) -> None:
        self.client.get(f"/drift_data?t={self.t}", name="/drift_data")
        self.t = (self.t + 1) % 500

    @task(2)
    def health(self) -> None:
        self.client.get("/health", name="/health")

    @task(1)
    def ready(self) -> None:
        self.client.get("/ready", name="/ready")

    @task(1)
    def metrics(self) -> None:
        self.client.get("/metrics", name="/metrics")
