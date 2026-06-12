"""
MLflow Model Registry lifecycle: Staging → Production, comparison gates, rollback.

Environment (shared with mlflow_log.py):
  MLFLOW_TRACKING_URI, MLFLOW_REGISTERED_MODEL, MLFLOW_PROMOTION_MIN_F1_DELTA
"""

from __future__ import annotations

import os
from typing import Any

HIGHER_IS_BETTER = ("overall_coarse_accuracy", "macro_f1_overall", "detection_rate")
LOWER_IS_BETTER = ("false_alarm_rate",)


def tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")


def registered_model_name() -> str:
    return os.environ.get("MLFLOW_REGISTERED_MODEL", "SAINT")


def _min_f1_delta() -> float:
    return float(os.environ.get("MLFLOW_PROMOTION_MIN_F1_DELTA", "0.0"))


def _client():
    from mlflow.tracking import MlflowClient

    return MlflowClient(tracking_uri=tracking_uri())


def _version_info(mv: Any) -> dict[str, Any]:
    return {
        "name": mv.name,
        "version": int(mv.version),
        "stage": mv.current_stage,
        "run_id": mv.run_id,
        "status": mv.status,
        "creation_timestamp": mv.creation_timestamp,
    }


def get_latest_version(stage: str | None = None) -> dict[str, Any] | None:
    """Return latest model version for a stage, or latest overall if stage is None."""
    client = _client()
    name = registered_model_name()
    if stage:
        versions = client.get_latest_versions(name, stages=[stage])
        if not versions:
            return None
        return _version_info(versions[0])

    all_v = client.search_model_versions(f"name='{name}'")
    if not all_v:
        return None
    latest = max(all_v, key=lambda v: int(v.version))
    return _version_info(latest)


def get_run_metrics(run_id: str) -> dict[str, float]:
    client = _client()
    run = client.get_run(run_id)
    raw = run.data.metrics
    return {k: float(v) for k, v in raw.items()}


def list_version_history(limit: int = 20) -> list[dict[str, Any]]:
    client = _client()
    name = registered_model_name()
    versions = sorted(
        client.search_model_versions(f"name='{name}'"),
        key=lambda v: int(v.version),
        reverse=True,
    )[:limit]
    out: list[dict[str, Any]] = []
    for mv in versions:
        info = _version_info(mv)
        try:
            info["metrics"] = get_run_metrics(mv.run_id)
        except Exception:
            info["metrics"] = {}
        out.append(info)
    return out


def compare_staging_vs_production() -> dict[str, Any]:
    """Side-by-side metrics: Staging candidate vs current Production."""
    staging = get_latest_version("Staging")
    production = get_latest_version("Production")

    staging_metrics: dict[str, float] = {}
    production_metrics: dict[str, float] = {}

    if staging:
        staging_metrics = get_run_metrics(staging["run_id"])
    if production:
        production_metrics = get_run_metrics(production["run_id"])

    deltas: dict[str, float | None] = {}
    all_keys = set(staging_metrics) | set(production_metrics)
    for key in sorted(all_keys):
        if key in staging_metrics and key in production_metrics:
            deltas[key] = staging_metrics[key] - production_metrics[key]
        else:
            deltas[key] = None

    gate = evaluate_promotion_gate(staging_metrics, production_metrics)

    return {
        "registered_model": registered_model_name(),
        "staging": staging,
        "production": production,
        "staging_metrics": staging_metrics,
        "production_metrics": production_metrics,
        "deltas": deltas,
        "promotion_gate": gate,
    }


def evaluate_promotion_gate(
    candidate: dict[str, float],
    production: dict[str, float] | None,
) -> dict[str, Any]:
    """Return whether Staging may promote to Production based on evaluation metrics."""
    result: dict[str, Any] = {
        "passed": False,
        "reasons": [],
        "required_min_f1_delta": _min_f1_delta(),
    }

    if not candidate:
        result["reasons"].append("No staging metrics on candidate run")
        return result

    if not production:
        result["passed"] = True
        result["reasons"].append("No production model exists — first promotion allowed")
        return result

    cand_score = candidate.get(
        "macro_f1_overall",
        candidate.get("overall_coarse_accuracy", 0.0),
    )
    prod_score = production.get(
        "macro_f1_overall",
        production.get("overall_coarse_accuracy", 0.0),
    )
    delta = cand_score - prod_score
    result["f1_delta"] = delta

    if delta >= _min_f1_delta():
        result["passed"] = True
        result["reasons"].append(
            f"Candidate score {cand_score:.4f} >= production {prod_score:.4f} "
            f"(delta {delta:+.4f}, min required {_min_f1_delta():+.4f})"
        )
    else:
        result["reasons"].append(
            f"Candidate score {cand_score:.4f} below production {prod_score:.4f} "
            f"(delta {delta:+.4f}, min required {_min_f1_delta():+.4f})"
        )

    if "false_alarm_rate" in candidate and "false_alarm_rate" in production:
        fa_delta = candidate["false_alarm_rate"] - production["false_alarm_rate"]
        result["false_alarm_delta"] = fa_delta
        if fa_delta > 0.05:
            result["passed"] = False
            result["reasons"].append(
                f"False alarm rate worsened by {fa_delta * 100:.1f}pp (max +5pp allowed)"
            )
        elif result["passed"]:
            result["reasons"].append(
                f"False alarm rate acceptable (delta {fa_delta * 100:+.1f}pp)"
            )

    return result


def _resolve_version_number(version: int | None) -> str | None:
    client = _client()
    name = registered_model_name()
    if version is not None:
        return str(version)
    versions = client.get_latest_versions(name, stages=["None"])
    if versions:
        return versions[0].version
    all_v = client.search_model_versions(f"name='{name}'")
    if not all_v:
        return None
    return str(max(int(v.version) for v in all_v))


def transition_stage(version: int, stage: str, archive_existing: bool = False) -> dict[str, Any]:
    client = _client()
    name = registered_model_name()
    client.transition_model_version_stage(
        name=name,
        version=str(version),
        stage=stage,
        archive_existing_versions=archive_existing,
    )
    return {"name": name, "version": version, "stage": stage}


def promote_to_staging(version: int | None = None) -> int | None:
    ver = _resolve_version_number(version)
    if ver is None:
        return None
    transition_stage(int(ver), "Staging", archive_existing=False)
    return int(ver)


def promote_staging_to_production(*, force: bool = False) -> dict[str, Any]:
    """
    Promote Staging → Production after comparison gate (unless force=True).
    Requires approved=true tag when MLFLOW_REQUIRE_APPROVAL=1.
    """
    staging = get_latest_version("Staging")
    if not staging:
        return {"ok": False, "error": "No model version in Staging"}

    production = get_latest_version("Production")
    staging_metrics = get_run_metrics(staging["run_id"])
    production_metrics = (
        get_run_metrics(production["run_id"]) if production else {}
    )

    gate = evaluate_promotion_gate(staging_metrics, production_metrics or None)
    if not force and not gate["passed"]:
        return {
            "ok": False,
            "error": "Promotion gate failed",
            "promotion_gate": gate,
            "staging": staging,
            "production": production,
        }

    if os.environ.get("MLFLOW_REQUIRE_APPROVAL", "0").lower() in ("1", "true", "yes"):
        client = _client()
        approval = client.get_model_version(staging["name"], str(staging["version"]))
        tag_map = {t.key: t.value for t in (approval.tags or [])}
        if tag_map.get("approved") != "true" and not force:
            return {
                "ok": False,
                "error": "Approval required: set model version tag approved=true in MLflow UI",
                "staging": staging,
            }

    transition_stage(staging["version"], "Production", archive_existing=True)
    return {
        "ok": True,
        "promoted_version": staging["version"],
        "archived_previous": production,
        "promotion_gate": gate,
        "forced": force,
    }


def approve_staging(version: int | None = None) -> dict[str, Any]:
    """Set approved=true tag on staging version (human approval gate)."""
    staging = get_latest_version("Staging")
    if version is None:
        if not staging:
            return {"ok": False, "error": "No Staging version"}
        version = staging["version"]
    client = _client()
    name = registered_model_name()
    client.set_model_version_tag(name, str(version), "approved", "true")
    client.set_model_version_tag(name, str(version), "approved_by", "api")
    return {"ok": True, "version": version, "approved": True}


def rollback_production() -> dict[str, Any]:
    """
    Revert Production to the highest archived version (previous production).
    """
    client = _client()
    name = registered_model_name()
    all_v = sorted(
        client.search_model_versions(f"name='{name}'"),
        key=lambda v: int(v.version),
        reverse=True,
    )
    production = next((v for v in all_v if v.current_stage == "Production"), None)
    archived = [v for v in all_v if v.current_stage == "Archived"]
    if not archived:
        return {"ok": False, "error": "No archived version to roll back to"}

    target = archived[0]
    if production:
        client.transition_model_version_stage(
            name=name,
            version=production.version,
            stage="Archived",
            archive_existing_versions=False,
        )
    client.transition_model_version_stage(
        name=name,
        version=target.version,
        stage="Production",
        archive_existing_versions=False,
    )
    return {
        "ok": True,
        "rolled_back_to": int(target.version),
        "previous_production": int(production.version) if production else None,
    }
