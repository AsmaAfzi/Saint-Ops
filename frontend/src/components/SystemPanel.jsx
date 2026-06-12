import { formatDetectorConfig } from "../utils/features";

export default function SystemPanel({ health, ready, streamInfo }) {
  const classifierVersion =
    ready?.classifier_version ?? streamInfo?.classifier_version ?? health?.classifier ?? "—";
  const detectorConfig =
    ready?.detector_config ?? streamInfo?.detector_config ?? null;

  const rows = [
    { label: "Health", value: health?.status ?? "—", ok: health?.status === "ok" },
    { label: "Readiness", value: ready?.status ?? "—", ok: ready?.status === "ready" },
    { label: "Artifacts loaded", value: ready?.artifacts_loaded ? "Yes" : "No", ok: ready?.artifacts_loaded },
    { label: "Drift data", value: ready?.drift_data_available ? "Available" : "Unavailable", ok: ready?.drift_data_available },
    { label: "Stream rows", value: ready?.stream_rows ?? streamInfo?.length ?? "—" },
    { label: "Stream source", value: ready?.stream_source ?? streamInfo?.source ?? "—" },
    { label: "Window size", value: streamInfo?.window_size ?? "—" },
    { label: "Window filter", value: streamInfo?.window_filter ?? "—" },
    { label: "Model source", value: ready?.model_source ?? "—" },
    { label: "Production URI", value: ready?.mlflow_model_uri ?? "—", mono: true },
    { label: "Shadow URI", value: ready?.shadow_model_uri ?? "—", mono: true },
    {
      label: "Shadow ready",
      value: ready?.shadow_ready ? "Yes" : "No",
      ok: ready?.shadow_ready,
      warn: ready?.shadow_ready === false,
    },
    { label: "Shadow traffic %", value: ready?.shadow_traffic_pct != null ? `${ready.shadow_traffic_pct}%` : "—" },
    { label: "Shadow error", value: ready?.shadow_error ?? "—", warn: !!ready?.shadow_error },
    { label: "Backend version", value: health?.version ?? "—" },
    { label: "Model type", value: health?.model ?? "—" },
    { label: "Classifier", value: classifierVersion, mono: true },
    { label: "Detector (v8)", value: formatDetectorConfig(detectorConfig), mono: true },
  ];

  return (
    <section className="panel system-panel">
      <div className="panel__header">
        <div>
          <h2>System status</h2>
          <p>Health, readiness, stream metadata, and shadow deployment state</p>
        </div>
      </div>

      <dl className="kv-grid">
        {rows.map((row) => (
          <div key={row.label} className="kv-item">
            <dt>{row.label}</dt>
            <dd
              className={[
                row.mono ? "mono" : "",
                row.ok ? "kv-ok" : "",
                row.warn ? "kv-warn" : "",
              ]
                .filter(Boolean)
                .join(" ")}
            >
              {row.value}
            </dd>
          </div>
        ))}
      </dl>

      {streamInfo?.features?.length > 0 && (
        <div className="feature-tags">
          <span className="feature-tags__label">Monitored features</span>
          <div className="feature-tags__list">
            {streamInfo.features.map((f) => (
              <span key={f} className="feature-tag">
                {f}
              </span>
            ))}
          </div>
        </div>
      )}

      {ready?.error && (
        <p className="panel-error">Readiness error: {ready.error}</p>
      )}
    </section>
  );
}
