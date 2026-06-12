export default function ModelRegistryPanel({
  modelsInfo,
  modelsCompare,
  loading,
  actionLoading,
  onRefresh,
  onApprove,
  onPromote,
  onRollback,
  onReload,
  promoteForce,
  onPromoteForceChange,
}) {
  const gate = modelsCompare?.promotion_gate;
  const versions = modelsInfo?.versions ?? [];

  const formatMetric = (v) => (v != null && !Number.isNaN(v) ? Number(v).toFixed(4) : "—");

  return (
    <section className="panel registry-panel">
      <div className="panel__header">
        <div>
          <h2>Model registry</h2>
          <p>MLflow lifecycle — approve, promote, rollback, and hot-reload</p>
        </div>
        <button type="button" className="btn btn--ghost btn--sm" onClick={onRefresh} disabled={loading}>
          Refresh
        </button>
      </div>

      {modelsInfo?.error && <p className="panel-error">{modelsInfo.error}</p>}

      {!modelsInfo?.error && (
        <>
          <div className="registry-meta">
            <span className="meta-chip">Model: {modelsInfo?.registered_model ?? "SAINT"}</span>
            <span className="meta-chip mono">Prod: {modelsInfo?.production_uri ?? "—"}</span>
            <span className="meta-chip mono">Shadow: {modelsInfo?.shadow_uri ?? "—"}</span>
          </div>

          <div className="registry-actions">
            <button type="button" className="btn btn--secondary btn--sm" onClick={onApprove} disabled={actionLoading}>
              Approve staging
            </button>
            <label className="force-check">
              <input
                type="checkbox"
                checked={promoteForce}
                onChange={(e) => onPromoteForceChange(e.target.checked)}
              />
              Force promote
            </label>
            <button type="button" className="btn btn--primary btn--sm" onClick={onPromote} disabled={actionLoading}>
              Promote to production
            </button>
            <button type="button" className="btn btn--ghost btn--sm" onClick={onRollback} disabled={actionLoading}>
              Rollback
            </button>
            <button type="button" className="btn btn--ghost btn--sm" onClick={onReload} disabled={actionLoading}>
              Hot reload
            </button>
          </div>

          {gate && (
            <div className={`gate-banner gate-banner--${gate.passed ? "pass" : "fail"}`}>
              <strong>Promotion gate: {gate.passed ? "Passed" : "Blocked"}</strong>
              <ul>
                {gate.reasons?.map((r, i) => (
                  <li key={i}>{r}</li>
                ))}
              </ul>
            </div>
          )}

          {modelsCompare && !modelsCompare.error && (
            <div className="compare-grid">
              <CompareCard
                title="Staging"
                version={modelsCompare.staging}
                metrics={modelsCompare.staging_metrics}
                formatMetric={formatMetric}
              />
              <CompareCard
                title="Production"
                version={modelsCompare.production}
                metrics={modelsCompare.production_metrics}
                formatMetric={formatMetric}
              />
            </div>
          )}

          <div className="table-wrap table-wrap--short">
            <table>
              <thead>
                <tr>
                  <th>Version</th>
                  <th>Stage</th>
                  <th>Status</th>
                  <th>Macro F1</th>
                  <th>Accuracy</th>
                  <th>Run ID</th>
                </tr>
              </thead>
              <tbody>
                {versions.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="empty-cell">
                      No registered versions found
                    </td>
                  </tr>
                ) : (
                  versions.map((v) => (
                    <tr key={v.version}>
                      <td>v{v.version}</td>
                      <td>
                        <span className={`stage-badge stage-badge--${(v.stage || "none").toLowerCase()}`}>
                          {v.stage || "None"}
                        </span>
                      </td>
                      <td>{v.status}</td>
                      <td className="mono">{formatMetric(v.metrics?.macro_f1_overall)}</td>
                      <td className="mono">{formatMetric(v.metrics?.overall_coarse_accuracy)}</td>
                      <td className="mono truncate">{v.run_id?.slice(0, 12)}…</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </>
      )}
    </section>
  );
}

function CompareCard({ title, version, metrics, formatMetric }) {
  const keys = ["macro_f1_overall", "overall_coarse_accuracy", "false_alarm_rate", "detection_rate"];

  return (
    <article className="compare-card">
      <h3>{title}</h3>
      {version ? (
        <p className="compare-version">
          v{version.version} · {version.stage}
        </p>
      ) : (
        <p className="compare-version muted">No version</p>
      )}
      <dl className="metric-dl">
        {keys.map((k) => (
          <div key={k}>
            <dt>{k.replace(/_/g, " ")}</dt>
            <dd className="mono">{formatMetric(metrics?.[k])}</dd>
          </div>
        ))}
      </dl>
    </article>
  );
}
