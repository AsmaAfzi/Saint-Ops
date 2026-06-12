export default function ShadowComparePanel({ data, timestep, loading, featureNames = [] }) {
  if (loading) {
    return (
      <section className="panel shadow-panel">
        <div className="panel__header">
          <div>
            <h2>Shadow inference comparison</h2>
            <p>Loading production vs staging at timestep {timestep}…</p>
          </div>
        </div>
      </section>
    );
  }

  if (!data || data.error) {
    return (
      <section className="panel shadow-panel">
        <div className="panel__header">
          <div>
            <h2>Shadow inference comparison</h2>
            <p>Side-by-side predictions at the current playback position</p>
          </div>
        </div>
        <p className="panel-muted">{data?.error ?? "No comparison data yet."}</p>
      </section>
    );
  }

  if (data.done) {
    return (
      <section className="panel shadow-panel">
        <div className="panel__header">
          <div>
            <h2>Shadow inference comparison</h2>
            <p>Stream ended — no further comparisons</p>
          </div>
        </div>
      </section>
    );
  }

  const { production, shadow, agreement, agreement_fine } = data;

  return (
    <section className="panel shadow-panel">
      <div className="panel__header">
        <div>
          <h2>Shadow inference comparison</h2>
          <p>Production vs staging at timestep {timestep}</p>
        </div>
        <div className="shadow-badges">
          <span className={`agreement-badge agreement-badge--${agreement ? "yes" : "no"}`}>
            Coarse {agreement ? "agree" : "disagree"}
          </span>
          <span
            className={`agreement-badge agreement-badge--${agreement_fine ? "yes" : "no"}`}
          >
            Fine {agreement_fine ? "agree" : "disagree"}
          </span>
        </div>
      </div>

      <div className="shadow-grid">
        <VariantCard label="Production" data={production} tone="prod" featureNames={featureNames} />
        <VariantCard label="Staging (shadow)" data={shadow} tone="shadow" featureNames={featureNames} />
      </div>
    </section>
  );
}

function VariantCard({ label, data, tone, featureNames }) {
  if (!data) {
    return (
      <article className={`variant-card variant-card--${tone}`}>
        <h3>{label}</h3>
        <p className="panel-muted">Unavailable</p>
      </article>
    );
  }

  const drift = data.predicted_drift;

  return (
    <article className={`variant-card variant-card--${tone}`}>
      <div className="variant-card__head">
        <h3>{label}</h3>
        <span className={`tag ${drift ? "tag--sensor" : "tag--stable"}`}>
          {drift ? data.drift_type : "No drift"}
        </span>
      </div>
      {drift && data.drift_type_fine && (
        <p className="mono fine-label variant-fine">{data.drift_type_fine}</p>
      )}
      <p className="mono variant-uri">{data.uri ?? "—"}</p>
      {data.classifier_version && (
        <p className="mono variant-meta">Classifier: {data.classifier_version}</p>
      )}
      {data.feature_error?.length > 0 && (
        <dl className="error-dl">
          {data.feature_error.map((err, i) => (
            <div key={i}>
              <dt>{featureNames[i] ?? `Feature ${i + 1}`}</dt>
              <dd className="mono">
                err {Number(err).toFixed(4)}
                {data.feature_norm_error?.[i] != null && (
                  <> · norm {Number(data.feature_norm_error[i]).toFixed(2)}x</>
                )}
                {data.feature_cusum?.[i] != null && (
                  <> · S+ {Number(data.feature_cusum[i]).toFixed(1)}</>
                )}
              </dd>
            </div>
          ))}
        </dl>
      )}
    </article>
  );
}
