import Plot from "react-plotly.js";

const CHART_THEME = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "rgba(15, 23, 42, 0.4)",
  font: { family: "Inter, system-ui, sans-serif", color: "#94a3b8", size: 12 },
  xaxis: {
    gridcolor: "rgba(148, 163, 184, 0.12)",
    tickangle: -35,
  },
  yaxis: {
    gridcolor: "rgba(148, 163, 184, 0.12)",
  },
};

export default function ExplainabilityPanel({ selectedLog, filteredExplanation }) {
  if (!selectedLog || !filteredExplanation.length) return null;

  const isSensor = selectedLog.drift_type === "Sensor Drift";
  const chartValues = filteredExplanation.map((f) =>
    f.norm_error != null ? f.norm_error : f.error
  );
  const chartYTitle =
    filteredExplanation.some((f) => f.norm_error != null)
      ? "Norm error (× val_p99)"
      : "Reconstruction error";

  return (
    <section className="explain-grid">
      <article className="panel explain-panel">
        <div className="panel__header">
          <div>
            <h2>{isSensor ? "Sensor drift analysis" : "Environmental drift analysis"}</h2>
            <p className="mono">
              {selectedLog.datetime}
              {selectedLog.drift_type_fine && (
                <span className="fine-label"> · {selectedLog.drift_type_fine}</span>
              )}
              {selectedLog.classifier_version && (
                <span className="fine-label"> · {selectedLog.classifier_version}</span>
              )}
            </p>
          </div>
        </div>
        <ul className="explain-list">
          {filteredExplanation.map((exp, idx) => (
            <li key={idx}>
              <div className="explain-feature">
                <span className="explain-name">{exp.feature_name}</span>
                <span className="explain-error">{exp.error.toFixed(4)}</span>
              </div>
              <p className="explain-detail">
                {exp.baseline != null && (
                  <span>val_p99 ref {exp.baseline.toFixed(4)} · </span>
                )}
                {exp.norm_error != null && (
                  <span>norm {exp.norm_error.toFixed(2)}x · </span>
                )}
                {exp.cusum != null && <span>CUSUM S+ {exp.cusum.toFixed(1)} · </span>}
                {exp.threshold != null && (
                  <span>Threshold {exp.threshold.toFixed(4)} · </span>
                )}
                {exp.reason}
              </p>
            </li>
          ))}
        </ul>
      </article>

      <article className="panel chart-panel">
        <div className="panel__header">
          <div>
            <h2>Feature contribution</h2>
            <p>{chartYTitle} by feature at selected timestep</p>
          </div>
        </div>
        <Plot
          data={[
            {
              x: filteredExplanation.map((f) => f.feature_name),
              y: chartValues,
              type: "bar",
              marker: {
                color: filteredExplanation.map((f) => f.error),
                colorscale: [
                  [0, "#06b6d4"],
                  [0.5, "#3b82f6"],
                  [1, "#f59e0b"],
                ],
                showscale: false,
              },
            },
          ]}
          layout={{
            ...CHART_THEME,
            height: 300,
            margin: { t: 8, b: 80, l: 48, r: 16 },
            yaxis: { ...CHART_THEME.yaxis, title: { text: chartYTitle } },
            autosize: true,
          }}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: "100%" }}
          useResizeHandler
        />
      </article>
    </section>
  );
}
