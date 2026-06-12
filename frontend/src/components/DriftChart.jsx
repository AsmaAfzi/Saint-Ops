import Plot from "react-plotly.js";

const CHART_THEME = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "rgba(15, 23, 42, 0.4)",
  font: { family: "Inter, system-ui, sans-serif", color: "#94a3b8", size: 12 },
  xaxis: {
    gridcolor: "rgba(148, 163, 184, 0.12)",
    zerolinecolor: "rgba(148, 163, 184, 0.2)",
    linecolor: "rgba(148, 163, 184, 0.2)",
  },
  yaxis: {
    gridcolor: "rgba(148, 163, 184, 0.12)",
    zerolinecolor: "rgba(148, 163, 184, 0.2)",
    linecolor: "rgba(148, 163, 184, 0.2)",
  },
};

export default function DriftChart({
  title,
  subtitle,
  xData,
  yData,
  selectedX,
  selectedY,
  lineColor,
  markerColor,
  statusLabel,
  statusTone,
  yAxisTitle = "Error",
  seriesLabel = "Reconstruction error",
}) {
  return (
    <article className="chart-card">
      <div className="chart-card__header">
        <div>
          <h3>{title}</h3>
          <p>{subtitle}</p>
        </div>
        <span className={`status-badge status-badge--${statusTone}`}>{statusLabel}</span>
      </div>

      <div className="chart-card__plot">
        <Plot
          data={[
            {
              x: xData,
              y: yData,
              type: "scatter",
              mode: "lines",
              name: seriesLabel,
              line: { color: lineColor, width: 2, shape: "spline" },
              fill: "tozeroy",
              fillcolor: `${lineColor}22`,
            },
            {
              x: selectedX != null ? [selectedX] : [],
              y: selectedY != null ? [selectedY] : [],
              type: "scatter",
              mode: "markers",
              name: "Selected event",
              marker: { color: markerColor, size: 10, symbol: "diamond", line: { color: "#fff", width: 1 } },
            },
          ]}
          layout={{
            ...CHART_THEME,
            height: 320,
            margin: { t: 8, b: 40, l: 48, r: 16 },
            showlegend: false,
            xaxis: { ...CHART_THEME.xaxis, title: { text: "Timestep", standoff: 8 } },
            yaxis: { ...CHART_THEME.yaxis, title: { text: yAxisTitle, standoff: 8 } },
            autosize: true,
          }}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: "100%" }}
          useResizeHandler
        />
      </div>
    </article>
  );
}
