export default function EventLog({ logs, selectedTime, onSelect, topK, envFeatures, sensorFeatures }) {
  return (
    <section className="panel event-log">
      <div className="panel__header">
        <div>
          <h2>Drift event log</h2>
          <p>Click a row to inspect feature-level explanations</p>
        </div>
        <span className="panel__count">{logs.length} events</span>
      </div>

      <div className="table-wrap">
        {logs.length === 0 ? (
          <div className="empty-state">
            <p>No drift events detected yet.</p>
            <span>Events appear here when the model flags an anomaly.</span>
          </div>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Datetime</th>
                <th>Type</th>
                <th>Top contributing features</th>
              </tr>
            </thead>
            <tbody>
              {logs.map((log, idx) => {
                const filtered = log.explanation.filter((exp) =>
                  log.drift_type === "Sensor Drift"
                    ? sensorFeatures.includes(exp.feature_name)
                    : envFeatures.includes(exp.feature_name)
                );
                const topFeatures = filtered
                  .sort((a, b) => b.error - a.error)
                  .slice(0, topK)
                  .map((f) => `${f.feature_name}: ${f.error.toFixed(4)}`)
                  .join(" · ");

                const isSelected = selectedTime === log.time;
                const typeClass =
                  log.drift_type === "Sensor Drift" ? "tag tag--sensor" : "tag tag--env";

                return (
                  <tr
                    key={`${log.time}-${idx}`}
                    className={isSelected ? "is-selected" : ""}
                    onClick={() => onSelect(log.time)}
                  >
                    <td className="mono">{log.datetime}</td>
                    <td>
                      <span className={typeClass}>{log.drift_type}</span>
                    </td>
                    <td className="features-cell">{topFeatures}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </section>
  );
}
