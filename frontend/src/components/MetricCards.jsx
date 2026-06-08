export default function MetricCards({
  currentStep,
  totalSteps,
  driftCount,
  sensorAlert,
  envAlert,
  isRunning,
}) {
  const sensorDrift = sensorAlert.includes("DRIFT");
  const envDrift = envAlert.includes("DRIFT");

  const cards = [
    {
      label: "Timestep",
      value: totalSteps ? `${currentStep} / ${totalSteps}` : String(currentStep),
      hint: isRunning ? "Streaming live" : "Paused",
      tone: "neutral",
    },
    {
      label: "Drift Events",
      value: String(driftCount),
      hint: driftCount === 0 ? "No anomalies yet" : "Logged this session",
      tone: driftCount > 0 ? "warning" : "neutral",
    },
    {
      label: "Sensor Channel",
      value: sensorDrift ? "Alert" : "Stable",
      hint: "Annulus pressure",
      tone: sensorDrift ? "danger" : "success",
    },
    {
      label: "Environment",
      value: envDrift ? "Alert" : "Stable",
      hint: "Downhole & wellhead",
      tone: envDrift ? "warning" : "success",
    },
  ];

  return (
    <section className="metric-grid" aria-label="Key metrics">
      {cards.map((card) => (
        <article key={card.label} className={`metric-card metric-card--${card.tone}`}>
          <span className="metric-label">{card.label}</span>
          <span className="metric-value">{card.value}</span>
          <span className="metric-hint">{card.hint}</span>
        </article>
      ))}
    </section>
  );
}
