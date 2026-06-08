export default function Header({ backendStatus, modelInfo }) {
  const isReady = backendStatus?.status === "ready";

  return (
    <header className="app-header">
      <div className="header-brand">
        <div className="brand-mark" aria-hidden="true">
          <svg viewBox="0 0 32 32" fill="none">
            <rect width="32" height="32" rx="8" fill="url(#brandGrad)" />
            <path
              d="M8 22V10l8 6 8-6v12h-3.5V14.5L16 19l-4.5-4.5V22H8z"
              fill="white"
              opacity="0.95"
            />
            <defs>
              <linearGradient id="brandGrad" x1="0" y1="0" x2="32" y2="32">
                <stop stopColor="#06b6d4" />
                <stop offset="1" stopColor="#2563eb" />
              </linearGradient>
            </defs>
          </svg>
        </div>
        <div>
          <p className="brand-eyebrow">SAINT-OPS</p>
          <h1>Live Drift Monitoring</h1>
          <p className="brand-subtitle">
            LSTM autoencoder · Volve production sensors · real-time anomaly detection
          </p>
        </div>
      </div>

      <div className="header-meta">
        <div className={`status-pill ${isReady ? "status-pill--ok" : "status-pill--warn"}`}>
          <span className="status-dot" />
          {isReady ? "System Ready" : backendStatus ? "Degraded" : "Connecting…"}
        </div>
        {modelInfo?.stream_source && (
          <span className="meta-chip">Stream: {modelInfo.stream_source}</span>
        )}
        {modelInfo?.model_source && (
          <span className="meta-chip">Model: {modelInfo.model_source}</span>
        )}
      </div>
    </header>
  );
}
