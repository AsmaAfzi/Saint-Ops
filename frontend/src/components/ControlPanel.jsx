export default function ControlPanel({ run, speed, onToggleRun, onSpeedChange, onExport, onReset, canExport }) {
  return (
    <section className="control-panel" aria-label="Playback controls">
      <button
        type="button"
        className={`btn btn--primary ${run ? "btn--pause" : ""}`}
        onClick={onToggleRun}
      >
        {run ? (
          <>
            <PauseIcon /> Pause stream
          </>
        ) : (
          <>
            <PlayIcon /> Resume stream
          </>
        )}
      </button>

      <div className="speed-control">
        <label htmlFor="speed-range">
          Playback speed
          <span className="speed-value">{speed.toFixed(1)}×</span>
        </label>
        <input
          id="speed-range"
          type="range"
          min="0.2"
          max="2.0"
          step="0.2"
          value={speed}
          onChange={(e) => onSpeedChange(parseFloat(e.target.value))}
        />
      </div>

      <div className="control-actions">
        <button type="button" className="btn btn--ghost" onClick={onReset}>
          <ResetIcon /> Reset
        </button>
        <button
          type="button"
          className="btn btn--secondary"
          onClick={onExport}
          disabled={!canExport}
        >
          <DownloadIcon /> Export CSV
        </button>
      </div>
    </section>
  );
}

function PlayIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
      <path d="M4 2.5v11l9-5.5-9-5.5z" />
    </svg>
  );
}

function PauseIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
      <path d="M4 2h3v12H4V2zm5 0h3v12H9V2z" />
    </svg>
  );
}

function DownloadIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
      <path d="M8 2v8M5 7l3 3 3-3M3 12v2h10v-2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function ResetIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
      <path d="M3 8a5 5 0 0 1 8.5-3.5M13 8a5 5 0 0 1-8.5 3.5" strokeLinecap="round" />
      <path d="M11 2v3h-3M5 14v-3h3" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
