import { useState, useEffect, useCallback } from "react";
import { api } from "./api/client";
import Header from "./components/Header";
import MetricCards from "./components/MetricCards";
import ControlPanel from "./components/ControlPanel";
import NotificationStack from "./components/NotificationStack";
import DriftChart from "./components/DriftChart";
import EventLog from "./components/EventLog";
import ExplainabilityPanel from "./components/ExplainabilityPanel";
import "./App.css";

const START_IDX = 10;
const TOP_K = 3;

const ENV_FEATURES = ["AVG_DOWNHOLE_PRESSURE", "AVG_DOWNHOLE_TEMPERATURE", "BORE_OIL_VOL", "AVG_WHP_P"];
const SENSOR_FEATURES = ["AVG_ANNULUS_PRESS"];

function App() {
  const [t, setT] = useState(START_IDX);
  const [run, setRun] = useState(true);
  const [speed, setSpeed] = useState(1.0);
  const [sensorData, setSensorData] = useState([]);
  const [envData, setEnvData] = useState([]);
  const [logs, setLogs] = useState([]);
  const [selectedTime, setSelectedTime] = useState(null);
  const [sensorAlert, setSensorAlert] = useState("Sensor Stable");
  const [envAlert, setEnvAlert] = useState("Environment Stable");
  const [notifications, setNotifications] = useState([]);
  const [backendStatus, setBackendStatus] = useState(null);
  const [streamInfo, setStreamInfo] = useState(null);

  useEffect(() => {
    const loadStatus = async () => {
      try {
        const [readyRes, infoRes] = await Promise.all([
          api.get("/ready"),
          api.get("/stream/info").catch(() => ({ data: null })),
        ]);
        setBackendStatus(readyRes.data);
        if (infoRes.data && !infoRes.data.error) {
          setStreamInfo({
            ...readyRes.data,
            length: infoRes.data.length,
          });
        } else {
          setStreamInfo(readyRes.data);
        }
      } catch {
        setBackendStatus({ status: "unavailable" });
      }
    };
    loadStatus();
    const interval = setInterval(loadStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  const showNotification = useCallback((title, message, type) => {
    setNotifications((prev) => [
      ...prev,
      { id: Date.now() + Math.random(), title, message, type },
    ]);
  }, []);

  const closeNotification = useCallback((id) => {
    setNotifications((prev) => prev.filter((n) => n.id !== id));
  }, []);

  useEffect(() => {
    const interval = setInterval(async () => {
      if (!run) return;

      try {
        const res = await api.get("/drift_data", { params: { t } });
        const data = res.data;
        if (data.done) return;

        setSensorData((prev) => [...prev, data.sensor_error]);
        setEnvData((prev) => [...prev, data.env_error]);

        if (data.predicted_drift) {
          setLogs((prev) => [...prev, data]);

          if (data.drift_type === "Sensor Drift") {
            setSensorAlert("Sensor Drift Detected");
            setEnvAlert("Environment Stable");
            showNotification("Sensor drift detected", data.datetime, "error");
          } else {
            setEnvAlert("Environmental Drift Detected");
            setSensorAlert("Sensor Stable");
            showNotification("Environmental drift detected", data.datetime, "warning");
          }
        } else {
          setSensorAlert("Sensor Stable");
          setEnvAlert("Environment Stable");
        }

        setT((prev) => prev + 1);
      } catch (err) {
        console.error(err);
      }
    }, 500 / speed);

    return () => clearInterval(interval);
  }, [t, run, speed, showNotification]);

  const exportCSV = () => {
    if (!logs.length) return;
    const headers = ["Datetime", "Drift Type", "Top Features"];
    const rows = logs.map((log) => {
      const filtered = log.explanation.filter((exp) =>
        log.drift_type === "Sensor Drift"
          ? SENSOR_FEATURES.includes(exp.feature_name)
          : ENV_FEATURES.includes(exp.feature_name)
      );
      const topFeatures = filtered
        .sort((a, b) => b.error - a.error)
        .slice(0, TOP_K)
        .map((f) => `${f.feature_name}: ${f.error.toFixed(4)}`)
        .join(" | ");
      return [log.datetime, log.drift_type, topFeatures];
    });

    const csv =
      "data:text/csv;charset=utf-8," +
      headers.join(",") +
      "\n" +
      rows.map((e) => e.join(",")).join("\n");

    const link = document.createElement("a");
    link.href = encodeURI(csv);
    link.download = "saint_ops_drift_logs.csv";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleReset = () => {
    setT(START_IDX);
    setSensorData([]);
    setEnvData([]);
    setLogs([]);
    setSelectedTime(null);
    setSensorAlert("Sensor Stable");
    setEnvAlert("Environment Stable");
    setNotifications([]);
    setRun(true);
  };

  const selectedLog = selectedTime !== null ? logs.find((l) => l.time === selectedTime) : null;

  const filteredExplanation = selectedLog
    ? selectedLog.explanation.filter((exp) =>
        selectedLog.drift_type === "Sensor Drift"
          ? SENSOR_FEATURES.includes(exp.feature_name)
          : ENV_FEATURES.includes(exp.feature_name)
      )
    : [];

  const xIndices = Array.from({ length: sensorData.length }, (_, i) => i + START_IDX);
  const sensorTone = sensorAlert.includes("Drift") ? "danger" : "success";
  const envTone = envAlert.includes("Drift") ? "warning" : "success";

  return (
    <div className="app">
      <NotificationStack notifications={notifications} onClose={closeNotification} />

      <div className="app-shell">
        <Header backendStatus={backendStatus} modelInfo={streamInfo} />

        <MetricCards
          currentStep={t}
          totalSteps={streamInfo?.stream_rows}
          driftCount={logs.length}
          sensorAlert={sensorAlert}
          envAlert={envAlert}
          isRunning={run}
        />

        <ControlPanel
          run={run}
          speed={speed}
          onToggleRun={() => setRun((r) => !r)}
          onSpeedChange={setSpeed}
          onExport={exportCSV}
          onReset={handleReset}
          canExport={logs.length > 0}
        />

        <section className="chart-grid">
          <DriftChart
            title="Sensor reconstruction error"
            subtitle="AVG_ANNULUS_PRESS — annulus pressure channel"
            xData={xIndices}
            yData={sensorData}
            selectedX={selectedLog ? selectedTime : null}
            selectedY={selectedLog ? sensorData[selectedTime - START_IDX] : null}
            lineColor="#f87171"
            markerColor="#ef4444"
            statusLabel={sensorAlert}
            statusTone={sensorTone}
          />
          <DriftChart
            title="Environmental reconstruction error"
            subtitle="Downhole pressure, temperature, oil volume & wellhead pressure"
            xData={xIndices}
            yData={envData}
            selectedX={selectedLog ? selectedTime : null}
            selectedY={selectedLog ? envData[selectedTime - START_IDX] : null}
            lineColor="#fbbf24"
            markerColor="#d97706"
            statusLabel={envAlert}
            statusTone={envTone}
          />
        </section>

        <EventLog
          logs={logs}
          selectedTime={selectedTime}
          onSelect={setSelectedTime}
          topK={TOP_K}
          envFeatures={ENV_FEATURES}
          sensorFeatures={SENSOR_FEATURES}
        />

        <ExplainabilityPanel
          selectedLog={selectedLog}
          filteredExplanation={filteredExplanation}
        />
      </div>

      <footer className="app-footer">
        <span>SAINT-OPS · Semantic AI for Predictive Maintenance</span>
        <span>Equinor Volve dataset · LSTM Autoencoder v1.1</span>
      </footer>
    </div>
  );
}

export default App;
