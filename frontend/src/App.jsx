import { useState, useEffect, useCallback, useMemo, useRef } from "react";
import { api } from "./api/client";
import {
  CHART_METRICS,
  aggregateChannelValues,
  filterExplanation,
  formatTopFeatures,
  getChannelIndices,
  getFeatureGroups,
} from "./utils/features";
import Header from "./components/Header";
import TabNav from "./components/TabNav";
import MetricCards from "./components/MetricCards";
import ControlPanel from "./components/ControlPanel";
import NotificationStack from "./components/NotificationStack";
import DriftChart from "./components/DriftChart";
import EventLog from "./components/EventLog";
import ExplainabilityPanel from "./components/ExplainabilityPanel";
import SystemPanel from "./components/SystemPanel";
import ModelRegistryPanel from "./components/ModelRegistryPanel";
import ShadowComparePanel from "./components/ShadowComparePanel";
import "./App.css";

const START_IDX = 10;
const TOP_K = 3;

const TABS = [
  { id: "monitor", label: "Drift monitor" },
  { id: "mlops", label: "MLOps & system" },
];

function App() {
  const [activeTab, setActiveTab] = useState("monitor");
  const [t, setT] = useState(START_IDX);
  const [run, setRun] = useState(true);
  const [speed, setSpeed] = useState(1.0);
  const [inferenceVariant, setInferenceVariant] = useState("auto");
  const [shadowTrafficPct, setShadowTrafficPct] = useState(0);
  const [chartMetric, setChartMetric] = useState("raw");
  const [sensorData, setSensorData] = useState({ raw: [], norm: [], cusum: [] });
  const [envData, setEnvData] = useState({ raw: [], norm: [], cusum: [] });
  const [logs, setLogs] = useState([]);
  const [selectedTime, setSelectedTime] = useState(null);
  const [sensorAlert, setSensorAlert] = useState("Sensor Stable");
  const [envAlert, setEnvAlert] = useState("Environment Stable");
  const [notifications, setNotifications] = useState([]);
  const [health, setHealth] = useState(null);
  const [backendStatus, setBackendStatus] = useState(null);
  const [streamInfo, setStreamInfo] = useState(null);
  const [modelsInfo, setModelsInfo] = useState(null);
  const [modelsCompare, setModelsCompare] = useState(null);
  const [shadowCompare, setShadowCompare] = useState(null);
  const [shadowCompareLoading, setShadowCompareLoading] = useState(false);
  const [servedVariant, setServedVariant] = useState(null);
  const [mlflowModelUri, setMlflowModelUri] = useState(null);
  const [featureNames, setFeatureNames] = useState([]);
  const [registryLoading, setRegistryLoading] = useState(false);
  const [actionLoading, setActionLoading] = useState(false);
  const [promoteForce, setPromoteForce] = useState(false);
  const shadowTrafficInitialized = useRef(false);
  const pollInFlightRef = useRef(false);
  const tRef = useRef(START_IDX);

  useEffect(() => {
    tRef.current = t;
  }, [t]);

  const featureGroups = useMemo(
    () => getFeatureGroups(featureNames.length ? featureNames : streamInfo?.features),
    [featureNames, streamInfo?.features]
  );

  const refreshSystem = useCallback(async () => {
    try {
      const [healthRes, readyRes, infoRes] = await Promise.all([
        api.get("/health"),
        api.get("/ready"),
        api.get("/stream/info").catch(() => ({ data: null })),
      ]);
      setHealth(healthRes.data);
      setBackendStatus(readyRes.data);
      if (infoRes.data && !infoRes.data.error) {
        setStreamInfo(infoRes.data);
        if (!featureNames.length && infoRes.data.features) {
          setFeatureNames(infoRes.data.features);
        }
      }
      if (
        !shadowTrafficInitialized.current &&
        readyRes.data?.shadow_traffic_pct != null
      ) {
        setShadowTrafficPct(readyRes.data.shadow_traffic_pct);
        shadowTrafficInitialized.current = true;
      }
    } catch {
      setBackendStatus({ status: "unavailable" });
    }
  }, [featureNames.length]);

  const refreshRegistry = useCallback(async () => {
    setRegistryLoading(true);
    try {
      const [infoRes, compareRes] = await Promise.all([
        api.get("/models/info"),
        api.get("/models/compare"),
      ]);
      setModelsInfo(infoRes.data);
      setModelsCompare(compareRes.data);
    } catch (err) {
      console.error(err);
    } finally {
      setRegistryLoading(false);
    }
  }, []);

  useEffect(() => {
    refreshSystem();
    const interval = setInterval(refreshSystem, 30000);
    return () => clearInterval(interval);
  }, [refreshSystem]);

  useEffect(() => {
    if (activeTab !== "mlops") return;
    refreshRegistry();
  }, [activeTab, refreshRegistry]);

  const showNotification = useCallback((title, message, type) => {
    setNotifications((prev) => [
      ...prev,
      { id: Date.now() + Math.random(), title, message, type },
    ]);
  }, []);

  const closeNotification = useCallback((id) => {
    setNotifications((prev) => prev.filter((n) => n.id !== id));
  }, []);

  const compareTimestep = selectedTime ?? Math.max(t - 1, START_IDX);

  useEffect(() => {
    if (activeTab !== "mlops") return;

    let cancelled = false;
    const loadShadow = async () => {
      setShadowCompareLoading(true);
      try {
        const res = await api.get("/models/shadow/compare", {
          params: { t: compareTimestep },
        });
        if (!cancelled) setShadowCompare(res.data);
      } catch (err) {
        if (!cancelled) setShadowCompare({ error: "Failed to load shadow comparison" });
        console.error(err);
      } finally {
        if (!cancelled) setShadowCompareLoading(false);
      }
    };

    loadShadow();
    return () => {
      cancelled = true;
    };
  }, [compareTimestep, activeTab, t]);

  useEffect(() => {
    if (!run) return;

    let cancelled = false;
    let timerId;

    const poll = async () => {
      if (cancelled || pollInFlightRef.current) return;
      pollInFlightRef.current = true;
      const currentT = tRef.current;

      try {
        const params = { t: currentT };
        if (inferenceVariant !== "auto") {
          params.variant = inferenceVariant;
        }
        const headers = {};
        if (inferenceVariant === "auto" && shadowTrafficPct > 0) {
          headers["X-Shadow-Traffic-Pct"] = String(shadowTrafficPct);
        }

        const res = await api.get("/drift_data", { params, headers });
        const data = res.data;

        if (cancelled) return;
        if (data.done) {
          setRun(false);
          return;
        }

        if (data.feature_names?.length) {
          setFeatureNames(data.feature_names);
        }
        if (data.served_variant) setServedVariant(data.served_variant);
        if (data.mlflow_model_uri) setMlflowModelUri(data.mlflow_model_uri);

        const { sensorIdx, envIdxs } = getChannelIndices(data.feature_names);
        const sensorIndices = sensorIdx >= 0 ? [sensorIdx] : [];
        const channelSample = {
          raw: {
            sensor: data.sensor_error,
            env: data.env_error,
          },
          norm: {
            sensor: aggregateChannelValues(data.feature_norm_error, sensorIndices),
            env: aggregateChannelValues(data.feature_norm_error, envIdxs),
          },
          cusum: {
            sensor: aggregateChannelValues(data.feature_cusum, sensorIndices, "max"),
            env: aggregateChannelValues(data.feature_cusum, envIdxs, "max"),
          },
        };
        setSensorData((prev) => ({
          raw: [...prev.raw, channelSample.raw.sensor],
          norm: [...prev.norm, channelSample.norm.sensor],
          cusum: [...prev.cusum, channelSample.cusum.sensor],
        }));
        setEnvData((prev) => ({
          raw: [...prev.raw, channelSample.raw.env],
          norm: [...prev.norm, channelSample.norm.env],
          cusum: [...prev.cusum, channelSample.cusum.env],
        }));

        if (data.predicted_drift) {
          setLogs((prev) => [...prev, data]);

          if (data.drift_type === "Sensor Drift") {
            setSensorAlert("Sensor Drift Detected");
            setEnvAlert("Environment Stable");
            showNotification(
              "Sensor drift detected",
              `${data.datetime}${data.drift_type_fine ? ` · ${data.drift_type_fine}` : ""}`,
              "error"
            );
          } else {
            setEnvAlert("Environmental Drift Detected");
            setSensorAlert("Sensor Stable");
            showNotification(
              "Environmental drift detected",
              `${data.datetime}${data.drift_type_fine ? ` · ${data.drift_type_fine}` : ""}`,
              "warning"
            );
          }
        } else {
          setSensorAlert("Sensor Stable");
          setEnvAlert("Environment Stable");
        }

        setT((prev) => prev + 1);
      } catch (err) {
        console.error(err);
      } finally {
        pollInFlightRef.current = false;
        if (!cancelled && run) {
          timerId = setTimeout(poll, 500 / speed);
        }
      }
    };

    poll();
    return () => {
      cancelled = true;
      clearTimeout(timerId);
      pollInFlightRef.current = false;
    };
  }, [run, speed, inferenceVariant, shadowTrafficPct, showNotification]);

  const runModelAction = async (label, request) => {
    setActionLoading(true);
    try {
      const res = await request();
      const data = res.data;
      if (data.ok === false || data.error) {
        showNotification(`${label} failed`, data.error ?? "Unknown error", "error");
      } else {
        showNotification(`${label} succeeded`, data.message ?? "Operation completed", "success");
        await Promise.all([refreshSystem(), refreshRegistry()]);
      }
      return data;
    } catch (err) {
      showNotification(`${label} failed`, err.message ?? "Request error", "error");
      return null;
    } finally {
      setActionLoading(false);
    }
  };

  const exportCSV = () => {
    if (!logs.length) return;
    const headers = ["Datetime", "Drift Type", "Fine Label", "Classifier", "Top Features (v8 rank)"];
    const rows = logs.map((log) => {
      const filtered = filterExplanation(log.explanation, log.drift_type, featureGroups);
      const topFeatures = formatTopFeatures(filtered, TOP_K);
      return [
        log.datetime,
        log.drift_type,
        log.drift_type_fine ?? "",
        log.classifier_version ?? "",
        topFeatures,
      ];
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
    setSensorData({ raw: [], norm: [], cusum: [] });
    setEnvData({ raw: [], norm: [], cusum: [] });
    setLogs([]);
    setSelectedTime(null);
    setSensorAlert("Sensor Stable");
    setEnvAlert("Environment Stable");
    setNotifications([]);
    setServedVariant(null);
    setMlflowModelUri(null);
    setShadowCompare(null);
    setRun(true);
  };

  const selectedLog = selectedTime !== null ? logs.find((l) => l.time === selectedTime) : null;
  const filteredExplanation = selectedLog
    ? filterExplanation(selectedLog.explanation, selectedLog.drift_type, featureGroups)
    : [];

  const chartConfig = CHART_METRICS[chartMetric] ?? CHART_METRICS.raw;
  const sensorSeries = sensorData[chartMetric] ?? sensorData.raw;
  const envSeries = envData[chartMetric] ?? envData.raw;
  const xIndices = Array.from({ length: sensorSeries.length }, (_, i) => i + START_IDX);
  const selectedSeriesIdx =
    selectedLog != null ? selectedLog.time - START_IDX : null;
  const sensorTone = /drift/i.test(sensorAlert) ? "danger" : "success";
  const envTone = /drift/i.test(envAlert) ? "warning" : "success";

  return (
    <div className="app">
      <NotificationStack notifications={notifications} onClose={closeNotification} />

      <div className="app-shell">
        <Header
          backendStatus={backendStatus}
          modelInfo={{ ...backendStatus, ...streamInfo, classifier: health?.classifier }}
          servedVariant={servedVariant}
          mlflowModelUri={mlflowModelUri}
        />

        <TabNav tabs={TABS} active={activeTab} onChange={setActiveTab} />

        {activeTab === "monitor" && (
          <>
            <MetricCards
              currentStep={t}
              totalSteps={streamInfo?.length ?? backendStatus?.stream_rows}
              driftCount={logs.length}
              sensorAlert={sensorAlert}
              envAlert={envAlert}
              isRunning={run}
            />

            <ControlPanel
              run={run}
              speed={speed}
              chartMetric={chartMetric}
              inferenceVariant={inferenceVariant}
              shadowTrafficPct={shadowTrafficPct}
              onToggleRun={() => setRun((r) => !r)}
              onSpeedChange={setSpeed}
              onChartMetricChange={setChartMetric}
              onVariantChange={setInferenceVariant}
              onShadowTrafficChange={setShadowTrafficPct}
              onExport={exportCSV}
              onReset={handleReset}
              canExport={logs.length > 0}
            />

            <section className="chart-grid">
              <DriftChart
                title={`Sensor channel · ${chartConfig.label}`}
                subtitle={featureGroups.sensorFeatures.join(", ")}
                xData={xIndices}
                yData={sensorSeries}
                selectedX={selectedLog ? selectedTime : null}
                selectedY={
                  selectedSeriesIdx != null && selectedSeriesIdx >= 0
                    ? sensorSeries[selectedSeriesIdx]
                    : null
                }
                lineColor="#f87171"
                markerColor="#ef4444"
                statusLabel={sensorAlert}
                statusTone={sensorTone}
                yAxisTitle={chartConfig.yTitle}
                seriesLabel={chartConfig.label}
              />
              <DriftChart
                title={`Environment channel · ${chartConfig.label}`}
                subtitle={featureGroups.envFeatures.join(", ")}
                xData={xIndices}
                yData={envSeries}
                selectedX={selectedLog ? selectedTime : null}
                selectedY={
                  selectedSeriesIdx != null && selectedSeriesIdx >= 0
                    ? envSeries[selectedSeriesIdx]
                    : null
                }
                lineColor="#fbbf24"
                markerColor="#d97706"
                statusLabel={envAlert}
                statusTone={envTone}
                yAxisTitle={chartConfig.yTitle}
                seriesLabel={chartConfig.label}
              />
            </section>

            <EventLog
              logs={logs}
              selectedTime={selectedTime}
              onSelect={setSelectedTime}
              topK={TOP_K}
              envFeatures={featureGroups.envFeatures}
              sensorFeatures={featureGroups.sensorFeatures}
            />

            <ExplainabilityPanel
              selectedLog={selectedLog}
              filteredExplanation={filteredExplanation}
            />
          </>
        )}

        {activeTab === "mlops" && (
          <div className="mlops-stack">
            <SystemPanel health={health} ready={backendStatus} streamInfo={streamInfo} />
            <ModelRegistryPanel
              modelsInfo={modelsInfo}
              modelsCompare={modelsCompare}
              loading={registryLoading}
              actionLoading={actionLoading}
              promoteForce={promoteForce}
              onPromoteForceChange={setPromoteForce}
              onRefresh={refreshRegistry}
              onApprove={() => runModelAction("Approve", () => api.post("/models/approve"))}
              onPromote={() =>
                runModelAction("Promote", () =>
                  api.post("/models/promote", { force: promoteForce })
                )
              }
              onRollback={() => runModelAction("Rollback", () => api.post("/models/rollback"))}
              onReload={() => runModelAction("Reload", () => api.post("/models/reload"))}
            />
            <ShadowComparePanel
              data={shadowCompare}
              timestep={compareTimestep}
              loading={shadowCompareLoading}
              featureNames={featureNames.length ? featureNames : streamInfo?.features ?? []}
            />
          </div>
        )}
      </div>

      <footer className="app-footer">
        <span>SAINT-OPS · Semantic AI for Predictive Maintenance</span>
        <span>Equinor Volve dataset · LSTM Autoencoder v1.3 · v8 fixed val_p99 detector</span>
      </footer>
    </div>
  );
}

export default App;
