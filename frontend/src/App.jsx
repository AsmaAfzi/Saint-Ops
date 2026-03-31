import React, { useState, useEffect } from "react";
import Plot from "react-plotly.js";
import axios from "axios";

const START_IDX = 800;
const TOP_K = 3;

// Feature filters based on drift type
const ENV_FEATURES = ["AVG_DOWNHOLE_PRESSURE", "AVG_DOWNHOLE_TEMPERATURE", "BORE_OIL_VOL", "AVG_WHP_P"];
const SENSOR_FEATURES = ["DP_CHOKE_SIZE", "AVG_DOWNHOLE_PRESSURE"];

function App() {
  const [t, setT] = useState(START_IDX);
  const [run, setRun] = useState(true);
  const [speed, setSpeed] = useState(1.0);
  const [sensorData, setSensorData] = useState([]);
  const [envData, setEnvData] = useState([]);
  const [logs, setLogs] = useState([]);
  const [selectedTime, setSelectedTime] = useState(null);
  const [sensorAlert, setSensorAlert] = useState("✅ Sensor Stable");
  const [envAlert, setEnvAlert] = useState("✅ Environment Stable");
  const [notifications, setNotifications] = useState([]);

  // Live streaming
  useEffect(() => {
    const interval = setInterval(async () => {
      if (!run) return;

      try {
        const res = await axios.get(`http://localhost:8000/drift_data?t=${t}`);
        const data = res.data;
        if (data.done) return;

        setSensorData(prev => [...prev, data.sensor_error]);
        setEnvData(prev => [...prev, data.env_error]);

        if (data.predicted_drift) {
          setLogs(prev => [...prev, data]);

          if (data.drift_type === "Sensor Drift") {
            setSensorAlert("⚠️ SENSOR DRIFT DETECTED");
            setEnvAlert("✅ Environment Stable");
            showNotification("⚠️ Sensor Drift Detected!", `Time: ${data.datetime}`, "error");
          } else {
            setEnvAlert("🌍 ENVIRONMENTAL DRIFT DETECTED");
            setSensorAlert("✅ Sensor Stable");
            showNotification("🌍 Environmental Drift Detected!", `Time: ${data.datetime}`, "warning");
          }
        } else {
          setSensorAlert("✅ Sensor Stable");
          setEnvAlert("✅ Environment Stable");
        }

        setT(prev => prev + 1);
      } catch (err) {
        console.error(err);
      }
    }, 500 / speed);

    return () => clearInterval(interval);
  }, [t, run, speed]);

  // Export CSV
  const exportCSV = () => {
    if (!logs.length) return;
    const headers = ["Datetime", "Drift Type", "Top Features"];
    const rows = logs.map(log => {
      // Filter features based on drift type
      const filteredExplanations = log.explanation.filter(exp =>
        log.drift_type === "Sensor Drift" ? SENSOR_FEATURES.includes(exp.feature_name) :
        ENV_FEATURES.includes(exp.feature_name)
      );

      const topFeatures = filteredExplanations
        .sort((a,b) => b.error - a.error)
        .slice(0, TOP_K)
        .map(f => `${f.feature_name}: ${f.error.toFixed(4)}`)
        .join(" | ");
      return [log.datetime, log.drift_type, topFeatures];
    });

    let csvContent =
      "data:text/csv;charset=utf-8," +
      headers.join(",") +
      "\n" +
      rows.map(e => e.join(",")).join("\n");

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "drift_logs.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const selectedLog = selectedTime !== null ? logs.find(l => l.time === selectedTime) : null;

  // Filter explanations for display based on drift type
  const filteredExplanation = selectedLog ? selectedLog.explanation.filter(exp =>
    selectedLog.drift_type === "Sensor Drift" ? SENSOR_FEATURES.includes(exp.feature_name) :
    ENV_FEATURES.includes(exp.feature_name)
  ) : [];

  // Show notification function - adds to stack
  const showNotification = (title, message, type) => {
    const newNotification = {
      id: Date.now() + Math.random(), // Unique ID
      title,
      message,
      type
    };

    setNotifications(prev => [...prev, newNotification]);
  };

  const closeNotification = (id) => {
    setNotifications(prev => prev.filter(notif => notif.id !== id));
  };

  return (
    <div style={{ 
      minHeight: "100vh",
      width: "100vw",
      background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
      padding: "20px",
      margin: 0,
      boxSizing: "border-box",
      fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
      overflowX: "hidden"
    }}>
      {/* Notification Stack */}
      <div style={{
        position: "fixed",
        top: "20px",
        right: "20px",
        zIndex: 9999,
        display: "flex",
        flexDirection: "column",
        gap: "12px",
        maxHeight: "calc(100vh - 40px)",
        overflowY: "auto",
        maxWidth: "400px"
      }}>
        {notifications.map((notification, index) => (
          <div 
            key={notification.id}
            style={{
              minWidth: "320px",
              background: notification.type === "error" ? 
                "linear-gradient(135deg, #fc8181 0%, #f56565 100%)" : 
                "linear-gradient(135deg, #fbd38d 0%, #ed8936 100%)",
              color: "white",
              padding: "20px 25px",
              borderRadius: "12px",
              boxShadow: "0 10px 40px rgba(0,0,0,0.3)",
              animation: "slideIn 0.3s ease-out",
              display: "flex",
              alignItems: "flex-start",
              justifyContent: "space-between",
              gap: "15px"
            }}
          >
            <div style={{ flex: 1 }}>
              <div style={{ 
                fontSize: "18px", 
                fontWeight: "700",
                marginBottom: "8px",
                letterSpacing: "0.3px"
              }}>
                {notification.title}
              </div>
              <div style={{ 
                fontSize: "14px",
                opacity: 0.95,
                fontWeight: "500"
              }}>
                {notification.message}
              </div>
            </div>
            <button
              onClick={() => closeNotification(notification.id)}
              style={{
                background: "rgba(255,255,255,0.2)",
                border: "none",
                color: "white",
                fontSize: "20px",
                cursor: "pointer",
                width: "28px",
                height: "28px",
                borderRadius: "6px",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                transition: "background 0.2s ease",
                flexShrink: 0
              }}
              onMouseOver={e => e.target.style.background = "rgba(255,255,255,0.3)"}
              onMouseOut={e => e.target.style.background = "rgba(255,255,255,0.2)"}
            >
              ×
            </button>
          </div>
        ))}
      </div>

      <div style={{ 
        width: "100%",
        background: "white",
        borderRadius: "20px",
        boxShadow: "0 20px 60px rgba(0,0,0,0.3)",
        padding: "40px",
        boxSizing: "border-box"
      }}>
        {/* Header */}
        <div style={{ 
          textAlign: "center", 
          marginBottom: "50px",
          borderBottom: "3px solid #667eea",
          paddingBottom: "20px"
        }}>
          <h1 style={{ 
            fontSize: "42px",
            fontWeight: "700",
            color: "#2d3748",
            margin: "0 0 10px 0",
            letterSpacing: "-0.5px"
          }}>
            🔍 Live Drift Monitoring System
          </h1>
          <p style={{ 
            fontSize: "16px",
            color: "#718096",
            margin: 0,
            fontWeight: "500"
          }}>
            Real-time Sensor & Environmental Drift Detection
          </p>
        </div>

        {/* Controls */}
        <div style={{ 
          display: "flex", 
          justifyContent: "center", 
          alignItems: "center",
          gap: "30px", 
          marginBottom: "40px",
          padding: "25px",
          background: "linear-gradient(135deg, #f6f8fb 0%, #e9ecef 100%)",
          borderRadius: "15px",
          boxShadow: "0 4px 6px rgba(0,0,0,0.07)"
        }}>
          <button 
            onClick={() => setRun(!run)} 
            style={{ 
              padding: "14px 32px", 
              fontSize: "16px",
              fontWeight: "600",
              border: "none",
              borderRadius: "10px",
              background: run ? "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)" : "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)",
              color: "white",
              cursor: "pointer",
              boxShadow: "0 4px 15px rgba(0,0,0,0.2)",
              transition: "all 0.3s ease",
              transform: "translateY(0)"
            }}
            onMouseOver={e => e.target.style.transform = "translateY(-2px)"}
            onMouseOut={e => e.target.style.transform = "translateY(0)"}
          >
            {run ? "⏸ Pause" : "▶️ Run"}
          </button>
          
          <div style={{
            background: "white",
            padding: "12px 20px",
            borderRadius: "10px",
            boxShadow: "0 2px 8px rgba(0,0,0,0.08)"
          }}>
            <label style={{ 
              fontSize: "15px",
              color: "#4a5568",
              fontWeight: "600"
            }}>
              Playback Speed: <span style={{ color: "#667eea", fontWeight: "700" }}>{speed}x</span>
              <input 
                type="range" 
                min="0.2" 
                max="2.0" 
                step="0.2" 
                value={speed} 
                onChange={e => setSpeed(parseFloat(e.target.value))} 
                style={{ 
                  marginLeft: "15px",
                  verticalAlign: "middle",
                  accentColor: "#667eea"
                }}
              />
            </label>
          </div>
          
          <button 
            onClick={exportCSV} 
            style={{ 
              padding: "14px 32px", 
              fontSize: "16px",
              fontWeight: "600",
              border: "none",
              borderRadius: "10px",
              background: "linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)",
              color: "white",
              cursor: "pointer",
              boxShadow: "0 4px 15px rgba(0,0,0,0.2)",
              transition: "all 0.3s ease",
              transform: "translateY(0)"
            }}
            onMouseOver={e => e.target.style.transform = "translateY(-2px)"}
            onMouseOut={e => e.target.style.transform = "translateY(0)"}
          >
            ⬇️ Export Logs
          </button>
        </div>

        {/* Plots */}
        <div style={{ 
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: "30px", 
          marginBottom: "40px" 
        }}>
          <div style={{ 
            background: "white",
            borderRadius: "15px",
            padding: "15px",
            boxShadow: "0 4px 20px rgba(0,0,0,0.08)",
            border: "1px solid #e2e8f0"
          }}>
            <Plot
              data={[
                { 
                  x: Array.from({length: sensorData.length}, (_, i) => i + START_IDX), 
                  y: sensorData, 
                  type: "scatter", 
                  mode: "lines+markers", 
                  name: "Sensor Error",
                  line: { color: "#f56565", width: 2 },
                  marker: { size: 6 }
                },
                { 
                  x: selectedLog ? [selectedTime] : [], 
                  y: selectedLog ? [sensorData[selectedTime - START_IDX]] : [], 
                  type: "scatter", 
                  mode: "markers", 
                  name: "Selected", 
                  marker: { color: "#e53e3e", size: 14, symbol: "diamond" } 
                }
              ]}
              layout={{ 
                title: {
                  text: "Sensor Drift Error",
                  font: { size: 18, color: "#2d3748", weight: 600 }
                },
                height: 450,
                margin: { t: 50, b: 50, l: 60, r: 30 },
                plot_bgcolor: "#fafafa",
                paper_bgcolor: "white",
                autosize: true
              }}
              config={{ displayModeBar: false, responsive: true }}
              style={{ width: "100%" }}
            />
            <div style={{ 
              marginTop: "15px", 
              padding: "12px 20px",
              fontWeight: "600",
              fontSize: "15px",
              borderRadius: "8px",
              textAlign: "center",
              background: sensorAlert.includes("⚠️") ? "#fed7d7" : "#c6f6d5",
              color: sensorAlert.includes("⚠️") ? "#c53030" : "#22543d",
              border: `2px solid ${sensorAlert.includes("⚠️") ? "#fc8181" : "#68d391"}`
            }}>
              {sensorAlert}
            </div>
          </div>

          <div style={{ 
            background: "white",
            borderRadius: "15px",
            padding: "15px",
            boxShadow: "0 4px 20px rgba(0,0,0,0.08)",
            border: "1px solid #e2e8f0"
          }}>
            <Plot
              data={[
                { 
                  x: Array.from({length: envData.length}, (_, i) => i + START_IDX), 
                  y: envData, 
                  type: "scatter", 
                  mode: "lines+markers", 
                  name: "Env Error",
                  line: { color: "#ed8936", width: 2 },
                  marker: { size: 6 }
                },
                { 
                  x: selectedLog ? [selectedTime] : [], 
                  y: selectedLog ? [envData[selectedTime - START_IDX]] : [], 
                  type: "scatter", 
                  mode: "markers", 
                  name: "Selected", 
                  marker: { color: "#c05621", size: 14, symbol: "diamond" } 
                }
              ]}
              layout={{ 
                title: {
                  text: "Environmental Drift Error",
                  font: { size: 18, color: "#2d3748", weight: 600 }
                },
                height: 450,
                margin: { t: 50, b: 50, l: 60, r: 30 },
                plot_bgcolor: "#fafafa",
                paper_bgcolor: "white",
                autosize: true
              }}
              config={{ displayModeBar: false, responsive: true }}
              style={{ width: "100%" }}
            />
            <div style={{ 
              marginTop: "15px", 
              padding: "12px 20px",
              fontWeight: "600",
              fontSize: "15px",
              borderRadius: "8px",
              textAlign: "center",
              background: envAlert.includes("🌍") ? "#feebc8" : "#c6f6d5",
              color: envAlert.includes("🌍") ? "#c05621" : "#22543d",
              border: `2px solid ${envAlert.includes("🌍") ? "#fbd38d" : "#68d391"}`
            }}>
              {envAlert}
            </div>
          </div>
        </div>

        {/* Drift Event Table */}
        <div style={{ marginBottom: "40px" }}>
          <h2 style={{ 
            textAlign: "center",
            fontSize: "28px",
            color: "#2d3748",
            marginBottom: "25px",
            fontWeight: "700"
          }}>
            📜 Drift Event Log
          </h2>
          <div style={{ 
            maxHeight: "350px", 
            overflowY: "auto", 
            borderRadius: "12px",
            boxShadow: "0 4px 20px rgba(0,0,0,0.08)",
            border: "1px solid #e2e8f0"
          }}>
            <table style={{ 
              width: "100%", 
              borderCollapse: "collapse", 
              textAlign: "center",
              background: "white"
            }}>
              <thead style={{ position: "sticky", top: 0, zIndex: 1 }}>
                <tr style={{ 
                  background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                  color: "white"
                }}>
                  <th style={{ 
                    border: "none", 
                    padding: "16px 12px",
                    fontWeight: "600",
                    fontSize: "15px",
                    letterSpacing: "0.5px"
                  }}>
                    Datetime
                  </th>
                  <th style={{ 
                    border: "none", 
                    padding: "16px 12px",
                    fontWeight: "600",
                    fontSize: "15px",
                    letterSpacing: "0.5px"
                  }}>
                    Drift Type
                  </th>
                  <th style={{ 
                    border: "none", 
                    padding: "16px 12px",
                    fontWeight: "600",
                    fontSize: "15px",
                    letterSpacing: "0.5px"
                  }}>
                    Top Features
                  </th>
                </tr>
              </thead>
              <tbody>
                {logs.map((log, idx) => {
                  const filteredFeatures = log.explanation.filter(exp =>
                    log.drift_type === "Sensor Drift" ? SENSOR_FEATURES.includes(exp.feature_name) :
                    ENV_FEATURES.includes(exp.feature_name)
                  );

                  const topFeatures = filteredFeatures
                    .sort((a,b) => b.error - a.error)
                    .slice(0, TOP_K)
                    .map(f => `${f.feature_name}: ${f.error.toFixed(4)}`)
                    .join(" | ");

                  return (
                    <tr 
                      key={idx} 
                      style={{ 
                        background: selectedTime===log.time ? "#e6fffa" : (idx % 2 === 0 ? "#fafafa" : "white"),
                        cursor: "pointer",
                        transition: "all 0.2s ease",
                        borderLeft: selectedTime===log.time ? "4px solid #38b2ac" : "4px solid transparent"
                      }}
                      onClick={() => setSelectedTime(log.time)}
                      onMouseOver={e => {
                        if (selectedTime !== log.time) {
                          e.currentTarget.style.background = "#f7fafc";
                        }
                      }}
                      onMouseOut={e => {
                        if (selectedTime !== log.time) {
                          e.currentTarget.style.background = idx % 2 === 0 ? "#fafafa" : "white";
                        }
                      }}
                    >
                      <td style={{ 
                        border: "1px solid #e2e8f0", 
                        padding: "14px 12px",
                        fontSize: "14px",
                        color: "#4a5568"
                      }}>
                        {log.datetime}
                      </td>
                      <td style={{ 
                        border: "1px solid #e2e8f0", 
                        padding: "14px 12px",
                        fontSize: "14px",
                        fontWeight: "600",
                        color: log.drift_type==="Sensor Drift" ? "#e53e3e" : "#dd6b20"
                      }}>
                        {log.drift_type}
                      </td>
                      <td style={{ 
                        border: "1px solid #e2e8f0", 
                        padding: "14px 12px",
                        fontSize: "13px",
                        color: "#718096",
                        fontFamily: "monospace"
                      }}>
                        {topFeatures}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* Explainability and Feature Contribution - Side by Side */}
        {selectedLog && filteredExplanation.length > 0 && (
          <div style={{ 
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: "30px", 
            marginBottom: "40px" 
          }}>
            {/* Explainability */}
            <div style={{ 
              padding: "25px 30px", 
              border: "2px solid #e2e8f0",
              borderRadius: "15px", 
              background: "linear-gradient(135deg, #fafafa 0%, #f7fafc 100%)",
              boxShadow: "0 4px 15px rgba(0,0,0,0.05)"
            }}>
              <h4 style={{ 
                margin: "0 0 20px 0",
                fontSize: "20px",
                color: "#2d3748",
                fontWeight: "700",
                borderBottom: "2px solid #cbd5e0",
                paddingBottom: "12px"
              }}>
                {selectedLog.drift_type === "Sensor Drift" ? "⚠️ Sensor Drift Explanation" : "🌍 Environmental Drift Explanation"}
              </h4>
              <ul style={{ 
                margin: 0,
                padding: "0 0 0 25px",
                lineHeight: "1.8"
              }}>
                {filteredExplanation.map((exp, idx) => (
                  <li key={idx} style={{ 
                    marginBottom: "12px",
                    fontSize: "15px",
                    color: "#4a5568"
                  }}>
                    <b style={{ 
                      color: "#2d3748",
                      fontSize: "16px",
                      fontWeight: "600"
                    }}>
                      {exp.feature_name}
                    </b>: {exp.error.toFixed(4)}
                    {exp.threshold ? ` > threshold ${exp.threshold.toFixed(4)}` : exp.baseline ? ` vs baseline ${exp.baseline.toFixed(4)}` : ""} → <span style={{ 
                      color: "#718096",
                      fontStyle: "italic"
                    }}>{exp.reason}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Feature Contribution */}
            <div style={{ 
              background: "white",
              borderRadius: "15px",
              padding: "25px",
              boxShadow: "0 4px 20px rgba(0,0,0,0.08)",
              border: "1px solid #e2e8f0"
            }}>
              <h4 style={{ 
                textAlign: "center",
                fontSize: "20px",
                color: "#2d3748",
                marginBottom: "20px",
                fontWeight: "700"
              }}>
                Feature Contribution at {selectedLog.datetime}
              </h4>
              <Plot
                data={[
                  {
                    x: filteredExplanation.map(f => f.feature_name),
                    y: filteredExplanation.map(f => f.error),
                    type: "bar",
                    marker: { 
                      color: filteredExplanation.map(f => f.error),
                      colorscale: "Viridis",
                      showscale: true
                    }
                  }
                ]}
                layout={{ 
                  height: 350,
                  margin: { t: 20, b: 100, l: 60, r: 60 },
                  xaxis: { 
                    tickangle: -45,
                    tickfont: { size: 12 }
                  },
                  yaxis: {
                    title: "Error Value",
                    titlefont: { size: 14 }
                  },
                  plot_bgcolor: "#fafafa",
                  paper_bgcolor: "white"
                }}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: "100%" }}
              />
            </div>
          </div>
        )}
      </div>
      
      {/* Inline CSS for animations */}
      <style>{`
        @keyframes slideIn {
          from {
            transform: translateX(400px);
            opacity: 0;
          }
          to {
            transform: translateX(0);
            opacity: 1;
          }
        }
      `}</style>
    </div>
  );
}

export default App;