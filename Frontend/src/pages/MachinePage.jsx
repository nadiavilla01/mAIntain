// pages/MachinePage.jsx
import React, { useMemo, useRef, useState, useEffect } from "react";
import { useParams } from "react-router-dom";
import AIAssistant from "../components/AIAssistant";
import SensorChart from "../components/SensorChart";
import { FiAlertCircle, FiDownload } from "react-icons/fi";
import "./MachinePage.css";

const API_BASE =
  import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "") || "http://localhost:8000";

const statusClass = (status) => {
  switch (status) {
    case "Normal": return "normal";
    case "Unstable": return "unstable";
    case "Critical": return "critical blink";
    default: return "unknown";
  }
};
const fmt = (v, dp = 2) => (typeof v === "number" && isFinite(v) ? v.toFixed(dp) : "–");

const intentStyles = {
  intent_sensor_drift:     { color: "#facc15", icon: "🌡️", severity: "warn" },
  intent_wrong_scaling:    { color: "#f59e0b", icon: "📏", severity: "warn" },
  intent_overestimate_rul: { color: "#ef4444", icon: "📈", severity: "critical", blink: true },
  intent_model_outdated:   { color: "#a855f7", icon: "📦", severity: "info" },
  intent_wrong_intent:     { color: "#94a3b8", icon: "❓", severity: "info" },
  intent_uncertain:        { color: "#94a3b8", icon: "⚠️", severity: "warn", blink: true },
};

function RulTrendChart({ aiSeries, baseSeries, rmse = 0 }) {
  const W = 100, H = 40, PAD = 6;

  const all = [...aiSeries, ...baseSeries];
  const min = Math.min(...all.map(Number), 0);
  const max = Math.max(...all.map(Number), 1);
  const y = (v) => {
    const t = (v - min) / (max - min || 1);
    return H - PAD - t * (H - PAD * 2);
  };
  const x = (i, n) => PAD + (i / (n - 1)) * (W - PAD * 2);

  const toPath = (arr) =>
    arr.map((v, i) => `${i === 0 ? "M" : "L"} ${x(i, arr.length)} ${y(v)}`).join(" ");

  // Confidence band (simple rmse around AI)
  const upper = aiSeries.map((v) => v + rmse);
  const lower = aiSeries.map((v) => v - rmse).slice().reverse();
  const bandPath =
    upper.map((v, i) => `${i === 0 ? "M" : "L"} ${x(i, upper.length)} ${y(v)}`).join(" ") +
    " " +
    lower.map((v, i) => `L ${x(upper.length - 1 - i, upper.length)} ${y(v)}`).join(" ") +
    " Z";

  // horizontal grid (5 lines)
  const gridYs = Array.from({ length: 5 }, (_, i) => PAD + (i / 4) * (H - PAD * 2));

  return (
    <svg className="rul-svg" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="img" aria-label="RUL Trend">
      {gridYs.map((gy, i) => (
        <line key={i} x1={PAD} y1={gy} x2={W - PAD} y2={gy} className="rul-grid" />
      ))}
      {rmse > 0 && <path d={bandPath} className="rul-band" />}
      <path d={toPath(baseSeries)} className="rul-line base" />
      <path d={toPath(aiSeries)} className="rul-line ai" />
    </svg>
  );
}

const makeSyntheticRul = (endValue = 120, n = 24, step = 1.5) => {
  const start = Math.max(endValue + step * (n - 1), endValue);
  return Array.from({ length: n }, (_, i) => {
    const base = start - i * step;
    const jitter = (Math.random() - 0.5) * step * 0.25;
    return Math.max(base + jitter, 0);
  });
};

export default function MachinePage({ machines }) {
  const { id } = useParams();
  const machine = machines.find((m) => m.id === parseInt(id, 10));
  const aiRef = useRef(null);

  const [alertTab, setAlertTab] = useState("all");
  const [alertQuery, setAlertQuery] = useState("");
  const [rangePct, setRangePct] = useState(100);
  const [smooth, setSmooth] = useState(false);
  const [seriesMode, setSeriesMode] = useState("compare"); // 'single' | 'compare'
  const [selectedSensor, setSelectedSensor] = useState(null);
  const [ack, setAck] = useState({});

  if (!machine) return <div style={{ color: "#fff", padding: 24 }}>Machine not found.</div>;

  const aiRul = machine.predicted_rul ?? machine.rul;
  const trend = machine.trend || {};

  // eslint-disable-next-line react-hooks/rules-of-hooks
  const availableSensors = useMemo(
    () => Object.keys(trend).filter((s) => trend[s]?.length),
    [trend]
  );

  // eslint-disable-next-line react-hooks/rules-of-hooks
  useEffect(() => {
    if (!selectedSensor || !availableSensors.includes(selectedSensor)) {
      setSelectedSensor(availableSensors[0] || null);
    }
  }, [availableSensors, selectedSensor]);

  const alertedSensors = new Set(
    (machine.alerts || [])
      .map((a) => {
        const text = a.text?.toLowerCase() || "";
        return availableSensors.find((s) => text.includes(s)) || null;
      })
      .filter(Boolean)
  );

  const isSensorNearThreshold = (sensor, val) => {
    if (!trend[sensor]?.length || val == null) return false;
    const sorted = [...trend[sensor]].sort((a, b) => a - b);
    const p95 = sorted[Math.floor(sorted.length * 0.95)];
    return val >= p95;
  };

  const filteredAlerts = useMemo(() => {
    const list = machine.alerts || [];
    const byTab =
      alertTab === "all"
        ? list
        : alertTab === "model"
        ? list.filter((a) => String(a.intent || "").includes("model"))
        : alertTab === "sensor"
        ? list.filter((a) => String(a.intent || "").includes("sensor"))
        : list.filter((a) => String(a.intent || "").includes("uncertain"));
    const q = alertQuery.trim().toLowerCase();
    return q ? byTab.filter((a) => (a.text || a.original || "").toLowerCase().includes(q)) : byTab;
  }, [machine.alerts, alertTab, alertQuery]);

  const exportAlertsAsCSV = (alerts) => {
    if (!alerts.length) return;
    const rows = [
      ["Text", "Intent", "Confidence", "Suggested Action", "Cause"],
      ...alerts.map((a) => [
        `"${(a.text || a.original || "").replace(/"/g, '""')}"`,
        a.intent || "",
        a.confidence ?? "",
        a.suggested_action || "-",
        a.cause || "-",
      ]),
    ];
    const blob = new Blob([rows.map((r) => r.join(",")).join("\n")], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `alerts_${machine.name.replace(/\s+/g, "_")}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const formatTimestamp = (iso) => {
    if (!iso) return "–";
    const date = new Date(iso);
    return date.toLocaleString([], { hour12: false });
  };
  const timeAgo = (iso) => {
    if (!iso) return "";
    const ms = Date.now() - new Date(iso).getTime();
    const s = Math.max(0, Math.floor(ms / 1000));
    if (s < 60) return `${s}s ago`;
    const m = Math.floor(s / 60);
    if (m < 60) return `${m}m ago`;
    const h = Math.floor(m / 60);
    return `${h}h ago`;
  };

  const copyId = async () => {
    try { await navigator.clipboard.writeText(String(machine.id)); } catch {}
  };

  const sensorDelta = (sensor) => {
    const arr = trend[sensor] || [];
    if (arr.length < 2) return 0;
    return arr[arr.length - 1] - arr[arr.length - 2];
  };

  const N = 24;
  const aiHistory = Array.isArray(machine.predicted_rul_history) && machine.predicted_rul_history.length >= 3
    ? machine.predicted_rul_history.slice(-N)
    : makeSyntheticRul(Number(aiRul ?? 120), N, 1.8);

  const baseHistory = Array.isArray(machine.rul_history) && machine.rul_history.length >= 3
    ? machine.rul_history.slice(-N)
    : makeSyntheticRul(Number(machine.rul ?? aiRul ?? 120), N, 1.4);

  const rmse = Number.isFinite(machine.rmse) ? Number(machine.rmse) : 0;

  const last7 = (arr) => arr.slice(-7);
  const slopePer7 = (arr) => {
    const a = last7(arr);
    if (a.length < 2) return 0;
    const delta = a[a.length - 1] - a[0];
    return delta / (a.length - 1); 
  };

  const aiSlope = slopePer7(aiHistory);
  const baseSlope = slopePer7(baseHistory);
  const aiDelta7 = (last7(aiHistory).at(-1) ?? 0) - (last7(aiHistory)[0] ?? 0);

  return (
    <div className="mp-page">
      <div className="mp-wrap">
        <div className="mp-header">
          <div className="mp-breadcrumb">Devices / {machine.name}</div>
          <div className="mp-head-actions">
            <button className="mp-btn-ghost" onClick={() => window.print()}>
              Generate Report
            </button>
          </div>
        </div>

        <div className="mp-hero">
          <div className="mp-hero-left">
            <div className="mp-title-row">
              <h1 className="mp-title">{machine.name}</h1>
              <span className={`mp-status ${statusClass(machine.status)}`}>{machine.status}</span>
            </div>
            <div className="mp-meta-row">
              <button className="mp-chip-lite" title="Copy Machine ID" onClick={copyId}>
                ID: <b className="num">{machine.id}</b>
              </button>
              <span className="mp-dot">•</span>
              <span className="mp-chip-lite">Section <b>{machine.section || "—"}</b></span>
            </div>
          </div>

          <div className="mp-hero-center">
            <div className="mp-rul-label">Predicted RUL</div>
            <div className="mp-rul-value num">
              {aiRul ?? "–"}<span> h</span>
            </div>
          </div>

          <div className="mp-hero-right">
            <div className="mp-upd">
              <div className="mp-upd-label">Last Update</div>
              <div className="mp-upd-value">
                {formatTimestamp(machine.last_updated)}{" "}
                <span className="mp-ago">({timeAgo(machine.last_updated)})</span>
              </div>
            </div>
            <div className="mp-kmini">
              <span className="mp-kchip">MAE <b className="num">{fmt(machine.mae)}</b></span>
              <span className="mp-kchip">RMSE <b className="num">{fmt(machine.rmse)}</b></span>
            </div>
          </div>
        </div>

        <div className="mp-main">
          <div className="mp-left">
            <div className="mp-card">
              <div className="mp-card-head">
                <h3>Sensor Comparison</h3>
                <div className="mp-head-controls">
                  {/* Range (keep top-right) */}
                  <div className="mp-seg" role="tablist" aria-label="Window range">
                    {[25, 50, 100].map((p) => (
                      <button
                        key={p}
                        className={`mp-seg-btn ${rangePct === p ? "is-active" : ""}`}
                        onClick={() => setRangePct(p)}
                        role="tab"
                        aria-pressed={rangePct === p}
                        title={`Show last ${p}%`}
                      >
                        {p}%
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              <SensorChart
                trend={trend}
                availableSensors={availableSensors}
                rangePct={rangePct}
                smooth={smooth}
                mode={seriesMode}
                selectedSensor={selectedSensor}
              />

              <div className="mp-chart-controls">
                <div className="mp-seg" role="tablist" aria-label="Series mode">
                  <button
                    className={`mp-seg-btn ${seriesMode === "single" ? "is-active" : ""}`}
                    onClick={() => setSeriesMode("single")}
                    role="tab"
                    aria-pressed={seriesMode === "single"}
                    title="View one sensor at a time"
                  >
                    Single
                  </button>
                  <button
                    className={`mp-seg-btn ${seriesMode === "compare" ? "is-active" : ""}`}
                    onClick={() => setSeriesMode("compare")}
                    role="tab"
                    aria-pressed={seriesMode === "compare"}
                    title="Compare all sensors"
                  >
                    Compare
                  </button>
                </div>

                {/* Smooth (right) */}
                <div className="mp-seg" aria-label="Smoothing">
                  <button
                    className={`mp-seg-btn ${smooth ? "is-active" : ""}`}
                    onClick={() => setSmooth((s) => !s)}
                    aria-pressed={smooth}
                    title="Toggle smoothing"
                  >
                    Smooth
                  </button>
                </div>
              </div>

              {seriesMode === "single" && (
                <div className="mp-sensor-pills" aria-label="Choose sensor">
                  {availableSensors.map((s) => (
                    <button
                      key={s}
                      className={`mp-chip ${selectedSensor === s ? "is-active" : ""}`}
                      onClick={() => setSelectedSensor(s)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" || e.key === " ") {
                          e.preventDefault();
                          setSelectedSensor(s);
                        }
                      }}
                      aria-pressed={selectedSensor === s}
                      title={`Show ${s}`}
                    >
                      {s}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>

          <div className="mp-right">
            <div className="mp-card">
              <div className="mp-card-head">
                <h3><FiAlertCircle style={{ marginRight: 8 }} /> Alerts</h3>
                <div className="mp-head-controls">
                  <button className="mp-btn-ghost" onClick={() => exportAlertsAsCSV(filteredAlerts)}>
                    <FiDownload style={{ marginRight: 6 }} />
                    Export
                  </button>
                  <button
                    className="mp-btn-ghost"
                    onClick={() =>
                      setAck((prev) => {
                        const next = { ...prev };
                        filteredAlerts.forEach((a, i) => {
                          const key = `${a.intent}-${a.text || a.original}-${i}`;
                          next[key] = true;
                        });
                        return next;
                      })
                    }
                  >
                    Acknowledge
                  </button>
                </div>
              </div>

              <div className="mp-tabs">
                <div className="mp-seg" role="tablist" aria-label="Alert filters">
                  {[
                    { k: "all", t: "All" },
                    { k: "model", t: "Model" },
                    { k: "sensor", t: "Sensor" },
                    { k: "uncertain", t: "Uncertain" },
                  ].map((t) => (
                    <button
                      key={t.k}
                      className={`mp-seg-btn ${alertTab === t.k ? "is-active" : ""}`}
                      onClick={() => setAlertTab(t.k)}
                      role="tab"
                      aria-pressed={alertTab === t.k}
                      title={`Filter: ${t.t}`}
                    >
                      {t.t}
                    </button>
                  ))}
                </div>
              </div>

              <div className="mp-alerts">
                {filteredAlerts.length === 0 ? (
                  <div className="mp-empty">✅ No current alerts</div>
                ) : (
                  filteredAlerts.map((alert, i) => {
                    const style = intentStyles[alert.intent] || {};
                    const key = `${alert.intent}-${alert.text || alert.original}-${i}`;
                    const isAck = !!ack[key];
                    const sev = style.severity || "info";
                    return (
                      <div
                        key={key}
                        className={`mp-alert ${style.blink ? "blink" : ""} ${isAck ? "is-ack" : ""}`}
                        style={{ borderLeftColor: style.color || "#334155" }}
                      >
                        <div className="mp-alert-title">
                          <span className={`mp-badge mp-badge-${sev}`}>{sev}</span>
                          <strong>
                            {style.icon || "⚠️"} {alert.text || alert.original}
                          </strong>
                          {!isAck && (
                            <button
                              className="mp-tag"
                              onClick={() => setAck((prev) => ({ ...prev, [key]: true }))}
                            >
                              Ack
                            </button>
                          )}
                        </div>
                        <div className="mp-alert-row">
                          <em>Intent:</em>&nbsp;{alert.intent}{" "}
                          {alert.confidence != null && `(${Math.round(alert.confidence * 100)}%)`}
                        </div>
                        <div className="mp-alert-row"><em>Cause:</em>&nbsp;{alert.cause || "–"}</div>
                        <div className="mp-alert-row"><em>Suggested Action:</em>&nbsp;{alert.suggested_action || "–"}</div>
                      </div>
                    );
                  })
                )}
              </div>
            </div>

            <div className="mp-card">
              <div className="mp-card-head"><h3>Sensor Readings</h3></div>
              <div className="mp-sensor-list">
                {availableSensors.map((s) => {
                  const val = machine.sensors[s];
                  const isAlert = alertedSensors.has(s);
                  const isHigh = isSensorNearThreshold(s, val);
                  const d = sensorDelta(s);
                  const arrow = d > 0 ? "▲" : d < 0 ? "▼" : "–";
                  const dirClass = d > 0 ? "up" : d < 0 ? "down" : "flat";
                  return (
                    <div className="mp-sensor-row" key={s}>
                      <span className="mp-sensor-name">{s.charAt(0).toUpperCase() + s.slice(1)}</span>
                      <span className={`mp-sensor-dir ${dirClass}`}>{arrow}</span>
                      <span className={`mp-sensor-val num ${isAlert ? "bad" : "good"}`}>
                        {fmt(val, s === "vibration" ? 4 : 3)}
                      </span>
                      {isHigh && !isAlert && <span className="mp-sensor-flag">95ᵗʰ</span>}
                    </div>
                  );
                })}
              </div>
              <div className="mp-note">Values near historical 95th percentile are marked.</div>
            </div>
          </div>

          <div className="mp-card mp-span">
            <div className="mp-card-head">
              <h3>RUL Trend &amp; Degradation</h3>
              <div className="rul-legend">
                <span className="rul-chip ai">AI <b className="num">{fmt(aiHistory.at(-1),0)} h</b></span>
                <span className="rul-chip base">Baseline <b className="num">{fmt(baseHistory.at(-1),0)} h</b></span>
                <span className="rul-chip ghost">Δ7d <b className={`num ${aiDelta7<0?"neg":"pos"}`}>{fmt(aiDelta7,1)} h</b></span>
                <span className="rul-chip ghost">Slope/7 <b className={`num ${aiSlope<0?"neg":"pos"}`}>{fmt(aiSlope,2)} h/d</b></span>
              </div>
            </div>
            <div className="rul-chart-wrap">
              <RulTrendChart aiSeries={aiHistory} baseSeries={baseHistory} rmse={rmse} />
            </div>
            <div className="rul-foot">
              <span className="muted">Shaded band = ±RMSE; lower values mean earlier failure.</span>
            </div>
          </div>
        </div>

        <div ref={aiRef}>
          <AIAssistant
            context={{
              id: machine.id,
              name: machine.name,
              sensors: machine.sensors,
              status: machine.status,
              alerts: machine.alerts,
              rul: machine.rul,
              predicted_rul: machine.predicted_rul,
            }}
            onSend={async (text, ctx) => {
              const res = await fetch(`${API_BASE}/api/ai-chat`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: text, context: ctx }),
              });
              if (!res.ok) {
                const t = await res.text().catch(() => "");
                throw new Error(`AI chat HTTP ${res.status} ${t}`);
              }
              const data = await res.json();
              return data.reply || "🤖 No reply.";
            }}
          />
        </div>
      </div>
    </div>
  );
}
