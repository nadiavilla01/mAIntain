import React, { useEffect, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip as ChartTooltip,
  ResponsiveContainer,
} from "recharts";
import { FiActivity, FiAlertTriangle, FiClock } from "react-icons/fi";
import "./insights.css";

const API_BASE =
  import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "") || "http://localhost:8000";

export default function Insights() {
  const [stats, setStats] = useState({
    anomaliesThisWeek: 0,
    criticalMachines: 0,
    avgRUL: 0,
  });
  const [rulTrend, setRulTrend] = useState([]);
  const [warnings, setWarnings] = useState([]);
  const [chat, setChat] = useState([
    { sender: "user", text: "Show RUL predictions for Mixer" },
    { sender: "ai", text: "Estimated 28 hours remaining. 📉" },
    { sender: "user", text: "What’s the likely cause of this anomaly?" },
    {
      sender: "ai",
      text:
        "The Robot Arm’s anomalous temperature is likely due to a cooling system fault. 🢨",
    },
  ]);
  const [message, setMessage] = useState("");
  const [typing, setTyping] = useState(false);

  // NEW: page loading state
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchInsights = async () => {
      const MIN_SPIN_MS = 900; // avoid flicker on fast responses
      const start = Date.now();
      setLoading(true);
      try {
        const res = await fetch(`${API_BASE}/insights`, { mode: "cors" });
        const json = await res.json();
        setStats({
          anomaliesThisWeek: json?.anomalies_this_week ?? 0,
          criticalMachines: json?.critical_count ?? 0,
          avgRUL: Math.round(json?.avg_rul ?? 0),
        });
        setRulTrend(json?.rul_trend ?? []);
        setWarnings(json?.warnings ?? []);
      } catch {
        /* noop */
      } finally {
        const elapsed = Date.now() - start;
        const wait = Math.max(0, MIN_SPIN_MS - elapsed);
        setTimeout(() => setLoading(false), wait);
      }
    };
    fetchInsights();
  }, []);

  const sendMessage = async (text) => {
    const userText = text.trim();
    if (!userText || typing) return;

    setChat((c) => [...c, { sender: "user", text: userText }]);
    setMessage("");
    setTyping(true);

    try {
      const res = await fetch(`${API_BASE}/insights/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userText }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setChat((c) => [...c, { sender: "ai", text: data?.reply || "🧠 No reply." }]);
    } catch {
      setChat((c) => [
        ...c,
        { sender: "ai", text: "⚠️ Chat backend not reachable or misconfigured." },
      ]);
    } finally {
      setTyping(false);
    }
  };

  const handleSend = () => sendMessage(message);
  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="ins-page">
      {loading && (
        <div className="ins-loader" aria-busy="true" aria-live="polite">
          <div className="ins-loader-card">
            <div className="ins-spinner" />
            <div className="ins-loader-title">Compiling insights…</div>
            <div className="ins-loader-sub">Fetching metrics & models</div>
          </div>
        </div>
      )}

      <div className={`ins-wrap ${loading ? "is-loading" : ""}`}>
        <div className="ins-stats">
          <div className="ins-stat-card" data-variant="anoms">
            <div className="ins-stat-top">
              <span className="ins-kpi-icon"><FiActivity size={18} /></span>
              <div className="ins-stat-body">
                <div className="ins-stat-label">Anomalies This Week</div>
                <div className="ins-stat-value">{stats.anomaliesThisWeek}</div>
              </div>
            </div>
          </div>

          <div className="ins-stat-card" data-variant="critical">
            <div className="ins-stat-top">
              <span className="ins-kpi-icon"><FiAlertTriangle size={18} /></span>
              <div className="ins-stat-body">
                <div className="ins-stat-label">Critical Machines</div>
                <div className="ins-stat-value">{stats.criticalMachines}</div>
              </div>
            </div>
          </div>

          <div className="ins-stat-card" data-variant="rul">
            <div className="ins-stat-top">
              <span className="ins-kpi-icon"><FiClock size={18} /></span>
              <div className="ins-stat-body">
                <div className="ins-stat-label">Avg RUL</div>
                <div className="ins-stat-value">{stats.avgRUL}h</div>
              </div>
            </div>
          </div>
        </div>

        <div className="ins-grid">
          <div className="ins-left">
            <div className="ins-card">
              <div className="ins-card-title">Overall System Health</div>
              <div className="ins-healthbar">
                <div className="g g1" />
                <div className="g g2" />
                <div className="g g3" />
              </div>
              <div className="ins-legend">Green = Normal · Yellow = Unstable · Red = Critical</div>
            </div>

            <div className="ins-card ins-scroll" style={{ minHeight: 0 }}>
              <div className="ins-card-title">🔍 Recent AI Warnings</div>
              <ul className="ins-warnings-list">
                {warnings.length === 0 ? (
                  <li className="ins-empty">No recent warnings</li>
                ) : (
                  warnings.map((w, i) => (
                    <li key={i} className="ins-warning">
                      <span className="ins-machine" style={{ color: w.color || "#facc15" }}>
                        {w.machine}
                      </span>
                      <span className="ins-bullet">•</span>
                      <span className="ins-msg">{w.message}</span>
                    </li>
                  ))
                )}
              </ul>
            </div>

            <div className="ins-card">
              <div className="ins-card-title">📈 Predicted RUL Trend</div>
              <div className="ins-chart">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={rulTrend}>
                    <XAxis dataKey="time" stroke="#94a3b8" tick={{ fontSize: 12 }} />
                    <YAxis stroke="#94a3b8" tick={{ fontSize: 12 }} />
                    <ChartTooltip />
                    <Line type="monotone" dataKey="rul" stroke="#38bdf8" strokeWidth={2} dot />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          <div className="ins-right">
            <div className="ins-card ins-chat">
              <div className="ins-card-title">💬 Live Chat</div>
              <div className="ins-sub">Ask about your data or anomalies.</div>

              <div className="chat-messages ins-scroll">
                {chat.map((m, i) => (
                  <div key={i} className={`chat-row ${m.sender === "user" ? "is-user" : "is-ai"}`}>
                    <div className={`chat-bubble ${m.sender}`}>{m.text}</div>
                  </div>
                ))}
                {typing && (
                  <div className="chat-row is-ai">
                    <div className="chat-bubble ai">Thinking…</div>
                  </div>
                )}
              </div>

              <div className="chat-input">
                <input
                  placeholder={loading ? "Loading insights…" : "Type a message..."}
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  onKeyDown={handleKeyDown}
                  disabled={loading}
                />
                <button onClick={handleSend} disabled={loading || !message.trim() || typing}>➤</button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
