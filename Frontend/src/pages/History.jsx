import React, { useEffect, useMemo, useState } from "react";
import {
  Memory,
  WarningAmber,
  Psychology,
  Description,
  Download,
  Close,
} from "@mui/icons-material";
import "./History.css";

const API_BASE =
  import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "") || "http://localhost:8000";

const typeMeta = {
  "Status Change": {
    color: "#38bdf8",
    icon: <Memory sx={{ fontSize: 20, color: "#38bdf8" }} />,
  },
  "AI Suggestion": {
    color: "#c084fc",
    icon: <Psychology sx={{ fontSize: 20, color: "#c084fc" }} />,
  },
  "Anomaly Detected": {
    color: "#facc15",
    icon: <WarningAmber sx={{ fontSize: 20, color: "#facc15" }} />,
  },
  "Report Generated": {
    color: "#60a5fa",
    icon: <Description sx={{ fontSize: 20, color: "#60a5fa" }} />,
  },
};

const labelForDate = (d) => {
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const target = new Date(d);
  target.setHours(0, 0, 0, 0);
  const diff = Math.round((today - target) / 86400000);
  if (diff === 0) return "Today";
  if (diff === 1) return "Yesterday";
  return target.toLocaleDateString(undefined, {
    day: "2-digit",
    month: "short",
    year: "numeric",
  });
};

export default function History() {
  const [events, setEvents] = useState([]);
  const [query, setQuery] = useState("");
  const [type, setType] = useState("All");
  const [range, setRange] = useState("7d");
  const [density, setDensity] = useState("cozy"); // 'cozy' | 'compact'

  useEffect(() => {
    const load = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/history`, { mode: "cors" });
        const json = await res.json();
        setEvents(Array.isArray(json) ? json : []);
      } catch (e) {
        console.error("Failed to fetch history:", e);
      }
    };
    load();
  }, []);

  const typeCounts = useMemo(() => {
    const c = {
      "Status Change": 0,
      "AI Suggestion": 0,
      "Anomaly Detected": 0,
      "Report Generated": 0,
    };
    events.forEach((e) => {
      if (c[e.type] !== undefined) c[e.type]++;
    });
    return c;
  }, [events]);

  const filtered = useMemo(() => {
    const now = Date.now();
    const msRange =
      range === "24h"
        ? 24 * 3600e3
        : range === "7d"
        ? 7 * 24 * 3600e3
        : range === "30d"
        ? 30 * 24 * 3600e3
        : Number.POSITIVE_INFINITY;

    const q = query.trim().toLowerCase();
    return events
      .filter((e) => {
        if (type !== "All" && e.type !== type) return false;
        if (msRange !== Infinity) {
          const t = Date.parse(e.timestamp);
          if (!isNaN(t) && now - t > msRange) return false;
        }
        if (!q) return true;
        const hay = `${e.machine || ""} ${e.detail || ""} ${e.type || ""}`.toLowerCase();
        return hay.includes(q);
      })
      .sort(
        (a, b) =>
          (Date.parse(b.timestamp || 0) || 0) -
          (Date.parse(a.timestamp || 0) || 0)
      );
  }, [events, type, range, query]);

  const grouped = useMemo(() => {
    const map = new Map();
    for (const e of filtered) {
      const key = new Date(e.timestamp || Date.now()).toISOString().slice(0, 10);
      if (!map.has(key)) map.set(key, []);
      map.get(key).push(e);
    }
    return Array.from(map.entries())
      .sort((a, b) => (a[0] < b[0] ? 1 : -1))
      .map(([k, items]) => ({ dateKey: k, label: labelForDate(k), items }));
  }, [filtered]);

  const exportCSV = () => {
    const headers = ["timestamp", "machine", "type", "detail"];
    const lines = [headers.join(",")];
    filtered.forEach((e) => {
      const line = [
        `"${(e.timestamp || "").replace(/"/g, '""')}"`,
        `"${(e.machine || "").replace(/"/g, '""')}"`,
        `"${(e.type || "").replace(/"/g, '""')}"`,
        `"${(e.detail || "").replace(/"/g, '""')}"`,
      ].join(",");
      lines.push(line);
    });
    const blob = new Blob([lines.join("\n")], {
      type: "text/csv;charset=utf-8;",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "history_filtered.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className={`hist ${density === "compact" ? "is-compact" : ""}`}>
      <div className="hist-board">
        <div className="hist-toolbar-grid">
          <div className="hist-left">
            <div className="hist-counts">
              <span className="pill soft">{events.length} total</span>
              <span className="pill c-cyan">
                {typeCounts["Status Change"]} status
              </span>
              <span className="pill c-orange">
                {typeCounts["Anomaly Detected"]} anomalies
              </span>
              <span className="pill c-violet">
                {typeCounts["AI Suggestion"]} AI
              </span>
              <span className="pill c-blue">
                {typeCounts["Report Generated"]} reports
              </span>
            </div>
          </div>

          <div className="hist-search">
            <div className="hist-search-inner">
              <input
                aria-label="Search history"
                placeholder="Search machine, detail, or type…"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
              />
              {query && (
                <button
                  className="hist-clear"
                  aria-label="Clear search"
                  onClick={() => setQuery("")}
                >
                  <Close sx={{ fontSize: 16 }} />
                </button>
              )}
            </div>
          </div>

          <div className="hist-controls">
            <div className="chip-row" role="tablist" aria-label="Type filter">
              {[
                "All",
                "Status Change",
                "Anomaly Detected",
                "AI Suggestion",
                "Report Generated",
              ].map((t) => (
                <button
                  key={t}
                  className={`chip ${type === t ? "is-active" : ""}`}
                  onClick={() => setType(t)}
                  role="tab"
                  aria-pressed={type === t}
                  title={`Filter: ${t}`}
                >
                  {t === "Status Change"
                    ? "Status"
                    : t === "Anomaly Detected"
                    ? "Anomaly"
                    : t === "AI Suggestion"
                    ? "AI"
                    : t === "Report Generated"
                    ? "Reports"
                    : "All"}
                </button>
              ))}
            </div>

            <div className="chip-row" role="tablist" aria-label="Time range">
              {["24h", "7d", "30d", "all"].map((r) => (
                <button
                  key={r}
                  className={`chip ${range === r ? "is-active" : ""}`}
                  onClick={() => setRange(r)}
                  role="tab"
                  aria-pressed={range === r}
                  title={`Range: ${r.toUpperCase()}`}
                >
                  {r.toUpperCase()}
                </button>
              ))}
            </div>

            <div className="chip-row" role="tablist" aria-label="Density">
              <button
                className={`chip ${density === "cozy" ? "is-active" : ""}`}
                onClick={() => setDensity("cozy")}
                title="Cozy density"
              >
                Cozy
              </button>
              <button
                className={`chip ${density === "compact" ? "is-active" : ""}`}
                onClick={() => setDensity("compact")}
                title="Compact density"
              >
                Compact
              </button>
            </div>

           
          </div>
        </div>

        <div className="hist-list">
          {grouped.map((g) => (
            <div key={g.dateKey} className="date-group">
              <div style={{display:"flex", justifyContent:"space-between"}}>
              <div className="date-header">{g.label}</div>
              <button className="btn-export" onClick={exportCSV} aria-label="Export CSV">
              <Download sx={{ fontSize: 18 }} />
              Export ({filtered.length})
            </button>
            </div>

              <div className="date-body">
                {g.items.map((e, i) => {
                  const meta =
                    typeMeta[e.type] || {
                      color: "#94a3b8",
                      icon: <Memory sx={{ fontSize: 20, color: "#94a3b8" }} />,
                    };
                  return (
                    <div key={i} className="event-row">
                      <div
                        className="ev-icon"
                        style={{
                          boxShadow: `0 0 0 2px ${meta.color}22 inset`,
                          background:
                            "linear-gradient(180deg,#0b1223,#0e1a2e)",
                        }}
                        aria-hidden
                      >
                        {meta.icon}
                      </div>

                      <div className="ev-main">
                        <div className="ev-top">
                          <div className="ev-machine">{e.machine || "—"}</div>
                          <div className="ev-time">
                            {e.timestamp
                              ? new Date(e.timestamp).toLocaleString()
                              : "—"}
                          </div>
                        </div>
                        <div className="ev-detail">{e.detail || "—"}</div>
                      </div>

                      <div
                        className="ev-type"
                        style={{ color: meta.color, borderColor: meta.color }}
                        title={e.type}
                      >
                        {e.type}
                      </div>
                    </div>
                  );
                })}
              </div>
              
            </div>
          ))}

          {grouped.length === 0 && (
            <div className="empty">No events match your filters.</div>
          )}
        </div>
      </div>
    </div>
  );
}
