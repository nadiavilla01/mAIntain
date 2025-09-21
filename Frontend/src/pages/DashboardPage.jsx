import React, { useEffect, useMemo, useState } from "react";
import MachineCard from "../components/MachineCard";
import { FiGrid, FiFilm, FiSearch } from "react-icons/fi";
import "./DashboardPage.css";

const statusOrder = { Critical: 3, Unstable: 2, Normal: 1, Unknown: 0 };

export default function DashboardPage({ machines = [] }) {
  const [status, setStatus] = useState("All");
  const [section, setSection] = useState("All");
  const [sortBy, setSortBy] = useState("status");
  const [density, setDensity] = useState("cozy");
  const [view, setView] = useState("grid");
  const [query, setQuery] = useState("");
  const [pinned, setPinned] = useState(() => {
    try { return new Set(JSON.parse(localStorage.getItem("pinned_machines") || "[]")); }
    catch { return new Set(); }
  });

  useEffect(() => {
    try { localStorage.setItem("pinned_machines", JSON.stringify([...pinned])); } catch {}
  }, [pinned]);

  const sections = useMemo(() => {
    const s = Array.from(new Set(machines.map((m) => m.section).filter(Boolean)));
    s.sort();
    return ["All", ...s];
  }, [machines]);

  const counts = useMemo(() => {
    const c = { Normal: 0, Unstable: 0, Critical: 0, Unknown: 0 };
    machines.forEach((m) => { c[m.status || "Unknown"] = (c[m.status || "Unknown"] || 0) + 1; });
    return c;
  }, [machines]);

  const filtered = useMemo(() => {
    let rows = machines;
    if (query.trim()) {
      const q = query.toLowerCase();
      rows = rows.filter(
        (m) =>
          m.name.toLowerCase().includes(q) ||
          (m.status || "").toLowerCase().includes(q) ||
          String(m.id).includes(q)
      );
    }
    if (status !== "All") rows = rows.filter((m) => (m.status || "Unknown") === status);
    if (section !== "All") rows = rows.filter((m) => m.section === section);
    if (sortBy === "status") {
      rows = [...rows].sort((a, b) => statusOrder[b.status || "Unknown"] - statusOrder[a.status || "Unknown"]);
    } else if (sortBy === "rulAsc") {
      rows = [...rows].sort((a, b) => (a.predicted_rul ?? a.rul ?? 0) - (b.predicted_rul ?? b.rul ?? 0));
    } else if (sortBy === "updatedDesc") {
      rows = [...rows].sort((a, b) => new Date(b.last_updated || 0) - new Date(a.last_updated || 0));
    }
    return rows;
  }, [machines, query, status, section, sortBy]);

  const pinnedList = filtered.filter((m) => pinned.has(m.id));
  const restList = filtered.filter((m) => !pinned.has(m.id));

  const [index, setIndex] = useState(0);
  useEffect(() => {
    if (view !== "carousel" || restList.length === 0) return;
    const id = setInterval(() => setIndex((i) => (i + 1) % restList.length), 5000);
    return () => clearInterval(id);
  }, [view, restList.length]);

  return (
    <div className="dash">
      <div className="dash-toolbar dash-toolbar-grid">
        <div className="tb-left">
          <div className="dash-summary">
            <span className="sum-pill sum-green">{counts.Normal} Normal</span>
            <span className="sum-pill sum-yellow">{counts.Unstable} Unstable</span>
            <span className="sum-pill sum-red">{counts.Critical} Critical</span>
          </div>
        </div>

        <div className="tb-search">
          <FiSearch className="dash-search-icon" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search by ID, name, or status..."
          />
        </div>

        <div className="tb-right">
          <div className="chip-row">
            {["All", "Normal", "Unstable", "Critical"].map((s) => (
              <button
                key={s}
                className={`chip ${status === s ? "is-active" : ""}`}
                onClick={() => setStatus(s)}
                data-color={s.toLowerCase()}
              >
                {s}
              </button>
            ))}
          </div>

          <select className="sel" value={section} onChange={(e) => setSection(e.target.value)}>
            {sections.map((s) => (
              <option key={s} value={s}>{s === "All" ? "All Sections" : s}</option>
            ))}
          </select>

          <select className="sel" value={sortBy} onChange={(e) => setSortBy(e.target.value)}>
            <option value="status">Sort: Status</option>
            <option value="rulAsc">Sort: RUL ↑</option>
            <option value="updatedDesc">Sort: Last Updated ↓</option>
          </select>

          <div className="density">
            <button
              className={`chip ${density === "cozy" ? "is-active" : ""}`}
              onClick={() => setDensity("cozy")}
            >Cozy</button>
            <button
              className={`chip ${density === "compact" ? "is-active" : ""}`}
              onClick={() => setDensity("compact")}
            >Compact</button>
          </div>

          <div className="seg">
            <button
              className={`seg-btn ${view === "grid" ? "is-active" : ""}`}
              onClick={() => setView("grid")}
              aria-label="Grid view"
            >
              <FiGrid />
            </button>
            <button
              className={`seg-btn ${view === "carousel" ? "is-active" : ""}`}
              onClick={() => setView("carousel")}
              aria-label="Carousel view"
            >
              <FiFilm />
            </button>
          </div>
        </div>
      </div>

      {pinnedList.length > 0 && (
        <div className="pinned-strip">
          {pinnedList.map((m) => (
            <MachineCard
              key={`p-${m.id}`}
              {...m}
              density={density}
              pinned
              onTogglePin={() =>
                setPinned((s) => {
                  const n = new Set(s);
                  n.delete(m.id);
                  return n;
                })
              }
            />
          ))}
        </div>
      )}

      {view === "carousel" && restList.length > 0 ? (
        <div className="carousel">
          <button
            className="nav prev"
            onClick={() => setIndex((i) => (i - 1 + restList.length) % restList.length)}
            aria-label="Previous"
          >
            ◀
          </button>

          <div className="carousel-row">
            {[-1, 0, 1].map((offset) => {
              const i = (index + offset + restList.length) % restList.length;
              const item = restList[i];
              const big = offset === 0;
              return (
                <div key={item.id} className={`car-item ${big ? "is-big" : ""}`}>
                  <MachineCard
                    {...item}
                    large={big}
                    density="cozy"
                    onTogglePin={() =>
                      setPinned((s) => {
                        const n = new Set(s);
                        if (n.has(item.id)) n.delete(item.id);
                        else n.add(item.id);
                        return n;
                      })
                    }
                  />
                </div>
              );
            })}
          </div>

          <button
            className="nav next"
            onClick={() => setIndex((i) => (i + 1) % restList.length)}
            aria-label="Next"
          >
            ▶
          </button>
        </div>
      ) : (
        <div className={`grid ${density === "compact" ? "is-compact" : ""}`}>
          {restList.map((m) => (
            <MachineCard
              key={m.id}
              {...m}
              density={density}
              onTogglePin={() =>
                setPinned((s) => {
                  const n = new Set(s);
                  if (n.has(m.id)) n.delete(m.id);
                  else n.add(m.id);
                  return n;
                })
              }
              pinned={pinned.has(m.id)}
            />
          ))}
        </div>
      )}
    </div>
  );
}
