import React, { useState, useEffect, useRef } from "react";
import { Routes, Route, Navigate, useLocation } from "react-router-dom";

import Sidebar from "./components/Sidebar";
import Topbar from "./components/Topbar";

import DashboardPage from "./pages/DashboardPage";
import MachinePage from "./pages/MachinePage";
import Machines from "./pages/Machines";
import Insights from "./pages/Insights";
import History from "./pages/History";
import FaultDetectionPage from "./pages/FaultDetectionPage";
import Settings from "./pages/Settings";

import Login from "./pages/Login";
import Boot from "./pages/Boot"; // shows Loader, then redirects to "/"

import Loader from "./components/Loader";

const API_BASE =
  import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "") || "http://localhost:8000";

const formatTimeAgo = (isoTime) => {
  if (!isoTime) return "–";
  try {
    const then = new Date(isoTime);
    const now = new Date();
    const diffMin = Math.floor((now - then) / 60000);
    return diffMin < 1 ? "Just now" : `${diffMin}m ago`;
  } catch {
    return "–";
  }
};

export default function App() {
  const [loading, setLoading] = useState(true);
  const [machines, setMachines] = useState([]);
  const [error, setError] = useState(null);
  const [mode, setMode] = useState("start");
  const abortRef = useRef(null);
  const mountedRef = useRef(true);

  // auth state (no backend yet)
  const authed = !!localStorage.getItem("auth_token");
  const location = useLocation();
  const isAuthScreen = location.pathname === "/login" || location.pathname === "/boot";

  const fetchMachines = async (signal) => {
    const res = await fetch(`${API_BASE}/machines?mode=${mode}`, {
      mode: "cors",
      signal,
      headers: { "cache-control": "no-cache" },
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const json = await res.json();
    return Array.isArray(json) ? json : [];
  };

  useEffect(() => {
    mountedRef.current = true;

    if (!authed) {
      setLoading(false);
      return;
    }

    const load = async () => {
      try {
        if (abortRef.current) abortRef.current.abort();
        const controller = new AbortController();
        abortRef.current = controller;
        const data = await fetchMachines(controller.signal);
        if (!mountedRef.current) return;
        const enriched = data.map((m) => ({
          ...m,
          last_updated_ago: formatTimeAgo(m.last_updated),
        }));
        setMachines(enriched);
        setError(null);
      } catch (err) {
        if (!mountedRef.current) return;
        setError(err.message || "Fetch error");
      } finally {
        if (mountedRef.current) setLoading(false);
      }
    };

    load();
    const id = setInterval(load, 10000);
    return () => {
      mountedRef.current = false;
      clearInterval(id);
      if (abortRef.current) abortRef.current.abort();
    };
  }, [mode, authed]);

  if (isAuthScreen) {
    return (
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/boot" element={<Boot />} />
        <Route path="*" element={<Navigate to="/login" replace />} />
      </Routes>
    );
  }

  if (authed && loading) return <Loader />;

  return (
    <div style={{ display: "flex" }}>
      <Sidebar />
      <div
        style={{
          marginLeft: "200px",
          width: "100%",
          background: "#030303",
          height: "100vh",
          display: "flex",
          flexDirection: "column",
          overflow: "hidden",
        }}
      >
        {error && (
          <div
            style={{
              background: "#3b1111",
              border: "1px solid #7f1d1d",
              color: "#fecaca",
              padding: "8px 16px",
              fontSize: 14,
              margin: "8px 16px",
              borderRadius: 10,
            }}
          >
            {error}
          </div>
        )}
        <div style={{ flex: 1, overflowY: "auto" }}>
          <Routes>
            {/* protected routes */}
            <Route
              path="/"
              element={authed ? <DashboardPage machines={machines} /> : <Navigate to="/login" replace />}
            />
            <Route
              path="/machines"
              element={authed ? <Machines machines={machines} /> : <Navigate to="/login" replace />}
            />
            <Route
              path="/machine/:id"
              element={authed ? <MachinePage machines={machines} /> : <Navigate to="/login" replace />}
            />
            <Route
              path="/insights"
              element={authed ? <Insights /> : <Navigate to="/login" replace />}
            />
            <Route
              path="/history"
              element={authed ? <History /> : <Navigate to="/login" replace />}
            />
            <Route
              path="/settings"
              element={authed ? <Settings /> : <Navigate to="/login" replace />}
            />
            <Route
              path="/fault-detection"
              element={authed ? <FaultDetectionPage /> : <Navigate to="/login" replace />}
            />

            <Route path="/login" element={<Navigate to="/" replace />} />
            <Route path="/boot" element={<Navigate to="/" replace />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </div>
      </div>
    </div>
  );
}
