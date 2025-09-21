import React, { useEffect, useMemo, useState } from "react";
import {
  MaterialReactTable,
  useMaterialReactTable,
} from "material-react-table";
import {
  Box,
  Button,
  IconButton,
  Tooltip,
  Drawer,
  Divider,
  Chip,
} from "@mui/material";
import { Edit, Delete, Download, Close } from "@mui/icons-material";
import MachineFormModal from "../components/MachineFormModal";
import ScheduleMaintenanceModal from "../components/ScheduleMaintenanceModal";
import "./machines.css";

const statusColors = {
  Normal: "#10b981",
  Unstable: "#facc15",
  Critical: "#ef4444",
  Unknown: "#94a3b8",
};
const statusOrder = { Unknown: 0, Normal: 1, Unstable: 2, Critical: 3 };
const fmt = (v, dp = 2) =>
  typeof v === "number" && isFinite(v) ? v.toFixed(dp) : "—";
const fmtH = (v) =>
  v !== undefined && v !== null && !isNaN(v) ? `${Math.round(v)} h` : "—";

export default function Machines({ machines }) {
  const [data, setData] = useState([]);
  const [modalOpen, setModalOpen] = useState(false);
  const [editRow, setEditRow] = useState(null);
  const [statusFilter, setStatusFilter] = useState("All");
  const [rowSelection, setRowSelection] = useState({});
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [drawerRow, setDrawerRow] = useState(null);
  const [moveOpen, setMoveOpen] = useState(false);
  const [newSection, setNewSection] = useState("");
  const [editingCell, setEditingCell] = useState(null);
  const [editingValue, setEditingValue] = useState("");
  const [flashId, setFlashId] = useState(null);

  // Maintenance storage keyed by machine id
  const [maintenanceById, setMaintenanceById] = useState({});
  const [schedOpen, setSchedOpen] = useState(false);

  useEffect(() => setData(machines || []), [machines]);

  const filteredData = useMemo(() => {
    const d =
      statusFilter === "All"
        ? data
        : data.filter((m) => (m.status || "Unknown") === statusFilter);
    return [...d].sort(
      (a, b) =>
        statusOrder[b.status || "Unknown"] - statusOrder[a.status || "Unknown"]
    );
  }, [data, statusFilter]);

  const counts = useMemo(() => {
    const c = { Normal: 0, Unstable: 0, Critical: 0, Unknown: 0 };
    data.forEach((m) => {
      c[m.status || "Unknown"] = (c[m.status || "Unknown"] || 0) + 1;
    });
    return c;
  }, [data]);

  const startInlineEdit = (row, key) => {
    setEditingCell({ id: row.original.id, key });
    setEditingValue(
      key === "status" ? row.original.status || "Unknown" : row.original.section || ""
    );
  };
  const commitInlineEdit = () => {
    if (!editingCell) return;
    setData((d) =>
      d.map((m) => {
        if (m.id !== editingCell.id) return m;
        const next = { ...m, [editingCell.key]: editingValue };
        if (editingCell.key === "status" && next.status === "Critical") {
          setFlashId(next.id);
          setTimeout(() => setFlashId(null), 1400);
        }
        return next;
      })
    );
    setEditingCell(null);
    setEditingValue("");
  };

  const handleAdd = () => {
    setEditRow(null);
    setModalOpen(true);
  };
  const handleEdit = (row) => {
    setEditRow(row.original);
    setModalOpen(true);
  };
  const handleDelete = (row) => {
    setData((d) => d.filter((m) => m.id !== row.original.id));
  };
  const handleSave = (newMachine) => {
    if (newMachine.id) {
      setData((d) => d.map((m) => (m.id === newMachine.id ? newMachine : m)));
      if (newMachine.status === "Critical") {
        setFlashId(newMachine.id);
        setTimeout(() => setFlashId(null), 1400);
      }
    } else {
      const maxId = data.length ? Math.max(...data.map((m) => m.id || 0)) : 0;
      const next = { ...newMachine, id: maxId + 1 };
      setData((d) => [...d, next]);
      if (next.status === "Critical") {
        setFlashId(next.id);
        setTimeout(() => setFlashId(null), 1400);
      }
    }
    setModalOpen(false);
  };

  const columns = useMemo(
    () => [
      {
        accessorKey: "name",
        header: "Name",
        size: 180,
        Cell: ({ row, cell }) => (
          <span
            className="mach-link"
            onClick={(e) => {
              e.stopPropagation();
              setDrawerRow(row.original);
              setDrawerOpen(true);
            }}
          >
            {cell.getValue()}
          </span>
        ),
      },
      {
        accessorKey: "status",
        header: "Status",
        size: 140,
        sortingFn: (a, b, id) =>
          statusOrder[a.getValue(id) || "Unknown"] -
          statusOrder[b.getValue(id) || "Unknown"],
        Cell: ({ row, cell }) => {
          const isEditing =
            editingCell &&
            editingCell.id === row.original.id &&
            editingCell.key === "status";
          const status = cell.getValue() || "Unknown";
          return isEditing ? (
            <select
              className="mach-select"
              value={editingValue}
              onChange={(e) => setEditingValue(e.target.value)}
              onBlur={commitInlineEdit}
              onKeyDown={(e) => e.key === "Enter" && commitInlineEdit()}
              autoFocus
            >
              {["Normal", "Unstable", "Critical", "Unknown"].map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          ) : (
            <span
              className="mach-pill"
              style={{ backgroundColor: statusColors[status] || "#94a3b8" }}
              onDoubleClick={(e) => {
                e.stopPropagation();
                startInlineEdit(row, "status");
              }}
            >
              {status === "Unstable" ? "⚠️ " : status === "Normal" ? "✅ " : ""}
              {status}
            </span>
          );
        },
      },
      {
        accessorKey: "section",
        header: "Location",
        size: 150,
        Cell: ({ row, cell }) => {
          const isEditing =
            editingCell &&
            editingCell.id === row.original.id &&
            editingCell.key === "section";
          return isEditing ? (
            <input
              className="mach-input"
              value={editingValue}
              onChange={(e) => setEditingValue(e.target.value)}
              onBlur={commitInlineEdit}
              onKeyDown={(e) => e.key === "Enter" && commitInlineEdit()}
              autoFocus
            />
          ) : (
            <span
              className="mach-cell-editable"
              onDoubleClick={(e) => {
                e.stopPropagation();
                startInlineEdit(row, "section");
              }}
            >
              {cell.getValue() || "—"}
            </span>
          );
        },
      },
      { accessorKey: "last_updated", header: "Last Updated", size: 200 },
      {
        accessorKey: "predicted_rul",
        header: "AI RUL (h)",
        size: 120,
        Cell: ({ cell }) => fmtH(cell.getValue()),
      },
      {
        accessorKey: "rul",
        header: "Baseline RUL (h)",
        size: 140,
        Cell: ({ cell }) => fmtH(cell.getValue()),
      },
      { accessorKey: "mae", header: "MAE", size: 90, Cell: ({ cell }) => fmt(cell.getValue(), 2) },
      { accessorKey: "rmse", header: "RMSE", size: 90, Cell: ({ cell }) => fmt(cell.getValue(), 2) },
      {
        id: "temperature",
        header: "Temp",
        size: 90,
        accessorFn: (row) => row.sensors?.temperature,
        Cell: ({ cell }) => fmt(cell.getValue(), 3),
      },
      {
        id: "vibration",
        header: "Vib",
        size: 90,
        accessorFn: (row) => row.sensors?.vibration,
        Cell: ({ cell }) => fmt(cell.getValue(), 4),
      },
      {
        id: "power",
        header: "Power",
        size: 90,
        accessorFn: (row) => row.sensors?.power,
        Cell: ({ cell }) => fmt(cell.getValue(), 3),
      },
      {
        id: "speed",
        header: "Speed",
        size: 90,
        accessorFn: (row) => row.sensors?.speed,
        Cell: ({ cell }) => fmt(cell.getValue(), 0),
      },
    ],
    [editingCell, editingValue]
  );

  const exportCSV = (rows) => {
    const headers = [
      "id","name","status","section","last_updated",
      "predicted_rul","rul","mae","rmse","temperature","vibration","power","speed",
    ];
    const lines = [headers.join(",")];
    rows.forEach((r) => {
      const m = r.original;
      const line = [
        m.id,
        `"${(m.name || "").replace(/"/g, '""')}"`,
        m.status || "",
        `"${(m.section || "").replace(/"/g, '""')}"`,
        m.last_updated || "",
        m.predicted_rul ?? "",
        m.rul ?? "",
        m.mae ?? "",
        m.rmse ?? "",
        m.sensors?.temperature ?? "",
        m.sensors?.vibration ?? "",
        m.sensors?.power ?? "",
        m.sensors?.speed ?? "",
      ].join(",");
      lines.push(line);
    });
    const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = "machines_filtered.csv"; a.click();
    URL.revokeObjectURL(url);
  };

  const table = useMaterialReactTable({
    columns,
    data: filteredData,
    enableRowActions: true,
    enableColumnResizing: true,
    enableStickyHeader: true,
    enableRowSelection: true,
    onRowSelectionChange: setRowSelection,
    state: { rowSelection },
    initialState: {
      density: "compact",
      pagination: { pageSize: 12, pageIndex: 0 },
      sorting: [{ id: "status", desc: true }],
      columnPinning: { left: ["mrt-row-select", "mrt-row-actions", "name", "status"] },
    },
    muiTablePaperProps: { elevation: 0, sx: { backgroundColor: "transparent", boxShadow: "none" } },
    muiTableContainerProps: {
      sx: {
        maxHeight: "calc(100vh - 230px)",
        overflow: "auto",
        borderRadius: "14px",
        border: "1px solid #223049",
        background: "linear-gradient(180deg, rgba(15,23,42,.85), rgba(8,15,32,.75))",
      },
    },
    muiTableHeadRowProps: { sx: { backgroundColor: "#0e162c" } },
    muiTableHeadCellProps: {
      sx: {
        color: "#cbd5e1",
        backgroundColor: "#0e162c",
        fontWeight: 800,
        textTransform: "uppercase",
        fontSize: "0.75rem",
        letterSpacing: "0.05em",
        borderBottom: "2px solid #223049",
      },
    },
    muiTableBodyRowProps: ({ row }) => ({
      onClick: () => { setDrawerRow(row.original); setDrawerOpen(true); },
      className: row.original.id === flashId ? "mach-row-flash" : undefined,
      sx: {
        cursor: "pointer",
        backgroundColor: row.index % 2 === 0 ? "#0f172a" : "#112034",
        transition: "background-color .15s ease",
        "&:hover": { backgroundColor: "#1e2b44 !important" },
      },
    }),
    muiTableBodyCellProps: {
      sx: { color: "#e6eefb", fontSize: "0.9rem", paddingY: "12px", borderBottom: "1px solid #162238" },
    },
    renderRowActions: ({ row }) => (
      <Box sx={{ display: "flex", gap: "0.25rem" }} onClick={(e) => e.stopPropagation()}>
        <Tooltip title="Edit">
          <IconButton size="small" onClick={() => handleEdit(row)}>
            <Edit fontSize="small" sx={{ color: "#38bdf8" }} />
          </IconButton>
        </Tooltip>
        <Tooltip title="Delete">
          <IconButton size="small" onClick={() => handleDelete(row)}>
            <Delete fontSize="small" sx={{ color: "#ef4444" }} />
          </IconButton>
        </Tooltip>
      </Box>
    ),
    renderTopToolbarCustomActions: ({ table }) => {
      const selected = table.getSelectedRowModel().flatRows;
      const filteredRows = table.getPrePaginationRowModel().rows;
      return (
        <div className="mach-toolbar">
          <button className="mach-btn" onClick={handleAdd}>+ Add Machine</button>
          <div className="mach-filters">
            {["All", "Normal", "Unstable", "Critical"].map((s) => (
              <button
                key={s}
                className={`mach-chip ${statusFilter === s ? "is-active" : ""}`}
                onClick={() => setStatusFilter(s)}
                style={s !== "All" ? { "--chip-bg": statusColors[s], "--chip-fg": "#04121e" } : undefined}
              >
                {s}
              </button>
            ))}
          </div>
          <div className="mach-kpis">
            <span className="mach-kpi" style={{ "--dot": statusColors.Normal }}>{counts.Normal} Normal</span>
            <span className="mach-kpi" style={{ "--dot": statusColors.Unstable }}>{counts.Unstable} Unstable</span>
            <span className="mach-kpi" style={{ "--dot": statusColors.Critical }}>{counts.Critical} Critical</span>
          </div>

          <div className="mach-spacer" />

          <div className="mach-bulk">
            <Button size="small" variant="outlined" disabled={!selected.length}
              onClick={() => {
                setData((d) => d.map((m) =>
                  selected.find((r) => r.original.id === m.id) ? { ...m, status: "Normal" } : m
                ));
                setRowSelection({});
              }}
              sx={{ color: "#10b981", borderColor: "#10b981", textTransform: "none" }}>
              Set Normal
            </Button>
            <Button size="small" variant="outlined" disabled={!selected.length}
              onClick={() => {
                setData((d) => d.map((m) =>
                  selected.find((r) => r.original.id === m.id) ? { ...m, status: "Unstable" } : m
                ));
                setRowSelection({});
              }}
              sx={{ color: "#facc15", borderColor: "#facc15", textTransform: "none", ml: 1 }}>
              Set Unstable
            </Button>
            <Button size="small" variant="outlined" disabled={!selected.length}
              onClick={() => {
                setData((d) => d.map((m) =>
                  selected.find((r) => r.original.id === m.id) ? { ...m, status: "Critical" } : m
                ));
                if (selected.length) { setFlashId(selected[0].original.id); setTimeout(() => setFlashId(null), 1400); }
                setRowSelection({});
              }}
              sx={{ color: "#ef4444", borderColor: "#ef4444", textTransform: "none", ml: 1 }}>
              Set Critical
            </Button>
            <Button size="small" variant="outlined" disabled={!selected.length}
              onClick={() => setMoveOpen(true)}
              sx={{ color: "#38bdf8", borderColor: "#38bdf8", textTransform: "none", ml: 1 }}>
              Move Section
            </Button>
            <Button size="small" variant="outlined" disabled={!selected.length}
              onClick={() => {
                const ids = new Set(selected.map((r) => r.original.id));
                setData((d) => d.filter((m) => !ids.has(m.id)));
                setRowSelection({});
              }}
              sx={{ color: "#ef4444", borderColor: "#ef4444", textTransform: "none", ml: 1 }}>
              Delete
            </Button>

            <Button size="small" startIcon={<Download />} onClick={() => exportCSV(filteredRows)}
              sx={{ color: "#e6eefb", borderColor: "#223049", textTransform: "none", ml: 2, border: "1px solid #223049" }}>
              Export CSV
            </Button>
          </div>
        </div>
      );
    },
    muiTopToolbarProps: { sx: { backgroundColor: "transparent", padding: "0 0 10px 0" } },
    muiBottomToolbarProps: { sx: { backgroundColor: "transparent", paddingTop: "10px", color: "#e6eefb" } },
    muiSearchTextFieldProps: {
      sx: {
        backgroundColor: "#0f172a",
        color: "#e6eefb",
        borderRadius: "10px",
        "& .MuiInputBase-input": { color: "#e6eefb" },
        "& fieldset": { borderColor: "#223049" },
        "&:hover fieldset": { borderColor: "#38bdf8" },
        "&.Mui-focused fieldset": { borderColor: "#38bdf8" },
      },
    },
  });

  const syntheticSpark = (v = 0, n = 24, jitter = 0.15) =>
    Array.from({ length: n }, (_, i) => ({
      t: i,
      y: Number(v) + (Math.sin(i / 3) + (Math.random() - 0.5) * 2) * (Math.abs(v) * jitter + 0.6),
    }));

  const alertDot = (sev = 0.5) =>
    sev >= 0.75 ? "#ef4444" : sev >= 0.4 ? "#facc15" : "#38bdf8";

  const getJobs = (id) => maintenanceById[id] || [];
  const removeJob = (id, jobId) =>
    setMaintenanceById((p) => ({
      ...p,
      [id]: (p[id] || []).filter((j) => j.id !== jobId),
    }));

  return (
    <div className="mach-page">
      <div className="mach-wrap">
        <div className="mach-header" />
        <div className="mach-tableCard">
          <MaterialReactTable table={table} />
        </div>
      </div>

      <MachineFormModal
        open={modalOpen}
        onClose={() => setModalOpen(false)}
        onSubmit={handleSave}
        initialData={editRow}
      />

      <ScheduleMaintenanceModal
        open={schedOpen}
        machine={drawerRow}
        onClose={() => setSchedOpen(false)}
        onSubmit={(job) => {
          setMaintenanceById((prev) => ({
            ...prev,
            [job.machineId]: [...(prev[job.machineId] || []), job],
          }));
          setSchedOpen(false);
        }}
      />

      <Drawer
        anchor="right"
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        PaperProps={{
          className: "mach-drawer",
          elevation: 0,
          sx: {
            width: 440,
            backgroundColor: "transparent",
            borderLeft: "1px solid var(--border)",
            boxShadow: "none",
          },
        }}
        ModalProps={{
          keepMounted: true,
          BackdropProps: {
            sx: {
              background:
                "radial-gradient(70% 50% at 60% 50%, rgba(2,6,23,.55), rgba(2,6,23,.85))",
              backdropFilter: "blur(4px)",
            },
          },
        }}
      >
        {drawerRow && (
          <div className="mach-drawer-inner">
            <div className="mach-drawer-head">
              <div className="mach-drawer-title">
                <div className="mach-avatar">{String(drawerRow.id ?? "?").slice(-2)}</div>
                <div>
                  <h2>{drawerRow.name}</h2>
                  <div className="mach-subtle">
                    {drawerRow.section || "—"} • Updated {new Date(drawerRow.last_updated || Date.now()).toLocaleTimeString()}
                  </div>
                </div>
              </div>

              <div className="mach-drawer-actions">
                <span
                  className="mach-pill"
                  style={{ backgroundColor: statusColors[drawerRow.status || "Unknown"] }}
                >
                  {drawerRow.status || "Unknown"}
                </span>
                <IconButton size="small" onClick={() => setDrawerOpen(false)} sx={{ ml: 1 }}>
                  <Close htmlColor="#9fb3d2" fontSize="small" />
                </IconButton>
              </div>
            </div>

            <Divider className="mach-div" />

            <div className="mach-drawer-grid">
              <div className="mach-drawer-kpi">
                <div className="k-label">AI RUL</div>
                <div className="k-value">{fmtH(drawerRow.predicted_rul)}</div>
              </div>
              <div className="mach-drawer-kpi">
                <div className="k-label">Baseline RUL</div>
                <div className="k-value">{fmtH(drawerRow.rul)}</div>
              </div>
              <div className="mach-drawer-kpi">
                <div className="k-label">MAE</div>
                <div className="k-value">{fmt(drawerRow.mae, 2)}</div>
              </div>
              <div className="mach-drawer-kpi">
                <div className="k-label">RMSE</div>
                <div className="k-value">{fmt(drawerRow.rmse, 2)}</div>
              </div>
            </div>

            <div className="mach-drawer-section">
              <div className="mach-drawer-sub">Scheduled</div>
              <div className="sched-list">
                {getJobs(drawerRow.id).length === 0 ? (
                  <div className="mach-subtle">No upcoming maintenance.</div>
                ) : (
                  getJobs(drawerRow.id).map((j) => (
                    <div key={j.id} className="sched-item">
                      <div className="sched-meta">
                        <span className="dot" />
                        <div className="sched-text">
                          <div className="t1">{j.name}</div>
                          <div className="t2">
                            {j.date} • {j.time}
                          </div>
                        </div>
                      </div>
                      <button
                        className="sched-xbtn"
                        onClick={() => removeJob(drawerRow.id, j.id)}
                        aria-label="Cancel schedule"
                        title="Cancel"
                      >
                        ×
                      </button>
                    </div>
                  ))
                )}
              </div>
            </div>

            {(drawerRow.alerts?.length || 0) > 0 && (
              <div className="mach-drawer-section">
                <div className="mach-drawer-sub">Open Alerts</div>
                <div className="mach-alerts">
                  {drawerRow.alerts.slice(0, 4).map((a, i) => (
                    <div key={i} className="mach-alert">
                      <span className="dot" style={{ background: alertDot(a.severity) }} />
                      <div className="text">
                        <div className="t1">{a.text}</div>
                        <div className="t2">
                          <span className="intent">{a.intent}</span>
                          <span className="sep">•</span>
                          <span className="meta">{a.cause}</span>
                        </div>
                      </div>
                      <Chip size="small" label={`${Math.round((a.confidence ?? 0) * 100)}%`} />
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="mach-drawer-section">
              <div className="mach-drawer-sub">Sensors</div>
              <div className="mach-sensors">
                {[
                  { k: "temperature", label: "Temp" },
                  { k: "vibration", label: "Vibration" },
                  { k: "power", label: "Power" },
                  { k: "speed", label: "Speed" },
                ].map((s) => {
                  const val = Number(drawerRow.sensors?.[s.k]);
                  const warn = isFinite(val) && Math.abs(val) > 1.5;
                  return (
                    <div key={s.k} className={`sensor-card ${warn ? "is-warn" : ""}`}>
                      <div className="sensor-head">
                        <span>{s.label}</span>
                        <Chip size="small" label={fmt(val, 2)} />
                      </div>
                      <svg viewBox="0 0 100 28" className="spark">
                        {syntheticSpark(val || 0.8).map((p, i, arr) => {
                          if (i === 0) return null;
                          const x1 = ((i - 1) / (arr.length - 1)) * 100;
                          const y1 = 14 - (arr[i - 1].y % 14);
                          const x2 = (i / (arr.length - 1)) * 100;
                          const y2 = 14 - (arr[i].y % 14);
                          return <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} />;
                        })}
                      </svg>
                    </div>
                  );
                })}
              </div>
            </div>

            <div className="mach-drawer-section">
              <div className="mach-drawer-sub">Quick Actions</div>
              <div className="mach-actions">
                <button
                  className="mach-action-btn"
                  onClick={() => setSchedOpen(true)}
                >
                  Schedule Maintenance
                </button>
                <button
                  className="mach-action-btn"
                  onClick={() => {
                    setData((d) =>
                      d.map((m) => (m.id === drawerRow.id ? { ...m, status: "Normal" } : m))
                    );
                    setDrawerRow((r) => ({ ...r, status: "Normal" }));
                  }}
                >
                  Mark as Normal
                </button>
                <button
                  className="mach-action-btn"
                  onClick={() => {
                    /* acknowledge alerts */
                  }}
                >
                  Acknowledge Alert
                </button>
              </div>
            </div>

            <div className="mach-drawer-section">
              <div className="mach-drawer-sub">Notes</div>
              <textarea className="mach-notes" placeholder="Add maintenance notes..." />
            </div>
          </div>
        )}
      </Drawer>
    </div>
  );
}
