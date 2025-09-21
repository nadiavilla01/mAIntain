import React, { useEffect, useRef, useState } from "react";
import "./ScheduleMaintenance.css";

export default function ScheduleMaintenanceModal({
  open,
  machine,
  onClose,
  onSubmit,
}) {
  const nameRef = useRef(null);

  const today = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  const dfltDate = `${today.getFullYear()}-${pad(today.getMonth() + 1)}-${pad(
    today.getDate()
  )}`;
  const dfltTime = `${pad(Math.min(9, today.getHours()))}:${pad(0)}`;

  const [name, setName] = useState("");
  const [date, setDate] = useState(dfltDate);
  const [time, setTime] = useState(dfltTime);

  useEffect(() => {
    if (open) {
      setName(`Maintenance – ${machine?.name ?? "Machine"}`);
      setTimeout(() => nameRef.current?.focus(), 50);
    }
  }, [open, machine]);

  useEffect(() => {
    const onEsc = (e) => e.key === "Escape" && open && onClose?.();
    window.addEventListener("keydown", onEsc);
    return () => window.removeEventListener("keydown", onEsc);
  }, [open, onClose]);

  if (!open) return null;

  const submit = (e) => {
    e?.preventDefault?.();
    if (!name.trim() || !date || !time) return;
    const localISO = new Date(`${date}T${time}`).toISOString();
    onSubmit?.({
      id: `${Date.now()}`,
      machineId: machine?.id,
      name: name.trim(),
      date,
      time,
      whenISO: localISO,
      createdAt: new Date().toISOString(),
    });
  };

  return (
    <div
      className="sched-overlay"
      onMouseDown={(e) => {
        if (e.target.classList.contains("sched-overlay")) onClose?.();
      }}
    >
      <form className="sched-card" onSubmit={submit}>
        <div className="sched-head">
          <div className="sched-title">
            <span className="dot" /> Schedule Maintenance
          </div>
          <button type="button" className="sched-x" onClick={onClose} aria-label="Close">×</button>
        </div>

        <div className="sched-sub">
          {machine?.name ? (
            <>You’re scheduling for <b>{machine.name}</b>.</>
          ) : (
            <>Create a maintenance entry.</>
          )}
        </div>

        <div className="sched-grid">
          <label className="sched-field">
            <span>Title</span>
            <input
              ref={nameRef}
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g., Quarterly inspection"
              required
            />
          </label>

          <label className="sched-field">
            <span>Date</span>
            <input
              type="date"
              value={date}
              onChange={(e) => setDate(e.target.value)}
              required
            />
          </label>

          <label className="sched-field">
            <span>Time</span>
            <input
              type="time"
              value={time}
              onChange={(e) => setTime(e.target.value)}
              required
            />
          </label>
        </div>

        <div className="sched-actions">
          <button type="button" className="btn-ghost" onClick={onClose}>
            Cancel
          </button>
          <button type="submit" className="btn-primary">
            Schedule
          </button>
        </div>
      </form>
    </div>
  );
}
