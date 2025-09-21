// src/pages/Settings.jsx
import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  FiUser,
  FiImage,
  FiMapPin,
  FiBriefcase,
  FiAtSign,
  FiPhone,
  FiGlobe,
  FiClock,
  FiSave,
  FiRotateCcw,
  FiShield,
  FiLock,
} from "react-icons/fi";
import "./Settings.css";

const STORAGE_KEY = "app_settings_v2";

const DEFAULT_FORM = {
  // profile
  avatar: "",
  displayName: "",
  username: "",
  email: "",
  phone: "",
  // company
  companyName: "",
  department: "",
  role: "",
  location: "",
  // prefs
  language: "English",
  timezone: "Europe/Rome",
  units: "metric", // metric | imperial
  notifyEmail: true,
  notifyAISuggestions: true,
  weeklySummary: false,
  // account (front-end only for now)
  currentPassword: "",
  newPassword: "",
  confirmPassword: "",
};

const LANGS = ["English", "Italiano", "Deutsch", "Español", "Français"];
const TIMEZONES = [
  "UTC",
  "Europe/Rome",
  "Europe/London",
  "America/New_York",
  "America/Los_Angeles",
  "Asia/Tokyo",
  "Asia/Singapore",
];

export default function Settings() {
  const [tab, setTab] = useState("profile"); // profile | account
  const [form, setForm] = useState(DEFAULT_FORM);
  const [toast, setToast] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const fileRef = useRef(null);
  const savedRef = useRef(JSON.stringify(DEFAULT_FORM));

  // Load from localStorage
  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) {
        const parsed = JSON.parse(raw);
        setForm({ ...DEFAULT_FORM, ...parsed });
        savedRef.current = JSON.stringify({ ...DEFAULT_FORM, ...parsed });
      }
    } catch {
      /* ignore */
    }
  }, []);

  const unsaved = useMemo(
    () => JSON.stringify(form) !== savedRef.current,
    [form]
  );

  const onChange = (k, v) => setForm((f) => ({ ...f, [k]: v }));

  const onAvatarPick = (e) => {
    const file = e.target?.files?.[0] || e.dataTransfer?.files?.[0];
    if (!file) return;
    const r = new FileReader();
    r.onload = () => setForm((f) => ({ ...f, avatar: r.result }));
    r.readAsDataURL(file);
  };

  const reset = () => {
    setForm((f) => {
      const next = { ...DEFAULT_FORM, language: f.language, timezone: f.timezone, units: f.units };
      return next;
    });
    setToast("Fields reset");
    setTimeout(() => setToast(""), 1600);
  };

  const save = () => {
    // Separate account fields from persisted profile/preferences
    const {
      currentPassword,
      newPassword,
      confirmPassword,
      ...toPersist
    } = form;

    // If on account tab and user typed a password, do simple front-end check
    if (tab === "account" && (newPassword || confirmPassword || currentPassword)) {
      if (!currentPassword) {
        setToast("Enter current password");
        setTimeout(() => setToast(""), 1600);
        return;
      }
      if (newPassword.length < 6) {
        setToast("New password too short");
        setTimeout(() => setToast(""), 1600);
        return;
      }
      if (newPassword !== confirmPassword) {
        setToast("Passwords do not match");
        setTimeout(() => setToast(""), 1600);
        return;
      }
      // No backend yet — pretend success
      setToast("Password updated (local only)");
      setTimeout(() => setToast(""), 1600);
    }

    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(toPersist));
      savedRef.current = JSON.stringify({ ...DEFAULT_FORM, ...toPersist });
      setToast("Saved changes");
      setTimeout(() => setToast(""), 1600);
    } catch {
      setToast("Save failed");
      setTimeout(() => setToast(""), 1600);
    }
  };

  return (
    <div className="settings">
      {/* Header */}
      <div className="set-head">
        <h1>Settings</h1>
        <div className="set-tabs">
          <button
            className={`set-tab ${tab === "profile" ? "is-active" : ""}`}
            onClick={() => setTab("profile")}
          >
            <FiUser /> Profile
          </button>
          <button
            className={`set-tab ${tab === "account" ? "is-active" : ""}`}
            onClick={() => setTab("account")}
          >
            <FiShield /> Account
          </button>
        </div>
      </div>

      {/* PROFILE TAB */}
      {tab === "profile" && (
        <div className="set-grid set-grid-vertical">
          {/* Avatar */}
          <div className="set-card set-area-avatar">
            <div className="set-card-title">
              <FiImage /> Avatar
            </div>

            <div
              className={`set-avatar-wrap ${dragOver ? "is-drag" : ""}`}
              onDragOver={(e) => {
                e.preventDefault();
                setDragOver(true);
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragOver(false);
                onAvatarPick(e);
              }}
            >
              <div
                className="set-avatar"
                style={{
                  backgroundImage: form.avatar ? `url(${form.avatar})` : undefined,
                }}
              >
                {!form.avatar && (
                  <span className="set-avatar-ini">
                    {(form.displayName || "U")[0]}
                  </span>
                )}
              </div>

              <div className="set-avatar-actions">
                <button className="chip" onClick={() => fileRef.current?.click()}>
                  <FiImage /> Upload
                </button>
                {form.avatar && (
                  <button
                    className="chip danger"
                    onClick={() => onChange("avatar", "")}
                  >
                    Remove
                  </button>
                )}
                <input
                  ref={fileRef}
                  type="file"
                  accept="image/*"
                  hidden
                  onChange={onAvatarPick}
                />
                <small className="hint">Drag & drop an image here</small>
              </div>
            </div>
          </div>

          {/* Company Info */}
          <div className="set-card set-area-company">
            <div className="set-card-title">
              <FiBuilding /> Company Info
            </div>
            <div className="set-form" style={{ gridTemplateColumns: "1fr 1fr" }}>
              <label>
                <span>Company name</span>
                <div className="inp">
                  <FiBuilding className="ic" />
                  <input
                    value={form.companyName}
                    onChange={(e) => onChange("companyName", e.target.value)}
                    placeholder="Company"
                  />
                </div>
              </label>

              <label>
                <span>Department</span>
                <div className="inp">
                  <FiBriefcase className="ic" />
                  <input
                    value={form.department}
                    onChange={(e) => onChange("department", e.target.value)}
                    placeholder="e.g., Maintenance"
                  />
                </div>
              </label>

              <label>
                <span>Role</span>
                <div className="inp">
                  <FiUser className="ic" />
                  <input
                    value={form.role}
                    onChange={(e) => onChange("role", e.target.value)}
                    placeholder="e.g., Operator"
                  />
                </div>
              </label>

              <label>
                <span>Location</span>
                <div className="inp">
                  <FiMapPin className="ic" />
                  <input
                    value={form.location}
                    onChange={(e) => onChange("location", e.target.value)}
                    placeholder="City / Plant / Line"
                  />
                </div>
              </label>
            </div>
          </div>

          {/* Personal Info */}
          <div className="set-card set-area-personal">
            <div className="set-card-title">
              <FiUser /> Personal Info
            </div>
            <div className="set-form">
              <label>
                <span>Display name</span>
                <div className="inp">
                  <FiUser className="ic" />
                  <input
                    value={form.displayName}
                    onChange={(e) => onChange("displayName", e.target.value)}
                    placeholder="Your name"
                  />
                </div>
              </label>

              <label>
                <span>Username</span>
                <div className="inp">
                  <FiUser className="ic" />
                  <input
                    value={form.username}
                    onChange={(e) => onChange("username", e.target.value)}
                    placeholder="username"
                  />
                </div>
              </label>

              <label>
                <span>Email</span>
                <div className="inp">
                  <FiAtSign className="ic" />
                  <input
                    type="email"
                    value={form.email}
                    onChange={(e) => onChange("email", e.target.value)}
                    placeholder="your@email.com"
                  />
                </div>
              </label>

              <label>
                <span>Phone</span>
                <div className="inp">
                  <FiPhone className="ic" />
                  <input
                    value={form.phone}
                    onChange={(e) => onChange("phone", e.target.value)}
                    placeholder="+39 …"
                  />
                </div>
              </label>
            </div>
          </div>

          {/* Preferences */}
          <div className="set-card set-area-prefs">
            <div className="set-card-title">
              <FiGlobe /> Preferences
            </div>

            <div className="set-form" style={{ gridTemplateColumns: "1fr 1fr" }}>
              <label>
                <span>Language</span>
                <div className="inp">
                  <FiGlobe className="ic" />
                  <select
                    value={form.language}
                    onChange={(e) => onChange("language", e.target.value)}
                  >
                    {LANGS.map((l) => (
                      <option key={l} value={l}>
                        {l}
                      </option>
                    ))}
                  </select>
                </div>
              </label>

              <label>
                <span>Timezone</span>
                <div className="inp">
                  <FiClock className="ic" />
                  <select
                    value={form.timezone}
                    onChange={(e) => onChange("timezone", e.target.value)}
                  >
                    {TIMEZONES.map((t) => (
                      <option key={t} value={t}>
                        {t}
                      </option>
                    ))}
                  </select>
                </div>
              </label>

              <label style={{ gridColumn: "1 / -1" }}>
                <span>Units</span>
                <div className="set-seg" role="tablist" aria-label="Units">
                  <button
                    className={`set-seg-btn ${
                      form.units === "metric" ? "is-active" : ""
                    }`}
                    onClick={() => onChange("units", "metric")}
                  >
                    Metric
                  </button>
                  <button
                    className={`set-seg-btn ${
                      form.units === "imperial" ? "is-active" : ""
                    }`}
                    onClick={() => onChange("units", "imperial")}
                  >
                    Imperial
                  </button>
                </div>
              </label>

              <div className="set-switches" style={{ gridColumn: "1 / -1" }}>
                <label className="sw">
                  <input
                    type="checkbox"
                    checked={form.notifyEmail}
                    onChange={(e) => onChange("notifyEmail", e.target.checked)}
                  />
                  Email alerts
                </label>
                <label className="sw">
                  <input
                    type="checkbox"
                    checked={form.notifyAISuggestions}
                    onChange={(e) =>
                      onChange("notifyAISuggestions", e.target.checked)
                    }
                  />
                  AI suggestions
                </label>
                <label className="sw">
                  <input
                    type="checkbox"
                    checked={form.weeklySummary}
                    onChange={(e) => onChange("weeklySummary", e.target.checked)}
                  />
                  Weekly summary
                </label>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ACCOUNT TAB (front-end only) */}
      {tab === "account" && (
        <div className="set-grid" style={{ gridTemplateColumns: "1fr 1fr" }}>
          <div className="set-card" style={{ gridColumn: "1 / -1" }}>
            <div className="set-card-title">
              <FiLock /> Change Password
            </div>
            <div className="set-form" style={{ gridTemplateColumns: "1fr 1fr" }}>
              <label style={{ gridColumn: "1 / -1" }}>
                <span>Current password</span>
                <div className="inp">
                  <FiLock className="ic" />
                  <input
                    type="password"
                    value={form.currentPassword}
                    onChange={(e) => onChange("currentPassword", e.target.value)}
                    placeholder="••••••••"
                  />
                </div>
              </label>

              <label>
                <span>New password</span>
                <div className="inp">
                  <FiLock className="ic" />
                  <input
                    type="password"
                    value={form.newPassword}
                    onChange={(e) => onChange("newPassword", e.target.value)}
                    placeholder="At least 6 characters"
                  />
                </div>
              </label>

              <label>
                <span>Confirm password</span>
                <div className="inp">
                  <FiLock className="ic" />
                  <input
                    type="password"
                    value={form.confirmPassword}
                    onChange={(e) =>
                      onChange("confirmPassword", e.target.value)
                    }
                    placeholder="Repeat new password"
                  />
                </div>
              </label>
            </div>
          </div>
        </div>
      )}

      {/* Sticky footer */}
      <div className="set-footer">
        <button className="chip" onClick={reset} title="Reset">
          <FiRotateCcw /> Reset
        </button>
        <div className="spacer" />
        <button
          className={`chip primary`}
          onClick={save}
          title="Save changes"
          disabled={!unsaved && tab !== "account"}
          style={{ opacity: !unsaved && tab !== "account" ? 0.7 : 1 }}
        >
          <FiSave /> Save Changes
        </button>
      </div>

      {/* Toast */}
      {toast && <div className="set-toast">{toast}</div>}
    </div>
  );
}
