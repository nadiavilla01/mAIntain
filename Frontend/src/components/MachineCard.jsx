import React from "react";
import { useNavigate } from "react-router-dom";
import { Sparklines, SparklinesLine } from "react-sparklines";
import { FiAlertCircle, FiStar } from "react-icons/fi";
import "./MachineCard.css";

const MAX_RUL = 125; 

export default function MachineCard({
  id,
  name,
  status,
  section,
  rul,
  predicted_rul,
  last_updated_ago,
  alerts = [],
  trend = {},
  large = false,
  density = "cozy",
  pinned = false,
  onTogglePin,
}) {
  const navigate = useNavigate();
  const hasAlerts = alerts.length > 0;

  const series =
    trend?.temperature?.length
      ? trend.temperature
      : trend?.power?.length
      ? trend.power
      : [0];

  const isNum = (v) => Number.isFinite(v);

  const usingPred = isNum(predicted_rul);
  const displayRul = usingPred
    ? Math.max(0, Math.round(predicted_rul))
    : isNum(rul)
    ? Math.max(0, Math.round(rul))
    : null;

  const rulLabel = usingPred ? "Pred RUL" : "RUL";


  let fillPercent = 0;
  if (isNum(predicted_rul) && isNum(rul) && rul > 0) {
    fillPercent = Math.min(Math.max((predicted_rul / rul) * 100, 0), 100);
  } else if (isNum(predicted_rul)) {
    fillPercent = Math.min(Math.max((predicted_rul / MAX_RUL) * 100, 0), 100);
  }

  return (
    <div
      className="machine-card"
      data-status={status || "Unknown"}
      data-density={density}
      data-pinned={pinned ? "true" : "false"}
      onClick={() => navigate(`/machine/${id}`)}
      style={large ? { transform: "scale(1.12)" } : undefined}
    >
      <div className="mc-head">
        <div className="mc-title">
          <span className="mc-name" title={name}>{name}</span>
          <span className="mc-type">Machine</span>
        </div>

        <div className="mc-actions" onClick={(e) => e.stopPropagation()}>
          <button
            className={`mc-pin ${pinned ? "is-on" : ""}`}
            title={pinned ? "Unpin" : "Pin"}
            onClick={onTogglePin}
          >
            <FiStar />
          </button>
          {hasAlerts && (
            <div className="mc-alert" title={`${alerts.length} alert(s)`}>
              <FiAlertCircle size={14} />
              <span className="mc-alert-count">{alerts.length}</span>
            </div>
          )}
          <span className="mc-status">{status || "Unknown"}</span>
        </div>
      </div>

      <div className="mc-sub">
        <span className="mc-section">{section || "—"}</span>
      </div>

      <div className="mc-graph">
        <Sparklines data={series} height={42} width={160} margin={4}>
          <SparklinesLine color="#38bdf8" style={{ fill: "none" }} />
        </Sparklines>
        <div className="mc-rul">
          <div className="mc-rul-fill" style={{ width: `${fillPercent}%` }} />
        </div>
      </div>

      <div className="mc-foot">
        <span className="mc-time">{last_updated_ago || "—"}</span>
        <span className="mc-rul-text">
          {rulLabel}: {displayRul !== null ? `${displayRul}h` : "—"}
        </span>
      </div>
    </div>
  );
}
