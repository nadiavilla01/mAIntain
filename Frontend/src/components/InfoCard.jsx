import React from "react";
import "./InfoCard.css";

const InfoCard = ({ label, value, unit, alert }) => {
  return (
    <div className="info-card">
      <div className="label">{label}</div>
      <div className="value">
        {value}
        {unit && <span className="unit">{unit}</span>}
        {alert && <span className="alert-dot" />}
      </div>
    </div>
  );
};

export default InfoCard;
