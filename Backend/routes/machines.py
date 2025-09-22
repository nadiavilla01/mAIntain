from fastapi import APIRouter, Query
from utils.data_loader import load_base_data, status_from_pred
from pretrainedDistilbert.intent_utils import infer_intent
from datetime import datetime
import numpy as np
import pandas as pd
import os
import math
from typing import Tuple, Optional

router = APIRouter()


SENSORS = ['s_2','s_3','s_4','s_7','s_8','s_9','s_11','s_12','s_13','s_14','s_15','s_17','s_20','s_21']

Q_LOW, Q_HIGH = 0.05, 0.95  


ENABLE_RUL_DIVERGENCE = False   
DROP_WINDOW = 5                 
DROP_THRESH_H = 15.0            
ZSCORE_WINDOW = 10            
MIN_Z_FOR_NOTE = 2.0           


FMEA_PATH = "../pretrainedDistilbert/FMEA.csv"
if os.path.exists(FMEA_PATH):
    fmea_df = pd.read_csv(FMEA_PATH)
else:
    fmea_df = pd.DataFrame(columns=["intent","cause","suggested_action"])


def safe_float(x):
    try:
        x = float(x)
        return None if np.isnan(x) else x
    except Exception:
        return None

def classify_alert(text: str, severity: float = 1.0):
    """
    Classify freeform alert text with DistilBERT and enrich from FMEA.
    """
    intent, confidence = infer_intent(text)
    fallback = "yes" if confidence < 0.75 else "no"

    row = fmea_df[fmea_df["intent"] == intent]
    cause = row["cause"].values[0] if not row.empty and "cause" in row.columns else "Unknown"
    action = row["suggested_action"].values[0] if not row.empty and "suggested_action" in row.columns else "Investigate manually"

    if fallback == "yes":
        try:
            with open("intent_fallback_log.csv", "a") as log_file:
                log_file.write(f"{intent},{confidence:.2f},{text}\n")
        except Exception as e:
            print(f"Could not log fallback intent: {e}")

    return {
        "text": text,
        "intent": intent,
        "confidence": round(float(confidence), 2),
        "fallback": fallback,
        "cause": cause,
        "suggested_action": action,
        "severity": float(max(0.0, min(1.0, severity))),
    }

def strongest_sensor_shift_z(df_unit: pd.DataFrame, center_cycle: int) -> Tuple[Optional[str], Optional[float]]:
    """
    Return (sensor_name, |z|) with the largest deviation at `center_cycle`
    compared to a short recent window; None if not enough data.
    """
    if df_unit.empty:
        return None, None
    lo = center_cycle - ZSCORE_WINDOW
    hi = center_cycle
    win = df_unit[(df_unit["time_cycles"] >= lo) & (df_unit["time_cycles"] <= hi)].copy()
    if win.empty:
        return None, None

    best_s, best_z = None, -1.0
    for s in SENSORS:
        vals = pd.to_numeric(win[s], errors="coerce").dropna()
        if len(vals) < 3:
            continue
        mu = vals.mean()
        sd = vals.std(ddof=1) or 1.0
        z = abs((vals.iloc[-1] - mu) / sd)
        if z > best_z:
            best_z = z
            best_s = s
    return (best_s, float(best_z)) if best_s is not None else (None, None)

def rul_based_alerts(unit_id: int,
                     current_cycle: int,
                     pred_df: pd.DataFrame | None,
                     df_unit: pd.DataFrame,
                     true_rul_at_cycle: float | None):
    """
    Build RUL alerts:
      • Rapid drop over last DROP_WINDOW predictions (only).
      • Optionally attach the strongest sensor shift IF |z| ≥ MIN_Z_FOR_NOTE.
      • Divergence alert is disabled.
    """
    out = []
    if pred_df is None or pred_df.empty:
        return out

    preds_u = pred_df[pred_df["unit_nr"] == unit_id].copy()
    if preds_u.empty:
        return out
    preds_u = preds_u.sort_values("time_cycles")

    # nearest predicted point to current cycle
    preds_u["diff"] = (preds_u["time_cycles"] - current_cycle).abs()
    p0 = preds_u.iloc[preds_u["diff"].argmin()]
    matched_cycle = int(p0["time_cycles"])
    yhat = float(p0["predicted_RUL"])

    # strongest sensor shift near this cycle (only note if ≥ 2σ)
    top_s, top_z = strongest_sensor_shift_z(df_unit, matched_cycle)
    sensor_note = f" · strongest sensor shift: {top_s} (~{top_z:.1f}σ)" if (top_s and top_z and top_z >= MIN_Z_FOR_NOTE) else ""

    #  divergence removed
    if ENABLE_RUL_DIVERGENCE and true_rul_at_cycle is not None:
        diff = float(abs(yhat - float(true_rul_at_cycle)))
   
    tail = preds_u[preds_u["time_cycles"] <= matched_cycle].tail(DROP_WINDOW + 1)
    if len(tail) >= DROP_WINDOW + 1:
        y_then = float(tail.iloc[0]["predicted_RUL"])
        drop = max(0.0, y_then - yhat)
        if drop >= DROP_THRESH_H:
            sev = min(drop / 25.0, 1.0)
            out.append(classify_alert(
                f"Rapid RUL decrease (~{drop:.0f}h over last ~{DROP_WINDOW} points){sensor_note}",
                severity=sev
            ))

    return out


@router.get("")
def get_machine_data(mode: str = Query("end", enum=["start", "middle", "end"])):
    df, pred_df, errs_df = load_base_data()
    if df is None:
        return []

    machines = []

    for unit_id, group in df.groupby("unit_nr", sort=True):
        group = group.reset_index(drop=True)

        idx = {"start": 0, "middle": len(group)//2, "end": -1}[mode]
        point = group.iloc[idx]
        current_cycle = int(point["time_cycles"])
        real_rul = safe_float(point.get("RUL"))

        # closest prediction for this unit
        predicted_rul = None
        matched_cycle = current_cycle
        if pred_df is not None and all(c in pred_df.columns for c in ["unit_nr","predicted_RUL","time_cycles"]):
            preds = pred_df[pred_df["unit_nr"] == unit_id].copy()
            if not preds.empty:
                preds["diff"] = (preds["time_cycles"] - current_cycle).abs()
                best = preds.sort_values("diff").iloc[0]
                predicted_rul = safe_float(best["predicted_RUL"])
                matched_cycle = int(best["time_cycles"])

        # error stats
        mae = rmse = None
        if errs_df is not None and "unit_nr" in errs_df.columns:
            match = errs_df[errs_df["unit_nr"] == unit_id]
            if not match.empty:
                mae = safe_float(match.iloc[0].get("avg_MAE"))
                rmse = safe_float(match.iloc[0].get("avg_RMSE"))

        # status from predicted RUL 
        status = status_from_pred(predicted_rul if predicted_rul is not None else real_rul, mae)

        # sensors now + perunit quantile thresholds
        sensors_now = {s: safe_float(point.get(s)) for s in SENSORS}
        q_hi = group[SENSORS].quantile(Q_HIGH)
        q_lo = group[SENSORS].quantile(Q_LOW)

        # sensor alerts 
        alerts = []
        for s, label in [("s_2","Temperature"), ("s_3","Vibration"), ("s_4","Power"), ("s_7","Speed")]:
            v = sensors_now.get(s)
            hi = safe_float(q_hi.get(s))
            lo = safe_float(q_lo.get(s))
            if v is None or hi is None or lo is None:
                continue
            if v > hi or v < lo:
                center = (hi + lo) / 2.0
                span = max(hi - lo, 1e-5)
                severity = float(min(abs(v - center) / span, 3.0) / 3.0)
                alerts.append(classify_alert(f"⚠️ {label} anomaly detected at cycle {matched_cycle}", severity))

        # fused RUL alerts
        alerts += rul_based_alerts(
            unit_id=int(unit_id),
            current_cycle=current_cycle,
            pred_df=pred_df,
            df_unit=group,
            true_rul_at_cycle=real_rul
        )

        # trends for sparkline
        def clip_series(series, i):
            lo = max(0, i - 8)
            hi = i + 8 if i >= 0 else len(series)
            return pd.to_numeric(series.iloc[lo:hi], errors="coerce").dropna().astype(float).round(4).tolist()

        trends = {
            "temperature": clip_series(group["s_2"], idx),
            "vibration":   clip_series(group["s_3"], idx),
            "power":       clip_series(group["s_4"], idx),
            "speed":       clip_series(group["s_7"], idx),
        }

        machines.append({
            "id": int(unit_id),
            "name": f"Machine {unit_id}",
            "section": f"Section {unit_id % 6 + 1}",
            "status": status,
            "rul": real_rul,  # baseline
            "predicted_rul": round(predicted_rul) if predicted_rul is not None else None,  # what UI should show!!! not the baseline 
            "mae": round(mae, 2) if mae is not None else None,
            "rmse": round(rmse, 2) if rmse is not None else None,
            "last_updated": datetime.utcnow().isoformat() + "Z",
            "trend": trends,
            "sensors": {
                "temperature": sensors_now.get("s_2"),
                "vibration":   sensors_now.get("s_3"),
                "power":       sensors_now.get("s_4"),
                "speed":       sensors_now.get("s_7"),
            },
            "alerts": alerts,
        })

    return machines