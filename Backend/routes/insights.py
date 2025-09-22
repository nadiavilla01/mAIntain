from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from collections import Counter
from datetime import datetime
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import os
import re

from utils.data_loader import load_base_data, status_from_pred
from pretrainedDistilbert.intent_utils import infer_intent


SENSORS = ['s_2','s_3','s_4','s_7','s_8','s_9','s_11','s_12','s_13','s_14','s_15','s_17','s_20','s_21']
Q_LOW, Q_HIGH = 0.05, 0.95
DROP_WINDOW = 5
DROP_THRESH_H = 15.0
ZSCORE_WINDOW = 10
MIN_Z_FOR_NOTE = 2.0
DEFAULT_MODE = "start"  


FMEA_PATH = "./FMEA.csv"
fmea_df = pd.read_csv(FMEA_PATH) if os.path.exists(FMEA_PATH) else pd.DataFrame(columns=["intent","cause","suggested_action"])


try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except Exception:
    pass

_client = None
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
try:
    if OPENAI_API_KEY:
        from openai import OpenAI
        _client = OpenAI(api_key=OPENAI_API_KEY)
    else:
        print("[Insights] OPENAI_API_KEY not set; LLM disabled.")
except Exception as e:
    print(f"[Insights] OpenAI client init failed: {e}")
    _client = None

router = APIRouter()


def _safe_float(x):
    try:
        v = float(x)
        return None if np.isnan(v) else v
    except Exception:
        return None

def generate_alert(text: str, severity: float = 1.0):
    intent, confidence = infer_intent(text)
    fallback = "yes" if confidence < 0.75 else "no"
    row = fmea_df[fmea_df["intent"] == intent]
    cause = row["cause"].values[0] if not row.empty and "cause" in row.columns else "Unknown"
    action = row["suggested_action"].values[0] if not row.empty and "suggested_action" in row.columns else "Investigate manually"
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
            best_z = z; best_s = s
    return (best_s, float(best_z)) if best_s is not None else (None, None)

def rul_based_alerts(unit_id: int, current_cycle: int, pred_df: Optional[pd.DataFrame], df_unit: pd.DataFrame, true_rul_at_cycle: Optional[float]):
    out = []
    if pred_df is None or pred_df.empty:
        return out
    preds_u = pred_df[pred_df["unit_nr"] == unit_id].copy()
    if preds_u.empty:
        return out
    preds_u = preds_u.sort_values("time_cycles")
    preds_u["diff"] = (preds_u["time_cycles"] - current_cycle).abs()
    p0 = preds_u.iloc[preds_u["diff"].argmin()]
    matched_cycle = int(p0["time_cycles"])
    yhat = float(p0["predicted_RUL"])

    top_s, top_z = strongest_sensor_shift_z(df_unit, matched_cycle)
    sensor_note = f" · strongest sensor shift: {top_s} (~{top_z:.1f}σ)" if (top_s and top_z and top_z >= MIN_Z_FOR_NOTE) else ""

    tail = preds_u[preds_u["time_cycles"] <= matched_cycle].tail(DROP_WINDOW + 1)
    if len(tail) >= DROP_WINDOW + 1:
        y_then = float(tail.iloc[0]["predicted_RUL"])
        drop = max(0.0, y_then - yhat)
        if drop >= DROP_THRESH_H:
            sev = min(drop / 25.0, 1.0)
            out.append(generate_alert(f"Rapid RUL decrease (~{drop:.0f}h over last ~{DROP_WINDOW} points){sensor_note}", sev))
    return out

def _compute_machine_snapshots(mode: str = DEFAULT_MODE) -> List[dict]:
    df, pred_df, errs_df = load_base_data()
    if df is None:
        return []

    idx_map = {"start": 0, "middle": None, "end": -1}

    machines = []
    for unit_id, group in df.groupby("unit_nr", sort=True):
        group = group.reset_index(drop=True)

        if mode == "middle":
            idx = len(group)//2
        else:
            idx = idx_map[mode]
        point = group.iloc[idx]
        current_cycle = int(point["time_cycles"])
        real_rul = _safe_float(point.get("RUL"))

        predicted_rul = None
        if pred_df is not None and all(c in pred_df.columns for c in ["unit_nr","predicted_RUL","time_cycles"]):
            preds = pred_df[pred_df["unit_nr"] == unit_id].copy()
            if not preds.empty:
                preds["diff"] = (preds["time_cycles"] - current_cycle).abs()
                best = preds.sort_values("diff").iloc[0]
                predicted_rul = _safe_float(best["predicted_RUL"])

        mae = rmse = None
        if errs_df is not None and "unit_nr" in errs_df.columns:
            match = errs_df[errs_df["unit_nr"] == unit_id]
            if not match.empty:
                mae = _safe_float(match.iloc[0].get("avg_MAE"))
                rmse = _safe_float(match.iloc[0].get("avg_RMSE"))

        status = status_from_pred(predicted_rul if predicted_rul is not None else real_rul, mae)

        q_hi = group[SENSORS].quantile(Q_HIGH)
        q_lo = group[SENSORS].quantile(Q_LOW)

        alerts = []
        def band_alert(s_key, label):
            v = _safe_float(point.get(s_key))
            hi = _safe_float(q_hi.get(s_key))
            lo = _safe_float(q_lo.get(s_key))
            if v is None or hi is None or lo is None:
                return
            if (v > hi) or (v < lo):
                center = (hi + lo) / 2.0
                span = max(hi - lo, 1e-5)
                severity = float(min(abs(v - center) / span, 3.0) / 3.0)
                alerts.append(generate_alert(f"⚠️ {label} anomaly detected at cycle {current_cycle}", severity))

        band_alert("s_2", "Temperature")
        band_alert("s_3", "Vibration")
        band_alert("s_4", "Power")
        band_alert("s_7", "Speed")

        alerts += rul_based_alerts(int(unit_id), current_cycle, pred_df, group, real_rul)

        machines.append({
            "id": int(unit_id),
            "name": f"Machine {unit_id}",
            "status": status,
            "rul": real_rul,
            "predicted_rul": round(predicted_rul) if predicted_rul is not None else None,
            "mae": round(mae, 2) if mae is not None else None,
            "rmse": round(rmse, 2) if rmse is not None else None,
            "alerts": alerts,
        })
    return machines

def _compose_context(question: str, mode: str = DEFAULT_MODE) -> str:
    machines = _compute_machine_snapshots(mode)
    if not machines:
        return "No dataset loaded."

    status_counter = Counter(m["status"] for m in machines)
    total = sum(status_counter.values())
    chunks = [f"Fleet summary — total {total} machines: "
              f"Normal {status_counter.get('Normal',0)}, "
              f"Unstable {status_counter.get('Unstable',0)}, "
              f"Critical {status_counter.get('Critical',0)}."]

    ids = sorted({int(x) for x in re.findall(r"(?:machine|unit|id)\s*#?\s*(\d{1,4})", question, flags=re.I)})
    sample = [m for m in machines if (not ids or m["id"] in ids)][:3]
    for m in sample:
        chunks.append(
            f"Machine {m['id']} — status {m['status']}; "
            f"RUL: {'unknown' if m['rul'] is None else f'{m['rul']:.0f} h'}; "
            f"AI RUL: {'n/a' if m['predicted_rul'] is None else f'{m['predicted_rul']} h'}; "
            f"MAE: {'n/a' if m['mae'] is None else m['mae']}; RMSE: {'n/a' if m['rmse'] is None else m['rmse']}."
        )

    if not fmea_df.empty and "intent" in fmea_df.columns:
        intents = ", ".join(sorted(set(fmea_df["intent"].astype(str).tolist()))[:10])
        chunks.append(f"Known FMEA intents: {intents}")

    chunks.append("Sensor aliases: temperature=s_2, vibration=s_3, power=s_4, speed=s_7.")
    return "\n\n".join(chunks)[:8000]

def _ask_llm(user_msg: str, context: str) -> str:
    if _client is None:
        return "LLM is not configured (missing OPENAI_API_KEY). Context I would have used:\n\n" + context[:1200]
    system = ("You are mAIntain, an industrial predictive maintenance copilot. "
              "Use the provided context to answer precisely. Include units. If uncertain, state assumptions and propose checks.")
    try:
        resp = _client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[{"role":"system","content":system},
                      {"role":"user","content":f"Question:\n{user_msg}\n\nContext:\n{context}"}],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM error: {e}"


@router.get("")
def get_ai_insights(mode: str = Query(DEFAULT_MODE, enum=["start","middle","end"])):
    machines = _compute_machine_snapshots(mode)
    if not machines:
        return {}

    warnings = []
    for m in machines:
        for a in m.get("alerts", []):
            warnings.append({
                "machine": m["name"],
                "message": a["text"],
                "color": "#ef4444" if a["fallback"] == "no" else "#facc15",
                "severity": a.get("severity", 0.5),
            })
    warnings.sort(key=lambda x: x.get("severity", 0.0), reverse=True)
    warnings = warnings[:30]

    status_counter = Counter(m["status"] for m in machines)
    total = sum(status_counter.values())
    pct = lambda v: round((v/total)*100) if total else 0

    # average of AI RUL when present, else fallback to baseline RUL
    rul_values = []
    for m in machines:
        if m["predicted_rul"] is not None:
            rul_values.append(m["predicted_rul"])
        elif m["rul"] is not None:
            rul_values.append(m["rul"])
    avg_rul = float(np.mean(rul_values)) if rul_values else 0.0

    rul_trend = [{"time": day, "rul": int(30 - i * 2)} for i, day in enumerate(["Mon","Tue","Wed","Thu","Fri"])]

    return {
        "anomalies_this_week": len(warnings),
        "critical_count": status_counter.get("Critical", 0),
        "avg_rul": round(avg_rul, 2),
        "rul_trend": rul_trend,
        "warnings": warnings,
        "health_ratio": {
            "normal": pct(status_counter.get("Normal", 0)),
            "unstable": pct(status_counter.get("Unstable", 0)),
            "critical": pct(status_counter.get("Critical", 0)),
        },
    }


class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

@router.post("/chat", response_model=ChatResponse)
def chat_to_insights(req: ChatRequest, mode: str = Query(DEFAULT_MODE, enum=["start","middle","end"])):
    msg = (req.message or "").strip()
    if not msg:
        raise HTTPException(status_code=400, detail="Empty message.")
    context = _compose_context(msg, mode=mode)
    reply = _ask_llm(msg, context)
    return ChatResponse(reply=reply)