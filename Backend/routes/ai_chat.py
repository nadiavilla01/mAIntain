from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any, List, Tuple, Optional, Union
import pandas as pd
import os
import math
import numpy as np

from intent_utils import infer_intent  # DistilBERT intent

router = APIRouter()


FMEA_PATH = "./FMEA.csv"
FMEA_COLS = ["intent", "cause", "suggested_action"]
if os.path.exists(FMEA_PATH):
    _fmea = pd.read_csv(FMEA_PATH)
    for col in FMEA_COLS:
        if col not in _fmea.columns:
            _fmea[col] = ""
    fmea_df = _fmea[FMEA_COLS].copy()
else:
    fmea_df = pd.DataFrame(columns=FMEA_COLS)

class AIChatRequest(BaseModel):
    message: str
    context: Dict[str, Any] = {}


ENABLE_RUL_DIVERGENCE = False   
ENABLE_RAPID_DROP     = True    
DROP_WINDOW           = 5       
DROP_THRESH_H         = 15.0    
ZSCORE_WINDOW         = 10     
MIN_TREND_Z_NOTE      = 2.0     

SENSOR_KEYMAP = {
    "temperature": "s_2",
    "vibration":   "s_3",
    "power":       "s_4",
    "speed":       "s_7",
}
SENSOR_ALIASES = {
    "temperature": {"temp", "temperature", "heat"},
    "vibration": {"vibration", "vibe"},
    "power": {"power", "watt", "current", "load"},
    "speed": {"speed", "rpm"}
}


def fmea_lookup(intent: str) -> Tuple[str, str]:
    row = fmea_df.loc[fmea_df["intent"] == intent]
    if row.empty:
        return "Unknown", "Investigate manually"
    r = row.iloc[0]
    return str(r.get("cause") or "Unknown"), str(r.get("suggested_action") or "Investigate manually")

def classify_alert(text: str, severity: float = 1.0) -> Dict[str, Any]:
    intent, confidence = infer_intent(text)
    fallback = "yes" if confidence < 0.75 else "no"
    cause, action = fmea_lookup(intent)
    if fallback == "yes":
        try:
            with open("intent_fallback_log.csv", "a") as log_file:
                log_file.write(f"{intent},{confidence:.2f},{text}\n")
        except Exception as e:
            print(f"⚠️ Could not log fallback intent: {e}")
    return {
        "text": text,
        "intent": intent,
        "confidence": round(float(confidence), 2),
        "fallback": fallback,
        "cause": cause,
        "suggested_action": action,
        "severity": float(max(0.0, min(1.0, severity))),
    }

def sensor_token_hits(text: str) -> Dict[str, int]:
    t = (text or "").lower()
    counts = {k: 0 for k in SENSOR_ALIASES}
    for sensor, words in SENSOR_ALIASES.items():
        if any(w in t for w in words):
            counts[sensor] += 1
    return counts

def strongest_sensor_shift_from_trend(trend: Dict[str, List[float]]) -> Tuple[Optional[str], Optional[float]]:
    if not trend:
        return None, None
    best_name, best_z = None, -1.0
    for name in ("temperature", "vibration", "power", "speed"):
        series = trend.get(name) or []
        if not isinstance(series, list) or len(series) < 3:
            continue
        tail = series[-min(ZSCORE_WINDOW, len(series)):]
        vals = np.array([v for v in tail if isinstance(v, (int, float))], dtype=float)
        if vals.size < 3:
            continue
        mu = float(vals.mean())
        sd = float(vals.std(ddof=1)) or 1.0
        z = abs((vals[-1] - mu) / sd)
        if z > best_z:
            best_z = z
            best_name = name
    if best_name is None:
        return None, None
    return best_name, float(best_z)

def pick_suspicious_sensor(alerts: List[Dict[str, Any]], sensors_point: Dict[str, float], trend: Dict[str, List[float]]) -> Tuple[str, str]:
    # 1) trend z-shift
    s_name, z = strongest_sensor_shift_from_trend(trend or {})
    if s_name and z is not None and z >= MIN_TREND_Z_NOTE:
        return s_name, f"largest recent shift (~{z:.1f}σ)"
    # 2) alert mentions
    mention_scores = {k: 0 for k in SENSOR_ALIASES}
    for a in alerts or []:
        c = sensor_token_hits(a.get("text") or a.get("original") or "")
        for k, v in c.items():
            mention_scores[k] += v
    by_mentions = sorted(mention_scores.items(), key=lambda x: x[1], reverse=True)
    if by_mentions and by_mentions[0][1] > 0:
        return by_mentions[0][0], "mentioned in recent alerts"
    # 3) pointwise deviation
    vals = {k: v for k, v in (sensors_point or {}).items() if isinstance(v, (int, float))}
    if not vals:
        return "unknown", "insufficient data"
    mu = sum(vals.values()) / len(vals)
    sigma = math.sqrt(sum((v - mu) ** 2 for v in vals.values()) / max(1, len(vals) - 1)) or 1.0
    scores = {k: abs((v - mu) / sigma) for k, v in vals.items()}
    s = max(scores.items(), key=lambda x: x[1])[0]
    return s, "largest deviation among current readings"

def rul_fusion_alerts_from_ctx(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build RUL alerts using context:
      - (divergence disabled)
      - rapid drop over last DROP_WINDOW points (if history provided)
    Add sensor note ONLY if the shift is clear (|z| >= MIN_TREND_Z_NOTE).
    """
    alerts: List[Dict[str, Any]] = []
    if not ENABLE_RAPID_DROP:
        return alerts

    trend = ctx.get("trend", {}) or ctx.get("sensor_trends", {}) or {}

    # accept any of these for history
    hist: Optional[Union[List[float], List[Dict[str, Any]]]] = (
        ctx.get("pred_history") or ctx.get("pred_series") or ctx.get("pred_points")
    )
    if isinstance(hist, list) and len(hist) >= DROP_WINDOW + 1:
        if isinstance(hist[0], dict):
            seq = [float(p.get("rul")) for p in hist if "rul" in p and isinstance(p.get("rul"), (int, float))]
        else:
            seq = [float(x) for x in hist if isinstance(x, (int, float))]
        if len(seq) >= DROP_WINDOW + 1:
            y_then = seq[-(DROP_WINDOW + 1)]
            y_now = seq[-1]
            drop = max(0.0, y_then - y_now)
            if drop >= DROP_THRESH_H:
                s_name, z = strongest_sensor_shift_from_trend(trend)
                sensor_note = ""
                if s_name and z is not None and z >= MIN_TREND_Z_NOTE:
                    sensor_note = f" · strongest sensor shift: {s_name} (~{z:.1f}σ)"
                sev = min(drop / 25.0, 1.0)
                alerts.append(classify_alert(
                    f"Rapid RUL decrease (~{drop:.0f}h over last ~{DROP_WINDOW} points){sensor_note}",
                    severity=sev
                ))
    return alerts

def summarize_alerts(alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[str, Dict[str, Any]] = {}
    for a in alerts or []:
        text = str(a.get("text") or a.get("original") or "").strip()
        intent = a.get("intent")
        conf = a.get("confidence")
        if not intent:
            intent, conf = infer_intent(text)
        try:
            conf = float(conf) if conf is not None else 0.5
        except Exception:
            conf = 0.5
        b = buckets.setdefault(intent, {"count": 0, "sum_conf": 0.0, "examples": []})
        b["count"] += 1
        b["sum_conf"] += conf
        if text and len(b["examples"]) < 2:
            b["examples"].append(text)

    out = []
    for intent, b in buckets.items():
        avg_conf = b["sum_conf"] / max(1, b["count"])
        cause, action = fmea_lookup(intent)
        out.append({
            "intent": intent,
            "count": b["count"],
            "avg_conf": avg_conf,
            "examples": b["examples"],
            "cause": cause,
            "action": action,
        })
    out.sort(key=lambda x: x["avg_conf"], reverse=True)
    return out

def normalize_question_intent(message: str, model_intent: str) -> str:
    m = (message or "").lower()
    if ("which" in m and "sensor" in m) or "most suspicious" in m:
        return "which_sensor"
    if "root cause" in m or ("why" in m and ("cause" in m or "happened" in m)):
        return "root_cause"
    if "what should i do" in m or "first step" in m or ("do" in m and "first" in m) or ("fix" in m and "first" in m):
        return "what_first"
    if "verify" in m or ("confirm" in m) or ("check" in m and "fix" in m):
        return "verify_fix"
    if "why" in m and ("alert" in m or "raised" in m or "trigger" in m):
        return "why_alert"
    return model_intent or "general"

def fmt_hours(x) -> str:
    try: return f"{int(round(float(x)))}h"
    except Exception: return "–"


@router.post("")
async def chat_with_ai(req: AIChatRequest):
    msg = (req.message or "").strip()
    ctx = req.context or {}

    name = ctx.get("name", "this machine")
    status = ctx.get("status", "unknown")
    predicted_rul = ctx.get("predicted_rul", ctx.get("rul", "–"))
    baseline_rul = ctx.get("rul", "–")
    alerts_in = ctx.get("alerts", []) or []
    sensors_point = ctx.get("sensors", {}) or {}
    trend = ctx.get("trend", {}) or ctx.get("sensor_trends", {}) or {}

    model_intent, model_conf = infer_intent(msg)
    q_intent = normalize_question_intent(msg, model_intent)

    lower_msg = msg.lower()
    brief = any(w in lower_msg for w in ["brief", "short", "concise"])
    detailed = any(w in lower_msg for w in ["detail", "explain", "long"])

    fused_alerts = rul_fusion_alerts_from_ctx(ctx)  
    all_alerts = (alerts_in or []) + fused_alerts

    groups = summarize_alerts(all_alerts)
    top = groups[0] if groups else None

    header_brief = f"📟 {name} — {status} · AI RUL {fmt_hours(predicted_rul)} (baseline {fmt_hours(baseline_rul)})"
    header_full = (
        f"**{name}** — status **{status}** · predicted RUL **{fmt_hours(predicted_rul)}** "
        f"(baseline {fmt_hours(baseline_rul)})\n"
        f" Your question maps to **{model_intent}** ({round(model_conf*100)}%)."
    )

    # intent-specific
    if q_intent == "why_alert":
        if top:
            ex = f" e.g., _{top['examples'][0]}_" if top["examples"] else ""
            return {"reply": f"{header_brief}\nReason: **{top['intent']}** ({round(top['avg_conf']*100)}%){ex}"}
        cause, _ = fmea_lookup(model_intent)
        return {"reply": f"{header_brief}\nReason: **{model_intent}** → {cause}"}

    if q_intent == "root_cause":
        if top: return {"reply": f"{header_brief}\nLikely cause: **{top['cause']}**"}
        cause, _ = fmea_lookup(model_intent)
        return {"reply": f"{header_brief}\nLikely cause: **{cause}**"}

    if q_intent == "what_first":
        if top: return {"reply": f"{header_brief}\nDo this first: **{top['action']}**"}
        _, action = fmea_lookup(model_intent)
        return {"reply": f"{header_brief}\nDo this first: **{action}**"}

    if q_intent == "which_sensor":
        sensor, why = pick_suspicious_sensor(all_alerts, sensors_point, trend)
        val = sensors_point.get(sensor)
        val_str = f"{val:.3f}" if isinstance(val, (int, float)) else "n/a"
        reply = f"{header_brief}\nMost suspicious sensor: **{sensor}** (value {val_str}, {why})."
        if detailed and top:
            reply += f"\nHint: {top['intent']} → {top['cause']} · Action: {top['action']}"
        return {"reply": reply}

    if q_intent == "verify_fix":
        lines = [
            header_brief,
            "To verify the fix:",
            "1) Re-run the pipeline and confirm alerts have cleared.",
            "2) Check the affected sensor returns within the 5–95% historical range.",
            "3) Compare AI RUL vs baseline — the gap should narrow, not widen.",
        ]
        if top: lines.append(f"4) Specific to **{top['intent']}**: {top['action']}")
        return {"reply": "\n".join(lines)}

    # fallback summaries
    if not groups:
        cause, action = fmea_lookup(model_intent)
        return {"reply": f"{header_brief}\nLikely cause: **{cause}** · Action: **{action}**"}

    if brief and not detailed:
        g = groups[0]
        ex = f" · e.g., _{g['examples'][0]}_" if g["examples"] else ""
        return {"reply": f"{header_brief}\nTop issue: **{g['intent']}** (avg conf {round(g['avg_conf']*100)}%) · {g['action']}{ex}"}

    k = 3 if detailed else 2
    lines = [header_full, "🔧 **Recommended next steps (FMEA + alerts):**"]
    for g in groups[:k]:
        lines.append(f"• **{g['intent']}** — {g['count']} alert(s), avg conf {round(g['avg_conf']*100)}%")
        lines.append(f"   ↳ Cause: **{g['cause']}**")
        lines.append(f"   ↳ Action: **{g['action']}**")
        if g["examples"]:
            lines.append(f"   ↳ Example: _{g['examples'][0]}_")
    return {"reply": "\n".join(lines)}