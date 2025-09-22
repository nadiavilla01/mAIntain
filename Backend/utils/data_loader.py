import pandas as pd
import os

def load_base_data():
    df = pred_df = errs_df = None

    # Load base CSV directly from Backend/
    try:
        df = pd.read_csv("./fd001_normalized_for_pretrained.csv")
    except Exception as e:
        print(f"❌ Failed loading base CSV: {e}")

    # Load predicted RULs from pretrainedCMAPSS/
    try:
        pred_df_path = "./pretrainedCMAPSS/predictions_fd001.csv"
        pred_df = pd.read_csv(pred_df_path)
        if "unit_nr" not in pred_df.columns or "predicted_RUL" not in pred_df.columns:
            print("⚠️ predictions_fd001.csv must have 'unit_nr' and 'predicted_RUL'")
            pred_df = None
    except Exception as e:
        print(f"❌ Could not load prediction file: {e}")
        pred_df = None

    # Load error stats from pretrainedCMAPSS/
    try:
        err_df_path = "./pretrainedCMAPSS/unit_error_stats.csv"
        errs_df = pd.read_csv(err_df_path)
    except Exception as e:
        print(f"❌ Could not load errors file: {e}")
        errs_df = None

    return df, pred_df, errs_df


def status_from_pred(predicted_rul, mae=None):
    if predicted_rul is None:
        return "Unknown"
    m = float(mae) if mae is not None else 0.0

    crit_cut = 20 + 0.2 * m        # Critical = RUL very low
    unstab_cut = 50 + 0.3 * m      # Unstable = medium RUL or bad MAE

    if predicted_rul <= crit_cut:
        return "Critical"
    if (predicted_rul <= unstab_cut) or (mae is not None and mae > 10):
        return "Unstable"
    return "Normal"


def enrich_alert(msg):
    msg_lower = msg.lower()
    if "temperature" in msg_lower:
        return {"type": "Temperature", "message": msg}
    elif "vibration" in msg_lower:
        return {"type": "Vibration", "message": msg}
    elif "power" in msg_lower:
        return {"type": "Power", "message": msg}
    return {"type": "General", "message": msg}
