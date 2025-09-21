import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../")

def load_base_data():
    df = pred_df = errs_df = None

    try:
        df = pd.read_csv(os.path.join(DATA_DIR, "fd001_normalized_for_pretrained.csv"))
    except Exception as e:
        print(f" Failed loading base CSV: {e}")

    try:
        pred_df_path = os.path.join(DATA_DIR, "predictions_fd001.csv")
        pred_df = pd.read_csv(pred_df_path)
        if "unit_nr" not in pred_df.columns or "predicted_RUL" not in pred_df.columns:
            print(" predictions_fd001.csv must have 'unit_nr' and 'predicted_RUL'")
            pred_df = None
    except Exception as e:
        print(f" Could not load prediction file: {e}")
        pred_df = None

    try:
        err_df_path = os.path.join(DATA_DIR, "unit_error_stats.csv")
        errs_df = pd.read_csv(err_df_path)
    except Exception as e:
        print(f"Could not load errors file: {e}")
        errs_df = None

    return df, pred_df, errs_df





def status_from_pred(predicted_rul, mae=None):
    if predicted_rul is None:
        return "Unknown"
    m = float(mae) if mae is not None else 0.0
    crit_cut = 20 + 0.2 * m    
    unstab_cut = 50 + 0.3 * m  

    if predicted_rul <= crit_cut:
        return "Critical"
    if (predicted_rul <= unstab_cut) or (mae is not None and mae > 10):
        return "Unstable"
    return "Normal"


def enrich_alert(msg):
    if "temperature" in msg.lower():
        return {"type": "Temperature", "message": msg}
    elif "vibration" in msg.lower():
        return {"type": "Vibration", "message": msg}
    elif "power" in msg.lower():
        return {"type": "Power", "message": msg}
    return {"type": "General", "message": msg}