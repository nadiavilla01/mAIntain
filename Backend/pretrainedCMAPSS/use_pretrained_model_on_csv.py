import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys
import json
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_percentage_error


sys.path.append(os.path.abspath("../CMAPSS-release"))  

SEQUENCE_LENGTH = 30
MODEL_PATH = "../CMAPSS-release/trials/model_FD001.pkl"
CSV_PATH = "../fd001_normalized_for_pretrained.csv"
MAX_RUL = 125
PREDICTIONS_CSV = "predictions_fd001.csv"
PLOT_PATH = "rul_predictions_plot.png"
METRICS_JSON = "metrics_fd001.json"


print("📦 Loading pre-trained model...")
model = torch.load(MODEL_PATH, map_location=torch.device("cpu"), weights_only=False)
model.eval()


print(f"📥 Loading dataset: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)


sensor_cols = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9',
               's_11', 's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21']


df["RUL"] = df["RUL"].clip(upper=MAX_RUL) / MAX_RUL


features = df[["unit_nr", "time_cycles"] + sensor_cols + ["RUL"]]


print(f"🔁 Creating sequences with length = {SEQUENCE_LENGTH}...")
X_seq, y_seq, meta_info = [], [], []

for unit in features["unit_nr"].unique():
    df_unit = features[features["unit_nr"] == unit].reset_index(drop=True)
    for i in range(len(df_unit) - SEQUENCE_LENGTH):
        seq = df_unit.loc[i:i+SEQUENCE_LENGTH-1, sensor_cols].values
        target = df_unit.loc[i+SEQUENCE_LENGTH, "RUL"]
        X_seq.append(seq)
        y_seq.append(target)
        meta_info.append([
            int(df_unit.loc[i+SEQUENCE_LENGTH, "unit_nr"]),
            int(df_unit.loc[i+SEQUENCE_LENGTH, "time_cycles"])
        ])

X_seq = torch.tensor(np.array(X_seq), dtype=torch.float32)
y_seq = torch.tensor(y_seq, dtype=torch.float32)

print(f"✅ Input shape: {X_seq.shape} | Targets: {y_seq.shape}")


print("🎯 Running predictions...")
with torch.no_grad():
    y_pred, *_ = model(X_seq)
    y_pred = y_pred.squeeze()


y_seq = y_seq * MAX_RUL
y_pred = y_pred * MAX_RUL


true_rul_np = y_seq.numpy()
pred_rul_np = y_pred.numpy()

rmse = torch.sqrt(torch.mean((y_pred - y_seq) ** 2)).item()
mae = torch.mean(torch.abs(y_pred - y_seq)).item()
r2 = r2_score(true_rul_np, pred_rul_np)
medae = median_absolute_error(true_rul_np, pred_rul_np)


non_zero_mask = true_rul_np != 0
if np.any(non_zero_mask):
    mape = mean_absolute_percentage_error(true_rul_np[non_zero_mask], pred_rul_np[non_zero_mask])
else:
    mape = float('nan')


print(f"\n📊 Evaluation Metrics:")
print(f"   RMSE : {rmse:.2f}")
print(f"   MAE  : {mae:.2f}")
print(f"   R²   : {r2:.3f}")
print(f"   MedAE: {medae:.2f}")
print(f"   MAPE : {mape*100:.2f}%")


print("\n🔍 Sample Predictions (first 10):")
for i in range(10):
    print(f"[Unit {meta_info[i][0]:>3}, Cycle {meta_info[i][1]:>4}] → "
          f"True RUL: {y_seq[i].item():>6.2f} | Predicted RUL: {y_pred[i].item():>6.2f}")


print(f"\n💾 Saving predictions to: {PREDICTIONS_CSV}")
pred_df = pd.DataFrame(meta_info, columns=["unit_nr", "time_cycles"])
pred_df["true_RUL"] = true_rul_np
pred_df["predicted_RUL"] = pred_rul_np
pred_df["abs_error"] = np.abs(pred_df["true_RUL"] - pred_df["predicted_RUL"])
pred_df.to_csv(PREDICTIONS_CSV, index=False)


metrics = {
    "rmse": float(round(rmse, 2)),
    "mae": float(round(mae, 2)),
    "r2": float(round(r2, 3)),
    "medae": float(round(medae, 2)),
    "mape_percent": float(round(mape * 100, 2)) if not np.isnan(mape) else None,
}

with open(METRICS_JSON, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"📊 Metrics saved to: {METRICS_JSON}")


plt.figure(figsize=(10, 5))
plt.plot(y_seq[:300], label="True RUL", alpha=0.8)
plt.plot(y_pred[:300], label="Predicted RUL", alpha=0.8)
plt.title("🔍 RUL Prediction vs Ground Truth")
plt.xlabel("Sample")
plt.ylabel("RUL")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_PATH)
plt.show()
print(f"\n Plot saved to: {PLOT_PATH}")