import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


INPUT_CSV = "predictions_fd001.csv"
TOP5_CSV = "top5_worst_predictions.csv"
UNIT_STATS_CSV = "unit_error_stats.csv"
HISTOGRAM_PNG = "abs_error_histogram.png"


print(f"📥 Loading predictions from: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)
df["abs_error"] = np.abs(df["true_RUL"] - df["predicted_RUL"])


top5 = df.sort_values(by="abs_error", ascending=False).head(5)
top5.to_csv(TOP5_CSV, index=False)
print("\n📊 Top 5 Worst Predictions:")
print(top5)


plt.figure(figsize=(10, 5))
plt.hist(df["abs_error"], bins=50, color="orange", edgecolor="black")
plt.title("Distribution of Absolute Errors")
plt.xlabel("Absolute Error (cycles)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.savefig(HISTOGRAM_PNG)
print(f"\n Histogram saved to: {HISTOGRAM_PNG}")


unit_stats = df.groupby("unit_nr").apply(
    lambda x: pd.Series({
        "avg_RMSE": np.sqrt(((x["predicted_RUL"] - x["true_RUL"])**2).mean()),
        "avg_MAE": x["abs_error"].mean()
    })
).reset_index()

unit_stats.to_csv(UNIT_STATS_CSV, index=False)
print(f"\n Per-unit error stats saved to: {UNIT_STATS_CSV}")


print("\n Per-unit Error (first 5 rows):")
print(unit_stats.head())

print("\n Evaluation complete!")