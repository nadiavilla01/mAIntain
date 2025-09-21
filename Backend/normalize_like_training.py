import pandas as pd
from sklearn.preprocessing import StandardScaler


CSV_PATH = "fd001_processed_for_lstm.csv"
OUTPUT_PATH = "fd001_normalized_for_pretrained.csv"


print("📥 Loading:", CSV_PATH)
df = pd.read_csv(CSV_PATH)


df = df.rename(columns={
    "unit": "unit_nr",
    "time": "time_cycles",
})


sensor_cols = ['s2', 's3', 's4', 's7', 's8', 's9',
               's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
sensor_cols_renamed = [f"s_{col[1:]}" for col in sensor_cols] 

sensor_rename_map = dict(zip(sensor_cols, sensor_cols_renamed))
df = df.rename(columns=sensor_rename_map)


scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[sensor_cols_renamed] = scaler.fit_transform(df_scaled[sensor_cols_renamed])


df_scaled.to_csv(OUTPUT_PATH, index=False)
print(f"Saved normalized dataset to: {OUTPUT_PATH}")