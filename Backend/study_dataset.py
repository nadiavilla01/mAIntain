import pandas as pd
import numpy as np
import os


DATA_DIR = "./CMAPSSData"  
train_path = os.path.join("./CMAPSS-release/CMAPSSData/train_FD001.txt")


column_names = ["unit", "time", "op_setting_1", "op_setting_2", "op_setting_3"] + \
               [f"s{i}" for i in range(1, 22)]  


df = pd.read_csv(train_path, sep="\s+", header=None, names=column_names)


useful_sensors = ['s2', 's3', 's4', 's7', 's8', 's9',
                  's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
df = df[["unit", "time"] + useful_sensors]


rul_df = df.groupby("unit")["time"].max().reset_index()
rul_df.columns = ["unit", "max_time"]
df = df.merge(rul_df, on="unit", how="left")
df["RUL"] = df["max_time"] - df["time"]
df.drop("max_time", axis=1, inplace=True)


df.to_csv("fd001_processed_for_lstm.csv", index=False)
print(" Saved fd001_processed_for_lstm.csv")