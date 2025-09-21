import os, sys, numpy as np, pandas as pd

BASE = os.getcwd()  # run from repo root
BASE_CSV = os.path.join(BASE, "fd001_normalized_for_pretrained.csv")
PRED_CSV = os.path.join(BASE, "predictions_fd001.csv")
ERRS_CSV = os.path.join(BASE, "unit_error_stats.csv")  


SENSORS = ['s_2','s_3','s_4','s_7','s_8','s_9','s_11','s_12','s_13','s_14','s_15','s_17','s_20','s_21']


U_UNSTABLE  = 990  
U_CRITICAL  = 991
U_TRENDING  = 992  

def make_unit_block(unit, start_cycle, n, last_rul, base_rul_start):
    """
    Block with mostly quiet sensors then a last-row spike to trigger alerts at 'end'.
    RUL: simple decreasing ramp ending at last_rul.
    """
    cycles = np.arange(start_cycle, start_cycle + n, dtype=int)
    rul = np.linspace(base_rul_start, last_rul, n).round(2)

    data = {
        "unit_nr": [unit]*n,
        "time_cycles": cycles,
        "RUL": rul,
    }
    for s in SENSORS:
        data[s] = np.random.normal(0.0, 0.02, size=n)

    data['s_2'][-1]  = 3.5   # temperature high
    data['s_3'][-1]  = 3.0   # vibration high
    data['s_4'][-1]  = 3.0   # power high
    data['s_7'][-1]  = 3.0   # speed high

    cols = ["unit_nr","time_cycles"] + SENSORS + ["RUL"]
    return pd.DataFrame(data)[cols]

def make_trending_degradation(unit, start_cycle, n, last_rul, base_rul_start,
                              ramp_cols=('s_2','s_3','s_4','s_7'),
                              ramp_len=12, ramp_peak=3.2):
    """
    Block where the last `ramp_len` cycles gradually ramp into an unstable regime
    across temperature/vibration/power/speed → alerts trigger and perceived health drops.
    """
    assert ramp_len < n, "ramp_len must be < n"
    cycles = np.arange(start_cycle, start_cycle + n, dtype=int)
    rul = np.linspace(base_rul_start, last_rul, n).round(2)

    data = {
        "unit_nr": [unit]*n,
        "time_cycles": cycles,
        "RUL": rul,
    }
    for s in SENSORS:
        data[s] = np.random.normal(0.0, 0.03, size=n)

    ramp_idx = np.arange(n - ramp_len, n)
    ramp = np.linspace(1.5, ramp_peak, ramp_len)  
    for col in ramp_cols:
        base = np.array(data[col])
        base[ramp_idx] = ramp + np.random.normal(0.0, 0.08, size=ramp_len) 
        data[col] = base

    cols = ["unit_nr","time_cycles"] + SENSORS + ["RUL"]
    return pd.DataFrame(data)[cols]

def safe_backup(path):
    if os.path.exists(path) and not os.path.exists(path + ".bak_testcases"):
        try:
            pd.read_csv(path).to_csv(path + ".bak_testcases", index=False)
            print(f"🧷 Backup → {path}.bak_testcases")
        except Exception as e:
            print(f"⚠️ backup failed for {path}: {e}")

def main():
    if not os.path.exists(BASE_CSV):
        print(f"❌ Base CSV not found: {BASE_CSV}")
        sys.exit(1)
    base_df = pd.read_csv(BASE_CSV)

    pred_df = pd.read_csv(PRED_CSV) if os.path.exists(PRED_CSV) else \
              pd.DataFrame(columns=["unit_nr","time_cycles","true_RUL","predicted_RUL","abs_error"])
    errs_df = pd.read_csv(ERRS_CSV) if os.path.exists(ERRS_CSV) else \
              pd.DataFrame(columns=["unit_nr","avg_RMSE","avg_MAE"])

    for p in (BASE_CSV, PRED_CSV, ERRS_CSV):
        if os.path.exists(p):
            safe_backup(p)

    test_ids = [U_UNSTABLE, U_CRITICAL, U_TRENDING]
    if "unit_nr" in base_df.columns:
        base_df = base_df[~base_df["unit_nr"].isin(test_ids)].copy()
    if "unit_nr" in pred_df.columns:
        pred_df = pred_df[~pred_df["unit_nr"].isin(test_ids)].copy()
    if "unit_nr" in errs_df.columns:
        errs_df = errs_df[~errs_df["unit_nr"].isin(test_ids)].copy()

    block_unstable = make_unit_block(U_UNSTABLE, start_cycle=180, n=21, last_rul=45, base_rul_start=65)
    block_critical = make_unit_block(U_CRITICAL, start_cycle=220, n=21, last_rul=12, base_rul_start=32)
    block_trend    = make_trending_degradation(
        U_TRENDING, start_cycle=260, n=28, last_rul=25, base_rul_start=58,
        ramp_cols=('s_2','s_3','s_4','s_7'), ramp_len=12, ramp_peak=3.2
    )

    base_df = pd.concat([base_df, block_unstable, block_critical, block_trend], ignore_index=True)
    base_df.to_csv(BASE_CSV, index=False)
    print(f" Appended units {U_UNSTABLE} (Unstable spike), {U_CRITICAL} (Critical spike), {U_TRENDING} (Trending-unstable) → {BASE_CSV}")

    rows = []
    for blk, uid, pred_rul in [
        (block_unstable, U_UNSTABLE, 42.0),
        (block_critical, U_CRITICAL, 12.0),
        (block_trend,    U_TRENDING, 25.0),
    ]:
        last = blk.iloc[-1]
        rows.append({
            "unit_nr": int(uid),
            "time_cycles": int(last["time_cycles"]),
            "true_RUL": float(last["RUL"]),
            "predicted_RUL": float(pred_rul),
            "abs_error": float(abs(pred_rul - float(last["RUL"]))),
        })

    pred_df = pd.concat([pred_df, pd.DataFrame(rows)], ignore_index=True)
    pred_df.to_csv(PRED_CSV, index=False)
    print(f" Injected predictions for {U_UNSTABLE}≈42h, {U_CRITICAL}≈12h, {U_TRENDING}≈25h → {PRED_CSV}")

    errs_df = pd.concat([
        errs_df,
        pd.DataFrame([
            {"unit_nr": U_UNSTABLE, "avg_RMSE": 15.0, "avg_MAE": 12.0},
            {"unit_nr": U_CRITICAL, "avg_RMSE": 18.0, "avg_MAE": 14.0},
            {"unit_nr": U_TRENDING, "avg_RMSE": 16.0, "avg_MAE": 11.0},
        ])
    ], ignore_index=True)
    errs_df.to_csv(ERRS_CSV, index=False)
    print(f" Updated/created error stats → {ERRS_CSV}")

    print("\n Ready. Start your API/UI (mode=end). Look for machines:")
    print(f"   • Machine {U_UNSTABLE}  → **Unstable** (last-row spikes).")
    print(f"   • Machine {U_CRITICAL}  → **Critical** (last-row spikes).")
    print(f"   • Machine {U_TRENDING}  → **Unstable** (multi-cycle ramp: temp/vibe/power/speed ↑).")

if __name__ == "__main__":
    np.random.seed(7)
    main()