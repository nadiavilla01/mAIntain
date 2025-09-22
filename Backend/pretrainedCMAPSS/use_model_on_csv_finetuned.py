import os, sys, json, math, random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, Subset
import torch.nn as nn
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import HuberRegressor


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.amp import autocast, GradScaler




sys.path.append("../CMAPSS-release")
CSV_PATH       = "../fd001_normalized_for_pretrained.csv"
MODEL_PATH     = "../CMAPSS-release/trials/model_FD001.pkl"           # pre-trained object
FINETUNED_PATH = "../CMAPSS-release/trials/model_FD001_finetuned.pkl"

PREDICTIONS_CSV = "predictions_fd001.csv"
METRICS_JSON    = "metrics_fd001.json"
PLOT_PATH       = "rul_predictions_plot.png"
SUMMARY_CSV     = "machines_fd001_summary.csv"
TOP5_WORST_CSV  = "top5_worst_predictions.csv"




SEQUENCE_LENGTH = 30
MAX_RUL = 125
SENSOR_COLS = ['s_2','s_3','s_4','s_7','s_8','s_9','s_11','s_12','s_13','s_14','s_15','s_17','s_20','s_21']
SMOOTH_ALPHA = 0.2




FINE_TUNE = True
EPOCHS = 14
HEAD_ONLY_EPOCHS = 1
BATCH_SIZE = 256
LR = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 4


LOSS_WEIGHT_BETA = 1.8        # increase weighttt
GAMMA_TOP = 0.20              # small weight at top region
HUBER_DELTA = 0.02
L2SP_ALPHA = 2e-5
LAMBDA_MONO = 0.02            # monotonic h
LAMBDA_SLOPE = 0.01           # slope-matching 
SLOPE_TOP_CUTOFF = 0.90       # enforce near EOL


GRAD_CLIP_NORM = 1.0          




USE_MC_DROPOUT_AVG = True
MC_AVG_PASSES = 30
MC_AVG_BATCH  = 512

CALIBRATE_GLOBAL = True
CALIBRATE_LAST   = True

USE_MC_DROPOUT_LAST = True
MC_DROPOUT_PASSES_LAST = 20

NUM_WORKERS = 0




def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

set_seed(7)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = torch.cuda.is_available()

print("📦 Loading pre-trained model…")
model = torch.load(MODEL_PATH, map_location=torch.device("cpu"), weights_only=False)
model.to(device)
pretrained_ref = {n: p.clone().detach().to(device) for n, p in model.named_parameters()}

print(f"📥 Loading dataset: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
df["RUL"] = df["RUL"].clip(upper=MAX_RUL) / MAX_RUL
features = df[["unit_nr","time_cycles"] + SENSOR_COLS + ["RUL"]].copy()

print(f"🔁 Creating sequences (L={SEQUENCE_LENGTH})…")
X_seq, y_seq, meta_info = [], [], []
for unit in features["unit_nr"].unique():
    df_u = features[features["unit_nr"] == unit].reset_index(drop=True)
    for i in range(len(df_u) - SEQUENCE_LENGTH):
        X_seq.append(df_u.loc[i:i+SEQUENCE_LENGTH-1, SENSOR_COLS].values)
        y_seq.append(df_u.loc[i+SEQUENCE_LENGTH, "RUL"])
        meta_info.append([int(df_u.loc[i+SEQUENCE_LENGTH,"unit_nr"]),
                          int(df_u.loc[i+SEQUENCE_LENGTH,"time_cycles"])])

X_seq = torch.tensor(np.array(X_seq), dtype=torch.float32)
y_seq = torch.tensor(np.array(y_seq), dtype=torch.float32)
N = len(meta_info)
print(f"✅ Input shape: {X_seq.shape} | Targets: {y_seq.shape}")

units_arr  = np.array([u for u,_ in meta_info])
cycles_arr = np.array([c for _,c in meta_info])


prev_idx = np.full(N, -1, dtype=np.int64)
for u in np.unique(units_arr):
    idxs = np.where(units_arr == u)[0]
    idxs = idxs[np.argsort(cycles_arr[idxs])]
    if len(idxs) > 1:
        prev_idx[idxs[1:]] = idxs[:-1]
prev_idx_t = torch.from_numpy(prev_idx)


all_units = features["unit_nr"].unique()
rng = np.random.default_rng(7); rng.shuffle(all_units)
split = int(len(all_units) * 0.8)
train_units, val_units = set(all_units[:split]), set(all_units[split:])
train_idx = np.where(np.isin(units_arr, list(train_units)))[0]
val_idx   = np.where(np.isin(units_arr, list(val_units)))[0]

meta_arr       = np.array(meta_info)
val_units_arr  = meta_arr[val_idx, 0]
val_cycles_arr = meta_arr[val_idx, 1]

all_idx = torch.arange(N, dtype=torch.long)
ds = TensorDataset(X_seq, y_seq, all_idx)
pin = torch.cuda.is_available()
train_loader = DataLoader(Subset(ds, train_idx), batch_size=BATCH_SIZE, shuffle=True,
                          drop_last=False, pin_memory=pin, num_workers=NUM_WORKERS)
val_loader   = DataLoader(Subset(ds, val_idx),   batch_size=BATCH_SIZE, shuffle=False,
                          drop_last=False, pin_memory=pin, num_workers=NUM_WORKERS)

def rmse_torch(yhat, y): return torch.sqrt(torch.mean((yhat - y)**2))

def last_linear_module(m: nn.Module):
    linear_layers = [mod for mod in m.modules() if isinstance(mod, nn.Linear)]
    return linear_layers[-1] if linear_layers else None

def set_requires_grad(m: nn.Module, flag: bool):
    for p in m.parameters(): p.requires_grad = flag

def huber(residual, delta=0.03):
    abs_r = residual.abs()
    quad = 0.5 * (abs_r**2)
    lin  = delta * (abs_r - 0.5*delta)
    return torch.where(abs_r <= delta, quad, lin)

def enable_mc_dropout(m: nn.Module):
    for mod in m.modules():
        name = mod.__class__.__name__.lower()
        if isinstance(mod, nn.Dropout) or "dropout" in name:
            mod.train()  # keep dropout active !!

def mc_avg_predict(model, X_tensor, passes=30, batch_size=512):
    model.eval(); enable_mc_dropout(model)
    N = X_tensor.shape[0]
    sum_pred = np.zeros(N, dtype=np.float64)
    dl = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for _ in range(passes):
            ofs = 0
            cur = np.empty(N, dtype=np.float32)
            for (xb,) in dl:
                xb = xb.to(device)
                yh, *_ = model(xb)
                b = yh.squeeze().detach().cpu().numpy()
                cur[ofs:ofs+len(b)] = b; ofs += len(b)
            sum_pred += cur
    avg = sum_pred / passes
    return np.clip(avg, 0.0, 1.0).astype(np.float32)




if FINE_TUNE:
    print("🛠️ Fine-tuning… (head warm-up → unfreeze, Huber + β(1-y)+γy, L2-SP, mono hinge + slope-match, cosine LR, AMP, grad clip, early stop on VAL-LAST MAE)")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR*0.1)
    scaler = GradScaler("cuda", enabled=AMP)

    head = last_linear_module(model)
    if head is not None:
        print("   └─ Stage 1: freeze backbone, train final Linear head")
        set_requires_grad(model, False); set_requires_grad(head, True)
    else:
        print("   └─ No explicit Linear head detected; training full model.")

    best_score, bad_epochs, best_state = math.inf, 0, None

    for epoch in range(1, EPOCHS+1):
        if epoch == HEAD_ONLY_EPOCHS + 1 and head is not None:
            print("   └─ Stage 2: unfreeze all")
            set_requires_grad(model, True)
            opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR*0.1)
            scaler = GradScaler("cuda", enabled=AMP)

        model.train()
        tr_loss, nobs = 0.0, 0
        beta_now = LOSS_WEIGHT_BETA * (epoch / max(1, EPOCHS))
        for xb, yb, ib in train_loader:
            xb, yb, ib = xb.to(device), yb.to(device), ib.to(device)
            opt.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=AMP):
                yhat, *_ = model(xb)
                yhat = yhat.squeeze()

                res  = yhat - yb
                w    = 1.0 + beta_now * (1.0 - yb) + GAMMA_TOP * yb
                mse_robust = huber(res, HUBER_DELTA)
                mse = (mse_robust * w).mean()

                # L2-SP to keep close to pretrained reference
                l2sp = 0.0
                for name, p in model.named_parameters():
                    if p.requires_grad and name in pretrained_ref:
                        ref = pretrained_ref[name]
                        l2sp = l2sp + torch.sum((p - ref)**2)

                # monotonic + slope consistency
                pos = {int(idx.item()): j for j, idx in enumerate(ib)}
                pairs = []
                prev_global = prev_idx_t[ib].cpu().tolist()
                for g, pidx in zip(ib.cpu().tolist(), prev_global):
                    if pidx != -1 and pidx in pos:
                        pairs.append((pos[g], pos[pidx]))

                if pairs:
                    cur_pos  = torch.tensor([i for i,_ in pairs], device=device, dtype=torch.long)
                    prev_pos = torch.tensor([j for _,j in pairs], device=device, dtype=torch.long)

                    mono = torch.relu(yhat[cur_pos] - yhat[prev_pos]).mean()

                    yprev_true = yb[prev_pos]
                    mask = (yprev_true >= SLOPE_TOP_CUTOFF)
                    if mask.any():
                        d_pred = (yhat[cur_pos] - yhat[prev_pos])[mask]
                        d_true = (yb[cur_pos]  - yb[prev_pos])[mask]
                        slope_loss = huber(d_pred - d_true, HUBER_DELTA).mean()
                    else:
                        slope_loss = yhat.new_zeros(())
                else:
                    mono = yhat.new_zeros(())
                    slope_loss = yhat.new_zeros(())

                loss = mse + L2SP_ALPHA * l2sp + LAMBDA_MONO * mono + LAMBDA_SLOPE * slope_loss

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)   
            scaler.step(opt); scaler.update()

            tr_loss += loss.item() * xb.size(0); nobs += xb.size(0)
        tr_loss /= max(nobs, 1)
        scheduler.step()

        # validation
        model.eval()
        with torch.no_grad():
            y_true, y_pred = [], []
            for xb, yb, _ in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                with autocast("cuda", enabled=AMP):
                    yh, *_ = model(xb)
                y_true.append(yb.detach().cpu()); y_pred.append(yh.squeeze().detach().cpu())
            y_true = torch.cat(y_true); y_pred = torch.clamp(torch.cat(y_pred), 0.0, 1.0)

            val_rmse = rmse_torch(y_pred, y_true).item()
            val_mae  = torch.mean(torch.abs(y_pred - y_true)).item()

            yt = (y_true.numpy() * MAX_RUL)
            yp = (y_pred.numpy() * MAX_RUL)
            val_df_epoch = pd.DataFrame({"unit": val_units_arr, "cycle": val_cycles_arr, "yt": yt, "yp": yp})
            last = val_df_epoch.sort_values(["unit","cycle"]).groupby("unit").tail(1)
            mae_last_epoch = float(np.mean(np.abs(last["yp"] - last["yt"])))
        print(f"  Epoch {epoch:02d} | train {tr_loss:.5f} | val RMSE {val_rmse:.4f} | val MAE {val_mae:.4f} | ↳ val-LAST MAE {mae_last_epoch:.3f}h")

        if mae_last_epoch < best_score - 1e-4:
            best_score = mae_last_epoch; bad_epochs = 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print("⏹ Early stopping."); break

    if best_state is not None:
        model.load_state_dict(best_state)

    model_cpu = model.to("cpu")
    torch.save(model_cpu, FINETUNED_PATH)
    print(f"💾 Saved fine-tuned model → {FINETUNED_PATH}")
    model.to(device)




print("🎯 Running predictions…")
model.eval()
if USE_MC_DROPOUT_AVG:
    print(f"   └─ MC-Dropout averaging: {MC_AVG_PASSES} passes")
    y_pred_norm = mc_avg_predict(model, X_seq, passes=MC_AVG_PASSES, batch_size=MC_AVG_BATCH)
    y_pred = torch.tensor(y_pred_norm, dtype=torch.float32)
else:
    with torch.no_grad():
        with autocast("cuda", enabled=AMP):
            y_pred, *_ = model(X_seq.to(device))
    y_pred = y_pred.squeeze().detach().cpu()
y_pred = torch.clamp(y_pred, 0.0, 1.0)




y_true_np_raw = (y_seq.numpy() * MAX_RUL)
y_pred_np_raw = (y_pred.numpy() * MAX_RUL)

global_cal = None
y_pred_np_glob = y_pred_np_raw.copy()
if CALIBRATE_GLOBAL:
    try:
        X_cal = y_pred_np_raw[train_idx].reshape(-1, 1)
        y_cal = y_true_np_raw[train_idx]
        global_cal = HuberRegressor(alpha=0.0, epsilon=1.35).fit(X_cal, y_cal)
        a, b = float(global_cal.coef_[0]), float(global_cal.intercept_)
        y_pred_np_glob = np.clip(a * y_pred_np_raw + b, 0.0, MAX_RUL)
        print(f"   └─ Global calibration: y' = {a:.4f} * y + {b:.4f}")
    except Exception as e:
        print(f"   └─ Global calibration skipped: {e}")
        global_cal = None

pred_df = pd.DataFrame(meta_info, columns=["unit_nr","time_cycles"])
pred_df["true_RUL_raw"] = y_true_np_raw
pred_df["pred_RUL_glob"] = y_pred_np_glob
pred_df["predicted_RUL"] = pred_df["pred_RUL_glob"]

last_cycle_cal, use_last_alt = None, False
if CALIBRATE_LAST:
    try:
        last_idx_all   = pred_df.groupby('unit_nr')['time_cycles'].idxmax()
        train_mask     = np.isin(pred_df["unit_nr"].values, list(train_units))
        last_idx_train = pred_df[train_mask].groupby('unit_nr')['time_cycles'].idxmax()
        val_mask_units = np.isin(pred_df["unit_nr"].values, list(val_units))
        last_idx_val   = pred_df[val_mask_units].groupby('unit_nr')['time_cycles'].idxmax()

        X_last = pred_df.loc[last_idx_train, "pred_RUL_glob"].to_numpy().reshape(-1,1)
        y_last = pred_df.loc[last_idx_train, "true_RUL_raw"].to_numpy()

        if np.std(y_last) < 1e-3 or np.std(X_last) < 1e-3:
            print("   └─ Keep base last-cycle (degenerate TRAIN-last variance).")
        else:
            last_cycle_cal = HuberRegressor(alpha=0.0, epsilon=1.35).fit(X_last, y_last)
            aL, bL = float(last_cycle_cal.coef_[0]), float(last_cycle_cal.intercept_)
            if not (0.5 <= aL <= 1.5) or abs(bL) > 15:
                print(f"   └─ Keep base last-cycle (reject a={aL:.4f}, b={bL:.4f}).")
            else:
                yt_last = pred_df.loc[last_idx_val, "true_RUL_raw"].to_numpy()
                yp_last_base = pred_df.loc[last_idx_val, "pred_RUL_glob"].to_numpy()
                yp_last_alt  = np.clip(aL * yp_last_base + bL, 0.0, MAX_RUL)
                mae_base = float(np.mean(np.abs(yp_last_base - yt_last)))
                mae_alt  = float(np.mean(np.abs(yp_last_alt  - yt_last)))
                gain = (mae_base - mae_alt) / max(mae_base, 1e-6)
                if mae_alt <= mae_base and gain >= 0.05:
                    all_last_base = pred_df.loc[last_idx_all, "pred_RUL_glob"].to_numpy()
                    all_last_alt  = np.clip(aL * all_last_base + bL, 0.0, MAX_RUL)
                    pred_df.loc[last_idx_all, "predicted_RUL"] = all_last_alt
                    use_last_alt = True
                    print("   └─ Last-cycle calibration ON (helpful).")
                else:
                    print("   └─ Keep base last-cycle (no clear gain).")
    except Exception as e:
        print(f"   └─ Last-cycle calibration skipped: {e}")




y_true_np = pred_df["true_RUL_raw"].to_numpy()
y_pred_np = pred_df["predicted_RUL"].to_numpy()
pred_df["abs_error"] = np.abs(pred_df["true_RUL_raw"] - pred_df["predicted_RUL"])

def smape(y_true: np.ndarray, y_pred: np.ndarray):
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom != 0
    return float(np.mean(2.0 * np.abs(y_pred - y_true)[mask] / denom[mask]) * 100.0) if mask.any() else None

def compute_metrics_on_indices(idxs, tag):
    yt = y_true_np[idxs]; yp = y_pred_np[idxs]
    rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
    mae  = float(np.mean(np.abs(yp - yt)))
    r2v  = float(r2_score(yt, yp))
    med  = float(median_absolute_error(yt, yp))
    mask = yt != 0
    mape = float(mean_absolute_percentage_error(yt[mask], yp[mask]) * 100) if mask.any() else None
    smp  = smape(yt, yp)
    print(f"[{tag}] RMSE {rmse:.2f} | MAE {mae:.2f} | R² {r2v:.3f} | MedAE {med:.2f} | MAPE {mape if mape is not None else 'n/a'} | sMAPE {smp if smp is not None else 'n/a'}")
    return {"rmse": round(rmse,2), "mae": round(mae,2), "r2": round(r2v,3),
            "medae": round(med,2), "mape_percent": round(mape,2) if mape is not None else None,
            "smape_percent": round(smp,2) if smp is not None else None}

print("\n📊 Evaluation Metrics (GLOBAL on all sequences):")
metrics_global = compute_metrics_on_indices(np.arange(len(y_true_np)), "GLOBAL")
print("📊 Evaluation Metrics (VAL split only):")
metrics_val = compute_metrics_on_indices(val_idx, "VAL")





pred_df_out = pred_df.rename(columns={"true_RUL_raw":"true_RUL"})[
    ["unit_nr","time_cycles","true_RUL","predicted_RUL","abs_error"]
]
pred_df_out.to_csv(PREDICTIONS_CSV, index=False)
print(f"💾 Predictions → {PREDICTIONS_CSV}")


val_mask_units = np.isin(pred_df["unit_nr"].values, list(val_units))
val_df_all = pred_df[val_mask_units].copy()
last_idx_val = val_df_all.groupby('unit_nr')['time_cycles'].idxmax()

val_last = val_df_all.loc[last_idx_val, ["true_RUL_raw","predicted_RUL"]].to_numpy()
yt_last, yp_last = val_last[:,0], val_last[:,1]
rmse_last = float(np.sqrt(np.mean((yp_last - yt_last)**2)))
mae_last  = float(np.mean(np.abs(yp_last - yt_last)))
r2_last   = float(r2_score(yt_last, yp_last))
med_last  = float(median_absolute_error(yt_last, yp_last))
mask_last = yt_last != 0
mape_last = float(mean_absolute_percentage_error(yt_last[mask_last], yp_last[mask_last]) * 100) if mask_last.any() else None
within5   = float(np.mean(np.abs(yp_last - yt_last) <= 5)  * 100)
within10  = float(np.mean(np.abs(yp_last - yt_last) <= 10) * 100)
print(f"[VAL-LAST] RMSE {rmse_last:.2f} | MAE {mae_last:.2f} | R² {r2_last:.3f} | MedAE {med_last:.2f}")
print(f"[VAL-LAST] ≤5h: {within5:.1f}% | ≤10h: {within10:.1f}%")

k = 5
val_tail5 = (val_df_all.sort_values(["unit_nr","time_cycles"]).groupby("unit_nr").tail(k))
yt5 = val_tail5["true_RUL_raw"].to_numpy()
yp5 = val_tail5["predicted_RUL"].to_numpy()
rmse_last5 = float(np.sqrt(np.mean((yp5 - yt5)**2)))
mae_last5  = float(np.mean(np.abs(yp5 - yt5)))
print(f"[VAL-LAST@{k}] RMSE {rmse_last5:.2f} | MAE {mae_last5:.2f}")


worst = (val_df_all.loc[last_idx_val]
         .assign(last_abs_error=lambda d: np.abs(d["true_RUL_raw"] - d["predicted_RUL"]))
         .sort_values("last_abs_error", ascending=False).head(5))
worst.to_csv(TOP5_WORST_CSV, index=False)


plt.figure(figsize=(10,5))
plt.plot(y_true_np[:300], label="True RUL", alpha=0.8)
plt.plot(y_pred_np[:300], label="Predicted RUL", alpha=0.8)
plt.title("RUL Prediction vs Ground Truth")
plt.xlabel("Sample"); plt.ylabel("RUL (hours)")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=120)
plt.close()
print(f"🖼️ Plot → {PLOT_PATH}")




def enforce_nonincreasing(arr: np.ndarray) -> np.ndarray:
    return np.minimum.accumulate(arr[::-1])[::-1]

pred_df['pred_smooth'] = (
    pred_df.groupby('unit_nr', group_keys=False)['predicted_RUL']
           .apply(lambda s: s.ewm(alpha=SMOOTH_ALPHA).mean())
)
pred_df['pred_mono'] = (
    pred_df.groupby('unit_nr', group_keys=False)['pred_smooth']
           .apply(lambda s: pd.Series(enforce_nonincreasing(s.values), index=s.index))
)
pred_df['abs_error'] = np.abs(pred_df['true_RUL_raw'] - pred_df['pred_mono'])

last_idx_all = pred_df.groupby('unit_nr')['time_cycles'].idxmax()
unit_last = pred_df.loc[last_idx_all, ['unit_nr','time_cycles','true_RUL_raw','pred_mono','abs_error']].rename(
    columns={'true_RUL_raw':'true_RUL','pred_mono':'pred_RUL','abs_error':'last_abs_error'}
)
per_unit = pred_df.groupby('unit_nr').agg(
    avg_RMSE=('abs_error', lambda x: float(np.sqrt(np.mean(x**2)))),
    avg_MAE =('abs_error', 'mean')
).reset_index()
summary = unit_last.merge(per_unit, on='unit_nr', how='left')

def map_status(rul, mae):
    if rul <= 20: return 'Critical'
    if (rul <= 50) or (mae > 10): return 'Unstable'
    return 'Normal'

summary['status'] = summary.apply(lambda r: map_status(r['pred_RUL'], r['avg_MAE']), axis=1)
summary['severity'] = (
    np.clip((50 - np.clip(summary['pred_RUL'], 0, 50)) / 50, 0, 1) * 70 +
    np.clip(summary['avg_MAE'] / 30, 0, 1) * 30
).round(1)

def mc_dropout_predict_last(model, X_seq_tensor, last_indices, passes=20):
    if passes <= 0: return np.zeros(len(last_indices), dtype=np.float32)
    model.eval(); enable_mc_dropout(model)
    X_last = X_seq_tensor[last_indices].to(device)
    preds = []
    with torch.no_grad():
        for _ in range(passes):
            yhat, *_ = model(X_last)
            preds.append(yhat.squeeze().detach().cpu().numpy())
    preds = np.stack(preds, axis=0)
    std_norm = preds.std(axis=0)
    return (std_norm * MAX_RUL).astype(np.float32)

if USE_MC_DROPOUT_LAST:
    try:
        last_indices_seq = last_idx_all.values
        unc_last_h = mc_dropout_predict_last(model, X_seq, last_indices_seq, passes=MC_DROPOUT_PASSES_LAST)
        last_unc_df = pred_df.loc[last_idx_all, ['unit_nr']].copy()
        last_unc_df['uncertainty_h'] = np.round(unc_last_h, 2)
        summary = summary.merge(last_unc_df, on='unit_nr', how='left')
    except Exception as e:
        print(f"⚠️ MC-Dropout uncertainty skipped: {e}")

summary = summary.rename(columns={'time_cycles':'last_cycle'}).sort_values(
    ['status','severity'], ascending=[False, False]
)
summary.to_csv(SUMMARY_CSV, index=False)
print(f"💾 Per-unit summary → {SUMMARY_CSV}")
print(f"💾 Worst-5 (VAL last-cycle) → {TOP5_WORST_CSV}")




metrics_json = {
    "global": metrics_global,
    "val": metrics_val,
    "val_last_cycle": {
        "rmse": round(rmse_last,2),
        "mae": round(mae_last,2),
        "r2": round(r2_last,3),
        "medae": round(med_last,2),
        "mape_percent": round(mape_last,2) if mape_last is not None else None,
        "pct_within_5h": round(within5,1),
        "pct_within_10h": round(within10,1),
        "last_k": 5,
        "last_k_rmse": round(rmse_last5,2),
        "last_k_mae": round(mae_last5,2)
    },
    "calibration": {
        "global": ({
            "enabled": True,
            "coef": float(global_cal.coef_[0]),
            "intercept": float(global_cal.intercept_)
        } if global_cal is not None else {"enabled": False}),
        "last_cycle": ({
            "enabled": True,
            "used": bool(use_last_alt),
            "coef": float(last_cycle_cal.coef_[0]),
            "intercept": float(last_cycle_cal.intercept_)
        } if last_cycle_cal is not None else {"enabled": False, "used": False})
    },
    "finetuned": bool(FINE_TUNE),
    "config": {
        "sequence_length": SEQUENCE_LENGTH,
        "max_rul": MAX_RUL,
        "sensors": SENSOR_COLS,
        "smooth_alpha": SMOOTH_ALPHA,
        "loss_weight_beta": LOSS_WEIGHT_BETA,
        "gamma_top": GAMMA_TOP,
        "huber_delta": HUBER_DELTA,
        "l2sp_alpha": L2SP_ALPHA,
        "lambda_mono": LAMBDA_MONO,
        "lambda_slope": LAMBDA_SLOPE,
        "slope_top_cutoff": SLOPE_TOP_CUTOFF,
        "head_only_epochs": HEAD_ONLY_EPOCHS,
        "scheduler": "CosineAnnealingLR",
        "grad_clip_norm": GRAD_CLIP_NORM,
        "amp": AMP,
        "mc_dropout_avg_passes": MC_AVG_PASSES if USE_MC_DROPOUT_AVG else 0,
        "mc_dropout_last_passes": MC_DROPOUT_PASSES_LAST if USE_MC_DROPOUT_LAST else 0
    }
}
with open(METRICS_JSON, "w") as f:
    json.dump(metrics_json, f, indent=2)
print(f"Metrics → {METRICS_JSON}")