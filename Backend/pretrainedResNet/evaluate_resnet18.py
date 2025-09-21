# fine_tune_resnet18.py
import os, json, math, argparse, random
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import models, transforms, datasets

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt

# ----------------------------
# Repro
# ----------------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----------------------------
# Tiny helpers
# ----------------------------
def stratified_per_class_split(targets, n_val_per_class=2):
    """Return train_idx, val_idx selecting exactly n_val_per_class per class for validation
       (if a class has <= n_val_per_class, keep at least 1 for train and 1 for val when possible)."""
    targets = np.asarray(targets)
    classes = np.unique(targets)
    train_idx, val_idx = [], []
    for c in classes:
        idx = np.where(targets == c)[0]
        np.random.shuffle(idx)
        if len(idx) <= 2:
            # edge case: keep 1/1 where possible
            split = 1
        else:
            split = min(n_val_per_class, max(1, len(idx)//4))  # safety on super tiny classes
        val_idx.extend(idx[:split].tolist())
        train_idx.extend(idx[split:].tolist())
    return np.array(train_idx), np.array(val_idx)

def make_sampler(labels):
    """Balanced sampler for training (helps a lot on tiny/imbalanced sets)."""
    counts = Counter(labels)
    class_weights = {c: 1.0 / counts[c] for c in counts}
    sample_weights = np.array([class_weights[l] for l in labels], dtype=np.float32)
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

def mixup(x, y, alpha=0.4):
    if alpha <= 0.0:
        return x, y, None, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def plot_confusion_matrix(cm, class_names, out_png="resnet18_confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True label', xlabel='Predicted label', title='ResNet-18 Confusion Matrix')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)
    fig.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./synthetic_images")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--val_per_class", type=int, default=2, help="how many images per class to hold out")
    parser.add_argument("--mixup", type=float, default=0.3, help="alpha; 0 disables")
    parser.add_argument("--lr_backbone", type=float, default=1e-4)
    parser.add_argument("--lr_head", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_model", type=str, default="fault_classifier_resnet18.pth")
    parser.add_argument("--out_json", type=str, default="resnet18_eval_metrics.json")
    parser.add_argument("--out_cm_png", type=str, default="resnet18_confusion_matrix.png")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Transforms ---
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.15, hue=0.02),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # --- Dataset & split ---
    full = datasets.ImageFolder(args.data, transform=None)
    print(f"📁 Loading dataset from {args.data}")
    print(f"  found {len(full)} images across {len(full.classes)} classes: {full.classes}")

    # make a deterministic stratified split per class
    np.random.seed(args.seed)
    all_targets = [t for _, t in full.samples]
    train_idx, val_idx = stratified_per_class_split(all_targets, n_val_per_class=args.val_per_class)

    # attach transforms per split
    full_train = datasets.ImageFolder(args.data, transform=train_tf)
    full_val   = datasets.ImageFolder(args.data, transform=val_tf)

    ds_train = Subset(full_train, train_idx)
    ds_val   = Subset(full_val,   val_idx)

    # class balance for sampler
    train_targets = [full.samples[i][1] for i in train_idx]
    sampler = make_sampler(train_targets)

    train_loader = DataLoader(ds_train, batch_size=args.batch, sampler=sampler, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(ds_val,   batch_size=args.batch, shuffle=False,   num_workers=2, pin_memory=True)

    print("🔎 Split summary (class : train/val):")
    counts_train = Counter(train_targets)
    counts_val   = Counter([full.samples[i][1] for i in val_idx])
    for c, name in enumerate(full.classes):
        print(f"  {name:12s}: {counts_train.get(c,0):>2d} / {counts_val.get(c,0):>2d}")

    # --- Model ---
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_f = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_f, len(full.classes))
    )
    model = model.to(device)

    # 2-stage fine-tuning: freeze backbone first, then unfreeze all
    for p in model.parameters():
        p.requires_grad = True
    for p in model.layer1.parameters(): p.requires_grad = False
    for p in model.layer2.parameters(): p.requires_grad = False
    for p in model.layer3.parameters(): p.requires_grad = False
    # layer4 + fc trainable first, then unfreeze everything later

    # discriminative LRs
    params = [
        {"params": model.layer4.parameters(), "lr": args.lr_backbone},
        {"params": model.fc.parameters(),     "lr": args.lr_head}
    ]
    optim = torch.optim.AdamW(params, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs, eta_min=args.lr_backbone*0.1)

    # label smoothing helps on tiny sets
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val = math.inf
    best_state = None
    bad = 0

    def evaluate():
        model.eval()
        y_true, y_pred = [], []
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                val_loss += criterion(out, yb).item() * xb.size(0)
                y_true.extend(yb.cpu().numpy().tolist())
                y_pred.extend(out.argmax(1).cpu().numpy().tolist())
        val_loss /= max(1, len(ds_val))
        return val_loss, np.array(y_true), np.array(y_pred)

    # --- Train ---
    E_UNFREEZE = max(2, args.epochs // 4)  # unfreeze after first quarter
    print("\n🚀 Training...")
    for epoch in range(1, args.epochs + 1):
        if epoch == E_UNFREEZE:
            for p in model.parameters(): p.requires_grad = True
            # reset optimizer with two groups (smaller LR for earlier layers)
            params = [
                {"params": model.layer1.parameters(), "lr": args.lr_backbone * 0.5},
                {"params": model.layer2.parameters(), "lr": args.lr_backbone * 0.75},
                {"params": model.layer3.parameters(), "lr": args.lr_backbone},
                {"params": model.layer4.parameters(), "lr": args.lr_backbone},
                {"params": model.fc.parameters(),     "lr": args.lr_head}
            ]
            optim = torch.optim.AdamW(params, weight_decay=args.weight_decay)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs - epoch + 1,
                                                               eta_min=args.lr_backbone*0.1)
            print("🔓 Unfroze backbone for full fine-tuning.")

        model.train()
        run_loss, right, seen = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            # MixUp (optional)
            if args.mixup > 0:
                xb, y_a, y_b, lam = mixup(xb, yb, alpha=args.mixup)
                out = model(xb)
                loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
                preds = out.argmax(1)
                # Use y_a for quick accuracy proxy
                right += (preds == y_a).sum().item()
            else:
                out = model(xb)
                loss = criterion(out, yb)
                preds = out.argmax(1)
                right += (preds == yb).sum().item()

            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            run_loss += loss.item() * xb.size(0)
            seen += xb.size(0)

        sched.step()
        tr_loss = run_loss / max(1, seen)
        tr_acc  = 100.0 * right / max(1, seen)

        val_loss, y_true, y_pred = evaluate()
        val_acc = 100.0 * accuracy_score(y_true, y_pred)

        print(f"Epoch {epoch:02d}/{args.epochs} | train loss {tr_loss:.4f} acc {tr_acc:5.1f}% | "
              f"val loss {val_loss:.4f} acc {val_acc:5.1f}%")

        if val_loss + 1e-5 < best_val:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print("⏹ Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # --- Final eval on VAL ---
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            y_true.extend(yb.cpu().numpy().tolist())
            y_pred.extend(out.argmax(1).cpu().numpy().tolist())
    y_true = np.array(y_true); y_pred = np.array(y_pred)

    cls_names = full.classes
    report = classification_report(y_true, y_pred, target_names=cls_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # save artifacts
    torch.save(model.state_dict(), args.out_model)
    print(f"\n💾 Model saved to {args.out_model}")

    plot_confusion_matrix(cm, cls_names, out_png=args.out_cm_png)
    print(f"🖼️ Confusion matrix → {args.out_cm_png}")

    # flatten metrics for JSON
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='micro', zero_division=0)

    out = {
        "num_images": len(full),
        "classes": cls_names,
        "split_counts": {
            "train": len(ds_train),
            "val": len(ds_val),
            "per_class_train": {cls_names[c]: int(counts_train.get(c, 0)) for c in range(len(cls_names))},
            "per_class_val":   {cls_names[c]: int(counts_val.get(c, 0))   for c in range(len(cls_names))},
        },
        "metrics": {
            "val_accuracy": round(accuracy_score(y_true, y_pred) * 100, 2),
            "macro_precision": round(macro_p * 100, 2),
            "macro_recall": round(macro_r * 100, 2),
            "macro_f1": round(macro_f1 * 100, 2),
            "micro_precision": round(micro_p * 100, 2),
            "micro_recall": round(micro_r * 100, 2),
            "micro_f1": round(micro_f1 * 100, 2),
        },
        "per_class": {
            cls: {
                "precision": round(report[cls]["precision"] * 100, 2),
                "recall":    round(report[cls]["recall"] * 100, 2),
                "f1":        round(report[cls]["f1-score"] * 100, 2),
                "support":   int(report[cls]["support"])
            } for cls in cls_names
        },
        "confusion_matrix": cm.tolist(),
        "config": {
            "epochs": args.epochs,
            "batch": args.batch,
            "val_per_class": args.val_per_class,
            "mixup_alpha": args.mixup,
            "lr_backbone": args.lr_backbone,
            "lr_head": args.lr_head,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "seed": args.seed
        }
    }
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"📊 Metrics → {args.out_json}")

if __name__ == "__main__":
    main()
