import os, json, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, Counter
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_confmat(cm, labels, out_png):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
           xticklabels=labels, yticklabels=labels, ylabel='True', xlabel='Predicted',
           title='Confusion Matrix')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    vmax = cm.max() if cm.size else 1
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = int(cm[i, j]) if cm.size else 0
            ax.text(j, i, str(v), ha="center", va="center",
                    color=("white" if v > vmax/2 else "black"))
    fig.tight_layout(); fig.savefig(out_png, bbox_inches="tight"); plt.close(fig)

def per_class_split(base, val_per_class, seed):
    per_class = defaultdict(list)
    for idx, (_, y) in enumerate(base.samples):
        per_class[y].append(idx)
    rng = np.random.default_rng(seed)
    train_idx, val_idx, train_counts, val_counts, warns = [], [], {}, {}, []
    for c in range(len(base.classes)):
        idxs = per_class[c]; rng.shuffle(idxs); n = len(idxs)
        if n <= 1:
            warns.append(f"(warning) class id {c} has only {n} image; placing it in TRAIN only.")
            n_val = 0
        else:
            n_val = min(val_per_class, n - 1)
        val_sp = idxs[:n_val]; tr_sp = idxs[n_val:]
        val_idx.extend(val_sp); train_idx.extend(tr_sp)
        train_counts[base.classes[c]] = len(tr_sp)
        val_counts[base.classes[c]] = len(val_sp)
    return train_idx, val_idx, train_counts, val_counts, warns

def mixup(x, y, alpha):
    if alpha <= 0.0: return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam*x + (1-lam)*x[idx], y, y[idx], float(lam)

def rand_bbox(W, H, lam):
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W); cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W); x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H); y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2

def cutmix(x, y, alpha):
    if alpha <= 0.0: return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x1, y1, x2, y2 = rand_bbox(x.size(3), x.size(2), lam)
    x_m = x.clone()
    x_m[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    box_area = (x2-x1)*(y2-y1)
    lam_adj = 1.0 - box_area/(x.size(2)*x.size(3))
    return x_m, y, y[idx], float(lam_adj)

def mixup_cutmix(x, y, mixup_alpha, cutmix_alpha, p_cutmix=0.5):
    if cutmix_alpha > 0.0 and np.random.rand() < p_cutmix:
        return cutmix(x, y, cutmix_alpha)
    return mixup(x, y, mixup_alpha)

def cross_entropy_ls(logits, targets, ls=0.0):
    return F.cross_entropy(logits, targets, label_smoothing=float(ls))

def eval_model(model, loader, device, tta):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            if tta and tta > 0:
                acc = logits
                for _ in range(tta):
                    acc = acc + model(torch.flip(x, dims=[3]))
                logits = acc / (1 + tta)
            y_true.extend(y.numpy())
            y_pred.extend(torch.argmax(logits, 1).cpu().numpy())
    return np.array(y_true), np.array(y_pred)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="./synthetic_images")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--val_per_class", type=int, default=60)
    ap.add_argument("--head_warmup", type=int, default=3)
    ap.add_argument("--unfreeze_l4", type=int, default=4)
    ap.add_argument("--unfreeze_all", type=int, default=12)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_backbone", type=float, default=3e-5)
    ap.add_argument("--weight_decay", type=float, default=5e-5)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--mixup", type=float, default=0.30)
    ap.add_argument("--cutmix", type=float, default=0.15)
    ap.add_argument("--mixup_final", type=float, default=0.10)
    ap.add_argument("--tta", type=int, default=4)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--balanced_sampler", action="store_true")
    ap.add_argument("--model_out", type=str, default="fault_classifier_resnet18.pth")
    ap.add_argument("--metrics_out_json", type=str, default="resnet18_eval_metrics.json")
    ap.add_argument("--confmat_png", type=str, default="resnet18_confusion_matrix.png")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else ("mps" if torch.backends.mps.is_available() else "cpu"))

    base_no_tf = datasets.ImageFolder(args.data)
    classes = base_no_tf.classes
    print(f"📁 Loading dataset from {args.data}")
    print(f"  found {len(base_no_tf)} images across {len(classes)} classes: {classes}")

    train_idx, val_idx, train_counts, val_counts, warns = per_class_split(base_no_tf, args.val_per_class, args.seed)
    for w in warns: print(w)
    print("  TRAIN counts → " + ", ".join([f"{k}:{v}" for k,v in train_counts.items()]))
    print("  VAL counts → " + ", ".join([f"{k}:{v}" for k,v in val_counts.items()]))

    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.25,0.25,0.25,0.1)], p=0.35),
        transforms.RandomApply([transforms.RandomPerspective(distortion_scale=0.25)], p=0.15),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.15),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.20, scale=(0.02,0.12), ratio=(0.3,3.3)),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    train_base = datasets.ImageFolder(args.data, transform=train_tfms)
    val_base   = datasets.ImageFolder(args.data, transform=val_tfms)
    train_ds = Subset(train_base, train_idx)
    val_ds   = Subset(val_base, val_idx)

    if args.balanced_sampler:
        y_tr = [train_base.samples[i][1] for i in train_idx]
        freq = Counter(y_tr)
        w = {c: 1.0/f for c, f in freq.items()}
        weights = [w[y] for y in y_tr]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    pin = bool(torch.cuda.is_available())
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=shuffle, sampler=sampler,
                              num_workers=0, pin_memory=pin, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=0, pin_memory=pin, drop_last=False)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model = model.to(device)

    params_backbone = [p for n,p in model.named_parameters() if not n.startswith("fc.")]
    params_layer4 = list(model.layer4.parameters())
    params_head = list(model.fc.parameters())

    for p in params_backbone: p.requires_grad = False
    opt = torch.optim.AdamW(params_head, lr=args.lr_head, weight_decay=args.weight_decay)

    best_state, best_macro_f1, bad = None, -1.0, 0

    print("\n🧪 Training...")
    for epoch in range(1, args.epochs+1):
        if epoch == args.unfreeze_l4:
            print("🔓 Unfreezing layer4…")
            for p in params_layer4: p.requires_grad = True
            opt = torch.optim.AdamW(
                [{"params": params_layer4, "lr": args.lr_backbone},
                 {"params": params_head,   "lr": args.lr_head}],
                weight_decay=args.weight_decay
            )
        if epoch == args.unfreeze_all:
            print("🔓 Unfreezing entire backbone…")
            for p in params_backbone: p.requires_grad = True
            opt = torch.optim.AdamW(
                [{"params": params_backbone, "lr": args.lr_backbone},
                 {"params": params_head,     "lr": args.lr_head}],
                weight_decay=args.weight_decay
            )

        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        if args.epochs > 1:
            mixup_alpha = args.mixup + (args.mixup_final - args.mixup) * ((epoch - 1)/(args.epochs - 1))
        else:
            mixup_alpha = args.mixup

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x_m, y_a, y_b, lam = mixup_cutmix(x, y, mixup_alpha, args.cutmix, p_cutmix=0.5)
            opt.zero_grad(set_to_none=True)
            logits = model(x_m)
            loss_a = cross_entropy_ls(logits, y_a, ls=args.label_smoothing)
            loss_b = cross_entropy_ls(logits, y_b, ls=args.label_smoothing)
            loss = lam * loss_a + (1 - lam) * loss_b
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item() * x.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += x.size(0)

        train_loss = loss_sum / max(1,total)
        train_acc = 100.0 * correct / max(1,total)

        y_true, y_pred = eval_model(model, val_loader, device, args.tta)
        val_acc = 100.0 * float((y_true == y_pred).mean()) if len(y_true) else 0.0
        macro_f1 = float(f1_score(y_true, y_pred, labels=list(range(len(classes))),
                                  average="macro", zero_division=0)) if len(y_true) else 0.0

        print(f"Epoch {epoch:02d}/{args.epochs} | train loss {train_loss:.4f} | train acc {train_acc:4.1f}% | val acc {val_acc:4.1f}% | macro-F1 {macro_f1:.3f}")

        improve = macro_f1 > best_macro_f1 + 1e-4
        if improve:
            best_macro_f1 = macro_f1
            best_state = {k: v.detach().cpu() for k,v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience and epoch >= args.head_warmup:
                print("⏹ Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), args.model_out)
    print(f"\n💾 Model saved to {args.model_out}")

    y_true, y_pred = eval_model(model, val_loader, device, args.tta)
    labels_idx = list(range(len(classes)))
    report = classification_report(y_true, y_pred, labels=labels_idx,
                                   target_names=classes, digits=3,
                                   zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)
    try:
        save_confmat(cm, classes, args.confmat_png)
        print(f"🖼️ Confusion matrix → {args.confmat_png}")
    except Exception as e:
        print(f"(warning) Could not save confusion matrix image: {e}")

    per_class = {}
    missing = []
    for i, name in enumerate(classes):
        r = report.get(name, {"precision":0.0, "recall":0.0, "f1-score":0.0, "support":0})
        sup = int(r.get("support", 0))
        if sup == 0: missing.append(name)
        per_class[name] = {
            "precision": round(float(r.get("precision", 0.0)), 3),
            "recall": round(float(r.get("recall", 0.0)), 3),
            "f1": round(float(r.get("f1-score", 0.0)), 3),
            "support": sup
        }

    train_counts_named = {classes[k]: v for k,v in Counter([train_base.samples[i][1] for i in train_idx]).items()}
    val_counts_named   = {classes[k]: v for k,v in Counter([val_base.samples[i][1] for i in val_idx]).items()}

    out = {
        "classes": classes,
        "counts": {"train": train_counts_named, "val": val_counts_named},
        "config": {
            "epochs": args.epochs, "batch": args.batch, "val_per_class": args.val_per_class,
            "seed": args.seed, "head_warmup": args.head_warmup,
            "unfreeze_l4": args.unfreeze_l4, "unfreeze_all": args.unfreeze_all,
            "lr_head": args.lr_head, "lr_backbone": args.lr_backbone,
            "weight_decay": args.weight_decay, "patience": args.patience,
            "mixup": args.mixup, "cutmix": args.cutmix, "mixup_final": args.mixup_final,
            "tta": args.tta, "label_smoothing": args.label_smoothing,
            "balanced_sampler": bool(args.balanced_sampler)
        },
        "val_metrics": {
            "accuracy_percent": round(float((y_true == y_pred).mean()*100.0), 2) if len(y_true) else None,
            "macro_f1": round(float(f1_score(y_true, y_pred, labels=labels_idx,
                                             average="macro", zero_division=0)), 3) if len(y_true) else None,
            "per_class": per_class,
            "missing_in_val": missing
        },
        "confusion_matrix_labels": classes,
        "confusion_matrix": cm.tolist()
    }
    with open(args.metrics_out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"📊 Metrics → {args.metrics_out_json}")

    print("\nClassification Report (VAL):")
    print(classification_report(y_true, y_pred, labels=labels_idx,
                                target_names=classes, digits=3, zero_division=0))

if __name__ == "__main__":
    main()
