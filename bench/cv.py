"""Patient-grouped 4-fold cross-validation (the Statistics section of the brief).

    python3 bench/cv.py --model densenet201 --task 5class
    python3 bench/cv.py --model dinov2_vitb14 --source hub-dinov2 --train_mode lora \
                        --task binary --epochs 50

All 2,363 frames of V8-KAUHC are pooled and split by PATIENT with GroupKFold(4),
so no patient appears in both the training and the held-out fold. Normal frames
carry no patient folder in the source archive (they are frame-level), so each
Normal frame is its own group - which reproduces the frame-level treatment the
published split already gives them, and is stated as such in the report.

Checkpoint selection inside each fold uses a grouped validation slice carved out
of the training folds, never the held-out fold.

Writes log/bench-cv-<task>-<name>/: per-fold summary + pooled out-of-fold
predictions, confusion matrix and metrics.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench import common as C                                      # noqa: E402
from bench import losses as L                                      # noqa: E402
from bench.train_dl import forward_logits, transforms_for          # noqa: E402
from src.modules import build_model                                # noqa: E402


class ListDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        with Image.open(p) as im:
            x = self.transform(im.convert("RGB"))
        return x, y


def pool_all(task, data_root=C.DATA_ROOT):
    items, groups = [], []
    for sp in C.SPLITS:
        for cls in sorted(os.listdir(os.path.join(data_root, sp))):
            # every task's mapping lives in one place; this used to be a local
            # 5class-or-binary conditional, which silently sent merged4 down the
            # binary branch and produced a binary CV labelled as merged4
            y = C.map_label(task, C.CLASSES_5.index(cls))
            d = os.path.join(data_root, sp, cls)
            for fn in sorted(os.listdir(d)):
                p = os.path.join(d, fn)
                items.append((p, y))
                groups.append(C.patient_of(p))
    return items, np.array(groups)


def fit_fold(args, tr_items, va_items, device, num_classes):
    tf_train, tf_eval = transforms_for(args.source, args.image_size, args.aug)
    tl = DataLoader(ListDataset(tr_items, tf_train), batch_size=args.batch_size,
                    shuffle=True, num_workers=args.workers, pin_memory=True)
    vl = DataLoader(ListDataset(va_items, tf_eval), batch_size=32, shuffle=False,
                    num_workers=args.workers, pin_memory=True)
    model = build_model(args.model, num_classes, source=args.source,
                        train_mode=args.train_mode).to(device)
    crit, _ = L.build_criterion(args.loss, [y for _, y in tr_items], num_classes,
                                device, label_smoothing=args.label_smoothing)
    crit = crit.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "adamw":
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9,
                              weight_decay=args.weight_decay)
    else:
        opt = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, args.epochs - args.warmup_epochs))
    if args.warmup_epochs > 0:
        warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1e-3,
                                                 total_iters=args.warmup_epochs)
        sch = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cos],
                                                    milestones=[args.warmup_epochs])
    else:
        sch = cos
    best_acc, best_sd = -1.0, None
    for ep in range(args.epochs):
        model.train()
        for xb, yb in tl:
            xb = xb.to(device, dtype=torch.float, non_blocking=True)
            yb = yb.to(device, dtype=torch.long, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = crit(forward_logits(model, xb), yb)
            loss.backward()
            opt.step()
        sch.step()
        model.eval()
        ok = n = 0
        with torch.no_grad():
            for xb, yb in vl:
                xb = xb.to(device, dtype=torch.float, non_blocking=True)
                p = forward_logits(model, xb).argmax(1).cpu()
                ok += (p == yb).sum().item()
                n += len(yb)
        acc = ok / max(n, 1)
        if acc > best_acc:
            best_acc = acc
            best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_sd)
    return model, best_acc


@torch.no_grad()
def predict(model, items, tf, device, workers):
    dl = DataLoader(ListDataset(items, tf), batch_size=32, shuffle=False,
                    num_workers=workers, pin_memory=True)
    P, Y = [], []
    model.eval()
    for xb, yb in dl:
        xb = xb.to(device, dtype=torch.float, non_blocking=True)
        P.append(torch.softmax(forward_logits(model, xb).float(), 1).cpu().numpy())
        Y.append(yb.numpy())
    return np.vstack(P), np.concatenate(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", default="5class", choices=["5class", "binary", "merged4"])
    ap.add_argument("--source", default="torchvision",
                    choices=["torchvision", "hub-dinov2", "open_clip"])
    ap.add_argument("--train_mode", default="full", choices=["full", "probe", "lora"])
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--loss", default="ce")
    # the cross-validation must train the SAME way the headline grid does, or the
    # CV numbers describe a model that appears nowhere in the leaderboard
    ap.add_argument("--optimizer", default="adam", choices=["adam", "adamw", "sgd"])
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_epochs", type=int, default=0)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--aug", default="light", choices=["light", "strong"])
    ap.add_argument("--protocol", default="tuned")
    ap.add_argument("--name", default=None)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    name = args.name or args.model
    C.assert_split()
    out_dir = os.path.join(C.LOG_ROOT, f"bench-cv-{args.task}-{name}")
    done = os.path.join(out_dir, "cv_summary.csv")
    if os.path.exists(done) and not args.force:
        print(f"[skip] {done} exists")
        return
    os.makedirs(out_dir, exist_ok=True)
    C.seed_everything()
    device = torch.device("cuda:0")
    classes = C.classes_for(args.task)
    K = len(classes)

    items, groups = pool_all(args.task)
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=args.folds)
    y_all = np.array([y for _, y in items])
    print(f"pooled {len(items)} frames, {len(set(groups))} groups "
          f"({sum(1 for g in set(groups) if g.startswith('P_'))} patients + "
          f"{sum(1 for g in set(groups) if not g.startswith('P_'))} frame-level Normal)",
          flush=True)

    oof_p = np.zeros((len(items), K))
    oof_fold = np.zeros(len(items), dtype=int)
    fold_rows = []
    rng = np.random.default_rng(C.SEED)
    t0 = time.time()
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(np.zeros(len(items)), y_all, groups), 1):
        tr_groups = np.array(sorted(set(groups[tr_idx])))
        rng.shuffle(tr_groups)
        n_val = max(1, int(round(args.val_frac * len(tr_groups))))
        val_groups = set(tr_groups[:n_val].tolist())
        va_sel = [i for i in tr_idx if groups[i] in val_groups]
        tr_sel = [i for i in tr_idx if groups[i] not in val_groups]
        tr_items = [items[i] for i in tr_sel]
        va_items = [items[i] for i in va_sel]
        te_items = [items[i] for i in te_idx]
        tf = transforms_for(args.source, args.image_size, args.aug)[1]
        tf0 = time.time()
        model, val_acc = fit_fold(args, tr_items, va_items, device, K)
        P, Y = predict(model, te_items, tf, device, args.workers)
        oof_p[te_idx] = P
        oof_fold[te_idx] = fold
        from sklearn.metrics import accuracy_score, f1_score
        acc = accuracy_score(Y, P.argmax(1))
        # a patient-grouped fold can legitimately contain no frames of a class;
        # averaging that class's F1 in as 0 measures fold composition, not the
        # model, so per-fold macro F1 covers the classes present in that fold
        present = sorted(set(np.unique(Y).tolist()))
        f1 = f1_score(Y, P.argmax(1), average="macro", zero_division=0,
                      labels=present)
        f1_all = f1_score(Y, P.argmax(1), average="macro", zero_division=0,
                          labels=list(range(K)))
        fold_rows.append({"fold": fold, "n_train": len(tr_items), "n_val": len(va_items),
                          "n_test": len(te_items),
                          "test_patients": len(set(groups[te_idx])),
                          "val_accuracy": round(val_acc, 6),
                          "test_accuracy": round(float(acc), 6),
                          "test_macro_f1": round(float(f1), 6),
                          "test_macro_f1_all_labels": round(float(f1_all), 6),
                          "classes_present": len(present),
                          "minutes": round((time.time() - tf0) / 60, 2)})
        print(f"  fold {fold}: n_test={len(te_items)} acc={acc:.4f} macroF1={f1:.4f} "
              f"({fold_rows[-1]['minutes']:.1f}min)", flush=True)
        del model
        torch.cuda.empty_cache()

    fdf = pd.DataFrame(fold_rows)
    paths = [p for p, _ in items]
    C.dump_split(out_dir, "oof", y_all, oof_p.argmax(1), paths, classes,
                 probs=oof_p, mean_ce=C.mean_cross_entropy(oof_p, y_all))
    pd.DataFrame({"path": paths, "fold": oof_fold, "y_true": y_all,
                  "y_pred": oof_p.argmax(1)}).to_csv(
        os.path.join(out_dir, "oof_predictions.csv"), index=False)
    summ = {"model": name, "task": args.task, "folds": args.folds,
            "protocol": args.protocol, "optimizer": args.optimizer, "lr": args.lr,
            "weight_decay": args.weight_decay, "warmup_epochs": args.warmup_epochs,
            "label_smoothing": args.label_smoothing, "aug": args.aug,
            "epochs": args.epochs,
            "grouping": "patient-disjoint GroupKFold; Normal frames are frame-level "
                        "(one group each), matching the published split",
            "mean_test_accuracy": round(float(fdf.test_accuracy.mean()), 6),
            "std_test_accuracy": round(float(fdf.test_accuracy.std(ddof=1)), 6),
            "mean_test_macro_f1": round(float(fdf.test_macro_f1.mean()), 6),
            "std_test_macro_f1": round(float(fdf.test_macro_f1.std(ddof=1)), 6),
            "pooled_oof_accuracy": round(float((oof_p.argmax(1) == y_all).mean()), 6),
            "pooled_oof_macro_f1": round(float(__import__("sklearn.metrics",
                fromlist=["f1_score"]).f1_score(y_all, oof_p.argmax(1),
                average="macro", zero_division=0)), 6),
            "headline": "pooled out-of-fold: every frame predicted exactly once, "
                        "by a model that never saw that patient",
            "total_minutes": round((time.time() - t0) / 60, 2)}
    fdf.to_csv(os.path.join(out_dir, "cv_folds.csv"), index=False)
    pd.DataFrame([summ]).to_csv(done, index=False)
    C.write_json(os.path.join(out_dir, "cv_summary.json"), summ)
    C.log_progress(f"CV {args.task:7s} {name:30s} "
                   f"acc={summ['mean_test_accuracy']:.4f}+-{summ['std_test_accuracy']:.4f} "
                   f"macroF1={summ['mean_test_macro_f1']:.4f} {summ['total_minutes']:.1f}min")
    print(pd.DataFrame([summ]).T.to_string())


if __name__ == "__main__":
    main()
