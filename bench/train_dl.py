"""The one config-driven entry point for every deep run in the v2 benchmark.

    python3 bench/train_dl.py --model efficientnet_b0 --task 5class
    python3 bench/train_dl.py --model dinov2_vitb14 --source hub-dinov2 \
                              --train_mode lora --task binary --epochs 50

Global protocol (BENCHMARK_BRIEF_V2.md, Phase A): 224x224, ImageNet norm, full
fine-tuning, Adam lr 1e-4, light flips on train only, cosine schedule, 100
epochs, batch 16 (halved on OOM and recorded), checkpoint = best val accuracy,
seed 1998, unshuffled deterministic evaluation.

Writes log/bench-<task>-<name>/: history.csv, summary_{train,val,test}.csv,
report_{split}.csv, preds_{split}.csv, PROBS_{split}.npy, cm_{split}.csv,
efficiency.json, best.pth (state_dict only), run_config.json.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench import common as C                                      # noqa: E402
from bench import efficiency as EFF                                # noqa: E402
from bench import losses as L                                      # noqa: E402
from src.modules import build_model                                # noqa: E402


def forward_logits(model, x):
    """googlenet returns a namedtuple of (main, aux2, aux1) in train mode."""
    out = model(x)
    if isinstance(out, tuple):
        return out[0]
    if hasattr(out, "logits"):
        return out.logits
    return out


def transforms_for(source, image_size):
    if source == "hub-dinov2":
        return C.dinov2_train_transform(image_size), C.dinov2_eval_transform(image_size)
    return C.train_transform(image_size), C.eval_transform(image_size)


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    P, Y = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True, dtype=torch.float)
        logits = forward_logits(model, xb)
        P.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        Y.append(yb.numpy())
    return np.vstack(P), np.concatenate(Y)


def train_one(args, out_dir, device):
    C.seed_everything(C.SEED)
    classes = C.classes_for(args.task)
    num_classes = len(classes)
    tf_train, tf_eval = transforms_for(args.source, args.image_size)

    train_ds_probe = C.image_folder("train", args.task, tf_eval)
    train_labels = [t for _, t in train_ds_probe.samples]

    sampler = None
    if args.sampler == "weighted":
        sampler = L.weighted_sampler(train_labels, num_classes)

    batch = args.batch_size
    history_path = os.path.join(out_dir, "history.csv")
    best_path = os.path.join(out_dir, "best.pth")

    while True:
        try:
            model = build_model(args.model, num_classes, source=args.source,
                                train_mode=args.train_mode,
                                feature_mode=args.feature_mode,
                                lora_r=args.lora_r, lora_alpha=args.lora_alpha,
                                lora_dropout=args.lora_dropout).to(device)
            _, tl = C.train_loader(args.task, tf_train, batch_size=batch,
                                   workers=args.workers, sampler=sampler)
            _, vl, _ = C.eval_loader("val", args.task, tf_eval,
                                     batch_size=max(batch, 16), workers=args.workers)
            criterion, loss_weights = L.build_criterion(args.loss, train_labels,
                                                        num_classes, device)
            criterion = criterion.to(device)
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(params, lr=args.lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                   T_max=args.epochs)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            history = []
            best_acc, best_epoch = -1.0, 0
            t_start = time.time()

            for epoch in range(1, args.epochs + 1):
                t0 = time.time()
                model.train()
                loss_sum, correct, seen = 0.0, 0, 0
                for xb, yb in tl:
                    xb = xb.to(device, non_blocking=True, dtype=torch.float)
                    yb = yb.to(device, non_blocking=True, dtype=torch.long)
                    optimizer.zero_grad(set_to_none=True)
                    logits = forward_logits(model, xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    optimizer.step()
                    loss_sum += loss.item() * xb.size(0)     # sample-weighted (fix 2)
                    correct += (logits.argmax(1) == yb).sum().item()
                    seen += xb.size(0)
                tr_loss, tr_acc = loss_sum / seen, correct / seen

                model.eval()
                vloss, vcorrect, vseen = 0.0, 0, 0
                vy, vp = [], []
                with torch.no_grad():
                    for xb, yb in vl:
                        xb = xb.to(device, non_blocking=True, dtype=torch.float)
                        yb = yb.to(device, non_blocking=True, dtype=torch.long)
                        logits = forward_logits(model, xb)
                        vloss += criterion(logits, yb).item() * xb.size(0)
                        pred = logits.argmax(1)
                        vcorrect += (pred == yb).sum().item()
                        vseen += xb.size(0)
                        vy.append(yb.cpu().numpy())
                        vp.append(pred.cpu().numpy())
                va_loss, va_acc = vloss / vseen, vcorrect / vseen
                from sklearn.metrics import precision_recall_fscore_support
                p, r, f1, _ = precision_recall_fscore_support(
                    np.concatenate(vy), np.concatenate(vp), average="macro",
                    zero_division=0, labels=list(range(num_classes)))
                epoch_lr = optimizer.param_groups[0]["lr"]
                scheduler.step()

                history.append({
                    "epoch": epoch, "loss": round(tr_loss, 6), "accuracy": round(tr_acc, 6),
                    "val_loss": round(va_loss, 6), "val_accuracy": round(va_acc, 6),
                    "val_macro_precision": round(float(p), 6),
                    "val_macro_recall": round(float(r), 6),
                    "val_macro_f1": round(float(f1), 6),
                    "lr": epoch_lr,
                    "time_s": round(time.time() - t0, 2),
                })
                # written every epoch: history.csv going missing is the reason
                # the smoke-test gate exists (addendum B6)
                pd.DataFrame(history).to_csv(history_path, index=False)

                if va_acc > best_acc:
                    best_acc, best_epoch = va_acc, epoch
                    torch.save(model.state_dict(), best_path)   # state_dict only
                if args.printing:
                    print(f"  ep {epoch:3d}/{args.epochs}  loss {tr_loss:.4f} acc {tr_acc:.4f}"
                          f"  val_loss {va_loss:.4f} val_acc {va_acc:.4f} val_mF1 {f1:.4f}"
                          f"  ({history[-1]['time_s']:.1f}s)", flush=True)

            wall_min = (time.time() - t_start) / 60.0
            peak_vram = (torch.cuda.max_memory_allocated() / 1024**2
                         if torch.cuda.is_available() else None)
            return model, dict(batch_size=batch, best_val_accuracy=best_acc,
                               best_epoch=best_epoch, train_wallclock_min=round(wall_min, 3),
                               peak_train_vram_mb=round(peak_vram, 1) if peak_vram else None,
                               loss_weights=loss_weights)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch <= 4:
                raise
            batch //= 2
            print(f"[OOM] retrying at batch_size={batch}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", default="5class", choices=["5class", "binary"])
    ap.add_argument("--source", default="torchvision",
                    choices=["torchvision", "hub-dinov2", "open_clip"])
    ap.add_argument("--train_mode", default="full", choices=["full", "probe", "lora"])
    ap.add_argument("--feature_mode", default="cls", choices=["cls", "cls+mean"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--loss", default="ce",
                    choices=["ce", "weighted_ce", "focal", "cb", "sampler"])
    ap.add_argument("--sampler", default="none", choices=["none", "weighted"])
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--name", default=None, help="run-dir name (default: --model)")
    ap.add_argument("--tier", default="", help="free-text tier label for the report")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--printing", type=int, default=1)
    ap.add_argument("--skip_cpu_latency", action="store_true")
    args = ap.parse_args()

    name = args.name or args.model
    C.assert_split()
    out_dir = C.run_dir(args.task, name)
    if C.already_done(args.task, name) and not args.force:
        print(f"[skip] {out_dir} already has summary_test.csv")
        return
    os.makedirs(out_dir, exist_ok=True)

    if not torch.cuda.is_available():
        raise SystemExit("refusing to train on CPU: no CUDA device visible.")
    device = torch.device("cuda:0")

    C.write_json(os.path.join(out_dir, "run_config.json"), vars(args))
    print(f"=== {args.task} / {name} "
          f"[{args.source}, {args.train_mode}, loss={args.loss}, sampler={args.sampler}] ===",
          flush=True)

    model, meta = train_one(args, out_dir, device)

    # reload the best checkpoint and evaluate deterministically
    model.load_state_dict(torch.load(os.path.join(out_dir, "best.pth"),
                                     map_location=device, weights_only=True))
    model.to(device).eval()
    classes = C.classes_for(args.task)
    _, tf_eval = transforms_for(args.source, args.image_size)
    summary = {}
    for split in C.SPLITS:
        _, dl, paths = C.eval_loader(split, args.task, tf_eval, batch_size=32,
                                     workers=args.workers)
        probs, y = evaluate(model, dl, device, len(classes))
        row = C.dump_split(out_dir, split, y, probs.argmax(1), paths, classes,
                           probs=probs, mean_ce=C.mean_cross_entropy(probs, y))
        summary[split] = row
        print(f"  [{split}] acc={row['accuracy']:.4f} macroF1={row['macro_f1']:.4f}",
              flush=True)

    eff = EFF.measure_all(model, args.image_size, skip_cpu=args.skip_cpu_latency)
    eff.update(meta)
    eff.update({"model": args.model, "name": name, "task": args.task,
                "source": args.source, "train_mode": args.train_mode,
                "tier": args.tier, "epochs": args.epochs, "lr": args.lr,
                "image_size": args.image_size, "seed": C.SEED,
                "loss": args.loss, "sampler": args.sampler,
                "torch": torch.__version__})
    C.write_json(os.path.join(out_dir, "efficiency.json"), eff)

    C.log_progress(f"{args.task:7s} {name:34s} test_acc={summary['test']['accuracy']:.4f} "
                   f"test_macroF1={summary['test']['macro_f1']:.4f} "
                   f"val_acc={meta['best_val_accuracy']:.4f} "
                   f"bs={meta['batch_size']} {meta['train_wallclock_min']:.1f}min")
    print(f"[done] {out_dir}")


if __name__ == "__main__":
    main()
