"""A2: lesion DETECTION, using the boxes the clinicians already drew.

    python3 bench/detect.py --model retinanet --task merged4

The classification benchmark is capped by 226 test frames and, on 5-class, by a
class with one training patient. Detection changes the denominator: it scores
LESIONS, and it answers the question a reader actually asks - where is it - which
frame-level accuracy never does.

The boxes are free. bench/masks_v21.py derives a mask for all 1,606 lesion frames
by diffing the V2.1 annotated export against the clean archive, so the drawn
ellipse becomes a bounding box with no manual annotation at all.

Protocol mirrors the classification grid: same patient-disjoint split, same
seed, boxes in 256x256 space, Normal frames kept as true negatives (a detector
that fires on healthy mucosa is useless), and mAP reported at [.5:.95] and .5.

This is also the pre-registered Track B trigger: if model families separate here
by more than the cross-validation fold spread, broader architecture search earns
its cost; if they do not, the ceiling is the data.
"""
import argparse
import json
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

MASKDIR = os.environ.get("BENCH_MASKDIR",
                         os.path.join(C.LOG_ROOT, "bench-lesion-masks-v21"))
SIZE = 256


def build_index(task):
    """path -> list of boxes. Normal frames appear with an empty list."""
    df = pd.read_csv(os.path.join(MASKDIR, "bench-lesion-bboxes.csv"))
    df = df[(df.reliable == 1) & (df.v8_split != "not-in-V8")]
    by_split = {s: {} for s in C.SPLITS}
    for _, r in df.iterrows():
        cls5 = C.CLASSES_5.index(r["class"])
        lab = C.map_label(task, cls5)
        p = os.path.join(C.DATA_ROOT, r["v8_split"], r["class"], r["filename"])
        if not os.path.exists(p):
            continue
        # +1: torchvision detectors reserve 0 for background
        by_split[r["v8_split"]].setdefault(p, []).append(
            [float(r.x0), float(r.y0), float(r.x1) + 1, float(r.y1) + 1, lab + 1])
    for sp in C.SPLITS:                       # Normal frames as true negatives
        d = os.path.join(C.DATA_ROOT, sp, "Normal")
        for fn in sorted(os.listdir(d)):
            by_split[sp].setdefault(os.path.join(d, fn), [])
    return by_split


class DetDataset(Dataset):
    def __init__(self, index, train=False):
        self.items = sorted(index.items())
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, boxes = self.items[i]
        with Image.open(path) as im:
            img = im.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(img, np.float32).transpose(2, 0, 1) / 255.0)
        b = torch.tensor([bb[:4] for bb in boxes], dtype=torch.float32).reshape(-1, 4)
        l = torch.tensor([int(bb[4]) for bb in boxes], dtype=torch.int64)
        if self.train and len(b) and torch.rand(1).item() < 0.5:      # hflip
            x = torch.flip(x, dims=[2])
            b = b.clone()
            b[:, [0, 2]] = SIZE - b[:, [2, 0]]
        return x, {"boxes": b, "labels": l}, path


def collate(batch):
    return [b[0] for b in batch], [b[1] for b in batch], [b[2] for b in batch]


def build_detector(name, num_classes):
    import torchvision
    from torchvision.models.detection import (retinanet_resnet50_fpn_v2,
                                              fasterrcnn_resnet50_fpn_v2,
                                              fcos_resnet50_fpn)
    if name == "retinanet":
        m = retinanet_resnet50_fpn_v2(weights="DEFAULT")
        from torchvision.models.detection.retinanet import RetinaNetClassificationHead
        a = m.anchor_generator.num_anchors_per_location()[0]
        m.head.classification_head = RetinaNetClassificationHead(
            in_channels=256, num_anchors=a, num_classes=num_classes,
            norm_layer=lambda c: torch.nn.GroupNorm(32, c))
        return m
    if name == "fcos":
        m = fcos_resnet50_fpn(weights="DEFAULT")
        from torchvision.models.detection.fcos import FCOSClassificationHead
        a = m.anchor_generator.num_anchors_per_location()[0]
        m.head.classification_head = FCOSClassificationHead(
            in_channels=256, num_anchors=a, num_classes=num_classes)
        return m
    if name == "fasterrcnn":
        m = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        inf = m.roi_heads.box_predictor.cls_score.in_features
        m.roi_heads.box_predictor = FastRCNNPredictor(inf, num_classes)
        return m
    raise ValueError(name)


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox",
                                  class_metrics=True)
    model.eval()
    for imgs, tgts, _ in loader:
        out = model([i.to(device) for i in imgs])
        metric.update([{k: v.detach().cpu() for k, v in o.items()} for o in out],
                      [{k: v for k, v in t.items()} for t in tgts])
    r = metric.compute()
    return {k: (float(v) if v.numel() == 1 else v.tolist())
            for k, v in r.items() if torch.is_tensor(v)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="retinanet",
                    choices=["retinanet", "fcos", "fasterrcnn"])
    ap.add_argument("--task", default="merged4", choices=list(C.TASKS))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--name", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    C.assert_split()
    C.seed_everything()
    name = args.name or args.model
    out = os.path.join(C.LOG_ROOT, f"bench-detect-{args.task}-{name}")
    if os.path.exists(os.path.join(out, "detect_test.json")) and not args.force:
        print(f"[skip] {out}")
        return
    os.makedirs(out, exist_ok=True)
    device = torch.device("cuda:0")

    idx = build_index(args.task)
    lesion_labels = sorted({b[4] for v in idx["train"].values() for b in v})
    num_classes = max(lesion_labels) + 1          # background + lesion classes
    print(f"lesion classes (1-based): {lesion_labels}  -> num_classes={num_classes}")
    for sp in C.SPLITS:
        nb = sum(len(v) for v in idx[sp].values())
        npos = sum(1 for v in idx[sp].values() if v)
        print(f"  {sp:6s} {len(idx[sp]):5d} frames  {npos:5d} with a lesion  {nb:5d} boxes")

    tr = DataLoader(DetDataset(idx["train"], train=True), batch_size=args.batch_size,
                    shuffle=True, num_workers=args.workers, collate_fn=collate)
    te = DataLoader(DetDataset(idx["test"]), batch_size=args.batch_size,
                    shuffle=False, num_workers=args.workers, collate_fn=collate)
    va = DataLoader(DetDataset(idx["val"]), batch_size=args.batch_size,
                    shuffle=False, num_workers=args.workers, collate_fn=collate)

    model = build_detector(args.model, num_classes).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.05)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    t0 = time.time()
    hist = []
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for imgs, tgts, _ in tr:
            imgs = [i.to(device) for i in imgs]
            tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]
            loss = sum(model(imgs, tgts).values())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss)
        sch.step()
        hist.append({"epoch": ep, "loss": round(tot / max(len(tr), 1), 5)})
        pd.DataFrame(hist).to_csv(os.path.join(out, "history.csv"), index=False)
        print(f"  ep {ep:3d}/{args.epochs}  loss {hist[-1]['loss']:.4f}", flush=True)

    res = {"val": evaluate(model, va, device, num_classes),
           "test": evaluate(model, te, device, num_classes)}
    res["meta"] = {"model": args.model, "task": args.task, "epochs": args.epochs,
                   "num_classes": num_classes, "seed": C.SEED,
                   "train_wallclock_min": round((time.time() - t0) / 60, 2),
                   "params_total": sum(p.numel() for p in model.parameters())}
    C.write_json(os.path.join(out, "detect_test.json"), res)
    t = res["test"]
    print(f"\nTEST  mAP@[.5:.95] {t.get('map'):.4f}   mAP@.5 {t.get('map_50'):.4f}   "
          f"mAR@100 {t.get('mar_100'):.4f}")
    C.log_progress(f"DET {args.task:7s} {name:22s} mAP={t.get('map'):.4f} "
                   f"mAP50={t.get('map_50'):.4f} {res['meta']['train_wallclock_min']:.1f}min")


if __name__ == "__main__":
    main()
