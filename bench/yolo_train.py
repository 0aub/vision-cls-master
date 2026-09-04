"""SOTA detectors on the WCE boxes: YOLOv8/v11/v12 and RT-DETR.

    python3 bench/yolo_train.py --model yolo11s --task merged4 --epochs 150

torchvision's RetinaNet/FCOS/Faster R-CNN are 2017-2019 designs and were only
ever the floor of this comparison. This adds the current single-stage family
(YOLOv8, v11, v12) and a real-time transformer detector (RT-DETR), which is what
a detection benchmark should be measured against.

Same split, same patient-disjointness, same class mapping as the classification
grid, via bench/yolo_export.py. Ultralytics is AGPL-3.0 - fine for research, and
recorded in the report so the licence is not a surprise later.
"""
import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench import common as C                                      # noqa: E402

WEIGHTS = {
    "yolov8n": "yolov8n.pt", "yolov8s": "yolov8s.pt", "yolov8m": "yolov8m.pt",
    "yolo11n": "yolo11n.pt", "yolo11s": "yolo11s.pt", "yolo11m": "yolo11m.pt",
    "yolo12n": "yolo12n.pt", "yolo12s": "yolo12s.pt",
    "rtdetr-l": "rtdetr-l.pt", "rtdetr-x": "rtdetr-x.pt",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(WEIGHTS))
    ap.add_argument("--task", default="merged4", choices=list(C.TASKS))
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    name = args.name or f"{args.model}_{args.imgsz}"
    out = os.path.join(C.LOG_ROOT, f"bench-detect-{args.task}-{name}")
    if os.path.exists(os.path.join(out, "detect_test.json")) and not args.force:
        print(f"[skip] {out}")
        return
    os.makedirs(out, exist_ok=True)
    data = os.path.abspath(f"data/yolo/{args.task}/data.yaml")
    if not os.path.exists(data):
        raise SystemExit(f"missing {data}; run bench/yolo_export.py --task {args.task}")

    from ultralytics import RTDETR, YOLO
    Cls = RTDETR if args.model.startswith("rtdetr") else YOLO
    model = Cls(WEIGHTS[args.model])
    t0 = time.time()
    model.train(data=data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                seed=C.SEED, deterministic=True, project=out, name="train",
                exist_ok=True, verbose=False, plots=False, val=True)
    mins = (time.time() - t0) / 60.0

    res = {}
    for split in ("val", "test"):
        m = model.val(data=data, split=split, imgsz=args.imgsz, batch=args.batch,
                      project=out, name=f"eval_{split}", exist_ok=True, verbose=False,
                      plots=False)
        b = m.box
        res[split] = {"map": float(b.map), "map_50": float(b.map50),
                      "map_75": float(b.map75),
                      "map_per_class": [float(x) for x in getattr(b, "maps", [])],
                      "precision": float(b.mp), "recall": float(b.mr)}
    n_params = sum(p.numel() for p in model.model.parameters())
    res["meta"] = {"model": args.model, "task": args.task, "epochs": args.epochs,
                   "imgsz": args.imgsz, "params_total": int(n_params),
                   "train_wallclock_min": round(mins, 2), "seed": C.SEED,
                   "framework": "ultralytics (AGPL-3.0)"}
    C.write_json(os.path.join(out, "detect_test.json"), res)
    t = res["test"]
    print(f"\nTEST  mAP@[.5:.95] {t['map']:.4f}  mAP@.5 {t['map_50']:.4f}  "
          f"P {t['precision']:.4f}  R {t['recall']:.4f}")
    C.log_progress(f"DET {args.task:7s} {name:22s} mAP={t['map']:.4f} "
                   f"mAP50={t['map_50']:.4f} {mins:.1f}min")


if __name__ == "__main__":
    main()
