"""Export the V8 split to YOLO format, reusing the clinicians' drawn boxes.

    python3 bench/yolo_export.py --task merged4

Ultralytics wants images/<split>/*.jpg with labels/<split>/*.txt holding
"cls cx cy w h" normalised. The split, the patient-disjointness and the class
mapping all come from bench/common.py, so the detection benchmark is scored on
exactly the same frames as the classification grid - no separate split, no
accidental patient leakage between the two.

Normal frames are exported with an EMPTY label file, which is how YOLO encodes a
true negative. A detector that fires on healthy mucosa is useless, so they must
be in the training set.
"""
import argparse
import os
import shutil
import sys

import pandas as pd
import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench import common as C                                      # noqa: E402

MASKDIR = os.environ.get("BENCH_MASKDIR",
                         os.path.join(C.LOG_ROOT, "bench-lesion-masks-v21"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="merged4", choices=list(C.TASKS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    C.assert_split()
    out = args.out or f"data/yolo/{args.task}"
    if os.path.exists(out) and not args.force:
        print(f"[skip] {out}")
        print(os.path.join(out, "data.yaml"))
        return
    shutil.rmtree(out, ignore_errors=True)

    df = pd.read_csv(os.path.join(MASKDIR, "bench-lesion-bboxes.csv"))
    df = df[(df.reliable == 1) & (df.v8_split != "not-in-V8")]
    boxes = {}
    for _, r in df.iterrows():
        key = (r["v8_split"], r["class"], r["filename"])
        boxes.setdefault(key, []).append(r)

    # lesion classes only; Normal is the background, encoded as an empty label
    names = [c for c in C.classes_for(args.task) if c != "Normal"]
    cls_id = {c: i for i, c in enumerate(names)}
    counts = {s: [0, 0] for s in C.SPLITS}

    for sp in C.SPLITS:
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(out, sub, sp), exist_ok=True)
        for cls in sorted(os.listdir(os.path.join(C.DATA_ROOT, sp))):
            for fn in sorted(os.listdir(os.path.join(C.DATA_ROOT, sp, cls))):
                src = os.path.join(C.DATA_ROOT, sp, cls, fn)
                stem = os.path.splitext(fn)[0]
                dst = os.path.join(out, "images", sp, stem + ".jpg")
                with Image.open(src) as im:
                    W, H = im.size
                    im.convert("RGB").save(dst, quality=95)
                lines = []
                for r in boxes.get((sp, cls, fn), []):
                    tgt = C.classes_for(args.task)[C.map_label(args.task,
                                                               C.CLASSES_5.index(cls))]
                    if tgt == "Normal":
                        continue
                    s = r["mask_size"]
                    cx = (r.x0 + r.x1 + 1) / 2 / s
                    cy = (r.y0 + r.y1 + 1) / 2 / s
                    w = (r.x1 - r.x0 + 1) / s
                    h = (r.y1 - r.y0 + 1) / s
                    lines.append(f"{cls_id[tgt]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                open(os.path.join(out, "labels", sp, stem + ".txt"), "w").write(
                    "\n".join(lines))
                counts[sp][0] += 1
                counts[sp][1] += len(lines)

    yaml.safe_dump({"path": os.path.abspath(out), "train": "images/train",
                    "val": "images/val", "test": "images/test",
                    "names": {i: n for n, i in cls_id.items()}},
                   open(os.path.join(out, "data.yaml"), "w"), sort_keys=False)
    for sp in C.SPLITS:
        print(f"  {sp:6s} {counts[sp][0]:5d} images  {counts[sp][1]:5d} boxes")
    print(f"classes: {names}")
    print(os.path.join(out, "data.yaml"))


if __name__ == "__main__":
    main()
