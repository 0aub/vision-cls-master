"""Lesion-centred crops as extra training data.

    python3 bench/lesion_crops.py --per_frame 2

The median lesion occupies 3.25% of a 256x256 frame, so resizing 512x480 to 224
leaves a lesion roughly 40 pixels across - and the network must also ignore the
95%+ of the frame that is patient-specific mucosa. That is precisely the
background the models were shown to memorise (Ulcer's single training patient).

A crop centred on the drawn ellipse, taken at a random scale around it, attacks
both at once: the lesion fills far more of the input, and the surrounding
patient-identifying context is mostly gone. Crops are TRAIN-split only and are
appended through --extra_train_dir, exactly like the copy-paste set.
"""
import argparse
import os
import random
import sys

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench import common as C                                      # noqa: E402

MASKDIR = os.environ.get("BENCH_MASKDIR",
                         os.path.join(C.LOG_ROOT, "bench-lesion-masks-v21"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_frame", type=int, default=2)
    ap.add_argument("--out", default="data/synthetic/V8-crops")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--min_ctx", type=float, default=1.6, help="min box-size multiple")
    ap.add_argument("--max_ctx", type=float, default=4.0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    C.assert_split()
    rng = random.Random(C.SEED)
    if os.path.exists(args.out) and not args.force:
        print(f"[skip] {args.out}")
        return

    df = pd.read_csv(os.path.join(MASKDIR, "bench-lesion-bboxes.csv"))
    df = df[(df.reliable == 1) & (df.v8_split == "train")]
    made = {}
    for _, r in df.iterrows():
        src = os.path.join(C.DATA_ROOT, "train", r["class"], r["filename"])
        if not os.path.exists(src):
            continue
        with Image.open(src) as im:
            W, H = im.size
            sx, sy = W / r["mask_size"], H / r["mask_size"]
            cx = (r.x0 + r.x1) / 2 * sx
            cy = (r.y0 + r.y1) / 2 * sy
            bw = max((r.x1 - r.x0) * sx, 8.0)
            bh = max((r.y1 - r.y0) * sy, 8.0)
            for k in range(args.per_frame):
                ctx = rng.uniform(args.min_ctx, args.max_ctx)
                half = max(bw, bh) * ctx / 2
                # jitter the centre so the lesion is not always dead centre,
                # which the network would otherwise learn as the cue
                jx = cx + rng.uniform(-0.15, 0.15) * half
                jy = cy + rng.uniform(-0.15, 0.15) * half
                l, t = jx - half, jy - half
                box = (int(max(0, l)), int(max(0, t)),
                       int(min(W, l + 2 * half)), int(min(H, t + 2 * half)))
                if box[2] - box[0] < 16 or box[3] - box[1] < 16:
                    continue
                crop = im.convert("RGB").crop(box).resize((args.size, args.size),
                                                          Image.BILINEAR)
                d = os.path.join(args.out, r["class"])
                os.makedirs(d, exist_ok=True)
                n = made.get(r["class"], 0)
                crop.save(os.path.join(d, f"CROP_{r['class']}_{n:05d}.png"))
                made[r["class"]] = n + 1
    C.write_json(os.path.join(args.out, "meta.json"),
                 {"per_frame": args.per_frame, "counts": made,
                  "context_multiple": [args.min_ctx, args.max_ctx],
                  "source": "TRAIN split only, boxes from " + MASKDIR})
    print("crops per class:", made, " total", sum(made.values()))


if __name__ == "__main__":
    main()
