"""Phase E: lesion copy-paste augmentation for the rare classes (module M3c).

    python3 bench/copypaste.py --per_class 400

Takes the ellipse-interior masks from bench/masks.py, cuts the lesion region out
of a TRAINING lesion frame, and blends it onto a TRAINING Normal frame, so the
rare classes gain samples whose background is no longer the one patient the
class was drawn from.

That targeting is the point. The measured failure of this corpus is not that the
lesions are hard, it is that Ulcer has a single training patient (P_117, 383
frames), so a model memorises that patient's mucosa and transfers nothing. Copy-
paste cannot manufacture a second patient's LESION, but it can break the
lesion-to-background correlation, which is the part of the shortcut that is
fixable without new data.

Sources are restricted to the TRAIN split - no val or test pixels are ever used -
and the output is a separate folder, never mixed into data/splitted/.
"""
import argparse
import os
import random
import sys

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench import common as C                                      # noqa: E402

MASKDIR = os.environ.get("BENCH_MASKDIR",
                         os.path.join(C.LOG_ROOT, "bench-lesion-masks"))
OUT = "data/synthetic/V8-CP"
RARE = ["Erosion", "Ulcer", "Xanthoma"]


def load_index():
    import csv
    rows = [r for r in csv.DictReader(open(os.path.join(MASKDIR, "bench-lesion-bboxes.csv")))
            if int(r["reliable"]) and r["v8_split"] == "train"]
    by_cls = {}
    for r in rows:
        p = os.path.join(C.DATA_ROOT, "train", r["class"], r["filename"])
        m = os.path.join(MASKDIR, "masks_filled", r["filename"].replace(".bmp", ".png"))
        if os.path.exists(p) and os.path.exists(m):
            by_cls.setdefault(r["class"], []).append((p, m))
    return by_cls


def paste(lesion_path, mask_path, bg_path, rng, size=256, jitter=True):
    les = Image.open(lesion_path).convert("RGB").resize((size, size), Image.BILINEAR)
    msk = Image.open(mask_path).convert("L").resize((size, size), Image.NEAREST)
    bg = Image.open(bg_path).convert("RGB").resize((size, size), Image.BILINEAR)
    m = np.asarray(msk) > 127
    ys, xs = np.nonzero(m)
    if len(ys) < 16:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    patch = les.crop((x0, y0, x1, y1))
    pm = Image.fromarray((m[y0:y1, x0:x1] * 255).astype(np.uint8))

    if jitter:                                   # scale and flip the graft a little
        s = rng.uniform(0.8, 1.25)
        w, h = max(8, int(patch.width * s)), max(8, int(patch.height * s))
        patch, pm = patch.resize((w, h), Image.BILINEAR), pm.resize((w, h), Image.NEAREST)
        if rng.random() < 0.5:
            patch, pm = (patch.transpose(Image.FLIP_LEFT_RIGHT),
                         pm.transpose(Image.FLIP_LEFT_RIGHT))
        if rng.random() < 0.5:
            patch, pm = (patch.transpose(Image.FLIP_TOP_BOTTOM),
                         pm.transpose(Image.FLIP_TOP_BOTTOM))

    # keep the graft near the centre: capsule frames vignette hard at the rim,
    # and a lesion pasted onto black border is not a plausible training sample
    cx = rng.randint(int(size * 0.30), int(size * 0.70))
    cy = rng.randint(int(size * 0.30), int(size * 0.70))
    ox, oy = int(cx - patch.width / 2), int(cy - patch.height / 2)
    ox = max(0, min(ox, size - patch.width))
    oy = max(0, min(oy, size - patch.height))

    # feather the alpha so the graft does not carry a hard synthetic edge, which
    # a CNN would learn instantly
    alpha = pm.filter(ImageFilter.GaussianBlur(radius=2.5))
    out = bg.copy()
    out.paste(patch, (ox, oy), alpha)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_class", type=int, default=400)
    ap.add_argument("--classes", nargs="*", default=RARE)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    C.assert_split()
    rng = random.Random(C.SEED)

    if os.path.exists(args.out) and not args.force:
        n = sum(len(f) for _, _, f in os.walk(args.out))
        print(f"[skip] {args.out} exists with {n} images")
        return

    by_cls = load_index()
    bgs = [os.path.join(C.DATA_ROOT, "train", "Normal", f)
           for f in sorted(os.listdir(os.path.join(C.DATA_ROOT, "train", "Normal")))]
    print(f"{len(bgs)} Normal training backgrounds; lesion sources: "
          + ", ".join(f"{c}={len(v)}" for c, v in sorted(by_cls.items())))

    total = 0
    for cls in args.classes:
        src = by_cls.get(cls, [])
        if not src:
            print(f"[warn] no masked training sources for {cls}")
            continue
        d = os.path.join(args.out, cls)
        os.makedirs(d, exist_ok=True)
        made = 0
        tries = 0
        while made < args.per_class and tries < args.per_class * 8:
            tries += 1
            lp, mp = src[rng.randrange(len(src))]
            bg = bgs[rng.randrange(len(bgs))]
            im = paste(lp, mp, bg, rng, args.size)
            if im is None:
                continue
            im.save(os.path.join(d, f"CP_{cls}_{made:05d}.png"))
            made += 1
        total += made
        print(f"  {cls}: {made} synthetic frames from {len(src)} masked sources")
    C.write_json(os.path.join(args.out, "meta.json"), {
        "per_class": args.per_class, "classes": args.classes, "size": args.size,
        "seed": C.SEED, "total": total,
        "sources": "TRAIN split lesion frames with reliable masks only",
        "backgrounds": "TRAIN split Normal frames only",
        "blend": "feathered alpha (Gaussian sigma 2.5) over the ellipse interior",
    })
    print(f"[done] {total} images -> {args.out}")


if __name__ == "__main__":
    main()
