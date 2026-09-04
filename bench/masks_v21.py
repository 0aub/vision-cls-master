"""A1: lesion masks at 100% coverage, from the V2.1 annotated export.

    python3 bench/masks_v21.py

bench/masks.py reconstructs masks by diffing the ringed pixels stored in the 2024
k-NN pickle against the clean archive - which only ever covered the 1,288 frames
that happened to be in that run's training split (80% of the corpus, and nothing
outside it).

V2.1 is the annotated export itself. Its flat lesion folders hold the SAME frames
as data/uncompressed/KAUHC-V2-full/_Prelabeling with the clinician's ring burned
in: matching is unambiguous (best-match L2 9-13 against a second-best of 40-54)
and the difference is 0.4-0.9% of pixels in dark brown-black. Diffing them gives
a mask for EVERY lesion frame, including every val and test frame.

Matching is a one-to-one assignment, not 1,606 independent nearest-neighbour
lookups: consecutive capsule frames are near-duplicates, so independent lookups
let two ringed frames claim the same clean frame. Same reasoning as the 2024
forensic run.

Writes log/bench-lesion-masks-v21/ with the same layout bench/masks.py produces
(masks/, masks_filled/, bench-lesion-bboxes.csv), so every downstream consumer -
cams.py, cam_geometry.py, copypaste.py - works unchanged.
"""
import argparse
import csv
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench import common as C                                      # noqa: E402
from bench.masks import MAX_RELIABLE_FRAC, fill_ellipse, v8_index, v8_name  # noqa: E402

V21 = "data/uncompressed/KAUHC-V2.1-raw/V2.1"
CLEAN = "data/uncompressed/KAUHC-V2-full/_Prelabeling"
OUT = os.path.join(C.LOG_ROOT, "bench-lesion-masks-v21")
SIZE = 256
THRESH = 2.0 / 255.0
CLASSES = ["AVM", "Erosion", "Ulcer", "Xanthoma"]


def rgb01(p, size=SIZE):
    with Image.open(p) as im:
        return np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR),
                          np.uint8).astype(np.float32) / 255.0


def clean_paths(cls):
    return sorted(os.path.join(dp, f)
                  for dp, _, fs in os.walk(os.path.join(CLEAN, cls))
                  for f in fs if f.lower().endswith(".bmp"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    C.assert_split()
    os.makedirs(os.path.join(OUT, "masks"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "masks_filled"), exist_ok=True)
    bbox_csv = os.path.join(OUT, "bench-lesion-bboxes.csv")
    if os.path.exists(bbox_csv) and not args.force:
        print(f"[skip] {bbox_csv} exists")
        return
    idx = v8_index()

    rows, unmatched = [], 0
    for cls in CLASSES:
        cp = clean_paths(cls)
        rp = sorted(os.path.join(V21, cls, f) for f in os.listdir(os.path.join(V21, cls))
                    if f.lower().endswith(".bmp"))
        print(f"--- {cls}: {len(rp)} ringed vs {len(cp)} clean", flush=True)
        clean = np.stack([rgb01(p) for p in cp])
        A = clean.reshape(len(cp), -1)
        B = np.stack([rgb01(p).reshape(-1) for p in rp])
        # ||a-b||^2 = ||a||^2 - 2a.b + ||b||^2 as one GEMM
        D = (np.einsum("ij,ij->i", A, A)[None, :] - 2.0 * (B @ A.T)
             + np.einsum("ij,ij->i", B, B)[:, None])
        np.maximum(D, 0, out=D)
        np.sqrt(D, out=D)
        ri, ci = linear_sum_assignment(D)
        for r, c in zip(ri, ci):
            ring = B[r].reshape(SIZE, SIZE, 3)
            cln = clean[c]
            diff = np.abs(ring - cln).max(axis=2)
            m = diff > THRESH
            ys, xs = np.nonzero(m)
            if len(ys) == 0:
                unmatched += 1
                continue
            fn = v8_name(os.path.relpath(cp[c], os.path.dirname(CLEAN)))
            split, _ = idx.get(fn, ("not-in-V8", cls))
            filled, method = fill_ellipse(m)
            Image.fromarray((m * 255).astype(np.uint8)).save(
                os.path.join(OUT, "masks", fn.replace(".bmp", ".png")))
            Image.fromarray((filled * 255).astype(np.uint8)).save(
                os.path.join(OUT, "masks_filled", fn.replace(".bmp", ".png")))
            rows.append({
                "filename": fn, "v8_split": split, "class": cls,
                "source_path": os.path.relpath(cp[c], "data/uncompressed/KAUHC-V2-full"),
                "flip_variant": "none",
                "y0": int(ys.min()), "y1": int(ys.max()),
                "x0": int(xs.min()), "x1": int(xs.max()),
                "stroke_px": int(m.sum()), "stroke_frac": round(float(m.mean()), 6),
                "lesion_px": int(filled.sum()), "lesion_frac": round(float(filled.mean()), 6),
                "fill_method": method,
                "reliable": int(filled.mean() <= MAX_RELIABLE_FRAC),
                "mask_size": SIZE, "match_l2": round(float(D[r, c]), 4),
                # the ringed frame is a real file here, not a row in the 2024
                # pickle; cams.py reads it directly. knn_row stays -1 to mark
                # "not from the pickle" and must never be used as an index.
                "ringed_path": os.path.relpath(rp[r], "."),
                "knn_row": -1,
            })
        del A, B, D, clean

    with open(bbox_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    by_split = Counter(r["v8_split"] for r in rows)
    C.write_json(os.path.join(OUT, "summary.json"), {
        "source": "V2.1 annotated export diffed against the clean archive",
        "masks_written": len(rows), "frames_with_no_difference": unmatched,
        "by_v8_split": dict(by_split),
        "by_class": dict(Counter(r["class"] for r in rows)),
        "reliable": int(sum(r["reliable"] for r in rows)),
        "median_match_l2": float(np.median([r["match_l2"] for r in rows])),
        "median_lesion_frac_pct": round(
            100 * float(np.median([r["lesion_frac"] for r in rows])), 3),
    })
    print(f"\n{len(rows)} masks; by split: {dict(by_split)}")
    print(f"[done] {OUT}")


if __name__ == "__main__":
    main()
