"""C1: overlay-derived lesion masks (module M3a).

GATE (addendum section C). First check whether an annotated-export dataset - the
ringed frames - exists for the CURRENT val/test lesion frames. If it does, build
masks for all splits. If it does not, report that and fall back to the 1,288
training lesion frames whose ringed pixels survive inside the archived
v7-knn/best.pkl, reconstructing each mask by diffing the stored (published)
pixels against the raw archive frame.

    python3 bench/masks.py

Writes log/bench-lesion-masks/:
    masks/<v8_filename>.png        the drawn STROKE - the pixels the clinician's
                                   pen changed, i.e. what a ring-trained model
                                   can key on
    masks_filled/<v8_filename>.png the ellipse INTERIOR - the lesion region the
                                   clinician circled, which is the localization
                                   ground truth for the pointing game
    bench-lesion-bboxes.csv        one row per mask (v8 split, class, bbox, areas,
                                   fill method, reliability flag)
    gate.txt / gate.json           the annotated-export check and what it decided

The stroke is a thin outline covering ~0.9% of the frame, so scoring "does the
CAM peak land on the annotation" against the stroke would measure pen-width, not
localization. The filled interior is the lesion region; both are kept because the
ring-vs-clean contrast figure needs the stroke as well.

Orientation: the pixels stored in the pickle are one of the 4 flip variants of
the source frame (the 2024 ML path was fed the AUGMENTED train loader). The diff
is therefore computed in the flipped frame and then un-flipped - every flip is
its own inverse - so masks and bboxes come out aligned to the RAW frame, which
is what the clean-trained models see.
"""
import argparse
import csv
import glob
import json
import os
import pickle
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench import common as C                                      # noqa: E402

FULL = "data/uncompressed/KAUHC-V2-full"
MATCHES = "archive/log/revision-train-matches.csv"
KNN_GLOB = "archive/log/v7-knn */best.pkl"
OUT = os.path.join(C.LOG_ROOT, "bench-lesion-masks")
SIZE = 256
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
THRESH = 2.0 / 255.0
FLIPS = ["none", "h", "v", "hv"]
# a "mask" covering more than this much of the frame is not a drawn ellipse
# (a handful of frames carry large edits); flagged unreliable, kept, not scored
MAX_RELIABLE_FRAC = 0.25
LESION_CLASSES = ["AVM", "Erosion", "Ulcer", "Xanthoma"]

# candidate locations for a ringed export of the current frames
ANNOTATED_CANDIDATES = [
    "data/uncompressed/KAUHC-V2-annotated",
    "data/uncompressed/KAUHC-V2-full/_Annotated",
    "data/uncompressed/KAUHC-V2-full/_Prelabeled",
    "data/uncompressed/annotated",
    "data/splitted/V8-KAUHC-annotated",
    "archive/data-uncompressed/KAUHC-V2-annotated",
]


class _Stub:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        elif isinstance(state, tuple) and len(state) == 2 and isinstance(state[1], dict):
            self.__dict__.update(state[1])


class SafeUnpickler(pickle.Unpickler):
    """Loads the numpy payload of a 2024 scikit-learn pickle under any version."""

    def find_class(self, module, name):
        if module.startswith(("numpy", "builtins", "copyreg", "_codecs")):
            return super().find_class(module, name)
        return type(name, (_Stub,), {})


def load_fit_matrix(path):
    with open(path, "rb") as f:
        obj = SafeUnpickler(f).load()
    for attr in ("_fit_X", "_X", "X_"):
        X = getattr(obj, attr, None)
        if isinstance(X, np.ndarray) and X.ndim == 2:
            return X
    for v in vars(obj).values():
        X = getattr(v, "_fit_X", None)
        if isinstance(X, np.ndarray) and X.ndim == 2:
            return X
    raise SystemExit(f"no 2-D feature matrix in {path}")


def rgb01(path, size=SIZE):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR),
                          np.uint8).astype(np.float32) / 255.0


def flipped(hwc, k):
    return (hwc, hwc[:, ::-1], hwc[::-1, :], hwc[::-1, ::-1])[k]


def fill_ellipse(mask):
    """Turn the drawn outline into the region it encircles.

    Close small pen gaps, then flood-fill the enclosed holes. When the contour is
    open the fill changes nothing, so fall back to the convex hull of the drawn
    pixels, which is the tightest region guaranteed to contain the lesion.
    Returns (filled_mask, method).
    """
    from scipy import ndimage
    closed = ndimage.binary_closing(mask, structure=np.ones((5, 5)), iterations=2)
    filled = ndimage.binary_fill_holes(closed)
    if filled.sum() >= 1.5 * mask.sum():
        return filled, "fill_holes"
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], axis=1)
    if len(pts) >= 3:
        try:
            from matplotlib.path import Path
            from scipy.spatial import ConvexHull
            hull = ConvexHull(pts)
            gy, gx = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
            grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
            inside = Path(pts[hull.vertices]).contains_points(grid)
            return inside.reshape(mask.shape) | mask, "convex_hull"
        except Exception:
            pass
    box = np.zeros_like(mask)
    box[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
    return box, "bbox"


def v8_name(source_path):
    """_Prelabeling/<cls>/<patient>/<group>/<frame>.bmp -> <cls>__<patient>__<group>__<frame>.bmp"""
    rel = source_path.replace(os.sep, "/")
    parts = rel.split("/")
    cls = parts[1]
    return f"{cls}__" + "__".join(parts[2:])


def v8_index(data_root=C.DATA_ROOT):
    """filename -> (split, class) for every frame in the live V8 split."""
    idx = {}
    for sp in C.SPLITS:
        for cls in sorted(os.listdir(os.path.join(data_root, sp))):
            for fn in os.listdir(os.path.join(data_root, sp, cls)):
                idx[fn] = (sp, cls)
    return idx


# --------------------------------------------------------------------------- #
# the gate
# --------------------------------------------------------------------------- #
def annotated_export_gate(idx):
    """Is there a ringed export covering the current val/test lesion frames?"""
    need = {fn for fn, (sp, cls) in idx.items()
            if sp in ("val", "test") and cls in LESION_CLASSES}
    checked = []
    for cand in ANNOTATED_CANDIDATES:
        exists = os.path.isdir(cand)
        found = 0
        if exists:
            have = set()
            for dp, _, fns in os.walk(cand):
                have.update(fns)
            found = len(need & have)
        checked.append({"path": cand, "exists": exists, "val_test_lesion_frames_found": found})
    best = max(checked, key=lambda c: c["val_test_lesion_frames_found"])
    return {
        "val_test_lesion_frames_required": len(need),
        "candidates_checked": checked,
        "annotated_export_available": best["val_test_lesion_frames_found"] >= len(need),
        "best_candidate": best,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    C.assert_split()
    os.makedirs(os.path.join(OUT, "masks"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "masks_filled"), exist_ok=True)
    idx = v8_index()

    gate = annotated_export_gate(idx)
    lines = ["C1 GATE - is there an annotated (ringed) export of the current frames?",
             "=" * 78]
    for c in gate["candidates_checked"]:
        lines.append(f"  {'FOUND ' if c['exists'] else 'absent'} {c['path']}"
                     + (f"   matching val/test lesion frames: "
                        f"{c['val_test_lesion_frames_found']}" if c["exists"] else ""))
    lines.append("")
    if gate["annotated_export_available"]:
        lines.append("RESULT: annotated export available - masks built for ALL splits.")
    else:
        lines.append(
            f"RESULT: NO annotated export of the current val/test lesion frames exists on\n"
            f"        this machine (0 of {gate['val_test_lesion_frames_required']} required "
            f"frames found in any candidate location).\n"
            f"        Proceeding with the addendum-C fallback: the only surviving copy of\n"
            f"        the burned-in overlay pixels is the 1,928 x 196,608 training matrix\n"
            f"        inside archive/log/v7-knn */best.pkl, which covers 1,288 lesion\n"
            f"        frames. Masks are reconstructed from it by diffing the stored\n"
            f"        published pixels against the raw archive frame.\n"
            f"        NOTE FOR THE USER: ask Dr. Hamza whether the annotated exports for\n"
            f"        all frames can be retrieved from the hospital workstation. If they\n"
            f"        arrive, rerun `python3 bench/masks.py --force` and then bench/cams.py\n"
            f"        to get C1/C2 at full val/test scope.")
    gate_txt = "\n".join(lines)
    print(gate_txt, flush=True)
    with open(os.path.join(OUT, "gate.txt"), "w") as f:
        f.write(gate_txt + "\n")
    C.write_json(os.path.join(OUT, "gate.json"), gate)

    bbox_csv = os.path.join(OUT, "bench-lesion-bboxes.csv")
    if os.path.exists(bbox_csv) and not args.force:
        print(f"[skip] {bbox_csv} exists")
        return

    rows = [r for r in csv.DictReader(open(MATCHES)) if r["bbox_y0y1x0x1"].strip()]
    print(f"\nreconstructing {len(rows)} lesion masks from the archived training matrix",
          flush=True)
    pkl = sorted(glob.glob(KNN_GLOB))[0]
    print(f"  loading {pkl} ...", flush=True)
    X = load_fit_matrix(pkl)
    print(f"  training matrix {X.shape}", flush=True)

    out_rows, missing = [], 0
    for n, r in enumerate(rows, 1):
        src = os.path.join(FULL, r["source_path"])
        if not os.path.exists(src):
            missing += 1
            continue
        k = FLIPS.index(r["flip"])
        train_hwc = (X[int(r["knn_row"])].reshape(3, SIZE, SIZE) * STD[:, None, None]
                     + MEAN[:, None, None]).transpose(1, 2, 0)
        arch_flipped = flipped(rgb01(src), k)
        diff = np.abs(train_hwc - arch_flipped).max(axis=2)
        mask_flipped = diff > THRESH
        mask = np.ascontiguousarray(flipped(mask_flipped, k))       # back to raw
        ys, xs = np.nonzero(mask)
        if len(ys) == 0:
            continue
        fn = v8_name(r["source_path"])
        split, cls = idx.get(fn, ("not-in-V8", r["class"]))
        filled, method = fill_ellipse(mask)
        Image.fromarray((mask * 255).astype(np.uint8)).save(
            os.path.join(OUT, "masks", fn.replace(".bmp", ".png")))
        Image.fromarray((filled * 255).astype(np.uint8)).save(
            os.path.join(OUT, "masks_filled", fn.replace(".bmp", ".png")))
        out_rows.append({
            "filename": fn, "v8_split": split, "class": r["class"],
            "source_path": r["source_path"], "flip_variant": r["flip"],
            "y0": int(ys.min()), "y1": int(ys.max()),
            "x0": int(xs.min()), "x1": int(xs.max()),
            "stroke_px": int(mask.sum()),
            "stroke_frac": round(float(mask.mean()), 6),
            "lesion_px": int(filled.sum()),
            "lesion_frac": round(float(filled.mean()), 6),
            "fill_method": method,
            "reliable": int(filled.mean() <= MAX_RELIABLE_FRAC),
            "mask_size": SIZE, "knn_row": r["knn_row"],
        })
        if n % 200 == 0:
            print(f"    {n}/{len(rows)}", flush=True)

    with open(bbox_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    from collections import Counter
    by_split = Counter(r["v8_split"] for r in out_rows)
    by_cls = Counter(r["class"] for r in out_rows)
    summary = {
        "masks_written": len(out_rows), "source_frames_missing": missing,
        "by_v8_split": dict(by_split), "by_class": dict(by_cls),
        "reliable": int(sum(r["reliable"] for r in out_rows)),
        "unreliable_excluded_from_scoring": int(len(out_rows)
                                                - sum(r["reliable"] for r in out_rows)),
        "fill_method_counts": dict(Counter(r["fill_method"] for r in out_rows)),
        "median_stroke_px": float(np.median([r["stroke_px"] for r in out_rows])),
        "median_stroke_frac_pct": round(
            100 * float(np.median([r["stroke_frac"] for r in out_rows])), 3),
        "median_lesion_px": float(np.median([r["lesion_px"] for r in out_rows])),
        "median_lesion_frac_pct": round(
            100 * float(np.median([r["lesion_frac"] for r in out_rows])), 3),
        "reliable_by_v8_split": dict(Counter(r["v8_split"] for r in out_rows
                                             if r["reliable"])),
    }
    C.write_json(os.path.join(OUT, "summary.json"), summary)
    print("\n" + json.dumps(summary, indent=2))
    print(f"[done] {OUT}")


if __name__ == "__main__":
    main()
