"""Where inside the ellipse does the attention peak land? (sharpens M3b)

    python3 bench/cam_geometry.py

The pointing game asks "is the CAM peak inside the lesion region". That question
cannot separate a clean-trained model from a ring-trained one, because the
clinician drew the ellipse AROUND the lesion: firing on the drawn stroke also
lands inside the region, which is why the archived ring-trained checkpoint scores
the HIGHEST plain hit-rate of any model.

The statistic that does separate them is WHERE inside. Split each lesion region
into a core (eroded interior - the lesion body) and an annulus (the rim next to
the drawn stroke), and ask which one the peak falls in:

    a lesion detector  peaks in the CORE
    a ring detector    peaks in the ANNULUS, hugging the pen stroke

Reads the per-frame CAM peaks already saved by bench/cams.py, so no model is
re-run. Writes log/bench-cam/bench-cam-geometry.csv and -summary.csv.
"""
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench import common as C                                      # noqa: E402

CAM = os.path.join(C.LOG_ROOT, "bench-cam")
MASKDIR = os.environ.get("BENCH_MASKDIR",
                         os.path.join(C.LOG_ROOT, "bench-lesion-masks"))
CORE_FRAC = 0.5          # erode by half the region's equivalent radius


def geometry(filled):
    """core / annulus split, centroid and equivalent radius of a filled region."""
    area = float(filled.sum())
    r_eq = np.sqrt(area / np.pi)
    it = max(1, int(round(CORE_FRAC * r_eq)))
    core = ndimage.binary_erosion(filled, iterations=it)
    if core.sum() == 0:                      # thin region: keep the single peak pixel
        dist = ndimage.distance_transform_edt(filled)
        core = dist >= dist.max() * 0.999
    return core, (filled & ~core), np.array(ndimage.center_of_mass(filled)), r_eq


def main():
    per_frame = os.path.join(CAM, "bench-pointing-game.csv")
    if not os.path.exists(per_frame):
        raise SystemExit(f"missing {per_frame}; run bench/cams.py first")
    df = pd.read_csv(per_frame)

    cache, rows = {}, []
    for _, r in df.iterrows():
        fn = r["filename"].replace(".bmp", ".png")
        if fn not in cache:
            p = os.path.join(MASKDIR, "masks_filled", fn)
            if not os.path.exists(p):
                continue
            filled = np.asarray(Image.open(p)) > 127
            cache[fn] = geometry(filled)
        core, annulus, centroid, r_eq = cache[fn]
        y, x = int(r["peak_y"]), int(r["peak_x"])
        if not (0 <= y < core.shape[0] and 0 <= x < core.shape[1]):
            continue
        in_core = bool(core[y, x])
        in_ann = bool(annulus[y, x])
        d = float(np.hypot(y - centroid[0], x - centroid[1]))
        rows.append({
            "model": r["model"], "ringed": r["ringed"], "v8_split": r["v8_split"],
            "class": r["class"], "filename": r["filename"],
            "peak_in_core": int(in_core), "peak_in_annulus": int(in_ann),
            "peak_outside": int(not (in_core or in_ann)),
            # distance from the region centroid in units of its own radius:
            # ~0 = dead centre, ~1 = on the rim, >1 = outside
            "peak_dist_over_radius": round(d / max(r_eq, 1e-6), 4),
        })

    g = pd.DataFrame(rows)
    if g.empty:
        raise SystemExit("no frames matched")
    g.to_csv(os.path.join(CAM, "bench-cam-geometry.csv"), index=False)

    def agg(d):
        inside = d.peak_in_core + d.peak_in_annulus
        return pd.Series({
            "n": len(d),
            "peak_in_core_rate": round(float(d.peak_in_core.mean()), 4),
            "peak_in_annulus_rate": round(float(d.peak_in_annulus.mean()), 4),
            "peak_outside_rate": round(float(d.peak_outside.mean()), 4),
            # of the peaks that land inside at all, what share is the lesion body
            "core_share_of_inside": round(float(d.peak_in_core.sum() / inside.sum()), 4)
            if inside.sum() else np.nan,
            "median_dist_over_radius": round(float(d.peak_dist_over_radius.median()), 4),
        })

    out = pd.concat([
        g.groupby(["model", "v8_split"]).apply(agg, include_groups=False)
         .reset_index(),
        g.groupby("model").apply(agg, include_groups=False)
         .reset_index().assign(v8_split="ALL"),
    ], ignore_index=True)
    out.to_csv(os.path.join(CAM, "bench-cam-geometry-summary.csv"), index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
