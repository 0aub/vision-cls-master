"""Build bench-results.zip (or a phase-partial zip).

    python3 bench/package.py --out bench-results-phaseA.zip --phase A
    python3 bench/package.py --out bench-results.zip

Contents, per the brief: the report, every summary/report/preds/cm CSV, every
history.csv and efficiency.json, PROBS files ONLY for tier winners, the C2
qualitative PNG panels, bench-trust.csv, bench-bootstrap-ci.csv and the
lesion-mask bbox CSV. No checkpoints (*.pth/*.pkl), nothing from archive/.
If the archive exceeds the size cap the PROBS files are dropped and the report
says so.
"""
import argparse
import glob
import json
import os
import sys
import zipfile

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench import common as C                                      # noqa: E402

LOG = C.LOG_ROOT
CAP_MB = 200

TOP_LEVEL = [
    "BENCH-REPORT.md", "bench-phase0-gates.txt", "bench-archive-index.txt",
    "bench-env-gate.txt", "bench-env-gate.json", "bench-environment.txt",
    "bench-progress.txt", "bench-phase-timing.txt", "bench-anomalies.md",
    "bench-trust.csv", "bench-reliability.csv", "bench-coverage.csv",
    "bench-bootstrap-ci.csv", "bench-mcnemar.txt", "bench-mcnemar.csv",
    "bench-longtail.csv", "bench-longtail-recall.csv",
    "bench-biomedclip-prompts-5class.json", "bench-biomedclip-prompts-binary.json",
    "bench-pareto-5class.png", "bench-pareto-binary.png",
]
RUN_FILES = ("history.csv", "efficiency.json", "run_config.json")
RUN_GLOBS = ("summary_*.csv", "report_*.csv", "preds_*.csv", "cm_*.csv")


def tier_winners():
    """(task, run-dir-name) of the best test accuracy in each tier, both tasks."""
    best = {}
    for d in sorted(glob.glob(os.path.join(LOG, "bench-*"))):
        base = os.path.basename(d)
        if not os.path.isdir(d) or "smoke_" in base or base.startswith("bench-cv-"):
            continue
        parts = base.split("-", 2)
        if len(parts) < 3 or parts[1] not in ("5class", "binary"):
            continue
        st = os.path.join(d, "summary_test.csv")
        ef = os.path.join(d, "efficiency.json")
        if not os.path.exists(st):
            continue
        acc = float(pd.read_csv(st).accuracy.iloc[0])
        tier = "unlabelled"
        if os.path.exists(ef):
            try:
                tier = json.load(open(ef)).get("tier") or "unlabelled"
            except Exception:
                pass
        k = (parts[1], tier)
        if k not in best or acc > best[k][1]:
            best[k] = (base, acc)
    return {v[0] for v in best.values()}


def add(z, path, arc, seen):
    if arc in seen or not os.path.exists(path):
        return
    z.write(path, arc)
    seen.add(arc)


def build(out, include_probs=True):
    winners = tier_winners()
    seen = set()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in TOP_LEVEL:
            add(z, os.path.join(LOG, f), f"log/{f}", seen)
        for d in sorted(glob.glob(os.path.join(LOG, "bench-*"))):
            if not os.path.isdir(d):
                continue
            base = os.path.basename(d)
            for f in RUN_FILES:
                add(z, os.path.join(d, f), f"log/{base}/{f}", seen)
            for g in RUN_GLOBS:
                for p in sorted(glob.glob(os.path.join(d, g))):
                    add(z, p, f"log/{base}/{os.path.basename(p)}", seen)
            for p in sorted(glob.glob(os.path.join(d, "*.json"))):
                add(z, p, f"log/{base}/{os.path.basename(p)}", seen)
            if include_probs and base in winners:
                for p in sorted(glob.glob(os.path.join(d, "PROBS_*.npy"))):
                    add(z, p, f"log/{base}/{os.path.basename(p)}", seen)
        # cross-validation
        for p in sorted(glob.glob(os.path.join(LOG, "bench-cv-*", "*.csv"))
                        + glob.glob(os.path.join(LOG, "bench-cv-*", "*.json"))):
            add(z, p, os.path.relpath(p, "."), seen)
        # phase C artefacts
        for p in sorted(glob.glob(os.path.join(LOG, "bench-cam", "*.csv"))
                        + glob.glob(os.path.join(LOG, "bench-cam", "panels", "*.png"))
                        + glob.glob(os.path.join(LOG, "bench-lesion-masks", "*.csv"))
                        + glob.glob(os.path.join(LOG, "bench-lesion-masks", "*.txt"))
                        + glob.glob(os.path.join(LOG, "bench-lesion-masks", "*.json"))):
            add(z, p, os.path.relpath(p, "."), seen)
        # the briefs, so the package is self-describing
        for f in ("BENCHMARK_PLAN_V2.md", "BENCHMARK_BRIEF_V2.md",
                  "BENCHMARK_BRIEF_V2_ADDENDUM.md"):
            add(z, f, f, seen)
    return len(seen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--phase", default="")
    args = ap.parse_args()
    n = build(args.out, include_probs=True)
    mb = os.path.getsize(args.out) / 1024**2
    if mb > CAP_MB:
        print(f"[size] {mb:.1f} MB exceeds the {CAP_MB} MB cap; rebuilding without PROBS")
        n = build(args.out, include_probs=False)
        mb = os.path.getsize(args.out) / 1024**2
        with open(os.path.join(LOG, "bench-anomalies.md"), "a") as f:
            f.write(f"\n- The results zip exceeded {CAP_MB} MB with the tier-winner "
                    f"PROBS files included, so they were dropped from "
                    f"`{os.path.basename(args.out)}` (final size {mb:.1f} MB). "
                    f"They remain on disk under `log/bench-<task>-<model>/`.\n")
    print(f"[done] {args.out}  {n} files, {mb:.1f} MB")
    # sanity: nothing forbidden slipped in
    with zipfile.ZipFile(args.out) as z:
        bad = [n for n in z.namelist()
               if n.startswith("archive/") or n.endswith((".pth", ".pkl"))]
    print("forbidden entries:", bad if bad else "none")


if __name__ == "__main__":
    main()
