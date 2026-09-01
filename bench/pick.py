"""Tiny helper the shell drivers use to ask "which model won?".

    python3 bench/pick.py --task 5class                    -> best deep model
    python3 bench/pick.py --task 5class --tier tier4-transformer
    python3 bench/pick.py --task 5class --top 2            -> two names, one per line
"""
import argparse
import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench import common as C                                      # noqa: E402

DEEP = ("tier2-classic-cnn", "tier3-efficient-cnn", "tier4-transformer",
        "tier5-foundation")


def rows(task):
    out = []
    for d in sorted(glob.glob(os.path.join(C.LOG_ROOT, "bench-*"))):
        base = os.path.basename(d)
        if not os.path.isdir(d) or "smoke_" in base or base.startswith("bench-cv-"):
            continue
        p = base.split("-", 2)
        if len(p) < 3 or p[1] != task:
            continue
        st = os.path.join(d, "summary_test.csv")
        sv = os.path.join(d, "summary_val.csv")
        ef = os.path.join(d, "efficiency.json")
        if not os.path.exists(st):
            continue
        tier = ""
        if os.path.exists(ef):
            try:
                tier = json.load(open(ef)).get("tier", "")
            except Exception:
                pass
        cfg = {}
        cp = os.path.join(d, "run_config.json")
        if os.path.exists(cp):
            try:
                cfg = json.load(open(cp))
            except Exception:
                pass
        out.append({"model": p[2], "tier": tier,
                    "arch": cfg.get("model", p[2]),
                    "protocol": cfg.get("protocol", "uniform"),
                    "train_mode": cfg.get("train_mode", ""),
                    "source": cfg.get("source", ""),
                    "has_ckpt": os.path.exists(os.path.join(d, "best.pth")),
                    "test_acc": float(pd.read_csv(st).accuracy.iloc[0]),
                    "val_acc": float(pd.read_csv(sv).accuracy.iloc[0])
                    if os.path.exists(sv) else 0.0})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="5class")
    ap.add_argument("--tier", default=None)
    ap.add_argument("--deep", action="store_true")
    ap.add_argument("--top", type=int, default=1)
    ap.add_argument("--by", default="test_acc", choices=["test_acc", "val_acc"])
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--source", default=None, help="filter on run_config source")
    ap.add_argument("--needs_ckpt", action="store_true",
                    help="only runs that saved a best.pth (Grad-CAM, CV)")
    ap.add_argument("--protocol", default=None, help="uniform | tuned")
    ap.add_argument("--field", default="model", choices=["model", "arch"],
                    help="'model' = run-directory name, 'arch' = the architecture "
                         "the run_config names (what --model expects)")
    args = ap.parse_args()
    df = rows(args.task)
    if df.empty:
        return
    if args.tier:
        df = df[df.tier == args.tier]
    if args.deep:
        df = df[df.tier.isin(DEEP)]
    if args.exclude:
        df = df[~df.model.isin(args.exclude)]
    if args.source:
        df = df[df.source == args.source]
    if args.needs_ckpt:
        df = df[df.has_ckpt]
    if args.protocol:
        df = df[df.protocol == args.protocol]
    df = df[df.tier != "hpo-sweep"]
    # tagged so the shell drivers can pull names out of the container's stdout,
    # which also carries the CUDA image banner
    for m in df.sort_values(args.by, ascending=False)[args.field].head(args.top):
        print("PICK\t" + str(m))


if __name__ == "__main__":
    main()
