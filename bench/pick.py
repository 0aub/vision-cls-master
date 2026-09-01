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
        out.append({"model": p[2], "tier": tier,
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
    for m in df.sort_values(args.by, ascending=False).model.head(args.top):
        print(m)


if __name__ == "__main__":
    main()
