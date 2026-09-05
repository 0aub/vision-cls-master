"""Bootstrap CIs and McNemar tests over the finished runs.

    python3 bench/stats.py --bootstrap --mcnemar

Bootstrap: 10,000 resamples of the test set (paired across models by row order,
which is the same unshuffled ImageFolder order everywhere), percentile 95% CIs
for accuracy and macro F1 of every model.

McNemar: exact binomial test on the discordant pairs, paired by image path,
between adjacent tier winners and between the best deep model and the best
embedding-based classical model.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench import common as C                                      # noqa: E402

N_BOOT = 10000


def iter_runs(pattern="log/bench-*"):
    for d in sorted(glob.glob(pattern)):
        base = os.path.basename(d)
        if not os.path.isdir(d) or not base.startswith("bench-") or "smoke_" in base:
            continue
        parts = base.split("-", 2)
        if len(parts) < 3 or parts[1] not in C.TASKS:
            continue
        f = os.path.join(d, "preds_test.csv")
        if not os.path.exists(f):
            continue
        eff = {}
        ef = os.path.join(d, "efficiency.json")
        if os.path.exists(ef):
            try:
                eff = json.load(open(ef))
            except Exception:
                eff = {}
        yield parts[1], parts[2], d, eff


def bootstrap(out_csv):
    from sklearn.metrics import accuracy_score, f1_score
    rng = np.random.default_rng(C.SEED)
    rows = []
    for task, name, d, eff in iter_runs():
        df = pd.read_csv(os.path.join(d, "preds_test.csv"))
        y, p = df.y_true.values, df.y_pred.values
        n = len(y)
        idx = rng.integers(0, n, size=(N_BOOT, n))
        accs = (y[idx] == p[idx]).mean(axis=1)
        K = len(C.classes_for(task))
        labels = list(range(K))
        # Vectorised macro F1 over all resamples at once. The per-resample
        # sklearn call was 10,000 python-level calls per run and blocked the
        # queue for hours; this counts TP/FP/FN with bincount on the flattened
        # resample index and is numerically identical.
        yb, pb = y[idx], p[idx]
        tp = np.zeros((N_BOOT, K)); fp = np.zeros((N_BOOT, K)); fn = np.zeros((N_BOOT, K))
        # NB: not `rows` - that name is the results accumulator in this function
        row_idx = np.repeat(np.arange(N_BOOT), n)
        for c in range(K):
            tp[:, c] = np.bincount(row_idx, weights=((yb == c) & (pb == c)).ravel(),
                                   minlength=N_BOOT)
            fp[:, c] = np.bincount(row_idx, weights=((yb != c) & (pb == c)).ravel(),
                                   minlength=N_BOOT)
            fn[:, c] = np.bincount(row_idx, weights=((yb == c) & (pb != c)).ravel(),
                                   minlength=N_BOOT)
        denom = 2 * tp + fp + fn
        f1c = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
        f1s = f1c.mean(axis=1)
        rows.append({
            "task": task, "model": name, "tier": eff.get("tier", ""), "n_test": n,
            "accuracy": round(float(accuracy_score(y, p)), 6),
            "accuracy_ci_lo": round(float(np.percentile(accs, 2.5)), 6),
            "accuracy_ci_hi": round(float(np.percentile(accs, 97.5)), 6),
            "macro_f1": round(float(f1_score(y, p, average="macro", zero_division=0,
                                             labels=labels)), 6),
            "macro_f1_ci_lo": round(float(np.percentile(f1s, 2.5)), 6),
            "macro_f1_ci_hi": round(float(np.percentile(f1s, 97.5)), 6),
            "n_bootstrap": N_BOOT,
        })
        print(f"  {task:7s} {name:34s} acc {rows[-1]['accuracy']:.4f} "
              f"[{rows[-1]['accuracy_ci_lo']:.4f}, {rows[-1]['accuracy_ci_hi']:.4f}]",
              flush=True)
    pd.DataFrame(rows).sort_values(["task", "accuracy"], ascending=[True, False]).to_csv(
        out_csv, index=False)
    print(f"[done] {out_csv}")


def mcnemar_pair(a, b):
    """a, b: boolean correctness arrays aligned by path. Exact binomial."""
    from statsmodels.stats.contingency_tables import mcnemar
    n01 = int((~a & b).sum())
    n10 = int((a & ~b).sum())
    tab = [[int((a & b).sum()), n10], [n01, int((~a & ~b).sum())]]
    res = mcnemar(tab, exact=True)
    return n10, n01, float(res.statistic), float(res.pvalue)


# the five real tiers; the sweep cells and the Phase D loss variants are not
# tiers and must not appear as "tier winners"
REAL_TIERS = ("tier1-classical", "tier2-classic-cnn", "tier3-efficient-cnn",
              "tier4-transformer", "tier5-foundation")


def pick_winners(task):
    """Best run per tier, by test accuracy."""
    best = {}
    for t, name, d, eff in iter_runs():
        if t != task or (eff.get("tier") or "") not in REAL_TIERS:
            continue
        s = pd.read_csv(os.path.join(d, "summary_test.csv"))
        acc = float(s.accuracy.iloc[0])
        tier = eff.get("tier") or "unlabelled"
        if tier not in best or acc > best[tier][1]:
            best[tier] = (name, acc, d)
    return best


def run_mcnemar(out_txt, out_csv):
    rows, lines = [], []
    for task in C.TASKS:
        best = pick_winners(task)
        if len(best) < 2:
            continue
        order = sorted(best.items(), key=lambda kv: kv[0])
        lines.append(f"\n=== {task} : tier winners ===")
        for tier, (name, acc, _) in order:
            lines.append(f"  {tier:24s} {name:34s} test acc {acc:.4f}")
        preds = {}
        for tier, (name, acc, d) in order:
            df = pd.read_csv(os.path.join(d, "preds_test.csv")).sort_values("path")
            preds[tier] = (name, df)
        pairs = [(order[i][0], order[i + 1][0]) for i in range(len(order) - 1)]
        # best deep vs best embedding-based classical
        deep = [(t, n, a) for t, (n, a, _) in best.items()
                if t.startswith(("tier2", "tier3", "tier4", "tier5"))]
        ml = [(t, n, a) for t, (n, a, _) in best.items() if t.startswith("tier1")]
        if deep and ml:
            pairs.append((max(deep, key=lambda x: x[2])[0], max(ml, key=lambda x: x[2])[0]))
        lines.append(f"\n  exact McNemar, paired by image path (n={len(preds[order[0][0]][1])})")
        for ta, tb in pairs:
            na, da = preds[ta]
            nb, db = preds[tb]
            assert (da.path.values == db.path.values).all()
            a = (da.y_true.values == da.y_pred.values)
            b = (db.y_true.values == db.y_pred.values)
            n10, n01, stat, p = mcnemar_pair(a, b)
            lines.append(f"    {na:30s} vs {nb:30s}  "
                         f"only-A-correct {n10:3d}  only-B-correct {n01:3d}  p={p:.4g}"
                         + ("  *" if p < 0.05 else ""))
            rows.append({"task": task, "tier_a": ta, "model_a": na,
                         "tier_b": tb, "model_b": nb,
                         "acc_a": round(float(a.mean()), 6),
                         "acc_b": round(float(b.mean()), 6),
                         "only_a_correct": n10, "only_b_correct": n01,
                         "p_value": p, "significant_at_0.05": bool(p < 0.05)})
    txt = "\n".join(lines)
    print(txt)
    with open(out_txt, "w") as f:
        f.write("McNemar exact tests between tier winners (v2 benchmark)\n")
        f.write("=" * 78 + "\n" + txt + "\n")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[done] {out_txt}, {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", action="store_true")
    ap.add_argument("--mcnemar", action="store_true")
    args = ap.parse_args()
    if args.bootstrap:
        bootstrap(os.path.join(C.LOG_ROOT, "bench-bootstrap-ci.csv"))
    if args.mcnemar:
        run_mcnemar(os.path.join(C.LOG_ROOT, "bench-mcnemar.txt"),
                    os.path.join(C.LOG_ROOT, "bench-mcnemar.csv"))


if __name__ == "__main__":
    main()
