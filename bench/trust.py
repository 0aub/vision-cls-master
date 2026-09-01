"""C3: calibration and selective prediction (module M4), post-hoc on saved PROBS.

    python3 bench/trust.py

For every run that has PROBS_val.npy and PROBS_test.npy:
  * temperature scaling fitted on validation by NLL (L-BFGS on a single scalar);
  * ECE with 15 equal-width confidence bins, before and after scaling;
  * reliability-diagram data (per bin: confidence, accuracy, count);
  * accuracy-vs-coverage: keep the most confident x% of test predictions and
    report accuracy there, at 100/90/80/70% coverage and on a fine grid.

Temperature scaling from stored probabilities is exact: softmax(log p / T) equals
softmax(logits / T), because the per-row log-partition term is constant across
classes and cancels in the softmax.

Writes log/bench-trust.csv, log/bench-reliability.csv, log/bench-coverage.csv.
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

N_BINS = 15
COVERAGES = [1.00, 0.90, 0.80, 0.70]


def ece(probs, y, n_bins=N_BINS):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    e, rows = 0.0, []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i else (conf >= lo) & (conf <= hi)
        n = int(m.sum())
        if n:
            acc, cf = correct[m].mean(), conf[m].mean()
            e += n / len(y) * abs(acc - cf)
        else:
            acc = cf = np.nan
        rows.append({"bin": i + 1, "bin_lo": round(lo, 4), "bin_hi": round(hi, 4),
                     "count": n, "mean_confidence": None if n == 0 else round(float(cf), 6),
                     "accuracy": None if n == 0 else round(float(acc), 6)})
    return float(e), rows


def fit_temperature(probs_val, y_val, grid=None):
    """Minimise validation NLL over T with a fine 1-D search (robust, no autograd)."""
    logp = np.log(np.clip(probs_val, 1e-12, 1.0))
    if grid is None:
        grid = np.concatenate([np.linspace(0.05, 1.0, 96)[:-1], np.linspace(1.0, 10.0, 181)])
    best_T, best_nll = 1.0, None
    for T in grid:
        z = logp / T
        z = z - z.max(axis=1, keepdims=True)
        nll = float(-(z[np.arange(len(y_val)), y_val]
                      - np.log(np.exp(z).sum(axis=1))).mean())
        if best_nll is None or nll < best_nll:
            best_T, best_nll = float(T), nll
    return best_T, best_nll


def apply_temperature(probs, T):
    z = np.log(np.clip(probs, 1e-12, 1.0)) / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def coverage_curve(probs, y, coverages=COVERAGES):
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y).astype(float)
    order = np.argsort(-conf)
    out = []
    for c in coverages:
        k = max(1, int(round(c * len(y))))
        idx = order[:k]
        out.append({"coverage": c, "n_kept": k,
                    "accuracy": round(float(correct[idx].mean()), 6),
                    "min_confidence": round(float(conf[idx].min()), 6),
                    "risk": round(float(1 - correct[idx].mean()), 6)})
    return out


def fine_coverage(probs, y, step=0.05):
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y).astype(float)
    order = np.argsort(-conf)
    rows = []
    c = 1.0
    while c >= 0.1 - 1e-9:
        k = max(1, int(round(c * len(y))))
        idx = order[:k]
        rows.append({"coverage": round(c, 3), "n_kept": k,
                     "accuracy": round(float(correct[idx].mean()), 6)})
        c -= step
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="log/bench-*")
    args = ap.parse_args()

    trust, reliability, coverage = [], [], []
    for d in sorted(glob.glob(args.pattern)):
        if not os.path.isdir(d):
            continue
        base = os.path.basename(d)
        if not base.startswith("bench-") or "smoke_" in base:
            continue
        pv = os.path.join(d, "PROBS_val.npy")
        pt = os.path.join(d, "PROBS_test.npy")
        if not (os.path.exists(pv) and os.path.exists(pt)):
            continue
        parts = base.split("-", 2)
        if len(parts) < 3:
            continue
        task, name = parts[1], parts[2]
        Pv, Pt = np.load(pv), np.load(pt)
        yv = C.labels_of("val", task)
        yt = C.labels_of("test", task)
        if len(yv) != len(Pv) or len(yt) != len(Pt):
            print(f"[warn] {base}: PROBS/label length mismatch, skipped")
            continue

        T, nll = fit_temperature(Pv, yv)
        Pt_cal = apply_temperature(Pt, T)
        e_raw, bins_raw = ece(Pt, yt)
        e_cal, bins_cal = ece(Pt_cal, yt)
        acc = float((Pt.argmax(1) == yt).mean())

        eff = {}
        ef = os.path.join(d, "efficiency.json")
        if os.path.exists(ef):
            eff = json.load(open(ef))

        row = {"task": task, "model": name, "tier": eff.get("tier", ""),
               "test_accuracy": round(acc, 6),
               "temperature": round(T, 4), "val_nll_at_T": round(nll, 6),
               "ece_before": round(e_raw, 6), "ece_after": round(e_cal, 6),
               "ece_reduction": round(e_raw - e_cal, 6),
               "mean_confidence_before": round(float(Pt.max(1).mean()), 6),
               "mean_confidence_after": round(float(Pt_cal.max(1).mean()), 6)}
        for c in coverage_curve(Pt_cal, yt):
            row[f"acc_at_coverage_{int(c['coverage']*100)}"] = c["accuracy"]
            row[f"minconf_at_coverage_{int(c['coverage']*100)}"] = c["min_confidence"]
        trust.append(row)

        for tag, bins in (("before", bins_raw), ("after", bins_cal)):
            for b in bins:
                reliability.append({"task": task, "model": name, "calibration": tag, **b})
        for c in fine_coverage(Pt_cal, yt):
            coverage.append({"task": task, "model": name, "calibration": "after", **c})
        for c in fine_coverage(Pt, yt):
            coverage.append({"task": task, "model": name, "calibration": "before", **c})
        print(f"  {task:7s} {name:34s} T={T:5.2f} ECE {e_raw:.4f} -> {e_cal:.4f} "
              f"acc@70%cov {row['acc_at_coverage_70']:.4f}", flush=True)

    os.makedirs(C.LOG_ROOT, exist_ok=True)
    pd.DataFrame(trust).sort_values(["task", "tier", "model"]).to_csv(
        os.path.join(C.LOG_ROOT, "bench-trust.csv"), index=False)
    pd.DataFrame(reliability).to_csv(os.path.join(C.LOG_ROOT, "bench-reliability.csv"),
                                     index=False)
    pd.DataFrame(coverage).to_csv(os.path.join(C.LOG_ROOT, "bench-coverage.csv"),
                                  index=False)
    print(f"[done] {len(trust)} runs -> log/bench-trust.csv")


if __name__ == "__main__":
    main()
