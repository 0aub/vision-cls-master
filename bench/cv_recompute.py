"""Recompute CV metrics from saved out-of-fold predictions - no retraining.

    python3 bench/cv_recompute.py

Per-fold macro F1 was averaged over ALL labels, so a class that a
patient-grouped fold legitimately contains none of scored 0 and dragged the
average down: it measured fold composition, not the model. This recomputes each
fold over the classes actually present, and adds the pooled out-of-fold figures -
every frame predicted exactly once by a model that never saw that patient, which
is the number the paper should quote.

Only the arithmetic changes; the predictions are the ones already on disk.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench import common as C                                      # noqa: E402


def main():
    rows = []
    for d in sorted(glob.glob(os.path.join(C.LOG_ROOT, "bench-cv-*"))):
        oofp = os.path.join(d, "oof_predictions.csv")
        summ = os.path.join(d, "cv_summary.json")
        if not (os.path.exists(oofp) and os.path.exists(summ)):
            continue
        base = os.path.basename(d).replace("bench-cv-", "")
        task = base.split("-", 1)[0]
        if task not in C.TASKS:
            continue
        oof = pd.read_csv(oofp)
        K = len(C.classes_for(task))
        per = []
        for f in sorted(oof.fold.unique()):
            s = oof[oof.fold == f]
            present = sorted(set(np.unique(s.y_true).tolist()))
            per.append({
                "fold": int(f), "n": len(s), "classes_present": len(present),
                "accuracy": float(accuracy_score(s.y_true, s.y_pred)),
                "macro_f1_present": float(f1_score(s.y_true, s.y_pred, average="macro",
                                                   zero_division=0, labels=present)),
                "macro_f1_all_labels": float(f1_score(s.y_true, s.y_pred,
                                                      average="macro", zero_division=0,
                                                      labels=list(range(K)))),
            })
        p = pd.DataFrame(per)
        meta = json.load(open(summ))
        rows.append({
            "task": task, "model": base.split("-", 1)[1], "folds": len(p),
            "min_classes_in_a_fold": int(p.classes_present.min()), "n_classes": K,
            "mean_accuracy": round(float(p.accuracy.mean()), 6),
            "std_accuracy": round(float(p.accuracy.std(ddof=1)), 6),
            "mean_macro_f1_present": round(float(p.macro_f1_present.mean()), 6),
            "mean_macro_f1_all_labels": round(float(p.macro_f1_all_labels.mean()), 6),
            "pooled_oof_accuracy": round(float(accuracy_score(oof.y_true, oof.y_pred)), 6),
            "pooled_oof_macro_f1": round(float(f1_score(oof.y_true, oof.y_pred,
                                                        average="macro",
                                                        zero_division=0)), 6),
            "protocol": meta.get("protocol", ""),
        })
        p.round(6).to_csv(os.path.join(d, "cv_folds_recomputed.csv"), index=False)
    out = pd.DataFrame(rows).sort_values(["task", "pooled_oof_accuracy"],
                                         ascending=[True, False])
    out.to_csv(os.path.join(C.LOG_ROOT, "bench-cv-summary.csv"), index=False)
    print(out.to_string(index=False))
    print(f"\n[done] {os.path.join(C.LOG_ROOT, 'bench-cv-summary.csv')}")


if __name__ == "__main__":
    main()
