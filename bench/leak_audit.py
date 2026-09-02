"""Global patient-disjointness audit, and leak-free test metrics for every run.

    python3 bench/leak_audit.py

The V8 split is patient-disjoint WITHIN each class - that is what the published
protocol claims and it holds. It is not patient-disjoint GLOBALLY: three patients
contribute frames to one class's training split and to another class's
evaluation split, so 29 test frames (12.8%) and 20 validation frames (9.0%) come
from a patient the model has seen, under a different label.

That is a milder thing than label leakage - the lesion type differs - but a model
can still recognise the patient's mucosa, and the merged Erosion+Ulcer task makes
it worse (P_105 would sit in both the merged training and merged test sets).

Writes log/bench-patient-leaks.csv (which patients, which frames) and
log/bench-leakfree.csv (every run's test metrics recomputed with those frames
dropped, alongside the headline numbers, from the saved predictions - nothing is
re-trained or re-inferred).
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench import common as C                                      # noqa: E402


def audit():
    per = defaultdict(lambda: defaultdict(set))          # patient -> split -> classes
    for sp in C.SPLITS:
        d = os.path.join(C.DATA_ROOT, sp)
        for cls in sorted(os.listdir(d)):
            for fn in os.listdir(os.path.join(d, cls)):
                p = C.patient_of(fn)
                if p.startswith("P_"):
                    per[p][sp].add(cls)
    rows = []
    for p, spl in sorted(per.items()):
        tr = spl.get("train", set())
        for sp in ("val", "test"):
            ev = spl.get(sp, set())
            if tr and ev:
                rows.append({"patient": p, "train_classes": " ".join(sorted(tr)),
                             "eval_split": sp, "eval_classes": " ".join(sorted(ev)),
                             "same_class": bool(tr & ev),
                             "n_eval_frames": sum(
                                 1 for cls in ev
                                 for fn in os.listdir(os.path.join(C.DATA_ROOT, sp, cls))
                                 if C.patient_of(fn) == p)})
    return pd.DataFrame(rows)


def leakfree_metrics():
    bad = C.leaked_paths("test")
    out = []
    for d in sorted(glob.glob(os.path.join(C.LOG_ROOT, "bench-*"))):
        base = os.path.basename(d)
        if not os.path.isdir(d) or "smoke_" in base or base.startswith("bench-cv-"):
            continue
        parts = base.split("-", 2)
        if len(parts) < 3 or parts[1] not in C.TASKS:
            continue
        f = os.path.join(d, "preds_test.csv")
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f)
        keep = ~df.path.isin(bad)
        if keep.sum() == 0 or keep.all():
            pass
        K = len(C.classes_for(parts[1]))
        labels = list(range(K))
        tier = ""
        ef = os.path.join(d, "efficiency.json")
        if os.path.exists(ef):
            try:
                tier = json.load(open(ef)).get("tier", "")
            except Exception:
                pass
        sub = df[keep]
        out.append({
            "task": parts[1], "model": parts[2], "tier": tier,
            "n_all": len(df), "n_leakfree": int(keep.sum()),
            "acc_all": round(float(accuracy_score(df.y_true, df.y_pred)), 6),
            "acc_leakfree": round(float(accuracy_score(sub.y_true, sub.y_pred)), 6),
            "macro_f1_all": round(float(f1_score(df.y_true, df.y_pred, average="macro",
                                                 zero_division=0, labels=labels)), 6),
            "macro_f1_leakfree": round(float(f1_score(sub.y_true, sub.y_pred,
                                                      average="macro", zero_division=0,
                                                      labels=labels)), 6),
        })
    d = pd.DataFrame(out)
    d["acc_delta"] = (d.acc_leakfree - d.acc_all).round(6)
    return d.sort_values(["task", "acc_all"], ascending=[True, False])


def main():
    a = audit()
    a.to_csv(os.path.join(C.LOG_ROOT, "bench-patient-leaks.csv"), index=False)
    print("PATIENTS APPEARING IN TRAIN AND IN AN EVALUATION SPLIT")
    print(a.to_string(index=False))
    cross = a[~a.same_class]
    print(f"\ncross-class leaks: {len(cross)}  "
          f"({int(cross.n_eval_frames.sum())} evaluation frames)")

    m = leakfree_metrics()
    m.to_csv(os.path.join(C.LOG_ROOT, "bench-leakfree.csv"), index=False)
    print("\nTEST METRICS WITH LEAKED FRAMES DROPPED (top of each task)")
    for t in C.TASKS:
        s = m[m.task == t].head(6)
        if len(s):
            print(f"\n--- {t}")
            print(s[["model", "n_all", "n_leakfree", "acc_all", "acc_leakfree",
                     "acc_delta", "macro_f1_all", "macro_f1_leakfree"]]
                  .to_string(index=False))


if __name__ == "__main__":
    main()
