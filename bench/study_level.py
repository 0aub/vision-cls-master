"""A6: evaluate at the clinical unit - the patient-study, not the frame.

    python3 bench/study_level.py

A gastroenterologist reads a study and decides about a patient; nobody acts on a
single frame. Frame-level accuracy is the wrong denominator for the clinical
claim, and it is also the pessimistic one: consecutive frames of the same lesion
are near-duplicates, so a model that is right about the lesion but wrong on three
blurry frames of it is scored three times for one mistake.

Aggregation is by (patient, study) over that study's frames, two ways:
  vote  - majority of the per-frame argmax
  mean  - argmax of the mean softmax, which respects confidence

TRAP, and the reason this module exists rather than a one-liner: Normal frames
carry no patient or study in their filename, so each becomes its own singleton
"study". Pooling those 80 singletons with the 14 real patient-studies takes
merged4 from a true 0.786 to a flattering 0.968. Real studies and singleton
Normal frames are therefore always reported separately, with n on every row, and
the lesion-study figure is the headline.

With n=14 test studies the confidence intervals are wide; they are reported
(Wilson) rather than hidden behind a point estimate.

Writes log/bench-study-level.csv.
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench import common as C                                      # noqa: E402

UNIT = re.compile(r"__(P_\d+)__(\d+)__")


def unit_of(path):
    """(unit id, is_a_real_patient_study)"""
    b = os.path.basename(path)
    m = UNIT.search(b)
    if m:
        return f"{m.group(1)}/{m.group(2)}", True
    return "NORMAL/" + b, False


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def aggregate(df, probs, K):
    df = df.copy()
    u = df.path.map(unit_of)
    df["unit"] = [a for a, _ in u]
    df["real"] = [b for _, b in u]
    rows = []
    for (unit, real), g in df.groupby(["unit", "real"]):
        vote = int(g.y_pred.value_counts().idxmax())
        if probs is not None:
            mean = int(np.argmax(probs[g.index.to_numpy()].mean(axis=0)))
        else:
            mean = vote
        rows.append({"unit": unit, "real_study": real, "n_frames": len(g),
                     "y_true": int(g.y_true.iloc[0]), "vote": vote, "mean": mean})
    return pd.DataFrame(rows)


def main():
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
        task, name = parts[1], parts[2]
        df = pd.read_csv(f).reset_index(drop=True)
        pf = os.path.join(d, "PROBS_test.npy")
        probs = np.load(pf) if os.path.exists(pf) else None
        K = len(C.classes_for(task))
        agg = aggregate(df, probs, K)
        les = agg[agg.real_study]
        tier = ""
        ef = os.path.join(d, "efficiency.json")
        if os.path.exists(ef):
            try:
                tier = json.load(open(ef)).get("tier", "")
            except Exception:
                pass
        row = {"task": task, "model": name, "tier": tier,
               "n_frames": len(df),
               "frame_accuracy": round(float((df.y_true == df.y_pred).mean()), 6),
               "n_lesion_studies": int(len(les)),
               "n_singleton_normal": int((~agg.real_study).sum())}
        for how in ("vote", "mean"):
            k = int((les.y_true == les[how]).sum())
            lo, hi = wilson(k, len(les))
            row[f"study_correct_{how}"] = k
            row[f"study_accuracy_{how}"] = round(k / len(les), 6) if len(les) else np.nan
            row[f"study_ci_lo_{how}"] = round(lo, 4)
            row[f"study_ci_hi_{how}"] = round(hi, 4)
        out.append(row)

    r = pd.DataFrame(out).sort_values(["task", "study_accuracy_vote", "frame_accuracy"],
                                      ascending=[True, False, False])
    r.to_csv(os.path.join(C.LOG_ROOT, "bench-study-level.csv"), index=False)
    for t in C.TASKS:
        s = r[r.task == t].head(5)
        if len(s):
            print(f"\n--- {t}   (lesion studies n={int(s.n_lesion_studies.iloc[0])}, "
                  f"singleton Normal frames excluded from the study figure)")
            print(s[["model", "frame_accuracy", "study_correct_vote",
                     "study_accuracy_vote", "study_ci_lo_vote", "study_ci_hi_vote",
                     "study_accuracy_mean"]].to_string(index=False))
    print(f"\n[done] {os.path.join(C.LOG_ROOT, 'bench-study-level.csv')}  ({len(r)} runs)")


if __name__ == "__main__":
    main()
