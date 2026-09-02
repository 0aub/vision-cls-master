"""Do ensembles buy anything here, and at what cost? (M1 trade-off, extended)

    python3 bench/ensemble.py

Uses the softmax probabilities already saved for every run - no retraining, no
re-inference. For each task it compares the single best model against mean-
probability ensembles of the top k, and against a deliberately DIVERSE ensemble
(one model per tier), because averaging correlated errors buys nothing and the
interesting question is whether different model families fail differently.

Cost is reported alongside accuracy: an ensemble of k models costs k forward
passes, so it only earns its place if the gain exceeds what a single larger model
would give for the same latency.

Writes log/bench-ensemble.csv.
"""
import glob
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench import common as C                                      # noqa: E402


def runs_for(task):
    out = []
    for d in sorted(glob.glob(os.path.join(C.LOG_ROOT, f"bench-{task}-*"))):
        p = os.path.join(d, "PROBS_test.npy")
        f = os.path.join(d, "preds_test.csv")
        e = os.path.join(d, "efficiency.json")
        if not (os.path.exists(p) and os.path.exists(f)):
            continue
        name = os.path.basename(d).replace(f"bench-{task}-", "")
        if "smoke_" in name or name.startswith("hpo_"):
            continue
        eff = json.load(open(e)) if os.path.exists(e) else {}
        df = pd.read_csv(f)
        probs = np.load(p)
        if len(df) != len(probs):
            continue
        out.append({"name": name, "tier": eff.get("tier", ""),
                    "lat": eff.get("gpu_latency_ms_b1"),
                    "probs": probs, "y": df.y_true.values,
                    "acc": float((df.y_pred.values == df.y_true.values).mean())})
    return sorted(out, key=lambda r: -r["acc"])


def score(members, K):
    y = members[0]["y"]
    P = np.mean([m["probs"] for m in members], axis=0)
    pred = P.argmax(1)
    lat = [m["lat"] for m in members if m["lat"]]
    return {"n_models": len(members),
            "accuracy": round(float(accuracy_score(y, pred)), 6),
            "macro_f1": round(float(f1_score(y, pred, average="macro",
                                             zero_division=0, labels=list(range(K)))), 6),
            "total_gpu_ms_b1": round(float(sum(lat)), 3) if lat else None,
            "members": " + ".join(m["name"][:26] for m in members)}


def main():
    rows = []
    for task in C.TASKS:
        R = runs_for(task)
        if len(R) < 2:
            continue
        K = len(C.classes_for(task))
        best = R[0]
        rows.append({"task": task, "kind": "single best", **score([best], K)})
        for k in (2, 3, 5):
            if len(R) >= k:
                rows.append({"task": task, "kind": f"top-{k}", **score(R[:k], K)})
        # one model per tier: different families should fail differently
        seen, div = set(), []
        for r in R:
            t = r["tier"] or "untiered"
            if t not in seen:
                seen.add(t)
                div.append(r)
        if len(div) >= 2:
            rows.append({"task": task, "kind": f"diverse ({len(div)} tiers)",
                         **score(div, K)})
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(C.LOG_ROOT, "bench-ensemble.csv"), index=False)
    for task in C.TASKS:
        s = d[d.task == task]
        if len(s):
            base = s[s.kind == "single best"].accuracy.iloc[0]
            s = s.assign(delta=(s.accuracy - base).round(4))
            print(f"\n--- {task}")
            print(s[["kind", "n_models", "accuracy", "delta", "macro_f1",
                     "total_gpu_ms_b1"]].to_string(index=False))
    print(f"\n[done] {os.path.join(C.LOG_ROOT, 'bench-ensemble.csv')}")


if __name__ == "__main__":
    main()
