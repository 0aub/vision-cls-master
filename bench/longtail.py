"""Phase D aggregation (module M2): pick the long-tail winner on validation.

    python3 bench/longtail.py --models efficientnet_b0 densenet201 dinov2_vitb14_lora

Assumes the five variants per model already ran through bench/train_dl.py under
the names <model>__ce, __weighted_ce, __focal, __cb, __sampler. The decision rule
is fixed in advance and stated in the report: the variant with the highest
VALIDATION macro F1 wins; its TEST numbers are the adopted result. Test numbers
of the losing variants are reported too, but they never drive the choice.
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench import common as C                                      # noqa: E402

VARIANTS = [("ce", "plain CE"), ("weighted_ce", "weighted CE"),
            ("focal", "focal, gamma=2"),
            ("cb", "class-balanced, effective number, beta=0.9999"),
            ("sampler", "weighted sampler + plain CE")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--task", default="5class")
    args = ap.parse_args()
    classes = C.classes_for(args.task)

    rows, recall_rows = [], []
    for base in args.models:
        for key, label in VARIANTS:
            name = f"{base}__{key}"
            d = C.run_dir(args.task, name)
            sv = os.path.join(d, "summary_val.csv")
            st = os.path.join(d, "summary_test.csv")
            if not (os.path.exists(sv) and os.path.exists(st)):
                print(f"[missing] {name}")
                continue
            v = pd.read_csv(sv).iloc[0]
            t = pd.read_csv(st).iloc[0]
            rows.append({"backbone": base, "variant": key, "variant_label": label,
                         "val_accuracy": round(float(v.accuracy), 6),
                         "val_macro_f1": round(float(v.macro_f1), 6),
                         "test_accuracy": round(float(t.accuracy), 6),
                         "test_macro_f1": round(float(t.macro_f1), 6),
                         "test_weighted_f1": round(float(t.weighted_f1), 6)})
            rep = pd.read_csv(os.path.join(d, "report_test.csv"), index_col=0)
            r = {"backbone": base, "variant": key, "variant_label": label}
            for c in classes:
                if c in rep.index:
                    r[f"recall_{c}"] = round(float(rep.loc[c, "recall"]), 4)
                    r[f"support_{c}"] = int(rep.loc[c, "support"])
            r["macro_recall"] = round(float(rep.loc["macro avg", "recall"]), 4)
            recall_rows.append(r)

    if not rows:
        print("nothing to aggregate")
        return
    df = pd.DataFrame(rows)
    winners = []
    for base, g in df.groupby("backbone"):
        w = g.sort_values("val_macro_f1", ascending=False).iloc[0]
        winners.append({"backbone": base, "selected_variant": w.variant,
                        "selected_label": w.variant_label,
                        "selection_metric": "validation macro F1",
                        "val_macro_f1": w.val_macro_f1,
                        "adopted_test_accuracy": w.test_accuracy,
                        "adopted_test_macro_f1": w.test_macro_f1,
                        "baseline_ce_test_macro_f1": float(
                            g[g.variant == "ce"].test_macro_f1.iloc[0])
                        if (g.variant == "ce").any() else None})
    df["is_winner"] = [
        any(w["backbone"] == r.backbone and w["selected_variant"] == r.variant
            for w in winners) for r in df.itertuples()]
    df.to_csv(os.path.join(C.LOG_ROOT, "bench-longtail.csv"), index=False)
    pd.DataFrame(recall_rows).to_csv(
        os.path.join(C.LOG_ROOT, "bench-longtail-recall.csv"), index=False)
    C.write_json(os.path.join(C.LOG_ROOT, "bench-longtail-decision.json"),
                 {"rule": "highest validation macro F1 wins; test numbers reported "
                          "for the adopted variant only as the headline",
                  "winners": winners})
    print(df.to_string(index=False))
    print()
    print(pd.DataFrame(winners).to_string(index=False))


if __name__ == "__main__":
    main()
