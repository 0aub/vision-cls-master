"""Pick one recipe per tier from the sweep, on VALIDATION macro F1.

    python3 bench/hpo_select.py

Reads every log/bench-5class-hpo_* run, ranks the cells of each tier's sweep by
validation macro F1 (never by test), and writes:
    log/bench-hpo.json   the chosen recipe per tier, machine-readable
    log/bench-hpo.md     the full sweep table for the report
The recipe is applied to every model in that tier by bench_tuned.sh.
"""
import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench import common as C                                      # noqa: E402

# which sweep representative stands for which tier
REPRESENTATIVE = {
    "resnet50": "tier2-classic-cnn",
    "convnext_tiny": "tier3-efficient-cnn",
    "vit_b_16": "tier4-transformer",
    "dinov2_vitb14_lora": "tier5-foundation",
}
KEYS = ("optimizer", "lr", "weight_decay", "warmup_epochs", "label_smoothing",
        "aug", "epochs")


def rows():
    out = []
    for d in sorted(glob.glob(os.path.join(C.LOG_ROOT, "bench-5class-hpo_*"))):
        st, sv = (os.path.join(d, f) for f in ("summary_test.csv", "summary_val.csv"))
        ef = os.path.join(d, "efficiency.json")
        if not all(os.path.exists(f) for f in (st, sv, ef)):
            continue
        e = json.load(open(ef))
        v = pd.read_csv(sv).iloc[0]
        t = pd.read_csv(st).iloc[0]
        name = os.path.basename(d).replace("bench-5class-hpo_", "")
        base = name.split("__")[0]
        out.append({
            "representative": base, "cell": name.split("__")[-1],
            "tier": REPRESENTATIVE.get(base, "?"),
            **{k: e.get(k) for k in KEYS},
            "best_epoch": e.get("best_epoch"),
            "val_accuracy": round(float(v.accuracy), 4),
            "val_macro_f1": round(float(v.macro_f1), 4),
            "test_accuracy": round(float(t.accuracy), 4),
            "test_macro_f1": round(float(t.macro_f1), 4),
        })
    return pd.DataFrame(out)


def main():
    df = rows()
    if df.empty:
        print("no sweep runs found")
        return
    chosen, lines = {}, []
    lines.append("Recipe selection is on **validation macro F1**, never on test. Each")
    lines.append("tier's recipe comes from one representative model swept over six cells")
    lines.append("(four for the LoRA tier's learning rate), 50 epochs each, and is then")
    lines.append("applied to every model in that tier.")
    lines.append("")
    for base, g in df.groupby("representative"):
        tier = REPRESENTATIVE.get(base, "?")
        g = g.sort_values("val_macro_f1", ascending=False)
        win = g.iloc[0]
        chosen[tier] = {"representative": base, "cell": win.cell,
                        **{k: win[k] for k in KEYS},
                        "val_macro_f1": float(win.val_macro_f1)}
        lines.append(f"**{tier}** - swept on `{base}`, winner `{win.cell}`"
                     f" (val macro F1 {win.val_macro_f1:.4f})")
        lines.append("")
        lines.append(g[["cell", "optimizer", "lr", "weight_decay", "warmup_epochs",
                        "label_smoothing", "aug", "best_epoch", "val_accuracy",
                        "val_macro_f1", "test_accuracy", "test_macro_f1"]]
                     .to_markdown(index=False))
        lines.append("")
    C.write_json(os.path.join(C.LOG_ROOT, "bench-hpo.json"), chosen)
    with open(os.path.join(C.LOG_ROOT, "bench-hpo.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    df.sort_values(["tier", "val_macro_f1"], ascending=[True, False]).to_csv(
        os.path.join(C.LOG_ROOT, "bench-hpo-sweep.csv"), index=False)
    print("\n".join(lines))
    # shell-consumable
    for tier, c in chosen.items():
        # warmup_epochs is an int flag; efficiency.json round-trips it as a float,
        # and argparse(type=int) rejects "0.0"
        print("RECIPE\t%s\t--optimizer %s --lr %g --weight_decay %g "
              "--warmup_epochs %d --label_smoothing %g --aug %s"
              % (tier, c["optimizer"], float(c["lr"]), float(c["weight_decay"]),
                 int(float(c["warmup_epochs"])), float(c["label_smoothing"]),
                 c["aug"]))


if __name__ == "__main__":
    main()
