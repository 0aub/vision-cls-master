"""Assemble log/BENCH-REPORT.md from whatever has finished so far.

    python3 bench/report.py [--phase A]

Re-runnable at every phase boundary; sections whose inputs do not exist yet are
marked "not yet run" rather than omitted, so a partial report is still honest.
"""
import argparse
import glob
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench import common as C                                      # noqa: E402

LOG = C.LOG_ROOT
TIER_ORDER = ["tier1-classical", "tier2-classic-cnn", "tier3-efficient-cnn",
              "tier4-transformer", "tier5-foundation", "phaseD-longtail",
              "phaseE-copypaste", "unlabelled"]
TIER_TITLE = {
    "tier1-classical": "Tier 1 - classical ML",
    "tier2-classic-cnn": "Tier 2 - classic CNNs",
    "tier3-efficient-cnn": "Tier 3 - efficient CNNs",
    "tier4-transformer": "Tier 4 - supervised transformers",
    "tier5-foundation": "Tier 5 - foundation models",
    "phaseD-longtail": "Phase D - long-tail objectives (5-class)",
    "phaseE-copypaste": "Phase E - lesion copy-paste augmentation (5-class)",
    "unlabelled": "Unlabelled",
}


def collect():
    rows = []
    for d in sorted(glob.glob(os.path.join(LOG, "bench-*"))):
        base = os.path.basename(d)
        if not os.path.isdir(d) or "smoke_" in base or base.startswith("bench-cv-"):
            continue
        parts = base.split("-", 2)
        if len(parts) < 3 or parts[1] not in ("5class", "binary"):
            continue
        task, name = parts[1], parts[2]
        st = os.path.join(d, "summary_test.csv")
        if not os.path.exists(st):
            continue
        s = pd.read_csv(st).iloc[0].to_dict()
        eff = {}
        ef = os.path.join(d, "efficiency.json")
        if os.path.exists(ef):
            try:
                eff = json.load(open(ef))
            except Exception:
                pass
        sv = os.path.join(d, "summary_val.csv")
        val_acc = float(pd.read_csv(sv).accuracy.iloc[0]) if os.path.exists(sv) else np.nan
        rows.append({
            "task": task, "model": name, "tier": eff.get("tier") or "unlabelled",
            "protocol": eff.get("protocol", "uniform"),
            "optimizer": eff.get("optimizer", ""), "lr": eff.get("lr"),
            "weight_decay": eff.get("weight_decay"),
            "warmup_epochs": eff.get("warmup_epochs"),
            "label_smoothing": eff.get("label_smoothing"),
            "aug": eff.get("aug", ""), "select_on": eff.get("select_on", ""),
            "epochs": eff.get("epochs"),
            "best_epoch": eff.get("best_epoch"),
            "train_mode": eff.get("train_mode", ""),
            "test_acc": s.get("accuracy"), "test_macro_f1": s.get("macro_f1"),
            "test_weighted_f1": s.get("weighted_f1"), "test_mean_ce": s.get("mean_ce"),
            "val_acc": val_acc,
            "params_M": (eff.get("params_total") or 0) / 1e6 or None,
            "trainable_M": (eff.get("params_trainable") or 0) / 1e6 or None,
            "gflops": eff.get("gflops_at_224"),
            "gpu_ms_b1": eff.get("gpu_latency_ms_b1"),
            "ips_b16": eff.get("gpu_throughput_ips_b16"),
            "cpu_ms_b1": eff.get("cpu_latency_ms_b1"),
            "vram_MB": eff.get("peak_train_vram_mb"),
            "train_min": eff.get("train_wallclock_min"),
            "batch": eff.get("batch_size"), "dir": d,
        })
    return pd.DataFrame(rows)


def fmt(df, cols, floats=4):
    d = df[cols].copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.{floats}f}")
    return d.to_markdown(index=False)


def leaderboard(df, task, protocol=None):
    out = []
    sub = df[df.task == task]
    if protocol is not None:
        sub = sub[sub.protocol == protocol]
    sub = sub[sub.tier != "hpo-sweep"]
    if sub.empty:
        return "  (no runs yet)\n"
    cols = ["model", "train_mode", "test_acc", "test_macro_f1", "test_weighted_f1",
            "params_M", "trainable_M", "gflops", "gpu_ms_b1", "ips_b16", "cpu_ms_b1",
            "vram_MB", "train_min", "batch"]
    for tier in TIER_ORDER:
        t = sub[sub.tier == tier].sort_values("test_acc", ascending=False)
        if t.empty:
            continue
        out.append(f"\n**{TIER_TITLE.get(tier, tier)}**\n")
        out.append(fmt(t, cols))
        out.append("")
    top = sub.sort_values("test_acc", ascending=False).head(10)
    out.append("\n**Overall top 10 by test accuracy**\n")
    out.append(fmt(top, ["model", "tier", "test_acc", "test_macro_f1", "params_M",
                         "gflops", "gpu_ms_b1"]))
    return "\n".join(out) + "\n"


def read_txt(p, missing="  (not yet run)\n"):
    return ("```\n" + open(p).read().rstrip() + "\n```\n") if os.path.exists(p) else missing


def csv_md(p, cols=None, sort=None, head=None, missing="  (not yet run)\n"):
    if not os.path.exists(p):
        return missing
    d = pd.read_csv(p)
    if sort:
        d = d.sort_values(sort, ascending=False)
    if cols:
        d = d[[c for c in cols if c in d.columns]]
    if head:
        d = d.head(head)
    return d.round(4).to_markdown(index=False) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="")
    args = ap.parse_args()
    df = collect()
    L = []
    w = L.append

    w("# KAUHC V2 benchmark - results report (v2 study plan)")
    w("")
    w(f"generated: {datetime.now().isoformat(timespec='seconds')}"
      + (f"  |  phase checkpoint: {args.phase}" if args.phase else ""))
    w("")
    w("Study design: `BENCHMARK_PLAN_V2.md`. Execution brief: `BENCHMARK_BRIEF_V2.md`")
    w("plus `BENCHMARK_BRIEF_V2_ADDENDUM.md` (the addendum overrides on conflict).")
    w("")
    w("## 0. Protocol")
    w("")
    w("| item | value |")
    w("|---|---|")
    w("| dataset | `data/splitted/V8-KAUHC`, patient-disjoint, seed 1998 |")
    w("| split | train 1916 / val 221 / test 226 (verified against the archived composition CSV at every run) |")
    w("| tasks | `5class` (AVM, Erosion, Normal, Ulcer, Xanthoma) and `binary` (Normal vs Lesion) |")
    w("| input | 224x224, ImageNet normalisation (DINOv2 runs use the official DINOv2 eval transform) |")
    w("| training | full fine-tuning, Adam lr 1e-4, cosine schedule, 100 epochs, batch 16 |")
    w("| augmentation | random horizontal + vertical flip, train split only |")
    w("| checkpoint | best validation accuracy |")
    w("| evaluation | unshuffled deterministic loaders; `preds_*.csv` row order == `PROBS_*.npy` row order |")
    w("| seed | 1998 |")
    w("")
    w("Every run directory `log/bench-<task>-<model>/` holds `history.csv`,")
    w("`summary_{train,val,test}.csv`, `report_{split}.csv`, `preds_{split}.csv`,")
    w("`PROBS_{split}.npy`, `cm_{split}.csv`, `efficiency.json`, `run_config.json`.")
    w("")

    w("## 1. Phase 0 - cleanup and gates")
    w("")
    w(read_txt(os.path.join(LOG, "bench-phase0-gates.txt")))
    w("### Archived artefact index")
    w("")
    w(read_txt(os.path.join(LOG, "bench-archive-index.txt")))

    w("## 2. Environment gate - which SOTA checkpoints loaded")
    w("")
    w(read_txt(os.path.join(LOG, "bench-env-gate.txt")))
    w("Full environment manifest (pip freeze, torch/CUDA, GPU, driver):")
    w("`log/bench-environment.txt`.")
    w("")

    w("## 3. Leaderboards")
    w("")
    w("Cost columns: `params_M` total parameters, `trainable_M` parameters actually")
    w("updated (this is where LoRA and the linear probes separate from full")
    w("fine-tuning), `gflops` at 224 from fvcore, `gpu_ms_b1` median GPU latency at")
    w("batch 1 over 500 timed iterations after 50 warm-ups, `ips_b16` images/s at")
    w("batch 16, `cpu_ms_b1` median CPU latency at batch 1 over 100 iterations after")
    w("10 warm-ups, `vram_MB` peak training VRAM, `train_min` training wall-clock.")
    w("")
    tuned_present = (df.protocol == "tuned").any()
    if tuned_present:
        w("Two protocols are reported. **Protocol B (tuned)** is the headline: one")
        w("validation-selected recipe per tier, chosen by the sweep in section 3.3.")
        w("**Protocol A (uniform)** is the brief's single-recipe-for-everything grid,")
        w("kept as a sensitivity ablation - it quantifies how much of a WCE benchmark's")
        w("ranking is an artefact of sharing one learning rate across every tier.")
        w("")
        w("### 3.1 Five-class - Protocol B (tuned, headline)")
        w(leaderboard(df, "5class", "tuned"))
        w("### 3.2 Binary - Protocol B (tuned, headline)")
        w(leaderboard(df, "binary", "tuned"))
        w("### 3.1a Five-class - Protocol A (uniform, ablation)")
        w(leaderboard(df, "5class", "uniform"))
        w("### 3.2a Binary - Protocol A (uniform, ablation)")
        w(leaderboard(df, "binary", "uniform"))
    else:
        w("### 3.1 Five-class")
        w(leaderboard(df, "5class"))
        w("### 3.2 Binary (lesion detection)")
        w(leaderboard(df, "binary"))
    w("### 3.3 Hyper-parameter sweep (per-tier recipe selection)")
    w("")
    w(read_txt(os.path.join(LOG, "bench-hpo.md"), missing="  (not yet run)\n"))

    w("## 4. Foundation models (tier 5)")
    w("")
    f5 = df[(df.tier == "tier5-foundation")]
    if f5.empty:
        w("  (Phase B not yet run)")
    else:
        w(fmt(f5.sort_values(["task", "test_acc"], ascending=[True, False]),
              ["task", "model", "train_mode", "test_acc", "test_macro_f1",
               "params_M", "trainable_M", "gflops", "gpu_ms_b1"]))
        w("")
        for task in ("5class", "binary"):
            p = os.path.join(LOG, f"bench-biomedclip-prompts-{task}.json")
            if os.path.exists(p):
                w(f"**BiomedCLIP zero-shot prompts ({task})** - all prompts, as required:")
                w("")
                w("```json")
                w(json.dumps(json.load(open(p)), indent=2))
                w("```")
                w("")
    w("")

    w("## 5. Long-tail objectives (Phase D, module M2)")
    w("")
    w(csv_md(os.path.join(LOG, "bench-longtail.csv")))
    w("")
    w("Per-class recall for every variant:")
    w("")
    w(csv_md(os.path.join(LOG, "bench-longtail-recall.csv")))
    w("")

    w("## 5b. Lesion copy-paste augmentation (Phase E, module M3c)")
    w("")
    w(read_txt(os.path.join(LOG, "bench-phaseE.md"), missing="  (not run)\n"))

    w("## 6. Localization faithfulness (Phase C, module M3)")
    w("")
    w("### 6.1 C1 gate - annotated exports")
    w("")
    w(read_txt(os.path.join(LOG, "bench-lesion-masks", "gate.txt")))
    p = os.path.join(LOG, "bench-lesion-masks", "summary.json")
    if os.path.exists(p):
        w("Masks built:")
        w("")
        w("```json")
        w(json.dumps(json.load(open(p)), indent=2))
        w("```")
        w("")
    w("### 6.2 Pointing game and CAM-region IoU")
    w("")
    w("Scored against the ellipse INTERIOR (the lesion region the clinician")
    w("circled), with the drawn stroke reported alongside. Chance level is the")
    w("mean lesion-region fraction, about 3.3% of the frame.")
    w("")
    w("The archived ring-trained checkpoint appears TWICE, on the same frames:")
    w("once on the reconstructed ringed input it was trained on and once on the")
    w("clean version. That pairing is the shortcut-learning test - scoring it on")
    w("ringed frames alone cannot show the shortcut, because the ellipse was drawn")
    w("around the lesion, so firing on the stroke lands inside the region anyway.")
    w("")
    w(csv_md(os.path.join(LOG, "bench-cam", "bench-pointing-game-summary.csv")))
    w("")
    w("**Where inside the ellipse the attention peak falls** (core = eroded lesion")
    w("body, annulus = rim beside the drawn stroke):")
    w("")
    w(csv_md(os.path.join(LOG, "bench-cam", "bench-cam-geometry-summary.csv")))
    w("")
    panels = sorted(glob.glob(os.path.join(LOG, "bench-cam", "panels", "*.png")))
    w(f"Qualitative panels: {len(panels)} PNG(s) under `log/bench-cam/panels/`.")
    w("")

    w("## 7. Calibration and selective prediction (Phase C, module M4)")
    w("")
    w(csv_md(os.path.join(LOG, "bench-trust.csv"),
             cols=["task", "tier", "model", "test_accuracy", "temperature",
                   "ece_before", "ece_after", "acc_at_coverage_100",
                   "acc_at_coverage_90", "acc_at_coverage_80", "acc_at_coverage_70"],
             sort=["task", "test_accuracy"]))
    w("")
    w("Reliability-diagram data: `log/bench-reliability.csv`. Full accuracy-vs-coverage")
    w("curves at 5% steps: `log/bench-coverage.csv`.")
    w("")

    w("## 8. Patient-grouped 4-fold cross-validation")
    w("")
    cvs = sorted(glob.glob(os.path.join(LOG, "bench-cv-*", "cv_summary.csv")))
    if not cvs:
        w("  (not yet run)")
    else:
        w(pd.concat([pd.read_csv(f) for f in cvs], ignore_index=True)
            .round(4).to_markdown(index=False))
    w("")

    w("## 9. Significance")
    w("")
    w(read_txt(os.path.join(LOG, "bench-mcnemar.txt")))
    w("### Bootstrap 95% CIs (10,000 resamples)")
    w("")
    w(csv_md(os.path.join(LOG, "bench-bootstrap-ci.csv"), sort=["task", "accuracy"]))
    w("")

    w("## 10. Anomalies, deviations and open items")
    w("")
    w(read_txt(os.path.join(LOG, "bench-anomalies.md"), missing="  (none recorded)\n"))

    w("## 11. Wall-clock")
    w("")
    if not df.empty and df.train_min.notna().any():
        by = df.groupby("tier").train_min.agg(["count", "sum", "mean"]).round(2)
        by.columns = ["runs", "total_min", "mean_min"]
        w(by.reset_index().to_markdown(index=False))
        w("")
        w(f"Total training wall-clock across all recorded runs: "
          f"{df.train_min.sum():.1f} min ({df.train_min.sum()/60:.1f} h).")
    w("")
    w("Per-phase timing log: `log/bench-phase-timing.txt`. Per-run progress log:")
    w("`log/bench-progress.txt`.")
    w("")
    w(read_txt(os.path.join(LOG, "bench-phase-timing.txt"), missing=""))

    out = os.path.join(LOG, "BENCH-REPORT.md")
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"[done] {out}  ({len(df)} runs summarised)")


if __name__ == "__main__":
    main()
