"""Re-measure every run's cost columns in one clean pass, machine otherwise idle.

    python3 bench/remeasure.py --deep          # torchvision / DINOv2 / BiomedCLIP
    python3 bench/remeasure.py --classical     # tier-1 sklearn models
    python3 bench/remeasure.py --deep --classical

Latency is the one number in efficiency.json that depends on what else the
machine was doing when it was taken. During the grid the classical tier ran
alongside the GPU queue, and comparing the same architecture's 5-class and
binary measurements showed up to 2.5x disagreement in CPU latency - noise, not
signal. This pass recomputes params, GFLOPs, GPU latency/throughput and CPU
latency for every finished run under identical idle conditions and writes them
back, preserving the fields only training can supply (peak train VRAM, training
wall-clock, batch size, selected epoch).

Run it with nothing else on the box.
"""
import argparse
import glob
import json
import os
import pickle
import sys
import time
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench import common as C                                      # noqa: E402
from bench import efficiency as EFF                                # noqa: E402
from src.modules import build_model                                # noqa: E402

# supplied by training, never by a latency pass
KEEP = ("peak_train_vram_mb", "train_wallclock_min", "batch_size",
        "best_val_accuracy", "best_epoch", "loss_weights", "feature_extraction_min",
        "selected_params", "val_selected", "val_macro_f1_of_selection",
        "model_size_mb", "selection", "embedding_meta", "svm_probability_note")


def runs():
    for d in sorted(glob.glob(os.path.join(C.LOG_ROOT, "bench-*"))):
        base = os.path.basename(d)
        if not os.path.isdir(d) or "smoke_" in base or base.startswith("bench-cv-"):
            continue
        p = base.split("-", 2)
        if len(p) < 3 or p[1] not in ("5class", "binary"):
            continue
        ef = os.path.join(d, "efficiency.json")
        cf = os.path.join(d, "run_config.json")
        if not (os.path.exists(ef) and os.path.exists(cf)):
            continue
        yield d, p[1], p[2], json.load(open(ef)), json.load(open(cf))


def remeasure_deep(d, task, name, eff, cfg):
    ck = os.path.join(d, "best.pth")
    if not os.path.exists(ck):
        return None
    K = len(C.classes_for(task))
    model = build_model(cfg.get("model", name), K,
                        source=cfg.get("source", "torchvision"),
                        train_mode=cfg.get("train_mode", "full"),
                        feature_mode=cfg.get("feature_mode", "cls"),
                        lora_r=cfg.get("lora_r", 8),
                        lora_alpha=cfg.get("lora_alpha", 16),
                        lora_dropout=cfg.get("lora_dropout", 0.05))
    model.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True))
    if torch.cuda.is_available():
        model = model.cuda()
    out = EFF.measure_all(model, int(cfg.get("image_size", 224)))
    del model
    torch.cuda.empty_cache()
    return out


def remeasure_classical(d, task, name, eff, cfg):
    pk = os.path.join(d, "best.pkl")
    if not os.path.exists(pk):
        return None
    from bench.train_ml import load_features
    X, _, _ = load_features("test", task, cfg.get("features", "raw"), workers=8)
    with open(pk, "rb") as f:
        model = pickle.load(f)
    n = min(100, len(X))
    for i in range(min(10, n)):                     # warm-up
        model.predict(X[i:i + 1])
    ts = []
    for i in range(n):
        t = time.perf_counter()
        model.predict(X[i:i + 1])
        ts.append((time.perf_counter() - t) * 1000.0)
    del model, X
    return {"cpu_latency_ms_b1": round(float(np.median(ts)), 4),
            "cpu_threads": torch.get_num_threads()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep", action="store_true")
    ap.add_argument("--classical", action="store_true")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    if not (args.deep or args.classical):
        args.deep = args.classical = True

    stamp = datetime.now().isoformat(timespec="seconds")
    n_ok = n_skip = 0
    for d, task, name, eff, cfg in runs():
        classical = eff.get("source") == "sklearn"
        if classical and not args.classical:
            continue
        if not classical and not args.deep:
            continue
        try:
            new = (remeasure_classical if classical else remeasure_deep)(
                d, task, name, eff, cfg)
        except Exception as e:
            print(f"  [fail] {task}/{name}: {type(e).__name__}: {e}", flush=True)
            n_skip += 1
            continue
        if new is None:
            n_skip += 1
            continue
        before = eff.get("gpu_latency_ms_b1"), eff.get("cpu_latency_ms_b1")
        merged = dict(eff)
        merged.update({k: v for k, v in new.items() if k not in KEEP})
        merged["efficiency_remeasured"] = stamp
        merged["efficiency_remeasure_note"] = (
            "params / GFLOPs / GPU latency+throughput / CPU latency re-taken in a "
            "single idle pass so every row in the cost table is comparable; "
            "peak_train_vram_mb, train_wallclock_min and batch_size are the "
            "originals from training")
        if not args.dry:
            C.write_json(os.path.join(d, "efficiency.json"), merged)
        print(f"  {task:7s} {name:34s} gpu {before[0]} -> "
              f"{merged.get('gpu_latency_ms_b1')}   cpu {before[1]} -> "
              f"{merged.get('cpu_latency_ms_b1')}", flush=True)
        n_ok += 1
    print(f"[done] re-measured {n_ok}, skipped {n_skip}")


if __name__ == "__main__":
    main()
