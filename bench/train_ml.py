"""Classical-ML tier (tier 1) in the same standardized output format.

    python3 bench/train_ml.py --model svm --task 5class --features raw
    python3 bench/train_ml.py --model svm --task binary --features embed:dinov2_vitb14

Two deliberate departures from the 2024 runs, both required by the brief:
  * features come from the DETERMINISTIC eval transform, not the augmented
    train loader (addendum B2) - the 2024 code fitted sklearn on one random
    flip of each training image;
  * inputs are 224x224 (150,528 dims raw), matching the rest of the v2 grid.

--features raw          flattened 224x224x3 pixels after ImageNet normalisation
--features embed:<name> penultimate features cached by bench/embed.py
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench import common as C                                      # noqa: E402

ML_MODELS = ["logistic_regression", "decision_tree", "random_forest", "svm", "knn",
             "naive_bayes", "adaboost", "lda", "qda", "mlp"]
SVM_MAX_ITER = int(os.environ.get("BENCH_SVM_MAX_ITER", 20_000_000))

# val-selected grids (Phase B embeddings). Raw-pixel runs use the defaults only.
GRIDS = {
    "logistic_regression": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
    "svm":                 [{"C": c} for c in (0.1, 1.0, 10.0)],
    "knn":                 [{"n_neighbors": k} for k in (1, 3, 5, 9, 15)],
    "random_forest":       [{"n_estimators": n} for n in (100, 300)],
    "decision_tree":       [{"max_depth": d} for d in (None, 5, 10, 20)],
    "adaboost":            [{"n_estimators": n} for n in (50, 100)],
    "mlp":                 [{"hidden_layer_sizes": h} for h in ((100,), (256,))],
    "lda":                 [{}],
    "qda":                 [{"reg_param": r} for r in (0.0, 0.01, 0.1)],
    "naive_bayes":         [{}],
}


def make_model(name, params=None, raw_pixels=False):
    from src.modules import get_ml_model
    m = get_ml_model(name)
    if name == "svm" and raw_pixels:
        # THE fix for this cell. With sklearn's default 200 MB kernel cache the
        # fit ran 9 h 50 m without finishing; with 2 GB it converges in ~1 min.
        # The reason is dimensionality, not problem size: every cache miss costs
        # a fresh kernel row, O(n_samples x 150,528) = ~3e8 multiply-adds, so
        # libsvm's shrinking heuristic thrashing the cache is ruinous here even
        # though the Gram matrix itself is small. Measured n_iter_ is 204-1374,
        # so the SMO bound below is a safety net that never binds; both it and
        # the convergence flag are recorded in efficiency.json.
        m.set_params(cache_size=2000, max_iter=SVM_MAX_ITER)
        # SVC(probability=True) refits the model five more times for Platt
        # scaling. On 150,528 raw dimensions that is hours, and it changes only
        # the probability estimates - never the decision boundary, never the
        # predictions. Probabilities come from softmax(decision_function)
        # instead, which the M4 temperature scaling recalibrates anyway.
        m.set_params(probability=False)
    if params:
        m.set_params(**params)
    return m


def raw_features(split, task, image_size=224, workers=4):
    import torch
    tf = C.eval_transform(image_size)
    ds, dl, paths = C.eval_loader(split, task, tf, batch_size=64, workers=workers)
    X, y = [], []
    for xb, yb in dl:
        X.append(xb.view(xb.size(0), -1).numpy().astype(np.float32))
        y.append(yb.numpy())
    return np.vstack(X), np.concatenate(y), paths


def embed_features(split, task, embed_name):
    d = os.path.join(C.LOG_ROOT, "bench-embeddings", embed_name)
    X = np.load(os.path.join(d, f"X_{split}.npy"))
    paths = [l.rstrip("\n") for l in open(os.path.join(d, f"paths_{split}.txt"))]
    y = C.labels_of(split, task)
    return X.astype(np.float32), y, paths


def load_features(split, task, features, workers=4):
    if features == "raw":
        return raw_features(split, task, workers=workers)
    if features.startswith("embed:"):
        return embed_features(split, task, features.split(":", 1)[1])
    raise ValueError(f"unknown --features {features!r}")


def probs_of(model, X, num_classes):
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if p.shape[1] == num_classes:
            return p
        out = np.zeros((len(X), num_classes), dtype=np.float64)
        for j, c in enumerate(model.classes_):
            out[:, int(c)] = p[:, j]
        return out
    d = model.decision_function(X)
    if d.ndim == 1:
        d = np.stack([-d, d], axis=1)
    e = np.exp(d - d.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=ML_MODELS)
    ap.add_argument("--task", default="5class", choices=["5class", "binary"])
    ap.add_argument("--features", default="raw")
    ap.add_argument("--select", action="store_true",
                    help="tune on val with the grid above (default: library defaults)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--tier", default="tier1-classical")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    tag = "raw" if args.features == "raw" else args.features.split(":", 1)[1]
    name = args.name or (f"ml_{args.model}_{tag}")
    C.assert_split()
    out_dir = C.run_dir(args.task, name)
    if C.already_done(args.task, name) and not args.force:
        print(f"[skip] {out_dir} already has summary_test.csv")
        return
    os.makedirs(out_dir, exist_ok=True)
    C.seed_everything()
    classes = C.classes_for(args.task)
    K = len(classes)

    t0 = time.time()
    data = {sp: load_features(sp, args.task, args.features, args.workers)
            for sp in C.SPLITS}
    feat_s = time.time() - t0
    Xtr, ytr, _ = data["train"]
    print(f"=== {args.task} / {name}  features {Xtr.shape} ({feat_s:.1f}s) ===", flush=True)

    from sklearn.metrics import f1_score
    raw = args.features == "raw"
    chosen, best_f1 = {}, None
    grid = GRIDS.get(args.model, [{}]) if args.select else [{}]
    t0 = time.time()
    if len(grid) > 1:
        Xva, yva, _ = data["val"]
        for params in grid:
            m = make_model(args.model, params, raw)
            m.fit(Xtr, ytr)
            f1 = f1_score(yva, m.predict(Xva), average="macro", zero_division=0)
            print(f"    grid {params} -> val macroF1 {f1:.4f}", flush=True)
            if best_f1 is None or f1 > best_f1:
                best_f1, chosen = f1, params
    model = make_model(args.model, chosen, raw)
    model.fit(Xtr, ytr)
    fit_s = time.time() - t0
    with open(os.path.join(out_dir, "best.pkl"), "wb") as f:
        pickle.dump(model, f)

    summary = {}
    lat = None
    for sp in C.SPLITS:
        X, y, paths = data[sp]
        p = probs_of(model, X, K)
        row = C.dump_split(out_dir, sp, y, p.argmax(1), paths, classes, probs=p,
                           mean_ce=C.mean_cross_entropy(p, y))
        summary[sp] = row
        print(f"  [{sp}] acc={row['accuracy']:.4f} macroF1={row['macro_f1']:.4f}", flush=True)
        if sp == "test":
            t = time.time()
            for i in range(min(100, len(X))):
                model.predict(X[i:i + 1])
            lat = (time.time() - t) / min(100, len(X)) * 1000.0

    n_params = None
    for attr in ("coef_", "feature_log_prob_", "coefs_"):
        v = getattr(model, attr, None)
        if v is not None:
            n_params = int(np.sum([np.asarray(a).size for a in
                                   (v if isinstance(v, list) else [v])]))
            break
    eff = {
        "model": args.model, "name": name, "task": args.task,
        "source": "sklearn", "train_mode": "fit", "tier": args.tier,
        "features": args.features, "feature_dim": int(Xtr.shape[1]),
        "params_total": n_params, "params_trainable": n_params,
        "gflops_at_224": None,
        "gflops_tool": "n/a (classical model; complexity is not a FLOP count)",
        "gpu_latency_ms_b1": None, "gpu_throughput_ips_b16": None,
        "cpu_latency_ms_b1": round(lat, 4) if lat else None,
        "cpu_threads": int(os.environ.get("OMP_NUM_THREADS", 0)) or None,
        "peak_train_vram_mb": None,
        "train_wallclock_min": round(fit_s / 60.0, 3),
        "feature_extraction_min": round(feat_s / 60.0, 3),
        "selected_params": chosen, "val_selected": bool(args.select),
        "svm_n_iter": ([int(v) for v in np.atleast_1d(model.n_iter_)]
                       if getattr(model, "n_iter_", None) is not None else None),
        "svm_max_iter_budget": SVM_MAX_ITER if (args.model == "svm" and raw) else None,
        "svm_cache_size_mb": (getattr(model, "cache_size", None)
                              if (args.model == "svm" and raw) else None),
        "svm_converged": (bool(np.all(np.atleast_1d(model.n_iter_) < SVM_MAX_ITER))
                          if (args.model == "svm" and raw
                              and getattr(model, "n_iter_", None) is not None)
                          else None),
        "svm_probability_note": ("softmax(decision_function): SVC(probability=True) "
                                 "is intractable on 150,528 raw dims and does not "
                                 "change predictions")
        if (args.model == "svm" and raw) else None,
        "val_macro_f1_of_selection": best_f1,
        "model_size_mb": round(os.path.getsize(os.path.join(out_dir, "best.pkl")) / 1024**2, 2),
        "seed": C.SEED,
    }
    C.write_json(os.path.join(out_dir, "efficiency.json"), eff)
    C.write_json(os.path.join(out_dir, "run_config.json"), vars(args))
    C.log_progress(f"{args.task:7s} {name:34s} test_acc={summary['test']['accuracy']:.4f} "
                   f"test_macroF1={summary['test']['macro_f1']:.4f} "
                   f"fit={fit_s/60:.1f}min feat={feat_s/60:.1f}min")
    print(f"[done] {out_dir}")


if __name__ == "__main__":
    main()
