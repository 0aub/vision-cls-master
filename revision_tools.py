# revision_tools.py
# Place this file in the ROOT of the vision-cls-master repo (next to src/) and run it there.
# It does NOT retrain anything and does NOT touch your existing code.
#
# Subcommands:
#   eval      : re-evaluate one saved model (best.pth / best.pkl) on train/val/test with
#               macro + weighted + per-class metrics, correct loss, and per-sample predictions
#   embed-ml  : train the 10 classical ML models on frozen CNN embeddings (reviewer comment 7)
#   mcnemar   : McNemar significance test between two models' preds_test.csv files
#
# Requirements: torch, torchvision, scikit-learn, pandas (all already in the repo env).
#               statsmodels only for mcnemar (pip install statsmodels).
#
# Typical usage (from the repo root, dataset at data/splitted/V7-KAUHC):
#   python revision_tools.py eval --log "log/v7-alexnet 2024-11-19 21-17-42"
#   for d in log/v7-*/ ; do python revision_tools.py eval --log "$d" ; done
#   python revision_tools.py embed-ml --backbone efficientnet_b0
#   python revision_tools.py embed-ml --backbone resnet152 --tune
#   python revision_tools.py mcnemar --a "<log>/preds_test.csv" --b "<log>/preds_test.csv"

import argparse, os, sys, pickle, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import (classification_report, precision_recall_fscore_support,
                             accuracy_score, confusion_matrix)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.modules import pretrained_network, get_ml_model  # noqa: E402

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def require_cuda(what):
    """The CNN paths must not silently fall back to CPU - it takes hours and looks
    like success. The classical models are CPU-only by design and never call this."""
    if DEVICE.type != "cuda" and os.environ.get("ALLOW_CPU_DL") != "1":
        raise SystemExit(f"refusing to run {what} on CPU: no CUDA device visible. "
                         f"Use the GPU image with --gpus all, or set ALLOW_CPU_DL=1 "
                         f"to override deliberately.")
ML_MODELS = ['logistic_regression', 'decision_tree', 'random_forest', 'svm', 'knn',
             'naive_bayes', 'adaboost', 'lda', 'qda', 'mlp']

# ---------------------------------------------------------------- data ----

def eval_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

def make_loader(data_root, split, image_size, batch_size=32):
    ds = ImageFolder(os.path.join(data_root, split), transform=eval_transform(image_size))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    paths = [p for p, _ in ds.samples]
    return ds, loader, paths

# ------------------------------------------------------------- metrics ----

def dump_all_metrics(y_true, y_pred, paths, classes, out_dir, split, extra=None):
    os.makedirs(out_dir, exist_ok=True)
    rep = classification_report(y_true, y_pred, target_names=classes, digits=4,
                                output_dict=True, zero_division=0)
    pd.DataFrame(rep).T.to_csv(os.path.join(out_dir, f"report_{split}.csv"))

    row = {"accuracy": accuracy_score(y_true, y_pred)}
    for avg in ("macro", "weighted"):
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=avg, zero_division=0)
        row.update({f"{avg}_P": p, f"{avg}_R": r, f"{avg}_F1": f1})
    if extra:
        row.update(extra)
    pd.DataFrame([row]).round(6).to_csv(os.path.join(out_dir, f"summary_{split}.csv"), index=False)

    pd.DataFrame({"path": paths, "y_true": y_true, "y_pred": y_pred}).to_csv(
        os.path.join(out_dir, f"preds_{split}.csv"), index=False)

    np.savetxt(os.path.join(out_dir, f"cm_{split}.csv"),
               confusion_matrix(y_true, y_pred), fmt="%d", delimiter=",")
    print(f"  [{split}] acc={row['accuracy']:.4f} "
          f"macroF1={row['macro_F1']:.4f} weightedF1={row['weighted_F1']:.4f}"
          + (f" loss={extra['loss']:.4f}" if extra and 'loss' in extra else ""))

# ---------------------------------------------------------------- eval ----

def parse_args_txt(log_dir):
    cfg = {}
    with open(os.path.join(log_dir, "args.txt")) as f:
        for line in f:
            if ':' in line:
                k, v = line.split(':', 1)
                v = v.strip()
                cfg[k.strip()] = None if v == "None" else v
    return cfg

def flatten_features(loader):
    xs, ys = [], []
    for inputs, targets in loader:
        xs.append(inputs.view(inputs.size(0), -1).numpy().astype(np.float32))
        ys.append(targets.numpy())
    return np.vstack(xs), np.concatenate(ys)

def cmd_eval(args):
    log_dir = args.log.rstrip('/').rstrip('\\')
    cfg = parse_args_txt(log_dir)
    model_name = cfg["model_name"]
    image_size = int(cfg.get("image_size", 256))
    is_ml = model_name in ML_MODELS
    out_dir = args.out or os.path.join("log", f"revision-eval-{model_name}")
    print(f"[eval] {model_name} ({'ML' if is_ml else 'DL'}), image_size={image_size} -> {out_dir}")

    if is_ml:
        with open(os.path.join(log_dir, "best.pkl"), "rb") as f:
            model = pickle.load(f)
        for split in ("train", "val", "test"):
            ds, loader, paths = make_loader(args.data, split, image_size)
            X, y = flatten_features(loader)
            y_pred = model.predict(X)
            dump_all_metrics(y, y_pred, paths, ds.classes, out_dir, split)
    else:
        require_cuda(f"the CNN {model_name}")
        ds0, _, _ = make_loader(args.data, "test", image_size)
        # rebuild exactly the architecture the run was trained with, attention included
        attn = cfg.get("attention_name")
        attn_idx = int(cfg.get("attention_index") or 4)
        model = pretrained_network(model_name, attn, attn_idx, len(ds0.classes))
        state = torch.load(os.path.join(log_dir, "best.pth"), map_location=DEVICE)
        model.load_state_dict(state)          # strict=True: a mismatch must fail loudly
        model.to(DEVICE).eval()
        criterion = nn.CrossEntropyLoss(reduction="sum")
        for split in ("train", "val", "test"):
            ds, loader, paths = make_loader(args.data, split, image_size)
            y_true, y_pred, loss_sum = [], [], 0.0
            with torch.no_grad():
                for inputs, labels in loader:
                    inputs = inputs.to(DEVICE, dtype=torch.float)
                    labels = labels.to(DEVICE, dtype=torch.long)
                    outputs = model(inputs)
                    loss_sum += criterion(outputs, labels).item()
                    y_pred += outputs.argmax(1).cpu().tolist()
                    y_true += labels.cpu().tolist()
            # correctly normalized mean cross-entropy: sum over samples / N
            dump_all_metrics(np.array(y_true), np.array(y_pred), paths, ds.classes,
                             out_dir, split, extra={"loss": loss_sum / len(ds)})

# ------------------------------------------------------------ embed-ml ----

def make_extractor(model_name, n_classes=5):
    """Frozen ImageNet backbone with the final classification layer removed."""
    model = pretrained_network(model_name, None, 4, n_classes)
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):          # resnet, googlenet
        model.fc = nn.Identity()
    elif hasattr(model, "classifier"):
        clf = model.classifier
        if isinstance(clf, nn.Linear):                                    # densenet
            model.classifier = nn.Identity()
        elif isinstance(clf, nn.Sequential):                              # alexnet, vgg, efficientnet
            for i in range(len(clf) - 1, -1, -1):
                if isinstance(clf[i], nn.Linear):
                    clf[i] = nn.Identity()
                    break
    else:
        raise ValueError(f"don't know how to strip the head of {model_name}")
    return model.to(DEVICE).eval()

def extract_embeddings(model_name, data_root, image_size=256):
    require_cuda(f"the {model_name} feature extractor")
    ext = make_extractor(model_name)
    out = {}
    for split in ("train", "val", "test"):
        ds, loader, paths = make_loader(data_root, split, image_size)
        feats, ys = [], []
        with torch.no_grad():
            for inputs, targets in loader:
                f = ext(inputs.to(DEVICE, dtype=torch.float))
                feats.append(f.view(f.size(0), -1).cpu().numpy().astype(np.float32))
                ys.append(targets.numpy())
        out[split] = (np.vstack(feats), np.concatenate(ys), paths, ds.classes)
        print(f"  [{split}] embeddings: {out[split][0].shape}")
    return out

TUNE_GRIDS = {
    "logistic_regression": {"C": [0.01, 0.1, 1, 10]},
    "decision_tree":       {"max_depth": [5, 10, 20, None]},
    "random_forest":       {"n_estimators": [100, 300], "max_depth": [10, None]},
    "svm":                 {"C": [0.1, 1, 10], "gamma": ["scale", "auto"]},
    "knn":                 {"n_neighbors": [3, 5, 9, 15]},
    "naive_bayes":         {"var_smoothing": [1e-9, 1e-7, 1e-5]},
    "adaboost":            {"n_estimators": [50, 200], "learning_rate": [0.5, 1.0]},
    "lda":                 {},
    "qda":                 {"reg_param": [0.0, 0.1, 0.5]},
    "mlp":                 {"hidden_layer_sizes": [(100,), (256, 64)], "alpha": [1e-4, 1e-2]},
}

def cmd_embed_ml(args):
    from sklearn.model_selection import ParameterGrid
    from sklearn.base import clone
    out_dir = args.out or os.path.join("log", f"revision-embed-{args.backbone}" + ("-tuned" if args.tune else ""))
    os.makedirs(out_dir, exist_ok=True)
    print(f"[embed-ml] backbone={args.backbone} tune={args.tune} -> {out_dir}")
    E = extract_embeddings(args.backbone, args.data, args.image_size)
    (Xtr, ytr, _, classes) = E["train"]
    (Xva, yva, _, _) = E["val"]
    summary = []
    for name in ML_MODELS:
        base = get_ml_model(name)
        if args.tune and TUNE_GRIDS.get(name):
            best_f1, best_params = -1, None
            for params in ParameterGrid(TUNE_GRIDS[name]):
                m = clone(base).set_params(**params)
                m.fit(Xtr, ytr)
                _, _, f1, _ = precision_recall_fscore_support(
                    yva, m.predict(Xva), average="macro", zero_division=0)
                if f1 > best_f1:
                    best_f1, best_params = f1, params
            model = clone(base).set_params(**best_params)
            print(f"  {name}: best params {best_params} (val macroF1={best_f1:.4f})")
        else:
            model = clone(base)
        model.fit(Xtr, ytr)
        mdir = os.path.join(out_dir, name)
        for split in ("train", "val", "test"):
            X, y, paths, _ = E[split]
            dump_all_metrics(y, model.predict(X), paths, classes, mdir, split)
        s = pd.read_csv(os.path.join(mdir, "summary_test.csv")).iloc[0].to_dict()
        s["model"] = name
        summary.append(s)
    cols = ["model", "accuracy", "macro_P", "macro_R", "macro_F1",
            "weighted_P", "weighted_R", "weighted_F1"]
    pd.DataFrame(summary)[cols].round(4).to_csv(os.path.join(out_dir, "summary_test_all.csv"), index=False)
    print(f"[embed-ml] test-set summary: {out_dir}/summary_test_all.csv")

# ------------------------------------------------------------- mcnemar ----

def cmd_mcnemar(args):
    from statsmodels.stats.contingency_tables import mcnemar
    a = pd.read_csv(args.a); b = pd.read_csv(args.b)
    m = a.merge(b, on="path", suffixes=("_a", "_b"))
    assert len(m) == len(a) == len(b), "prediction files do not cover the same samples"
    assert (m["y_true_a"] == m["y_true_b"]).all(), "the two files disagree on the ground truth"
    ok_a = m["y_true_a"] == m["y_pred_a"]
    ok_b = m["y_true_b"] == m["y_pred_b"]
    n11 = int((ok_a & ok_b).sum());  n01 = int((ok_a & ~ok_b).sum())
    n10 = int((~ok_a & ok_b).sum()); n00 = int((~ok_a & ~ok_b).sum())
    res = mcnemar([[n11, n01], [n10, n00]], exact=True)
    disc = n01 + n10

    lines = [
        "McNemar exact test (test split, paired by image path)",
        "=" * 70,
        f"model A            : {args.name_a or args.a}",
        f"model B            : {args.name_b or args.b}",
        f"paired samples     : {len(m)}",
        "",
        "contingency table (counts of test images)",
        f"{'':<22}{'B correct':>12}{'B wrong':>12}",
        f"{'A correct':<22}{n11:>12}{n01:>12}",
        f"{'A wrong':<22}{n10:>12}{n00:>12}",
        "",
        f"accuracy A         : {(n11 + n01) / len(m):.4f}  ({n11 + n01}/{len(m)})",
        f"accuracy B         : {(n11 + n10) / len(m):.4f}  ({n11 + n10}/{len(m)})",
        f"discordant pairs   : {disc}   (A right/B wrong = {n01}, A wrong/B right = {n10})",
        "",
        f"McNemar exact p    : {res.pvalue:.6f}",
        f"statistic          : {res.statistic}",
        "",
    ]
    alpha = 0.05
    if res.pvalue < alpha:
        lines.append(f"At alpha={alpha}: REJECT H0 - the two models' error rates differ significantly.")
    else:
        lines.append(f"At alpha={alpha}: FAIL TO REJECT H0 - no significant difference in error rates.")
    if disc == 0:
        lines.append("NOTE: zero discordant pairs - the two models made identical predictions on")
        lines.append("      every test image, so the test carries no information (p = 1 by construction).")
    elif disc < 10:
        lines.append(f"NOTE: only {disc} discordant pairs; the exact test is valid but has very low power.")

    out = "\n".join(lines)
    print(out)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write(out + "\n")
        print(f"\n[written] {args.out}")

# ---------------------------------------------------------------- main ----

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("eval")
    e.add_argument("--log", required=True, help="existing run folder containing args.txt and best.pth/best.pkl")
    e.add_argument("--data", default="data/splitted/V7-KAUHC")
    e.add_argument("--out", default=None)

    g = sub.add_parser("embed-ml")
    g.add_argument("--backbone", required=True,
                   help="e.g. efficientnet_b0, resnet152, densenet201")
    g.add_argument("--data", default="data/splitted/V7-KAUHC")
    g.add_argument("--image_size", type=int, default=256)
    g.add_argument("--tune", action="store_true", help="small grid search selected on the val split")
    g.add_argument("--out", default=None)

    n = sub.add_parser("mcnemar")
    n.add_argument("--a", required=True); n.add_argument("--b", required=True)
    n.add_argument("--name_a", default=None); n.add_argument("--name_b", default=None)
    n.add_argument("--out", default="log/revision-mcnemar.txt")

    args = ap.parse_args()
    {"eval": cmd_eval, "embed-ml": cmd_embed_ml, "mcnemar": cmd_mcnemar}[args.cmd](args)
