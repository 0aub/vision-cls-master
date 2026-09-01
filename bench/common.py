"""Shared plumbing for the v2 benchmark: split guard, loaders, metric dumps.

Three repo pitfalls are fixed here rather than in src/, so the existing Trainer
keeps working unchanged (BENCHMARK_BRIEF_V2_ADDENDUM.md section B):

  B1  src/datasets.py builds every DataLoader with shuffle=True, val and test
      included, which scrambles preds_*.csv row order. eval_loader() below is
      always shuffle=False and the row order is the ImageFolder sample order.
  B2  prepare_set_ml() feeds the AUGMENTED train loader to sklearn. Classical
      models here are fitted on eval-transform (deterministic) features.
  B3  the splitfolders path cannot reproduce the published split. assert_split()
      refuses to run if data/splitted/V8-KAUHC is not exactly 1916/221/226, and
      nothing in bench/ ever calls the splitting code.
"""
import json
import os
import random
import re

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

DATA_ROOT = "data/splitted/V8-KAUHC"
SPLITS = ("train", "val", "test")
EXPECTED_SIZES = {"train": 1916, "val": 221, "test": 226}
CLASSES_5 = ["AVM", "Erosion", "Normal", "Ulcer", "Xanthoma"]
CLASSES_BIN = ["Normal", "Lesion"]
NORMAL_INDEX_5 = CLASSES_5.index("Normal")
SEED = 1998
IMAGENET_NORM = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
LOG_ROOT = "log"
PROGRESS = os.path.join(LOG_ROOT, "bench-progress.txt")


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #
def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# split guard (addendum B3)
# --------------------------------------------------------------------------- #
def assert_split(data_root=DATA_ROOT):
    sizes = {}
    for sp in SPLITS:
        d = os.path.join(data_root, sp)
        if not os.path.isdir(d):
            raise SystemExit(f"[SPLIT GATE] missing {d}")
        sizes[sp] = sum(len(os.listdir(os.path.join(d, c))) for c in sorted(os.listdir(d)))
    if sizes != EXPECTED_SIZES:
        raise SystemExit(
            f"[SPLIT GATE] {data_root} is {sizes}, expected {EXPECTED_SIZES}. "
            "Refusing to run: the splitfolders path cannot reproduce the published "
            "split (it yields 1923/239/244). Restore the existing folders."
        )
    return sizes


# --------------------------------------------------------------------------- #
# tasks
# --------------------------------------------------------------------------- #
def classes_for(task):
    return CLASSES_5 if task == "5class" else CLASSES_BIN


def target_transform_for(task):
    if task == "5class":
        return None
    # binary: Normal -> 0, any lesion class -> 1
    return lambda y: 0 if y == NORMAL_INDEX_5 else 1


# --------------------------------------------------------------------------- #
# transforms
# --------------------------------------------------------------------------- #
def eval_transform(image_size=224, norm=IMAGENET_NORM):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(*norm),
    ])


def train_transform(image_size=224, norm=IMAGENET_NORM, aug=True):
    ops = [transforms.Resize((image_size, image_size))]
    if aug:                                   # "light flips (train only)"
        ops += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
    ops += [transforms.ToTensor(), transforms.Normalize(*norm)]
    return transforms.Compose(ops)


def dinov2_eval_transform(image_size=224):
    """The official DINOv2 classification eval transform.

    dinov2/data/transforms.py: resize the short side to 256 with BICUBIC, centre
    crop to 224, then normalise with the ImageNet constants (DINOv2 publishes
    IMAGENET_DEFAULT_MEAN/STD; what differs from the torchvision default used by
    the rest of the grid is the bicubic resize + centre crop, not the constants).
    """
    resize = int(image_size * 256 / 224)
    return transforms.Compose([
        transforms.Resize(resize, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(*IMAGENET_NORM),
    ])


def dinov2_train_transform(image_size=224, aug=True):
    resize = int(image_size * 256 / 224)
    ops = [transforms.Resize(resize, interpolation=transforms.InterpolationMode.BICUBIC),
           transforms.CenterCrop(image_size)]
    if aug:
        ops += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
    ops += [transforms.ToTensor(), transforms.Normalize(*IMAGENET_NORM)]
    return transforms.Compose(ops)


# --------------------------------------------------------------------------- #
# datasets / loaders
# --------------------------------------------------------------------------- #
def image_folder(split, task, tf, data_root=DATA_ROOT):
    return ImageFolder(os.path.join(data_root, split), transform=tf,
                       target_transform=target_transform_for(task))


def eval_loader(split, task, tf, batch_size=32, workers=4, data_root=DATA_ROOT):
    """Unshuffled. preds_*.csv row order == ds.samples order (addendum B1)."""
    ds = image_folder(split, task, tf, data_root)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers,
                    pin_memory=torch.cuda.is_available())
    paths = [p for p, _ in ds.samples]
    return ds, dl, paths


def train_loader(task, tf, batch_size=16, workers=4, data_root=DATA_ROOT, sampler=None):
    ds = image_folder("train", task, tf, data_root)
    g = torch.Generator()
    g.manual_seed(SEED)
    if sampler is not None:
        dl = DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=workers,
                        pin_memory=torch.cuda.is_available(), drop_last=False)
    else:
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=workers,
                        pin_memory=torch.cuda.is_available(), generator=g, drop_last=False)
    return ds, dl


def labels_of(split, task, data_root=DATA_ROOT):
    ds = ImageFolder(os.path.join(data_root, split),
                     target_transform=target_transform_for(task),
                     loader=lambda p: None)
    return np.array([t for _, t in ds.samples]) if task == "5class" else \
        np.array([0 if t == NORMAL_INDEX_5 else 1 for _, t in ds.samples])


# --------------------------------------------------------------------------- #
# patient ids (grouped CV) - filenames are Class__P_<id>__<study>__<frame>.bmp,
# Normal frames are Normal__Normal_<timestamp>.bmp and have no patient folder.
# --------------------------------------------------------------------------- #
_PAT = re.compile(r"__(P_\d+)__")


def patient_of(path):
    m = _PAT.search(os.path.basename(path))
    if m:
        return m.group(1)
    return "NORMAL__" + os.path.splitext(os.path.basename(path))[0]


# --------------------------------------------------------------------------- #
# metric dumps - one standard format for every model in the grid
# --------------------------------------------------------------------------- #
def dump_split(out_dir, split, y_true, y_pred, paths, classes, probs=None,
               mean_ce=None, save_probs=True):
    import pandas as pd
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, precision_recall_fscore_support)
    os.makedirs(out_dir, exist_ok=True)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    rep = classification_report(y_true, y_pred, labels=list(range(len(classes))),
                                target_names=classes, digits=4, output_dict=True,
                                zero_division=0)
    pd.DataFrame(rep).T.round(4).to_csv(os.path.join(out_dir, f"report_{split}.csv"))

    row = {"accuracy": accuracy_score(y_true, y_pred)}
    for avg in ("macro", "weighted"):
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=avg, zero_division=0,
            labels=list(range(len(classes))))
        row[f"{avg}_precision"] = p
        row[f"{avg}_recall"] = r
        row[f"{avg}_f1"] = f1
    if mean_ce is not None:
        row["mean_ce"] = float(mean_ce)
    row["n"] = int(len(y_true))
    pd.DataFrame([row]).round(6).to_csv(os.path.join(out_dir, f"summary_{split}.csv"),
                                        index=False)

    pd.DataFrame({"path": paths, "y_true": y_true, "y_pred": y_pred}).to_csv(
        os.path.join(out_dir, f"preds_{split}.csv"), index=False)

    np.savetxt(os.path.join(out_dir, f"cm_{split}.csv"),
               confusion_matrix(y_true, y_pred, labels=list(range(len(classes)))),
               fmt="%d", delimiter=",")

    if probs is not None and save_probs:
        np.save(os.path.join(out_dir, f"PROBS_{split}.npy"),
                np.asarray(probs, dtype=np.float32))
    return row


def mean_cross_entropy(probs, y_true, eps=1e-12):
    p = np.clip(np.asarray(probs, dtype=np.float64), eps, 1.0)
    return float(-np.log(p[np.arange(len(y_true)), np.asarray(y_true)]).mean())


# --------------------------------------------------------------------------- #
# resume + progress log (addendum D1, D4)
# --------------------------------------------------------------------------- #
def run_dir(task, name):
    return os.path.join(LOG_ROOT, f"bench-{task}-{name}")


def already_done(task, name):
    f = os.path.join(run_dir(task, name), "summary_test.csv")
    return os.path.exists(f) and os.path.getsize(f) > 0


def log_progress(line):
    os.makedirs(LOG_ROOT, exist_ok=True)
    with open(PROGRESS, "a") as f:
        f.write(line.rstrip() + "\n")
    print("[progress] " + line.rstrip(), flush=True)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
