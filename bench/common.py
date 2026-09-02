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
# Erosion and Ulcer are both mucosal breaks differing in depth; the Lewis Score
# and CECDAI group them, and Ulcer alone has a single training patient. Merging
# takes the worst-supported class from 1 training patient to 6.
CLASSES_MERGED4 = ["AVM", "ErosionUlcer", "Normal", "Xanthoma"]
MERGED4_MAP = {0: 0, 1: 1, 2: 2, 3: 1, 4: 3}   # from the CLASSES_5 index
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
TASKS = ("5class", "binary", "merged4")


def classes_for(task):
    return {"5class": CLASSES_5, "binary": CLASSES_BIN,
            "merged4": CLASSES_MERGED4}[task]


def map_label(task, y):
    """CLASSES_5 index -> this task's label."""
    if task == "5class":
        return y
    if task == "binary":
        return 0 if y == NORMAL_INDEX_5 else 1
    return MERGED4_MAP[y]


def target_transform_for(task):
    if task == "5class":
        return None
    return lambda y: map_label(task, y)


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
    """aug: True/'light' = the brief's flips-only policy; 'strong' adds scale,
    rotation and photometric jitter; False/'none' = deterministic."""
    mode = "light" if aug is True else ("none" if aug in (False, None) else str(aug))
    if mode == "strong":
        ops = [
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0),
                                         ratio=(0.85, 1.18)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply([transforms.RandomRotation(20)], p=0.5),
            transforms.ColorJitter(brightness=0.25, contrast=0.25,
                                   saturation=0.25, hue=0.03),
        ]
    else:
        ops = [transforms.Resize((image_size, image_size))]
        if mode == "light":
            ops += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
    ops += [transforms.ToTensor(), transforms.Normalize(*norm)]
    if mode == "strong":
        ops += [transforms.RandomErasing(p=0.25, scale=(0.02, 0.12))]
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


def train_patients(task, data_root=DATA_ROOT):
    """Patients appearing anywhere in the TRAIN split, regardless of class."""
    d = os.path.join(data_root, "train")
    out = set()
    for cls in sorted(os.listdir(d)):
        for fn in os.listdir(os.path.join(d, cls)):
            p = patient_of(fn)
            if p.startswith("P_"):
                out.add(p)
    return out


def leaked_paths(split, data_root=DATA_ROOT):
    """Eval frames whose PATIENT also appears in train under some other class.

    The published split is patient-disjoint within each class but not globally:
    P_105 is an AVM and Erosion training patient and an Ulcer test patient, P_90
    an Erosion training patient and an AVM test patient, P_19 an AVM training
    patient and a Xanthoma validation patient. Any task that merges classes - and
    strictly, any task at all - should be able to report numbers with those
    frames excluded.
    """
    tp = train_patients(data_root)
    d = os.path.join(data_root, split)
    out = set()
    for cls in sorted(os.listdir(d)):
        for fn in sorted(os.listdir(os.path.join(d, cls))):
            if patient_of(fn) in tp:
                out.add(os.path.join(d, cls, fn))
    return out


def eval_loader(split, task, tf, batch_size=32, workers=4, data_root=DATA_ROOT,
                exclude_leaks=False):
    """Unshuffled. preds_*.csv row order == ds.samples order (addendum B1)."""
    ds = image_folder(split, task, tf, data_root)
    if exclude_leaks and split != "train":
        bad = leaked_paths(split, data_root)
        items = [(p, map_label(task, y)) for p, y in ds.samples if p not in bad]
        ds2 = ListDataset(items, tf)
        dl = DataLoader(ds2, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=torch.cuda.is_available())
        return ds2, dl, [p for p, _ in items]
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers,
                    pin_memory=torch.cuda.is_available())
    paths = [p for p, _ in ds.samples]
    return ds, dl, paths


def train_loader(task, tf, batch_size=16, workers=4, data_root=DATA_ROOT, sampler=None,
                 extra_dir=None):
    base = image_folder("train", task, tf, data_root)
    extra = extra_items(extra_dir, task)
    if extra:
        items = [(p, (y if task == "5class" else (0 if y == NORMAL_INDEX_5 else 1)))
                 for p, y in base.samples] + extra
        ds = ListDataset(items, tf)
    else:
        ds = base
    g = torch.Generator()
    g.manual_seed(SEED)
    if sampler is not None:
        dl = DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=workers,
                        pin_memory=torch.cuda.is_available(), drop_last=False)
    else:
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=workers,
                        pin_memory=torch.cuda.is_available(), generator=g, drop_last=False)
    return ds, dl


class ListDataset(torch.utils.data.Dataset):
    """(path, label) pairs, for training sets assembled from more than one tree."""

    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from PIL import Image
        p, y = self.items[i]
        with Image.open(p) as im:
            return self.transform(im.convert("RGB")), y


def extra_items(extra_dir, task):
    """Class-foldered images outside data/splitted, mapped to this task's labels."""
    items = []
    if not extra_dir or not os.path.isdir(extra_dir):
        return items
    for cls in sorted(os.listdir(extra_dir)):
        d = os.path.join(extra_dir, cls)
        if not os.path.isdir(d) or cls not in CLASSES_5:
            continue
        y = CLASSES_5.index(cls) if task == "5class" else (0 if cls == "Normal" else 1)
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith((".png", ".bmp", ".jpg", ".jpeg")):
                items.append((os.path.join(d, fn), y))
    return items


def labels_of(split, task, data_root=DATA_ROOT):
    ds = ImageFolder(os.path.join(data_root, split), loader=lambda p: None)
    return np.array([map_label(task, t) for _, t in ds.samples])


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
