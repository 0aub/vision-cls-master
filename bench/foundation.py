"""Phase B tier-5 runs that need no gradient descent.

    python3 bench/foundation.py --mode dinov2-probe --backbone dinov2_vitb14 --task 5class
    python3 bench/foundation.py --mode dinov2-knn   --backbone dinov2_vitb14 --task 5class
    python3 bench/foundation.py --mode biomedclip-zeroshot --task 5class
    python3 bench/foundation.py --mode biomedclip-probe    --task 5class

Features come from bench/embed.py's cache. efficiency.json carries the REAL
inference cost (the frozen backbone plus the tiny head), not just the head, so
these points sit honestly on the Pareto plots.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench import common as C                                      # noqa: E402
from bench import efficiency as EFF                                # noqa: E402
from bench.prompts import prompts_for                              # noqa: E402
from src.modules import build_model                                # noqa: E402

EMB = os.path.join(C.LOG_ROOT, "bench-embeddings")


def load_emb(embed_name, task):
    d = os.path.join(EMB, embed_name)
    if not os.path.exists(os.path.join(d, "X_test.npy")):
        raise SystemExit(f"missing embeddings {d}; run bench/embed.py first")
    out = {}
    for sp in C.SPLITS:
        X = np.load(os.path.join(d, f"X_{sp}.npy")).astype(np.float32)
        paths = [l.rstrip("\n") for l in open(os.path.join(d, f"paths_{sp}.txt"))]
        out[sp] = (X, C.labels_of(sp, task), paths)
    meta = json.load(open(os.path.join(d, "meta.json")))
    return out, meta


def backbone_cost(mode, backbone, num_classes, feature_mode):
    """Params / GFLOPs / latency of the deployed graph (frozen encoder + head)."""
    try:
        if mode.startswith("dinov2"):
            m = build_model(backbone, num_classes, source="hub-dinov2",
                            train_mode="probe", feature_mode=feature_mode)
        else:
            m = build_model("biomedclip", num_classes, source="open_clip")
        if torch.cuda.is_available():
            m = m.cuda()
        eff = EFF.measure_all(m, 224)
        del m
        torch.cuda.empty_cache()
        return eff
    except Exception as e:
        return {"cost_measurement_error": f"{type(e).__name__}: {e}"}


def run_probe(data, task, C_grid=(0.001, 0.01, 0.1, 1.0, 10.0, 100.0), normalize=True):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler
    Xtr, ytr, _ = data["train"]
    Xva, yva, _ = data["val"]
    scaler = StandardScaler().fit(Xtr) if normalize else None
    tr = scaler.transform(Xtr) if scaler else Xtr
    va = scaler.transform(Xva) if scaler else Xva
    best, best_f1, trace = None, None, []
    for c in C_grid:
        clf = LogisticRegression(C=c, max_iter=2000, n_jobs=-1)
        clf.fit(tr, ytr)
        f1 = f1_score(yva, clf.predict(va), average="macro", zero_division=0)
        trace.append({"C": c, "val_macro_f1": round(float(f1), 6)})
        print(f"    C={c:<8g} val macroF1 {f1:.4f}", flush=True)
        if best_f1 is None or f1 > best_f1:
            best, best_f1 = c, f1
    clf = LogisticRegression(C=best, max_iter=2000, n_jobs=-1).fit(tr, ytr)
    return clf, scaler, {"selected_C": best, "val_macro_f1": best_f1, "grid": trace}


def run_knn(data, task, k_grid=(1, 3, 5, 9, 15, 25)):
    from sklearn.metrics import f1_score
    from sklearn.neighbors import KNeighborsClassifier
    Xtr, ytr, _ = data["train"]
    Xva, yva, _ = data["val"]
    # cosine distance on L2-normalised features, the standard DINOv2 k-NN recipe
    nrm = lambda A: A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)  # noqa: E731
    tr, va = nrm(Xtr), nrm(Xva)
    best, best_f1, trace = None, None, []
    for k in k_grid:
        clf = KNeighborsClassifier(n_neighbors=k, metric="cosine")
        clf.fit(tr, ytr)
        f1 = f1_score(yva, clf.predict(va), average="macro", zero_division=0)
        trace.append({"k": k, "val_macro_f1": round(float(f1), 6)})
        print(f"    k={k:<3d} val macroF1 {f1:.4f}", flush=True)
        if best_f1 is None or f1 > best_f1:
            best, best_f1 = k, f1
    clf = KNeighborsClassifier(n_neighbors=best, metric="cosine").fit(tr, ytr)
    return clf, nrm, {"selected_k": best, "val_macro_f1": best_f1, "grid": trace}


def run_zeroshot(task, out_dir, workers=4):
    """BiomedCLIP zero-shot with averaged class-prompt text embeddings."""
    import open_clip
    from src.modules import BIOMEDCLIP_HF
    from torchvision import transforms
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, _ = open_clip.create_model_from_pretrained(BIOMEDCLIP_HF)
    tokenizer = open_clip.get_tokenizer(BIOMEDCLIP_HF)
    model.to(device).eval()
    classes = C.classes_for(task)
    prompts = prompts_for(task)
    C.write_json(os.path.join(out_dir, "prompts.json"), prompts)
    C.write_json(os.path.join(C.LOG_ROOT, f"bench-biomedclip-prompts-{task}.json"), prompts)

    with torch.no_grad():
        W = []
        for cls in classes:
            t = tokenizer(prompts[cls]).to(device)
            e = model.encode_text(t).float()
            e = e / e.norm(dim=-1, keepdim=True)
            e = e.mean(dim=0)
            W.append(e / e.norm())
        W = torch.stack(W, dim=1)                       # (dim, K)
        logit_scale = model.logit_scale.exp().item()

    tf = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711)),
    ])
    out = {}
    with torch.no_grad():
        for sp in C.SPLITS:
            _, dl, paths = C.eval_loader(sp, task, tf, batch_size=32, workers=workers)
            P, Y = [], []
            for xb, yb in dl:
                f = model.encode_image(xb.to(device, dtype=torch.float)).float()
                f = f / f.norm(dim=-1, keepdim=True)
                P.append(torch.softmax(logit_scale * f @ W, dim=1).cpu().numpy())
                Y.append(yb.numpy())
            out[sp] = (np.vstack(P), np.concatenate(Y), paths)
    del model
    torch.cuda.empty_cache()
    return out, {"logit_scale": logit_scale,
                 "n_prompts_per_class": {c: len(prompts[c]) for c in classes}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["dinov2-probe", "dinov2-knn", "biomedclip-probe",
                             "biomedclip-zeroshot"])
    ap.add_argument("--backbone", default="dinov2_vitb14")
    ap.add_argument("--task", default="5class", choices=["5class", "binary", "merged4"])
    ap.add_argument("--feature_mode", default="cls", choices=["cls", "cls+mean"])
    ap.add_argument("--embed_name", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--tier", default="tier5-foundation")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    suffix = "" if args.feature_mode == "cls" else "_clsmean"
    if args.mode.startswith("dinov2"):
        embed_name = args.embed_name or (args.backbone + suffix)
        default_name = f"{args.backbone}{suffix}_" + args.mode.split("-")[1]
    else:
        embed_name = args.embed_name or "biomedclip"
        default_name = "biomedclip_" + args.mode.split("-")[1]
    name = args.name or default_name

    C.assert_split()
    out_dir = C.run_dir(args.task, name)
    if C.already_done(args.task, name) and not args.force:
        print(f"[skip] {out_dir} already has summary_test.csv")
        return
    os.makedirs(out_dir, exist_ok=True)
    C.seed_everything()
    classes = C.classes_for(args.task)
    K = len(classes)
    print(f"=== {args.task} / {name} [{args.mode}] ===", flush=True)
    t0 = time.time()
    sel = {}

    if args.mode == "biomedclip-zeroshot":
        results, sel = run_zeroshot(args.task, out_dir, args.workers)
        emb_meta = {"source": "open_clip", "note": "no fitting; text-prompt classifier"}
        dim = None
    else:
        data, emb_meta = load_emb(embed_name, args.task)
        dim = emb_meta["dim"]
        if args.mode == "dinov2-knn":
            clf, prep, sel = run_knn(data, args.task)
            results = {sp: (clf.predict_proba(prep(data[sp][0])), data[sp][1], data[sp][2])
                       for sp in C.SPLITS}
        else:
            clf, scaler, sel = run_probe(data, args.task)
            tx = (lambda A: scaler.transform(A)) if scaler is not None else (lambda A: A)
            results = {sp: (clf.predict_proba(tx(data[sp][0])), data[sp][1], data[sp][2])
                       for sp in C.SPLITS}

    summary = {}
    for sp in C.SPLITS:
        p, y, paths = results[sp]
        if p.shape[1] != K:                       # sklearn drops absent classes
            full = np.zeros((len(p), K))
            for j, c in enumerate(clf.classes_):
                full[:, int(c)] = p[:, j]
            p = full
        row = C.dump_split(out_dir, sp, y, p.argmax(1), paths, classes, probs=p,
                           mean_ce=C.mean_cross_entropy(p, y))
        summary[sp] = row
        print(f"  [{sp}] acc={row['accuracy']:.4f} macroF1={row['macro_f1']:.4f}", flush=True)

    eff = backbone_cost(args.mode, args.backbone, K, args.feature_mode)
    head_params = (dim * K + K) if dim else 0
    if args.mode == "dinov2-knn":
        head_params = 0
    eff.update({
        "model": args.backbone if args.mode.startswith("dinov2") else "BiomedCLIP",
        "name": name, "task": args.task, "tier": args.tier, "mode": args.mode,
        "source": "hub-dinov2" if args.mode.startswith("dinov2") else "open_clip",
        "train_mode": "frozen+" + args.mode.split("-")[1],
        "feature_mode": args.feature_mode, "feature_dim": dim,
        "params_trainable": int(head_params),
        "head_params": int(head_params),
        "selection": sel, "embedding_meta": emb_meta,
        "train_wallclock_min": round((time.time() - t0) / 60.0, 3),
        "peak_train_vram_mb": None, "seed": C.SEED,
    })
    if eff.get("params_total"):
        eff["params_trainable_pct"] = round(100.0 * head_params / eff["params_total"], 6)
    C.write_json(os.path.join(out_dir, "efficiency.json"), eff)
    C.write_json(os.path.join(out_dir, "run_config.json"), vars(args))
    C.log_progress(f"{args.task:7s} {name:34s} test_acc={summary['test']['accuracy']:.4f} "
                   f"test_macroF1={summary['test']['macro_f1']:.4f} "
                   f"{(time.time()-t0)/60:.1f}min")
    print(f"[done] {out_dir}")


if __name__ == "__main__":
    main()
