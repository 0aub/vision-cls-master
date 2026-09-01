"""Cache penultimate / frozen features for the embedding-based runs.

    # DINOv2 frozen features (CLS, and the CLS+mean-patch variant)
    python3 bench/embed.py --name dinov2_vitb14 --source hub-dinov2 --feature_mode cls
    python3 bench/embed.py --name dinov2_vitb14 --source hub-dinov2 --feature_mode cls+mean

    # a fine-tuned Phase A backbone, head stripped
    python3 bench/embed.py --name densenet201 --source torchvision \
        --ckpt log/bench-5class-densenet201/best.pth --num_classes 5 --out densenet201_ft5

    # BiomedCLIP frozen image encoder
    python3 bench/embed.py --name biomedclip --source open_clip

Writes log/bench-embeddings/<out>/X_{split}.npy + paths_{split}.txt + meta.json.
Features are always taken with the deterministic eval transform and an
unshuffled loader, so row i of X_<split> is row i of preds_<split>.csv.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench import common as C                                      # noqa: E402
from src.modules import (build_model, load_biomedclip, load_dinov2,   # noqa: E402
                         strip_head)

ROOT = os.path.join(C.LOG_ROOT, "bench-embeddings")


@torch.no_grad()
def extract(model_fn, tf, device, workers=4, batch_size=32):
    out = {}
    for split in C.SPLITS:
        _, dl, paths = C.eval_loader(split, "5class", tf, batch_size=batch_size,
                                     workers=workers)
        feats = []
        for xb, _ in dl:
            xb = xb.to(device, dtype=torch.float, non_blocking=True)
            feats.append(model_fn(xb).float().cpu().numpy())
        out[split] = (np.vstack(feats), paths)
        print(f"  {split}: {out[split][0].shape}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--source", default="torchvision",
                    choices=["torchvision", "hub-dinov2", "open_clip"])
    ap.add_argument("--feature_mode", default="cls", choices=["cls", "cls+mean"])
    ap.add_argument("--ckpt", default=None, help="fine-tuned state_dict to load first")
    ap.add_argument("--num_classes", type=int, default=5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_name = args.out or (args.name if args.feature_mode == "cls"
                            else f"{args.name}_clsmean")
    out_dir = os.path.join(ROOT, out_name)
    if os.path.exists(os.path.join(out_dir, "X_test.npy")) and not args.force:
        print(f"[skip] {out_dir} already cached")
        return
    C.assert_split()
    C.seed_everything()
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    if args.source == "torchvision":
        model = build_model(args.name, args.num_classes, train_mode="full")
        if args.ckpt:
            model.load_state_dict(torch.load(args.ckpt, map_location="cpu",
                                             weights_only=True))
        dim = strip_head(model)
        model.to(device).eval()
        tf = C.eval_transform(224)
        fn = lambda x: model(x)                                    # noqa: E731
    elif args.source == "hub-dinov2":
        backbone, d = load_dinov2(args.name)
        backbone.to(device).eval()
        dim = d * (2 if args.feature_mode == "cls+mean" else 1)
        tf = C.dinov2_eval_transform(224)

        def fn(x):
            o = backbone.forward_features(x)
            if args.feature_mode == "cls+mean":
                return torch.cat([o["x_norm_clstoken"],
                                  o["x_norm_patchtokens"].mean(dim=1)], dim=-1)
            return o["x_norm_clstoken"]
    else:
        clip, preprocess, _, dim = load_biomedclip()
        clip.to(device).eval()
        from torchvision import transforms
        tf = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711)),
        ])
        fn = lambda x: clip.encode_image(x)                        # noqa: E731

    print(f"=== embeddings {out_name} (dim {dim}) ===", flush=True)
    feats = extract(fn, tf, device, args.workers)
    for split, (X, paths) in feats.items():
        np.save(os.path.join(out_dir, f"X_{split}.npy"), X.astype(np.float32))
        with open(os.path.join(out_dir, f"paths_{split}.txt"), "w") as f:
            f.write("\n".join(paths) + "\n")
    C.write_json(os.path.join(out_dir, "meta.json"), {
        "name": args.name, "source": args.source, "feature_mode": args.feature_mode,
        "ckpt": args.ckpt, "dim": int(dim), "out": out_name,
        "transform": str(tf), "seconds": round(time.time() - t0, 1),
        "shapes": {s: list(feats[s][0].shape) for s in feats},
    })
    print(f"[done] {out_dir}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
