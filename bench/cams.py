"""C2: attention faithfulness against the overlay-derived lesion masks (M3b).

    python3 bench/cams.py --models densenet201 resnet50 --dinov2 dinov2_vitb14_lora \
                          --panels 8

Grad-CAM on the last convolutional block for CNNs; Grad-CAM on the final
transformer block's first norm (with a patch-token reshape) for ViT/Swin/MaxViT
and DINOv2. For every frame that has a mask (bench/masks.py) we score:

  pointing game : does the CAM argmax fall inside the ellipse mask?
  IoU           : intersection-over-union of the top-20% CAM region with the mask

and we do it per class and per V8 split. The archived ring-trained
efficientnet_b0 (v7, 2024) is scored on the RECONSTRUCTED RINGED frames - the
pixels it was actually trained on - while the clean-trained models are scored on
the raw frames, which is exactly the contrast the signature figure shows.

Writes log/bench-cam/: bench-pointing-game.csv (per frame), bench-pointing-game-
summary.csv (per model x class x split), and panels/*.png.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench import common as C                                      # noqa: E402
from bench import masks as M                                       # noqa: E402
from src.modules import build_model, pretrained_network            # noqa: E402

OUT = os.path.join(C.LOG_ROOT, "bench-cam")
MASKDIR = os.path.join(C.LOG_ROOT, "bench-lesion-masks")
V7_CKPT_GLOB = "archive/log/v7-efficientnet_b0 */best.pth"
TOP_FRAC = 0.20


# --------------------------------------------------------------------------- #
# where to hook
# --------------------------------------------------------------------------- #
def seq_reshape(h, w):
    """(B, 1+N, C) token sequence -> (B, C, h, w), dropping the CLS token."""
    def f(t):
        t = t[:, -h * w:, :]
        return t.reshape(t.size(0), h, w, t.size(2)).permute(0, 3, 1, 2)
    return f


def hwc_reshape(t):
    """(B, H, W, C) -> (B, C, H, W); torchvision Swin keeps channels last."""
    return t.permute(0, 3, 1, 2)


def target_layer(model, name, source="torchvision"):
    if source == "hub-dinov2":
        return [model.backbone.blocks[-1].norm1], seq_reshape(16, 16)
    if name.startswith(("efficientnet", "convnext", "mobilenet_v3", "vgg", "alexnet")):
        return [model.features[-1]], None
    if name.startswith("resnet"):
        return [model.layer4[-1]], None
    if name.startswith("densenet"):
        return [model.features[-1]], None
    if name.startswith("shufflenet"):
        return [model.conv5], None
    if name.startswith("googlenet"):
        return [model.inception5b], None
    if name.startswith("swin"):
        return [model.features[-1][-1].norm1], hwc_reshape
    if name.startswith("vit_"):
        return [model.encoder.layers[-1].ln_1], seq_reshape(14, 14)
    if name.startswith("maxvit"):
        return [model.blocks[-1]], None
    raise ValueError(f"no Grad-CAM target layer registered for {name}")


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def score_cam(cam, mask):
    """cam and mask are float/bool arrays of the same HxW."""
    if cam.shape != mask.shape:
        cam = np.asarray(Image.fromarray(cam.astype(np.float32)).resize(
            (mask.shape[1], mask.shape[0]), Image.BILINEAR))
    peak = np.unravel_index(int(np.argmax(cam)), cam.shape)
    hit = bool(mask[peak])
    thr = np.quantile(cam, 1.0 - TOP_FRAC)
    region = cam >= thr
    inter = np.logical_and(region, mask).sum()
    union = np.logical_or(region, mask).sum()
    return hit, (float(inter / union) if union else 0.0), peak


def overlay(img01, cam, alpha=0.45):
    import matplotlib
    if cam.shape != img01.shape[:2]:
        cam = np.asarray(Image.fromarray(cam.astype(np.float32)).resize(
            (img01.shape[1], img01.shape[0]), Image.BILINEAR))
    cam = (cam - cam.min()) / (float(np.ptp(cam)) + 1e-9)
    heat = matplotlib.colormaps["jet"](cam)[..., :3]
    return np.clip((1 - alpha) * img01 + alpha * heat, 0, 1)


# --------------------------------------------------------------------------- #
# frame sources
# --------------------------------------------------------------------------- #
def masked_frames():
    df = pd.read_csv(os.path.join(MASKDIR, "bench-lesion-bboxes.csv"))
    rows = []
    for _, r in df.iterrows():
        mp = os.path.join(MASKDIR, "masks", r["filename"].replace(".bmp", ".png"))
        if not os.path.exists(mp) or r["v8_split"] == "not-in-V8":
            continue
        rows.append(dict(filename=r["filename"], split=r["v8_split"], cls=r["class"],
                         source_path=r["source_path"], flip=r["flip_variant"],
                         knn_row=int(r["knn_row"]), mask_path=mp))
    return rows


def ringed_image(X, knn_row, flip):
    """Reconstruct the published (ringed) 256x256 frame, raw orientation."""
    k = M.FLIPS.index(flip)
    hwc = (X[knn_row].reshape(3, M.SIZE, M.SIZE) * M.STD[:, None, None]
           + M.MEAN[:, None, None]).transpose(1, 2, 0)
    return np.clip(M.flipped(hwc, k), 0, 1)


# --------------------------------------------------------------------------- #
def build_cam(model, layers, rt):
    from pytorch_grad_cam import GradCAM
    try:
        return GradCAM(model=model, target_layers=layers, reshape_transform=rt)
    except TypeError:                                   # older signature
        return GradCAM(model=model, target_layers=layers, reshape_transform=rt,
                       use_cuda=torch.cuda.is_available())


def run_model(spec, frames, X_ring, device, panels_wanted, task="5class"):
    """spec: dict(name=, source=, ckpt=, image_size=, ringed=, label=)"""
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    classes = C.classes_for(task)
    if spec["source"] == "v7-legacy":
        model = pretrained_network(spec["name"], None, 4, 5)
    else:
        model = build_model(spec["name"], len(classes), source=spec["source"],
                            train_mode=spec.get("train_mode", "full"))
    sd = torch.load(spec["ckpt"], map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    model.to(device).eval()
    layers, rt = target_layer(model, spec["name"], spec["source"])
    cam = build_cam(model, layers, rt)

    size = spec["image_size"]
    if spec["source"] == "hub-dinov2":
        tf = C.dinov2_eval_transform(size)
    else:
        tf = C.eval_transform(size)

    rows, panels = [], []
    for i, fr in enumerate(frames):
        mask = np.asarray(Image.open(fr["mask_path"])) > 127
        if spec["ringed"]:
            img01 = ringed_image(X_ring, fr["knn_row"], fr["flip"])
            pil = Image.fromarray((img01 * 255).astype(np.uint8))
        else:
            pil = Image.open(os.path.join(M.FULL, fr["source_path"])).convert("RGB")
            img01 = np.asarray(pil.resize((M.SIZE, M.SIZE), Image.BILINEAR),
                               np.float32) / 255.0
        x = tf(pil).unsqueeze(0).to(device).requires_grad_(True)
        y = classes.index(fr["cls"]) if fr["cls"] in classes else 1
        g = cam(input_tensor=x, targets=[ClassifierOutputTarget(y)])[0]
        with torch.no_grad():
            pred = int(model(x.detach()).argmax(1).item())
        hit, iou, peak = score_cam(g, mask)
        rows.append(dict(model=spec["label"], ringed=spec["ringed"],
                         filename=fr["filename"], v8_split=fr["split"],
                         **{"class": fr["cls"]}, y_true=y, y_pred=pred,
                         pointing_hit=int(hit), iou_top20=round(iou, 6),
                         peak_y=int(peak[0]), peak_x=int(peak[1]),
                         mask_px=int(mask.sum())))
        if len(panels) < panels_wanted:
            panels.append((fr, img01, g, mask))
    del cam, model
    torch.cuda.empty_cache()
    return pd.DataFrame(rows), panels


def save_panels(bundles, path, title):
    """bundles: list of (label, frame, img01, cam, mask) columns per row."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(bundles)
    if n == 0:
        return
    ncol = 2 + len(bundles[0]["cams"])
    fig, axes = plt.subplots(n, ncol, figsize=(2.6 * ncol, 2.6 * n))
    axes = np.atleast_2d(axes)
    for r, b in enumerate(bundles):
        axes[r, 0].imshow(b["img"])
        axes[r, 0].set_ylabel(f"{b['cls']}\n{b['split']}", fontsize=8)
        if r == 0:
            axes[r, 0].set_title("raw frame", fontsize=9)
        axes[r, 1].imshow(b["mask"], cmap="gray")
        if r == 0:
            axes[r, 1].set_title("clinician ellipse\n(mask)", fontsize=9)
        for c, (label, img, cam) in enumerate(b["cams"]):
            axes[r, 2 + c].imshow(overlay(img, cam))
            if r == 0:
                axes[r, 2 + c].set_title(label, fontsize=9)
        for a in axes[r]:
            a.set_xticks([])
            a.set_yticks([])
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[],
                    help="clean-trained torchvision run names (log/bench-5class-<name>)")
    ap.add_argument("--dinov2", default=None, help="a DINOv2 run name to include")
    ap.add_argument("--task", default="5class")
    ap.add_argument("--panels", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="debug: only N frames")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    C.assert_split()
    os.makedirs(os.path.join(OUT, "panels"), exist_ok=True)
    per_frame_csv = os.path.join(OUT, "bench-pointing-game.csv")
    if os.path.exists(per_frame_csv) and not args.force:
        print(f"[skip] {per_frame_csv} exists")
        return
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    frames = masked_frames()
    if args.limit:
        frames = frames[:args.limit]
    print(f"{len(frames)} lesion frames carry a mask", flush=True)

    specs = []
    for m in args.models:
        ck = os.path.join(C.run_dir(args.task, m), "best.pth")
        if os.path.exists(ck):
            specs.append(dict(name=m, source="torchvision", ckpt=ck, image_size=224,
                              ringed=False, label=f"{m} (clean)", train_mode="full"))
    if args.dinov2:
        ck = os.path.join(C.run_dir(args.task, args.dinov2), "best.pth")
        base = args.dinov2.split("_lora")[0].split("_probe")[0]
        if os.path.exists(ck):
            specs.append(dict(name=base, source="hub-dinov2", ckpt=ck, image_size=224,
                              ringed=False, label=f"{args.dinov2} (clean)",
                              train_mode="lora" if "lora" in args.dinov2 else "probe"))
    v7 = sorted(glob.glob(V7_CKPT_GLOB))
    if v7:
        specs.append(dict(name="efficientnet_b0", source="v7-legacy", ckpt=v7[0],
                          image_size=256, ringed=True,
                          label="efficientnet_b0 v7 (ring-trained, ringed input)"))

    X_ring = None
    if any(s["ringed"] for s in specs):
        pkl = sorted(glob.glob(M.KNN_GLOB))[0]
        print(f"loading ringed pixels from {pkl} ...", flush=True)
        X_ring = M.load_fit_matrix(pkl)

    all_rows, panel_store = [], {}
    for spec in specs:
        print(f"--- Grad-CAM: {spec['label']}", flush=True)
        df, panels = run_model(spec, frames, X_ring, device, args.panels, args.task)
        all_rows.append(df)
        panel_store[spec["label"]] = panels
        print(f"    pointing hit-rate {df.pointing_hit.mean():.3f}  "
              f"mean IoU {df.iou_top20.mean():.3f}", flush=True)

    df = pd.concat(all_rows, ignore_index=True)
    df.to_csv(per_frame_csv, index=False)

    g = (df.groupby(["model", "v8_split", "class"])
           .agg(n=("pointing_hit", "size"),
                pointing_hit_rate=("pointing_hit", "mean"),
                mean_iou_top20=("iou_top20", "mean"))
           .reset_index().round(4))
    overall = (df.groupby(["model", "v8_split"])
                 .agg(n=("pointing_hit", "size"),
                      pointing_hit_rate=("pointing_hit", "mean"),
                      mean_iou_top20=("iou_top20", "mean"))
                 .reset_index().round(4))
    overall["class"] = "ALL"
    allsplit = (df.groupby(["model"])
                  .agg(n=("pointing_hit", "size"),
                       pointing_hit_rate=("pointing_hit", "mean"),
                       mean_iou_top20=("iou_top20", "mean"))
                  .reset_index().round(4))
    allsplit["class"] = "ALL"
    allsplit["v8_split"] = "ALL"
    out = pd.concat([g, overall, allsplit], ignore_index=True)
    out.to_csv(os.path.join(OUT, "bench-pointing-game-summary.csv"), index=False)
    print(out.to_string(index=False))

    # qualitative panels: same frames, every model side by side
    labels = list(panel_store.keys())
    bundles = []
    for i in range(min(args.panels, len(frames))):
        fr, img01, _, mask = panel_store[labels[0]][i]
        cams = []
        for lab in labels:
            f2, img2, cam2, _ = panel_store[lab][i]
            cams.append((lab.replace(" (", "\n("), img2, cam2))
        bundles.append(dict(cls=fr["cls"], split=fr["split"], img=img01,
                            mask=mask, cams=cams))
    for start in range(0, len(bundles), 4):
        chunk = bundles[start:start + 4]
        save_panels(chunk, os.path.join(OUT, "panels",
                                        f"cam_contrast_{start//4 + 1}.png"),
                    "Grad-CAM vs clinician ellipse: clean-trained vs ring-trained")
    print(f"[done] {OUT}")


if __name__ == "__main__":
    main()
