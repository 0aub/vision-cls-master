"""Phase 0 environment gate: which SOTA checkpoints actually load on this machine.

Nothing here is silently skipped. Every entry lands in log/bench-env-gate.json /
.txt as loaded / unavailable with the exact failure, and the report quotes it.
"""
import json
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS = []


def check(name, fn, optional=False):
    entry = {"name": name, "optional": optional}
    try:
        info = fn()
        entry.update(status="loaded", detail=info)
        print(f"[OK]   {name}: {info}", flush=True)
    except Exception as e:
        entry.update(status="unavailable",
                     detail=f"{type(e).__name__}: {e}".replace("\n", " ")[:600])
        print(f"[FAIL] {name}: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc(limit=2)
    RESULTS.append(entry)


def dinov2(entry):
    def f():
        m = torch.hub.load("facebookresearch/dinov2", entry, trust_repo=True)
        m.eval()
        with torch.no_grad():
            out = m.forward_features(torch.zeros(1, 3, 224, 224))
        return (f"cls dim {out['x_norm_clstoken'].shape[-1]}, "
                f"patch tokens {tuple(out['x_norm_patchtokens'].shape[1:])}, "
                f"params {sum(p.numel() for p in m.parameters())/1e6:.1f}M")
    return f


def dinov3(entry):
    def f():
        m = torch.hub.load("facebookresearch/dinov3", entry, trust_repo=True)
        m.eval()
        return f"params {sum(p.numel() for p in m.parameters())/1e6:.1f}M"
    return f


def dinov3_hf(repo):
    def f():
        from transformers import AutoModel
        m = AutoModel.from_pretrained(repo)
        return f"HF {repo}, params {sum(p.numel() for p in m.parameters())/1e6:.1f}M"
    return f


def biomedclip():
    import open_clip
    from src.modules import BIOMEDCLIP_HF
    model, preprocess = open_clip.create_model_from_pretrained(BIOMEDCLIP_HF)
    tok = open_clip.get_tokenizer(BIOMEDCLIP_HF)
    model.eval()
    with torch.no_grad():
        im = model.encode_image(torch.zeros(1, 3, 224, 224))
        tx = model.encode_text(tok(["an endoscopy image"]))
    return (f"image embed {im.shape[-1]}, text embed {tx.shape[-1]}, "
            f"params {sum(p.numel() for p in model.parameters())/1e6:.1f}M, "
            f"preprocess {preprocess}")


def flops_tool():
    from fvcore.nn import FlopCountAnalysis
    import torchvision.models as M
    m = M.resnet18()
    return f"fvcore resnet18@224 = {FlopCountAnalysis(m, torch.zeros(1,3,224,224)).total()/1e9:.3f} GFLOPs"


def main():
    check("torch/cuda", lambda: f"torch {torch.__version__}, cuda {torch.cuda.is_available()}, "
                                f"device {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
    check("fvcore", flops_tool)
    check("thop", lambda: __import__("thop").__name__ + " importable")
    check("peft", lambda: "peft " + __import__("peft").__version__)
    check("pytorch-grad-cam", lambda: "grad-cam importable: " +
          str(hasattr(__import__("pytorch_grad_cam"), "GradCAM")))
    check("timm", lambda: "timm " + __import__("timm").__version__)
    check("open_clip", lambda: "open_clip_torch " + __import__("open_clip").__version__)
    check("dinov2_vits14 (torch.hub)", dinov2("dinov2_vits14"))
    check("dinov2_vitb14 (torch.hub)", dinov2("dinov2_vitb14"))
    check("BiomedCLIP (open_clip hf-hub)", biomedclip)
    check("dinov3_vits16 (torch.hub)", dinov3("dinov3_vits16"), optional=True)
    check("dinov3 (HF facebook/dinov3-vits16-pretrain-lvd1689m)",
          dinov3_hf("facebook/dinov3-vits16-pretrain-lvd1689m"), optional=True)

    os.makedirs("log", exist_ok=True)
    with open("log/bench-env-gate.json", "w") as f:
        json.dump(RESULTS, f, indent=2)
    with open("log/bench-env-gate.txt", "w") as f:
        f.write("PHASE 0 ENVIRONMENT GATE\n" + "=" * 78 + "\n")
        for r in RESULTS:
            tag = "LOADED     " if r["status"] == "loaded" else "UNAVAILABLE"
            opt = " (optional)" if r["optional"] else ""
            f.write(f"{tag} {r['name']}{opt}\n           {r['detail']}\n")
    required_fail = [r["name"] for r in RESULTS
                     if r["status"] != "loaded" and not r["optional"]]
    print("\n" + open("log/bench-env-gate.txt").read())
    if required_fail:
        print("[GATE] required components unavailable: " + ", ".join(required_fail))
    else:
        print("[GATE] all required components loaded")


if __name__ == "__main__":
    main()
