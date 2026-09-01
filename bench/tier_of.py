"""Print the benchmark tier of an architecture, for the shell drivers."""
import sys

T2 = ("efficientnet_b0 resnet50 densenet201 alexnet vgg16 googlenet resnet152 "
      "densenet121 efficientnet_b7").split()
T3 = "efficientnet_v2_s convnext_tiny mobilenet_v3_large shufflenet_v2_x1_0".split()
T4 = "vit_b_16 swin_t swin_s maxvit_t".split()

m = sys.argv[1] if len(sys.argv) > 1 else ""
tier = ("tier2-classic-cnn" if m in T2 else
        "tier3-efficient-cnn" if m in T3 else
        "tier4-transformer" if m in T4 else
        "tier5-foundation" if m.startswith("dinov2") or m.startswith("biomedclip")
        else "tier2-classic-cnn")
print("TIER\t" + tier)
