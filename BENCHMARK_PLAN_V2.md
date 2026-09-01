# KAUHC V2 study plan v2: benchmark + four novelty modules

Supersedes BENCHMARK_PLAN.md. The core trade-off grid (Prof. Wadii's request) stays,
but it becomes the backbone of a study with four claims no existing WCE paper makes
together. Honesty constraint that shapes everything: 35 lesion patients total, so
novelty must come from METHOD and ANALYSIS, not from pretending the dataset can
support architecture records.

## The four novelty claims

N1. First WCE benchmark spanning the full complexity ladder, classical ML, classic
    CNNs, efficient CNNs, supervised transformers, and self-supervised FOUNDATION
    models, under one patient-disjoint protocol, with a complete accuracy vs
    runtime vs complexity trade-off including parameter-efficient fine-tuning.
N2. Quantified shortcut-learning analysis: burned-in annotation overlays and
    frame-level splits each measured in a controlled ablation (already done).
    To our knowledge the first quantification of either artifact in WCE.
N3. Clinician overlays repurposed as weak localization ground truth: the drawn
    ellipses (the very artifact removed from training) give per-lesion regions,
    used to (a) quantitatively score whether each model's attention lands on the
    lesion (Grad-CAM pointing game and IoU), including the striking contrast of
    ring-trained vs clean-trained models, and (b) drive lesion-focused
    augmentation for the rare classes.
N4. Clinical-deployment analysis: calibration (ECE, reliability diagrams) and
    selective prediction (accuracy vs coverage curves: "defer to the
    gastroenterologist" operating points), reported per model tier.

## Detailed model grid

Tier 1, classical ML (10 models x 3 inputs): AdaBoost, Decision Tree, KNN, LDA,
  Logistic Regression, MLP, Naive Bayes, QDA, Random Forest, SVM, on
  (a) raw 224x224 pixels, (b) best-CNN embeddings, (c) DINOv2 embeddings.

Tier 2, classic CNNs (torchvision, ImageNet-1k weights): AlexNet, VGG16,
  GoogLeNet, ResNet50, ResNet152, DenseNet121, DenseNet201, EfficientNet B0,
  EfficientNet B7.

Tier 3, efficient CNNs: MobileNet V3-Large, ShuffleNet V2 x1.0,
  EfficientNet V2-S, ConvNeXt-Tiny.

Tier 4, supervised transformers: ViT-B/16, Swin-T, Swin-S, MaxViT-T.

Tier 5, foundation models (the SOTA axis):
  - DINOv2 ViT-S/14 and ViT-B/14 (torch.hub facebookresearch/dinov2), three
    regimes each: frozen features + linear probe; frozen + k-NN; LoRA
    fine-tuning (r=8, alpha=16, attention q/v matrices, via peft).
  - BiomedCLIP (microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 via
    open_clip): zero-shot with class-descriptive text prompts, AND frozen
    image-encoder features + linear probe.
  - DINOv3 if the checkpoint downloads cleanly on the machine; otherwise noted
    as unavailable, not silently skipped.
  Hypothesis worth publishing either way: with 35 lesion patients, frozen
  foundation features + tiny heads match or beat fully fine-tuned supervised
  backbones at a fraction of trainable parameters.

## Module details

M1 Trade-off (all tiers): params, trainable params (full FT vs LoRA vs probe),
  GFLOPs, GPU latency b1, CPU latency b1, throughput b16, peak training VRAM,
  training wall-clock. Pareto figures per task; the LoRA and probe points give
  the resources story real depth.

M2 Long-tail rescue (5-class task, top 3 backbones + DINOv2-B LoRA):
  controlled comparison of plain CE vs weighted CE vs focal loss (gamma 2) vs
  class-balanced loss (effective number, beta 0.9999) vs weighted sampler,
  selected on validation macro F1, adopted variant rerun on test. Target metric:
  ulcer and xanthoma recall, currently the failure point.

M3 Overlay-derived weak localization:
  a. Build lesion region maps: pixel diff between annotated export and raw frame
     gives each ellipse; store per-frame binary masks and bounding boxes (the
     earlier forensic run already produced bboxes for all 1,288 training lesions;
     extend to val/test lesion frames the same way).
  b. Attention faithfulness: Grad-CAM (CNNs) and attention rollout (ViTs) on
     clean-trained models; score pointing game hit-rate and CAM-region IoU
     against the ellipse regions, per class, per tier. Include the contrast
     figure: the archived ring-trained checkpoint's CAM (fires on the ring
     location) vs the clean-trained model (fires on the lesion). This is the
     paper's signature figure.
  c. Lesion-focused augmentation for rare classes: copy-paste augmentation,
     lesion regions (from the masks) blended onto normal mucosa backgrounds with
     Poisson or alpha blending, applied only to ulcer/erosion/xanthoma training
     data; evaluated as one controlled variant. Optional stretch, only if
     copy-paste shows promise: diffusion-based synthesis (Stable Diffusion +
     LoRA per class) with FID sanity checks; clearly labeled exploratory.

M4 Trust and deployment: temperature scaling on validation; ECE and reliability
  diagrams per tier winner; selective prediction curves (accuracy at 100/90/80/70
  percent coverage by confidence); McNemar between tier winners; patient-grouped
  4-fold CV for tier winners; 95 percent bootstrap CIs on everything.

## Protocol (unchanged foundations)

Clean overlay-free frames, deduplicated, patient-disjoint split (seed 1998, the
same V8 split), resolution 224x224 for the entire grid, identical macro,
weighted, and per-class metrics for every model, logits saved for every test
prediction (required by M4), history.csv mandatory (training curves).
Tasks: binary lesion detection and 5-class. CV per M4.

## What lands in the paper

Sections: benchmark table grouped by tier with cost columns; two Pareto figures;
foundation-model subsection (zero-shot, probe, LoRA); long-tail table; the
localization-faithfulness figure and pointing-game table; calibration and
selective-prediction figure; CV and significance; the protocol-ablation
subsection already written. Abstract and contributions updated around claims
N1 to N4. The rev2 manuscript package stays the skeleton.

## Order of execution and fallbacks

Phase A: core grid tiers 2 to 4 plus tier 1 (Wadii's ask, paper-critical).
Phase B: tier 5 foundation models (cheap: mostly frozen features and LoRA).
Phase C: M3a+M3b (localization; needs no training) and M4 (post-hoc on saved
logits). Phase D: M2 long-tail variants. Phase E: M3c augmentation, optional.
Each phase produces a self-contained result; if time runs out after C, the paper
already carries N1, N2, N3(b), N4. Nothing in the writing depends on Phase E.
