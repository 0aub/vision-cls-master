# BENCHMARK BRIEF v2 for Claude Code (local session)

Supersedes BENCHMARK_BRIEF.md. Execute BENCHMARK_PLAN_V2.md end to end. Prior
sessions left forensic artifacts you must reuse, not recompute. Work autonomously;
stop only at the named gates. Nothing is ever deleted; superseded material moves to
archive/.

## Phase 0: cleanup and gates (same as v1, plus two additions)

1. Create archive/ and move into it: log/v7-*, log/revision-*, log/revision2*,
   revision/, old briefs and one-off scripts, temporary data dirs. Keep:
   data/uncompressed/KAUHC-V2-full (source of truth) and data/splitted/V8-KAUHC
   (the split; verify composition 1916/221/226 against the archived
   revision2-split-composition.csv).
2. Verify the src fixes (metric-mode validation, sample-weighted loss).
3. RETAIN AND INDEX these archived artifacts, later phases need them:
   - archive/**/v7-* checkpoints (ring-trained models, needed for the CAM
     contrast figure in Phase C)
   - archive/**/revision-train-matches.csv (lesion bboxes for 1,288 training
     frames) and the annotated-vs-raw matching methodology
   - archive/**/revision2-* summaries (protocol-ablation numbers, do not rerun)
4. Environment gate: pip install as needed: fvcore or thop, peft, open_clip_torch,
   pytorch-grad-cam, timm (fallback loader). Try torch.hub load of
   facebookresearch/dinov2 (vits14, vitb14) and open_clip BiomedCLIP
   (hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224). Record in
   the report exactly which loaded; a checkpoint that cannot download is reported
   as unavailable, never silently skipped. Also attempt DINOv3; optional.
5. Smoke-test gate: 2-epoch efficientnet_b0 run at 224 through the new runner;
   history.csv, all summary/report/preds/cm files, logits file, efficiency.json
   must all appear. Fix until green.

## Phase A: core grid (paper-critical, run first)

Runner: one config-driven entry point. Global protocol: 224x224, ImageNet norm,
full fine-tuning, Adam LR 1e-4, light flips (train only), cosine schedule, 100
epochs, batch 16 (halve for models that OOM and record it), checkpoint = best val
accuracy, seed 1998, unshuffled deterministic eval. Every run writes
log/bench-<task>-<model>/: history.csv, summary_{train,val,test}.csv (acc, macro
P/R/F1, weighted P/R/F1, correct mean CE), report_{split}.csv, preds_test.csv
(path, y_true, y_pred), PROBS_test.npy (softmax probabilities, row-aligned with
preds_test.csv; REQUIRED for Phase C/M4), cm_{split}.csv, efficiency.json.

Efficiency measurement per model (efficiency.json): params_total,
params_trainable, gflops@224 (record the tool), gpu_latency_ms_b1 (median of 500
after 50 warmup, cuda.synchronize), gpu_throughput_ips_b16, cpu_latency_ms_b1
(median of 100 after 10 warmup, note thread count), peak_train_vram_mb,
train_wallclock_min.

Model order (5-class first, then binary, both tasks for every deep model):
  A1: efficientnet_b0, resnet50, densenet201
  A2: efficientnet_v2_s, convnext_tiny, mobilenet_v3_large, shufflenet_v2_x1_0
  A3: vit_b_16, swin_t, swin_s, maxvit_t
  A4: alexnet, vgg16, googlenet, resnet152, densenet121, efficientnet_b7
Classical ML at 224 (after A1): 10 sklearn models on raw pixels (150,528 dims);
embeddings runs move to Phase B so they can use the best backbone and DINOv2.

Sanity gate after A1: 5-class test accuracy in [0.55, 0.80] and no
majority-class collapse; investigate anomalies before continuing.

## Phase B: foundation models (tier 5)

For DINOv2 ViT-S/14 and ViT-B/14 (input 224, DINOv2 normalization):
  1. Frozen features (CLS token, and CLS+mean-patch concat as a documented
     variant): linear probe (logistic regression, val-selected C) and k-NN
     (k val-selected). Both tasks.
  2. LoRA fine-tuning: peft, r=8, alpha=16, dropout 0.05, target q and v
     projections, head trained, everything else frozen; same schedule as Phase A
     but 50 epochs (converges faster); record trainable param count.
For BiomedCLIP:
  3. Zero-shot: encode class prompts (write 3 to 5 clinically phrased prompts
     per class, average text embeddings; e.g. "an endoscopy image of small bowel
     angiodysplasia with visible vascular lesion" for AVM; document all prompts
     in the report). Both tasks (binary via lesion/normal prompt sets).
  4. Frozen image features + linear probe.
Classical-ML embedding runs: 10 sklearn models on (a) the best Phase A backbone's
penultimate features and (b) DINOv2-B features; default + val-selected grids.
All outputs in the same standardized format (bench-<task>-<name>/).

## Phase C: localization faithfulness and deployment analysis (no training)

C1. Lesion masks: for every lesion frame in val and test, compute the pixel diff
    between the annotated export and the raw frame (reuse the archived matching
    method); save binary ellipse masks and bboxes to log/bench-lesion-masks/.
    Training-frame bboxes already exist in revision-train-matches.csv.
C2. Attention maps: Grad-CAM (CNNs; last conv block) and attention rollout or
    Grad-CAM on final norm layer (ViTs/DINOv2) for all test lesion frames, for:
    the top-2 Phase A models, the top DINOv2 variant, and the archived
    ring-trained efficientnet_b0 (v7 checkpoint, evaluated on the ANNOTATED
    versions of the same frames). Score per model: pointing-game hit rate (CAM
    argmax inside mask) and IoU of the top-20-percent CAM region vs mask, per
    class. Save per-frame scores CSV plus 6 to 10 side-by-side qualitative
    panels (raw frame, mask, clean-model CAM, ring-model CAM) as PNGs; these
    become the paper's signature figure.
C3. Calibration and selective prediction from saved PROBS_test.npy: temperature
    scaling fitted on val; ECE (15 bins) before and after; reliability-diagram
    data CSV; accuracy-vs-coverage curves at confidence thresholds (100/90/80/70
    percent coverage) per tier winner. One consolidated bench-trust.csv.

## Phase D: long-tail variants (5-class)

On efficientnet_b0, the best Phase A backbone if different, and DINOv2-B LoRA:
plain CE vs weighted CE vs focal (gamma 2) vs class-balanced loss (effective
number, beta 0.9999) vs weighted sampler. Select on val macro F1, rerun the
winner on test. Deliver one table: per-class recall for every variant. The
decision rule and outcome go in the report.

## Phase E (optional, only after A-D are zipped): lesion copy-paste augmentation

Using C1 masks: blend ulcer/erosion/xanthoma lesion regions onto normal training
backgrounds (alpha or Poisson blending), one controlled variant on the best
5-class model. If and only if it helps val macro F1 by >0.02, keep and report;
otherwise report the negative result in one sentence. Diffusion synthesis: skip
unless explicitly told otherwise; note it as future work.

## Statistics (rolls up A, B, D)

Patient-grouped 4-fold CV (both tasks) for: best classic CNN, best efficient CNN,
best transformer, best foundation variant, efficientnet_b0. McNemar exact tests,
paired by path, between tier winners (adjacent pairs) and best-deep vs
best-embedding-ML. Bootstrap 95 percent CIs (10,000 resamples) for accuracy and
macro F1 of every model into bench-bootstrap-ci.csv.

## Report and package

log/BENCH-REPORT.md: cleanup log, environment gate outcomes (which SOTA
checkpoints loaded), full per-task leaderboards grouped by tier with cost
columns, foundation-model results incl. zero-shot, long-tail table, pointing
game table, calibration summary, CV, McNemar, anomalies, wall-clock per phase.
Zip as bench-results.zip: the report, every summary/report/preds/cm CSV, every
history.csv and efficiency.json, PROBS files ONLY for tier winners (size), the
C2 qualitative PNG panels, bench-trust.csv, bench-bootstrap-ci.csv, lesion-mask
bbox CSV. No checkpoints, nothing from archive/. If the zip exceeds ~200 MB,
drop the PROBS files and say so in the report.
