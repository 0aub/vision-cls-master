# KAUHC V2 deep training roadmap - design

Date: 2026-09-02
Status: approved in chat, pending spec review
Supersedes nothing; extends `BENCHMARK_PLAN_V2.md` after the v2 benchmark completed.

## 1. Why this document exists

The v2 benchmark is finished: 235 runs across three tasks, two protocols, five
tiers, plus cross-validation, calibration, localisation and long-tail analysis.
It answered the question it was asked. It also closed off most of the obvious
next moves, and this document records which, so the roadmap does not spend
machine time on them.

### 1.1 What the benchmark measured, and what follows

| finding | evidence | consequence for the roadmap |
|---|---|---|
| Architecture separates only on the well-posed task, and only below the top | On **5-class and binary**, McNemar between every pair of tier winners gives p = 0.06-1.00: nothing separates. On **merged4** three pairs do separate (p = 9e-6, 7e-5, 0.011) - the task is well-posed enough to tell a good model from a poor one. But the top two remain a statistical tie: dinov2_vits14 LoRA 0.841 vs a 16,384-parameter LDA 0.832, p = 0.87. | **Bounded architecture search only.** The headroom above the current best is not statistically visible; what is visible is that poor choices are punishable. |
| Scale does not help | Spearman rho(parameters, accuracy) = **-0.002**; rho(GFLOPs, accuracy) = **-0.17**. efficientnet_b7 (64 M) 0.615 vs densenet121 (7 M) 0.681. Best merged4 tier-1 model is a **16,384-parameter LDA** at 0.832. | No deeper/wider models. Report the negative correlation as an N1 result. |
| Longer training does not help | Best epoch 4-38 of 100. The 100-epoch tuned grid was worse than 50 (ViT macro F1 0.485 vs 0.544). | Keep 50-epoch schedules. |
| Hyper-parameters help the foundation tier, little else | 24-cell sweep: DINOv2 LoRA validation macro F1 0.456 -> 0.588 at lr 1e-3; ViT macro F1 +0.07; efficient-CNN tier unchanged. | Tuning is done. Reuse the per-tier recipes. |
| **Patient count is the binding constraint** | Ulcer: 1 training patient; **86% of every Ulcer frame in the archive belongs to P_117**. Ulcer recall 0.000 in 20 of 24 sweep cells, never above 0.16. | Data acquisition is the only lever that moves subtyping. |
| Merging Erosion+Ulcer is the one in-data remedy | Erosion<->Ulcer is 20.5% of all test errors. Merging pools 4+8 patients. CV macro F1 +0.07 to +0.22, fold variance roughly halved. | `merged4` is the deployable task. |
| The 221-frame validation split is unreliable | Phase E: validation endorsed a variant 12 points worse on test; the pre-registered rule rejected it by 0.0003. Phase D: class-balanced loss won validation (0.602) and lost test (0.413). | **Select on patient-grouped CV, never on validation.** |
| Study-level performance is much stronger than frame-level, and under-reported | Binary: **14/14 held-out lesion studies correct**. merged4 11/14. Frame-level 0.973 / 0.841. | Make the study the reported clinical unit. |

### 1.2 What V2.1 changed

V2.1 is the annotated export previously believed lost. Its flat lesion folders
match the clean archive frames 1:1 (best-match L2 9-13 against second-best
40-54) with 0.36-0.92% of pixels altered in dark brown-black - the drawn rings.

- **Masks can go from 80% to 100% coverage** (1,246 -> 1,606 lesion frames).
- **Detection boxes come free**, removing the need for manual annotation.
- **Erythema** is a genuinely new class: 166 frames, 5 patients, with paired
  `Prelabeling` (clean) and `Postlabeling` (drawn) trees. Adds 3 new patients.
- Hemangioma (37), LymphoidHyperplasia (151), Neoplasm (105) are **ringed-only,
  no clean version, no patient IDs** - unusable until Dr. Hamza exports their
  `Prelabeling` folders, which Erythema proves his workflow produces.

## 2. Scope and sequencing

**Track A** (data and tasks) -> **Track C** (attention slice, as a bounded
negative result) -> **Track B** (broader model search) held in reserve.

**Pre-registered trigger for Track B**, fixed now so it is not a judgement call
later. merged4 showed that a better-posed task *can* separate models, so the
trigger is not "nothing is separable" but "something above the current top is
reachable". Run Track B only if **either**:

- Track A's detection task (A2) separates model families with a gap larger than
  its own CV standard deviation - evidence that more capacity is exploitable; or
- Track C's attention slice clears the section-5 adoption rule - evidence that
  architectural change still buys something on this corpus.

If neither holds, the ceiling is the data, B is not run, and the paper says so
with these numbers behind it.

### 2.1 Decomposition into implementation plans

A2 is a new subsystem with its own metrics, data format and failure modes; the
rest is evaluation and data work on machinery that already exists. They are
therefore two plans, executed in order:

- **Plan 1 - data and evaluation:** A1, A3, A5, A6, then A4, then Track C.
  All of it extends `bench/` in place.
- **Plan 2 - detection:** A2, once Plan 1's A1 has produced full-coverage boxes.

A7 is a human ask and blocks neither.

## 3. Track A - work items

Each item states its acceptance criterion. Items are independent except where noted.

### A1. Full-coverage lesion masks from V2.1
Diff each V2.1 ringed frame against its matched clean archive frame (the
methodology already in `bench/masks.py`, which currently reconstructs from the
2024 k-NN pickle). Emit stroke masks, filled ellipse interiors and bounding
boxes for **all 1,606** lesion frames.
*Depends on:* nothing. *Cost:* ~1 h.
**Acceptance:** every lesion frame in train/val/test carries a mask; the C2
pointing game reruns at full scope with no `not-in-V8` bucket.

### A2. Detection task
Boxes from A1. Detector: torchvision's single-stage heads (RetinaNet and FCOS)
on backbones already in the grid, so no new dependency is introduced and the
per-tier tuned recipes carry over. Patient-grouped 4-fold CV, reported as
mAP@[.5:.95], mAP@.5 and per-lesion recall at a fixed operating point.
*Depends on:* A1. *Cost:* ~6 h.
**Acceptance:** a detection leaderboard with CV means and standard deviations;
an explicit statement of whether model families separate (this is the Track B
trigger).

### A3. Erythema class -> `6class` and `merged5` tasks
Ingest `_Erythema/Prelabeling` (clean) with its patient folders; keep
`Postlabeling` as annotation only, never as training pixels. Extend the task
enum; rebuild the split with Erythema's 5 patients.
*Depends on:* nothing. *Cost:* ~3 h.
**Acceptance:** `6class` and `merged5` run end to end; corpus at 31 lesion
patients; the split remains patient-disjoint per class and the global leak audit
is rerun.

### A4. Ulcer-targeted copy-paste, CV-selected
Phase E applied to **Ulcer only** (it moved Ulcer recall 0.000 -> 0.129 and
failed by wrecking Erosion), source pool widened by A1's full masks, selection on
patient-grouped CV rather than the validation split.
*Depends on:* A1, A5. *Cost:* ~2 h.
**Acceptance:** adopt only under the A5 rule; publish the number either way.

### A5. Cross-validation as the default selection protocol
Make patient-grouped CV the selection metric everywhere. Already fixed in this
pass: `cv.py` sent `merged4` down the binary branch, and per-fold macro F1
averaged absent classes as 0. Remaining work is to route Phase D/E-style variant
selection through CV.
Concretely: `bench/longtail.py` selects on `summary_val.csv` macro F1 and
`bench/hpo_select.py` on the sweep's validation macro F1. Both move to CV means
from `bench-cv-summary.csv`, with the one-standard-deviation rule of section 5.
*Depends on:* nothing. *Cost:* ~2 h.
**Acceptance:** no adoption decision anywhere reads `summary_val.csv`; each
selection writes the CV mean, the fold standard deviation and the margin it won
or lost by.

### A6. Study-level evaluation as a first-class metric
A clinician reads a study, not a frame. Aggregate frame predictions per
patient-study (majority vote and mean-probability), report accuracy, per-class
recall and bootstrap CIs over studies. Must separate real patient-studies from
singleton Normal frames - conflating them inflates the number
(0.968 vs a true 0.786 on merged4).
*Depends on:* nothing. *Cost:* ~3 h.
**Acceptance:** `bench-study-level.csv` with n_studies stated on every row and
CIs that honestly reflect n=14.

### A7. Data asks (human, blocking nothing)
1. **More Ulcer and Erosion patients** - the only fix for subtyping.
2. **`Prelabeling` exports for Hemangioma, LymphoidHyperplasia, Neoplasm** -
   293 frames, 3 classes, would make this an 8-class benchmark.

## 4. Track C - attention slice

The 15 attention modules already implemented in `src/modules.py` (`se_layer`,
`cbam`, `bam`, `eca`, `simam`, `coordinate_attention`, ...) applied to the two
best CNNs on `merged4`, patient-grouped CV.

This does not contradict "stop architecture search" in 1.1. That conclusion is
about *spending the roadmap on* new backbones. Track C is the opposite: it spends
six hours closing the question with evidence, on code that already exists, so the
paper can state the negative rather than leave a reviewer to ask.

Pre-registered framing: this is a **sensitivity analysis reported as a negative
result** unless it clears the A5 adoption rule. Expected outcome, stated in
advance: attention variants move accuracy by less than the fold-to-fold variance.
*Cost:* ~6 h.

## 5. Decision rules, fixed in advance

Phase E's near-miss - validation endorsed a variant 12 points worse on test, and
the pre-registered threshold rejected it by 0.0003 - is the reason these are
fixed before any result is seen.

1. **Selection metric:** patient-grouped CV mean macro F1. Never the 221-frame
   validation split.
2. **Adoption threshold:** CV mean improvement greater than one standard
   deviation of the fold spread.
3. **Every negative result is reported with its number.**
4. **Study-level numbers always state n_studies** and never pool singleton
   Normal frames with real patient-studies.

## 6. Interfaces

Everything extends `bench/`; nothing forks it.

- Task enum lives in `bench/common.py:TASKS`; `map_label` is the single mapping
  point. A3 adds `6class` and `merged5` there and nowhere else. Three bugs in one
  day came from constants duplicated instead of imported - `report.py`,
  `package.py`, `stats.py` and `remeasure.py` each hardcoded the task list and
  made `merged4` invisible to the whole pipeline. This is now a hard rule.
- A6 adds `bench/study_level.py`, reading `preds_*.csv` only - no retraining.
- A2 is the one genuinely new subsystem: `bench/detect_*.py` with its own metrics
  module, reusing the existing split, patient grouping and CV machinery.

## 7. What lands in the paper

- N1-N4 unchanged, strengthened by full-coverage localisation (A1).
- **N5 = detection** (A2).
- The protocol findings become a methods contribution in their own right:
  per-class versus global patient-disjointness; validation-split unreliability;
  the measured 20-25 point cost of the burned-in annotation shortcut; and the
  observation that **statistical separability is a property of task design, not
  just sample size** - the same 226 test frames separate models on merged4 and
  cannot on 5-class, because merging repaired a class with one training patient.
- Study-level results (A6) reported alongside frame-level throughout.

## 8. Risks

| risk | mitigation |
|---|---|
| Detection inherits the same patient ceiling and separates nothing | That outcome is itself the Track B trigger, and is reported. |
| n=14 test studies makes study-level CIs very wide | Report the CI, never the point estimate alone; state n on every row. |
| Erythema's 5 patients overlap existing ones (P_44, P_103 do) | Rerun the global leak audit after A3; the machinery exists. |
| Ringed-only classes never arrive | A3 proceeds on Erythema alone; the other three stay excluded and the exclusion is stated. |
