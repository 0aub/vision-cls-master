# ADDENDUM to BENCHMARK_BRIEF_V2.md (read together; this overrides where they conflict)

## A. Repository strategy (the repo is the base, extend it)

1. Work on a new git branch `bench-v2` off main. Commit at every phase boundary
   with a message naming the phase. Add/extend .gitignore: data/, log/, archive/,
   *.pth, *.pkl, *.npy. Never commit data or checkpoints.
2. Extend, do not rewrite: add a model registry in src/modules.py with three
   sources: `torchvision` (existing path), `hub-dinov2` (torch.hub), and
   `open_clip` (BiomedCLIP). Add config flags: `train_mode: full | probe | lora`
   and `task: 5class | binary`. Keep the existing Trainer working for the
   torchvision path; wrap rather than fork for the new sources.
3. Environment manifest: at the start of Phase A, write pip freeze, torch/CUDA
   versions, GPU name, and driver into log/bench-environment.txt. The paper needs
   this and it never gets recorded after the fact.

## B. Known pitfalls in this exact repo (verified in earlier sessions; fix in Phase 0)

1. src/datasets.py builds ALL DataLoaders with shuffle=True, including val and
   test. Evaluation must use unshuffled loaders or preds_test.csv row order is
   garbage. The bench runner must construct its own eval loaders.
2. prepare_set_ml() feeds the AUGMENTED train loader to sklearn fitting (random
   flips). Fit classical models on deterministic eval-transform features and note
   the change.
3. The splitfolders code path in datasets.py CANNOT reproduce the existing split
   (proven earlier: it yields 1923/239/244). Never let it run; always use the
   existing data/splitted/V8-KAUHC folders. Guard: assert split sizes 1916/221/226
   at startup of every run.
4. googlenet returns an aux-output tuple in train mode; the existing code path
   handled it at 256, verify at 224. inception_v3 crashed in the 2024 runs (299
   input, tuple output); it stays excluded.
5. vit_b_16 and maxvit_t hard-require 224 input; this is why the whole grid is
   224. DINOv2 uses patch 14 (224 divisible by 14, fine) and its own
   normalization constants, not ImageNet's torchvision transform defaults; use
   the official DINOv2 transforms for those runs.
6. history.csv was silently not written by the revision2 training path. The
   smoke-test gate exists because of this; do not skip it.

## C. Correction to Phase C (the earlier brief overstated what is possible)

The ANNOTATED (ringed) versions of val/test lesion frames may not exist on this
machine: the ringed pixels are only guaranteed to exist for the 1,288 TRAINING
frames, embedded in the archived v7-knn/best.pkl. Therefore:

1. C1 gate: first check whether an annotated-export dataset folder (ringed
   frames) exists for the current val/test lesion frames. If yes, build masks
   for all splits as written. If no, report that, and proceed with the fallback:
2. Fallback scope: build masks for the 1,288 training lesion frames by diffing
   the pickle-stored pixels against the raw frames (bboxes already exist in
   archive/**/revision-train-matches.csv; regenerate full binary masks the same
   way, un-flipping using the recorded flip variant).
3. Pointing-game evaluation then runs on training-split lesion frames. That is
   still meaningful for localization (the models were never given masks), but
   the report and the paper must state the split it was computed on. Qualitative
   CAM panels for val/test frames are still produced (no masks needed to look).
4. The signature contrast figure works either way: reconstruct the ringed
   training frames from the pickle, run the archived ring-trained
   efficientnet_b0 checkpoint's Grad-CAM on them, and the clean-trained model's
   Grad-CAM on the raw versions of the same frames.
5. Separately, tell the user to ask Dr. Hamza whether the annotated exports for
   all frames can be retrieved from the hospital workstation; if they arrive
   later, rerun C1/C2 at full scope with one command.

## D. Operational requirements (long multi-day runs fail; plan for it)

1. Resume: before starting any run, check for its summary_test.csv; if present
   and non-empty, skip. Re-invoking the whole brief must be idempotent.
2. Checkpoints: save model state_dict only (no optimizer) as best.pth per run;
   keep them locally (Phase C needs them; nothing goes in the zip). Check free
   disk at Phase 0; the grid needs roughly 40 to 80 GB for checkpoints.
3. Phase-boundary deliverables: at the end of Phases A, B, C, and D, write a
   partial zip (bench-results-phaseA.zip, ...) with that phase's text outputs,
   so partial results can be reviewed while later phases run.
4. Progress log: append one line per completed run (model, task, test acc,
   minutes) to log/bench-progress.txt so a human can glance at state.
5. OOM policy: halve batch to 8 then 4 before declaring a model infeasible;
   record final batch size in efficiency.json (it affects latency comparability
   notes, not the b1 latency measurements).
6. Budget guard: if projected total wall-clock exceeds 3 days at the end of
   Phase A, report projected times per remaining phase and continue; do not
   silently drop models.

## E. One-line answers to open questions

- Base repo: yes, github.com/0aub/vision-cls-master on branch bench-v2.
- The archived v7 checkpoints and the raw + splitted data must be present before
  starting; everything else the agent installs or downloads itself.
- Nothing in Phases A to D requires human input; the only human ask mid-run is
  C5 (annotated exports from Dr. Hamza), and it is optional.
