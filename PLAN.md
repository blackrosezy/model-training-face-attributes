# Data reorganization & relabeling plan

Goal: stop trusting the legacy folder labels (age / emotion / gender), re-label every image with a
VLM (Unsloth Studio, see `AGENTS.md`), and train from manifests instead of `ImageFolder`.

## Why the old layout was wrong
- Labels lived in folder names (`data/<task>/<split>/<class>/`), so changing a label meant moving files.
- Three separate datasets, one per task; an image could carry only one label.
- Train/val split fixed by folder, no record of how, no test set, possible person/near-duplicate leakage.
- Filenames collided (`ffhq_1.png` was a different image in each emotion class); stray junk files.
- Heavy imbalance (age 80plus 276 vs 01-10 1,981) and epoch size set by the biggest task (gender 66k vs age 7k).
- Class order implicit (alphabetical folder sort).

## Target layout
_(Partly superseded: labels now live in SQLite `~/.local/share/face-labeling/labels.db`, not `master.csv` / `vlm_labels.csv`; see `AGENTS.md`. Images are in `data/raw/` and `data/discarded/`, `data` -> `/home/cribpc/face-data`.)_

```
data/
├── raw/<source>/<source>_<NNNNNN>.<ext>   images, unique ids, never edited
├── processed/                              (optional) face-cropped, fixed size
├── labels/
│   ├── master.csv                          image, source, legacy_path, legacy_* (UNTRUSTED)
│   ├── vlm_labels.csv                      VLM age/emotion/gender per image
│   └── splits/                             train/val/test csv, seeded, grouped by person/source
├── configs/
│   ├── classes.yaml                        fixed class order
│   └── age_buckets.yaml                    (optional) only if exact ages exist
└── _stray/                                 non-image junk found in the old folders
tools/
├── build_manifest.py                       scan legacy layout -> master.csv (already run)
├── reorganize.py                           legacy folders -> raw/ (already run)
└── vlm_label.py                            label images with the VLM, resumable
```

## Status

### Done
- [x] `configs/classes.yaml`
- [x] `master.csv` built (116,691 images)
- [x] Images moved to `raw/` (ffhq 10,571 / image 23,756 / other 58,659 / utkface 23,705); old folders removed
- [x] `tools/vlm_label.py` working; first 100 random images labeled (all valid JSON, ~0.4 s/img, 2 workers)
- [x] Label store moved to SQLite (`tools/labeldb.py`); labeling GUI (`tools/gui.py`), new-image scan (`tools/scan.py`), dedup finders (`tools/dedup/`)
- [x] VLM also flags overlays, non-photo styles and multi-face images; those are moved to `data/discarded/`
- [x] ~192k images in `raw/` (192,239 at the 2026-11-21 check); full VLM labeling run **in progress** (GPU busy)

First-100 comparison with legacy labels: gender 73/75 agree, age 6/7, emotion 7/18.
VLM looks biased: `neutral` 55/100, `21-30` 59/100. Agreement is not accuracy.

### To do
1. **Check VLM quality**
   - View the disagreeing emotion images by eye; decide who is wrong, VLM or folder.
   - Try a confidence signal (ask twice / ask for a score); try a prompt less biased to the middle classes.
2. **Larger labeling run**
   - Stratified 1,000-image batch first (~7 min), then the full set (~13 h at 2 workers; test more workers).
   - Decide label policy: keep VLM label only, require VLM == legacy, or require confidence.
3. **Finish the structure**
   - Identify the source of `raw/other` (58k) — needed for source/person-level splits.
   - File-hash dedupe, then `labels/splits/` (train/val/test, seeded, grouped).
   - `age_buckets.yaml` only if exact ages are recoverable (UTKFace-style names encode age).
   - `processed/` face crop/align (optional).
4. **Update training code** (superseded by the "Training plan" section below)
   - `train_multitask.py` / `test_multitask.py`: read the label DB + splits, mask missing task labels
     (`ignore_index=-1`), class weights/sampler for imbalance, class order from `classes.yaml`.
   - Currently the scripts still expect the old `ImageFolder` layout and will not run.
5. **Housekeeping**
   - Update README "Data Structure" section; commit `tools/`, `configs/`, `PLAN.md` (`data/` is gitignored).

## Notes
- Moving files via WSL on `/mnt/d` is very slow (~4k files/min); use native Windows
  (`powershell.exe` + `[System.IO.File]::Move`) for bulk moves (~1k files/s).
- Unsloth Studio: API key is invalidated on Studio restart, and the model must be loaded (see `AGENTS.md`).

## Training plan: attribute model retraining (review of 2026-11-21)

Context: the browser app (`/mnt/d/bazilah/live-face-analytics`, ONNX Runtime Web on WebGPU, RTX 4080) runs
YOLOv8n-face (`yolov8n-face-lindevs.onnx`) and our multitask model (ResNet18 224px, 11.2M params, 44.7 MB fp32, opset 17, dynamic batch;
outputs `age_logits` with only **8** classes (old folder buckets, not our 21), `emotion_logits` 6, `gender_logits` 2; the JS reads
only emotion and gender, age is unused so far and will be shown later). Labeling of ~192k images is running (GPU busy): do NOT start
GPU work, full-dataset runs, or anything that touches the label DB until it is done.

### 1. Problems found in the current training code
- [ ] `train_multitask.py` / `test_multitask.py` still read the old `ImageFolder` layout (`data/<task>/train|val`); that layout is gone. Must read the SQLite label DB (`valid=1 AND loc='raw'`) and class order from `data/configs/classes.yaml`.
- [ ] Three separate loaders with round-robin optimizer steps; epoch length set by the biggest task. Replace by one joint batch with a masked loss (`ignore_index=-1`).
- [ ] No imbalance handling. First ~2k labels: emotion 65% neutral, angry/fear <1%; age 44% in `24-27`. Use class-balanced weights or a sampler.
- [ ] Training crop (`Resize(256)`, `CenterCrop(256)`, `RandomResizedCrop(224)`) does not match the browser crop (tight YOLO box, resize short side to 256, center-crop 224). Train on the same pipeline: YOLO box -> fixed margin -> square crop, with box jitter.
- [ ] Best checkpoint chosen by mean val loss (dominated by the biggest task) and accuracy-only metrics. Use macro-F1 / balanced accuracy for emotion, MAE for age, and a combined score.
- [ ] No person-level split, no test set. Group splits by person (use `face_id` embeddings/clusters), seeded, with a held-out test set.
- [ ] ONNX export: dynamic batch, ImageNet normalization done in JS. `test_multitask.py` hard-codes the same preprocessing and already lists 21 age classes, which does not match the deployed 8-class model.
- [ ] README "Data Structure" section is out of date.

### 2. Model choice (browser is RTX 4080 WebGPU, so FLOPs are not the limit)
- [ ] Add a pluggable backbone (timm) and `--img-size`. Candidates: ResNet18 (baseline), ResNet50, ConvNeXt-Tiny/Nano, MobileOne-S2/S3, EfficientNetV2-S. Check WebGPU op support (LayerNorm, GELU, depthwise 7x7) before committing to ConvNeXt.
- [ ] Keep 224 (or 192) input; webcam faces are small, resolution helps emotion/age.
- [ ] Try a face-pretrained init (InsightFace-style backbone) vs ImageNet.
- [ ] Pick by real in-browser ms per frame measured in the app next to YOLOv8n-face + the counter models, against accuracy. The attribute model runs ~3x/s on 1 face, so a bigger model costs little.
- [ ] Optional: distillation from a bigger teacher.

### 3. Age head (will be shown in the UI)
- [ ] Keep `age_logits` in the ONNX, now with the 21 buckets of `classes.yaml` (the deployed model has 8, so the JS needs the new class list/bucket centers when age is added).
- [ ] Loss: cross-entropy on a soft Gaussian target around the labeled bucket + an expected-age term (ordinal-aware).
- [ ] Metrics: MAE in years and accuracy within +-1 bucket, not exact-bucket accuracy.
- [ ] VLM labels are biased to `24-27` and neutral: class-balanced weighting for age too.
- [ ] UI later: show expected age (softmax-weighted) or a +-4 year range, smoothed per track over ~10 frames, never the raw argmax bucket.

### 4. Data / labels
- [ ] After a first model: cross-validated label-noise detection (confident disagreement with the VLM), review those in the GUI.
- [ ] Build a hand-verified gold test set (~500-1000 images, spread over all ages and emotions) so we measure against ground truth, not VLM mistakes.
- [ ] Crop-cache script: run `yolov8n-face-lindevs.onnx` once over the images and store crops (same margin as the app). Not to be run while labeling is using the GPU; use `--limit` first.
- [ ] Augmentation: flip, small rotation, color jitter, blur, JPEG compression, box jitter.
- [ ] Training: AdamW with warmup + cosine, AMP, channels_last, EMA, lower LR for the backbone than the heads.

### 5. ONNX export for the browser
- [ ] Put mean/std normalization inside the graph (browser sends 0-1 pixels).
- [ ] Fixed input shape (batch 1, or a small fixed N); no dynamic axes.
- [ ] Output logits only (softmax in JS); convert to fp16 and re-check accuracy (44.7 MB -> ~22 MB).
- [ ] Load-test in ORT-Web WebGPU: every op must have a WebGPU kernel (no CPU fallback).
- [ ] Update `test_multitask.py` to match the new preprocessing.

### 6. Live app changes (in `live-face-analytics`, NOT done, for later)
- [ ] `AttributeClassifier.ts`: crop with the same margin/square crop as training; today `BOX.PADDING` (1 px) is only used for drawing and the crop is the tight box.
- [ ] Read `age_logits`, compute expected age, add `age` to `Attributes` / `CLASS_NAMES` / `Visualizer`.
- [ ] Temporal smoothing per track (EMA of probabilities or vote over the last N) for emotion, gender and age; emotion flickers today.
- [ ] Move normalization into the ONNX and skip the JS loop (+ `getImageData`).
- [ ] Pin the `onnxruntime-web` version in `index.html` (currently an unpinned CDN `ort.min.js`) so WebGPU behavior can't change under us.
- [ ] Optional: the main loop awaits detector -> attributes -> counter in series; consider overlapping them if frame rate matters.

### Open questions
- Confirm `yolov8n-face-lindevs.onnx` is the detector to build the training crops with.
- Which crop margin the app should use (decide together with the training crops).
