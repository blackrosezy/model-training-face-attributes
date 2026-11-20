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
4. **Update training code**
   - `train_multitask.py` / `test_multitask.py`: read `master.csv` + splits, mask missing task labels
     (`ignore_index=-1`), class weights/sampler for imbalance, class order from `classes.yaml`.
   - Currently the scripts still expect the old `ImageFolder` layout and will not run.
5. **Housekeeping**
   - Update README "Data Structure" section; commit `tools/`, `configs/`, `PLAN.md` (`data/` is gitignored).

## Notes
- Moving files via WSL on `/mnt/d` is very slow (~4k files/min); use native Windows
  (`powershell.exe` + `[System.IO.File]::Move`) for bulk moves (~1k files/s).
- Unsloth Studio: API key is invalidated on Studio restart, and the model must be loaded (see `AGENTS.md`).
