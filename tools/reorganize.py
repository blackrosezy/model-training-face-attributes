"""Move legacy data/<task>/<split>/<class>/<img> into data/raw/<source>/<source>_<NNNNNN>.<ext>.

Names collided across the legacy folders (e.g. ffhq_1.png is a different image in each emotion
class), so images get fresh unique ids. The old path is kept in master.csv (legacy_path).
Writes data/labels/move_map.csv BEFORE moving so the move can be resumed or undone.
Usage: python tools/reorganize.py [--dry-run]
"""
import csv, sys, shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path("data")
dry = "--dry-run" in sys.argv
master = ROOT / "labels/master.csv"
rows = list(csv.DictReader(master.open()))
if "legacy_path" in rows[0]:
    sys.exit("master.csv already reorganized")

counter = defaultdict(int)
mapping = []
for r in rows:
    old = r["image"]; src = r["source"]; ext = Path(old).suffix.lower()
    counter[src] += 1
    new = f"raw/{src}/{src}_{counter[src]:06d}{ext}"
    mapping.append((old, new))

mp = ROOT / "labels/move_map.csv"
if not dry:
    with mp.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["old", "new"]); w.writerows(mapping)
    for old, new in mapping:
        dst = ROOT / new
        dst.parent.mkdir(parents=True, exist_ok=True)
        if (ROOT / old).exists():
            shutil.move(ROOT / old, dst)
        elif not dst.exists():
            sys.exit(f"missing both {old} and {new}")
    # stray non-image files
    stray = ROOT / "_stray"; stray.mkdir(exist_ok=True)
    for p in list(ROOT.glob("*/*/*/*")):
        if p.parts[1] in ("age", "emotion", "gender") and p.is_file():
            shutil.move(p, stray / p.name)

    m = dict(mapping)
    out = []
    for r, (old, new) in zip(rows, mapping):
        out.append({"image": new, "source": r["source"], "legacy_path": old,
                    "legacy_task": r["legacy_task"], "legacy_split": r["legacy_split"],
                    "legacy_label": r["legacy_label"]})
    with master.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    v = ROOT / "labels/vlm_labels.csv"
    vr = list(csv.DictReader(v.open()))
    for r in vr: r["image"] = m[r["image"]]
    with v.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(vr[0])); w.writeheader(); w.writerows(vr)
print(("DRY RUN " if dry else "") + f"{len(mapping)} images; per source: {dict(counter)}")
