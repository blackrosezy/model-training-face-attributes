"""Scan the legacy data/<task>/<split>/<class>/ layout and write data/labels/master.csv.

Legacy folder labels are stored in legacy_* columns and are NOT trusted.
Images are not moved or modified.
"""
import csv, re
from pathlib import Path

ROOT = Path("data")
EXTS = {".jpg", ".jpeg", ".png"}
TASKS = ("age", "emotion", "gender")

def source_of(name: str) -> str:
    if name.startswith("ffhq_"): return "ffhq"
    if re.match(r"^\d+_[01]_\d_\d+", name): return "utkface"
    if name.startswith("image"): return "image"
    return "other"

rows = []
for task in TASKS:
    for p in sorted((ROOT / task).glob("*/*/*")):
        if p.suffix.lower() not in EXTS:
            continue
        split, cls = p.parts[-3], p.parts[-2]
        rows.append({
            "image": p.relative_to(ROOT).as_posix(),
            "source": source_of(p.name),
            "legacy_task": task, "legacy_split": split, "legacy_label": cls,
        })

out = ROOT / "labels" / "master.csv"
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
print(f"wrote {len(rows)} rows -> {out}")
