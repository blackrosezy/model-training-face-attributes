"""Register new images dropped into data/raw/ in the label database.

    python tools/scan.py          # fast: re-lists only folders whose modification time changed
    python tools/scan.py --full   # ignore the remembered folder times and list everything

Why it is fast: reading /mnt/d from WSL is slow per file, so we (1) only list names (no per-file
stat), (2) skip folders whose mtime is unchanged since the last scan, and (3) diff the names against
the database in memory. New images are inserted as unlabeled, so the labeler and the duplicate
finder pick them up. Nothing is ever moved or deleted here.

Folder layout: data/raw/<source>/<file>   (the <source> folder name becomes the image's source).
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labeldb import DATA, LabelDB

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def scan(db: LabelDB, full: bool = False, log=print) -> dict:
    t0 = time.monotonic()
    known = db.known_paths("raw")
    old = {} if full else db.scan_state()
    new_state, new_rows = {}, []
    listed = skipped = seen = 0

    stack = ["raw"]
    while stack:
        rel = stack.pop()
        try:
            mtime = os.stat(DATA / rel).st_mtime_ns        # read BEFORE listing: a change during the listing is caught next time
        except FileNotFoundError:
            continue
        prev = old.get(rel)
        if prev and prev[0] == mtime:                       # folder unchanged: no listing, just walk its known subfolders
            skipped += 1
            subdirs = prev[1]
            new_state[rel] = (mtime, subdirs)
        else:
            listed += 1
            subdirs = []
            with os.scandir(DATA / rel) as it:
                for e in it:
                    if e.is_dir(follow_symlinks=False):
                        subdirs.append(e.name)
                    elif os.path.splitext(e.name)[1].lower() in EXTS:
                        seen += 1
                        path = f"{rel}/{e.name}"
                        if path not in known:
                            parts = path.split("/")
                            new_rows.append({"path": path, "loc": "raw", "source": parts[1] if len(parts) > 2 else "",
                                             "legacy_path": "", "legacy_task": "", "legacy_split": "", "legacy_label": ""})
            new_state[rel] = (mtime, sorted(subdirs))
            log(f"  listed {rel}: {seen:,} images so far, {len(new_rows):,} new")
        stack.extend(f"{rel}/{d}" for d in subdirs)

    db.scan_commit(new_rows, new_state)                     # one transaction: rows + folder times
    out = {"new": len(new_rows), "folders_listed": listed, "folders_skipped": skipped,
           "seconds": round(time.monotonic() - t0, 1)}
    by_src = {}
    for r in new_rows:
        by_src[r["source"]] = by_src.get(r["source"], 0) + 1
    out["by_source"] = by_src
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="re-list every folder, ignoring remembered mtimes")
    a = ap.parse_args()
    r = scan(LabelDB(), full=a.full)
    print(f"scan done in {r['seconds']}s: {r['new']} new image(s) "
          f"({r['folders_listed']} folder(s) listed, {r['folders_skipped']} unchanged and skipped)"
          + (f"  {r['by_source']}" if r["by_source"] else ""))


if __name__ == "__main__":
    main()
