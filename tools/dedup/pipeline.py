"""find -> choose keepers -> write the report. Back-end independent."""
import json
import random
from pathlib import Path
from typing import Callable, Dict, List

import sys

from . import registry, store
from .keeper import describe, pick_keeper


def candidate_images(data_dir: Path) -> List[str]:
    """Images still in raw/ (already discarded ones are not considered)."""
    from labeldb import LabelDB
    return LabelDB().candidates("raw", only_unlabeled=False)


def find(data_dir: Path, finder: str, params: Dict, limit: int, progress: Callable[[str, int, int], None]) -> Dict:
    images = candidate_images(data_dir)
    if limit and limit < len(images):
        images = random.Random(0).sample(images, limit)
    print(f"finder={finder} params={json.dumps(params)} images={len(images):,}", flush=True)
    groups = registry.create(finder, **params).find(data_dir, images, progress)

    # Keep the history of duplicates that were already moved, so "restore" keeps working after a new search.
    old_rows = store.read_report()
    moved_groups = {r["group_id"] for r in old_rows if r["status"] == "moved"}
    history = [r for r in old_rows if r["group_id"] in moved_groups]       # incl. the kept image of each group
    offset = max((int(r["group_id"]) for r in history), default=0)
    rows = list(history)
    print(f"found {len(groups):,} groups - reading image sizes to choose keepers", flush=True)
    ordered = sorted(groups, key=lambda g: g.members[0])
    for n, (gid, g) in enumerate(enumerate(ordered, offset + 1), 1):
        info = {m: describe(data_dir, m) for m in g.members}            # each file opened once
        keep = set(g.keep) if g.keep else {pick_keeper(data_dir, g.members, info)}
        for m in g.members:
            w, h, size = info[m]
            rows.append({"group_id": str(gid), "image": m, "keeper": "1" if m in keep else "0",
                         "score": f"{g.score:.4f}", "finder": finder, "status": "pending", "moved_to": "",
                         "width": w, "height": h, "bytes": size})
        if n % 20 == 0 or n == len(ordered):
            progress("review", n, len(ordered))
    print("writing report", flush=True)
    store.write_report(data_dir, rows)
    n_remove = sum(r["keeper"] == "0" and r["status"] == "pending" for r in rows)
    return {"groups": len(groups), "to_remove": n_remove, "images": len(images)}
