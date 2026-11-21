"""Link images of the same person: images.person_id (+ table persons).

person_id NULL = "not clustered yet"; treat such an image as its own person when splitting/training.
IDs are stable: once an image has a person_id it keeps it, a later run only fills or merges.

  python tools/persons.py stats                 # how many persons / images are linked
  python tools/persons.py from-dups             # dry run: what would be linked from the duplicate report
  python tools/persons.py from-dups --apply     # write it (DB only, no GPU, no files touched)
  python tools/persons.py set ID IMAGE [IMAGE]  # manual fix: put these images into person ID (use 'new' for a new one)

from-dups uses every group in the `dups` table (kept images AND the ones moved to discarded/). Run it after each
dedup find/apply; groups that share an image are merged into one person.
"""
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labeldb import LabelDB  # noqa: E402


# Finders whose groups really mean "same person": exact duplicates and ArcFace identity.
# (faiss_embed uses ImageNet features and also groups look-alike strangers, so it is not used.)
SAME_PERSON_FINDERS = ("torch_gpu", "face_id")


class UnionFind:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def dup_components(db: LabelDB):
    """Connected components (lists of images.path) from the duplicate report, plus the finders that made them."""
    known = {r[0]: r[1] for r in db.c.execute("SELECT path, person_id FROM images")}
    uf, finders = UnionFind(), defaultdict(set)
    missing = 0
    groups = defaultdict(list)
    for gid, image, moved_to, status, finder in db.c.execute("SELECT group_id, image, moved_to, status, finder FROM dups WHERE finder IN (%s)"
                                                                              % ",".join("?" * len(SAME_PERSON_FINDERS)), SAME_PERSON_FINDERS):
        path = next((p for p in (moved_to if status == "moved" else None, image, moved_to) if p and p in known), None)
        if path is None:
            missing += 1
            continue
        groups[gid].append(path)
        finders[gid].add(finder)
    for gid, members in groups.items():
        for m in members[1:]:
            uf.union(members[0], m)
    comps = defaultdict(list)
    comp_finders = defaultdict(set)
    for gid, members in groups.items():
        root = uf.find(members[0])
        comp_finders[root] |= finders[gid]
    for p in list(uf.p):
        comps[uf.find(p)].append(p)
    return [(sorted(v), comp_finders[r]) for r, v in comps.items() if len(v) > 1], known, missing


def from_dups(db: LabelDB, apply: bool) -> None:
    comps, known, missing = dup_components(db)
    _link(db, comps, known, missing, apply)


def link_groups(db: LabelDB, groups, finder: str) -> None:
    """Link groups the instant a finder returns them (groups: lists of images.path). Writes immediately."""
    if finder not in SAME_PERSON_FINDERS:
        return
    known = {r[0]: r[1] for r in db.c.execute("SELECT path, person_id FROM images")}
    uf = UnionFind()
    missing = 0
    for members in groups:
        members = [m for m in members if m in known]
        missing += 0 if members else 1
        for m in members[1:]:
            uf.union(members[0], m)
        if members:
            uf.find(members[0])
    comps = defaultdict(list)
    for p in list(uf.p):
        comps[uf.find(p)].append(p)
    _link(db, [(sorted(v), {finder}) for v in comps.values() if len(v) > 1], known, missing, apply=True)


def _link(db: LabelDB, comps, known, missing, apply: bool) -> None:
    next_id = (db.c.execute("SELECT COALESCE(MAX(person_id),0) FROM images").fetchone()[0] or 0)
    next_id = max(next_id, db.c.execute("SELECT COALESCE(MAX(person_id),0) FROM persons").fetchone()[0] or 0)
    new = merged = relinked = 0
    plan = []                                                  # (person_id, paths, source, old_ids_to_merge)
    for paths, fnd in comps:
        olds = sorted({known[p] for p in paths if known[p] is not None})
        if olds:
            pid = olds[0]
            merged += len(olds) - 1
        else:
            next_id += 1
            pid = next_id
            new += 1
        relinked += sum(known[p] != pid for p in paths)
        plan.append((pid, paths, "dups:" + "+".join(sorted(fnd)), olds[1:]))
    sizes = Counter(min(len(p), 10) for _, p, _, _ in plan)
    print(f"components with 2+ images: {len(plan):,} (new persons {new:,}, merges of old ids {merged:,}); "
          f"images to set: {relinked:,}; report rows not found in images: {missing:,}")
    print("group size histogram (10 = 10 or more):", dict(sorted(sizes.items())))
    if not apply:
        print("dry run - nothing written. Add --apply to write.")
        return
    with db.tx() as c:
        for pid, paths, source, merge_ids in plan:
            for i in range(0, len(paths), 500):
                part = paths[i:i + 500]
                c.execute("UPDATE images SET person_id=? WHERE path IN (%s)" % ",".join("?" * len(part)), [pid] + part)
            for old in merge_ids:
                c.execute("UPDATE images SET person_id=? WHERE person_id=?", (pid, old))
                c.execute("DELETE FROM persons WHERE person_id=?", (old,))
            c.execute("INSERT OR REPLACE INTO persons(person_id, source, n_images) VALUES(?,?,?)", (pid, source, len(paths)))
        c.execute("UPDATE persons SET n_images=(SELECT COUNT(*) FROM images i WHERE i.person_id=persons.person_id)")
    print("written.")


def stats(db: LabelDB) -> None:
    n_img, n_linked = db.c.execute("SELECT COUNT(*), COUNT(person_id) FROM images").fetchone()
    n_per = db.c.execute("SELECT COUNT(DISTINCT person_id) FROM images").fetchone()[0]
    print(f"images {n_img:,}; with person_id {n_linked:,}; distinct persons {n_per:,}; "
          f"images still unclustered (own person) {n_img - n_linked:,}")
    top = db.c.execute("SELECT person_id, COUNT(*) n FROM images WHERE person_id IS NOT NULL "
                       "GROUP BY 1 ORDER BY n DESC LIMIT 5").fetchall()
    print("largest persons:", [tuple(r) for r in top])


def set_person(db: LabelDB, pid: str, images) -> None:
    with db.tx() as c:
        if pid == "new":
            pid = (c.execute("SELECT MAX(COALESCE((SELECT MAX(person_id) FROM images),0),"
                             "COALESCE((SELECT MAX(person_id) FROM persons),0))").fetchone()[0] or 0) + 1
        pid = int(pid)
        n = sum(c.execute("UPDATE images SET person_id=? WHERE path=?", (pid, p)).rowcount for p in images)
        c.execute("INSERT OR IGNORE INTO persons(person_id, source) VALUES(?, 'manual')", (pid,))
        c.execute("UPDATE persons SET n_images=(SELECT COUNT(*) FROM images WHERE person_id=?) WHERE person_id=?", (pid, pid))
    print(f"{n} images -> person {pid}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stats", "from-dups", "set"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    db = LabelDB()
    if a.cmd == "stats":
        stats(db)
    elif a.cmd == "from-dups":
        from_dups(db, a.apply)
    else:
        set_person(db, a.args[0], a.args[1:])
