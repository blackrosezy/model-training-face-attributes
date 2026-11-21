"""SQLite store for image labels (replaces master.csv / vlm_labels.csv / duplicates.csv).

The database file lives on the fast WSL disk (default ~/.local/share/face-labeling/labels.db,
override with env FACE_LABELS_DB): SQLite on /mnt/d is ~200x slower and unreliable. The images
themselves stay in data/. CSV copies can be written any time with:  python tools/labeldb.py export

  python tools/labeldb.py migrate   # one-time import of the old CSV files
  python tools/labeldb.py export    # write CSVs to data/labels/export/
  python tools/labeldb.py stats
"""
import csv
import json
import os
import sqlite3
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Optional

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
DEFAULT_DB = Path.home() / ".local" / "share" / "face-labeling" / "labels.db"
FIELDS = ("age", "emotion", "gender")

SCHEMA = """
CREATE TABLE IF NOT EXISTS images(
  path TEXT PRIMARY KEY, loc TEXT NOT NULL, source TEXT NOT NULL,
  legacy_path TEXT, legacy_task TEXT, legacy_split TEXT, legacy_label TEXT,
  age TEXT, emotion TEXT, gender TEXT, overlay TEXT,
  valid INTEGER,            -- NULL = not labeled yet, 1 = ok, 0 = failed
  raw TEXT, seq INTEGER     -- seq = order in which the image was labeled
);
CREATE INDEX IF NOT EXISTS idx_images_view ON images(loc, valid, seq);
CREATE TABLE IF NOT EXISTS dups(
  group_id INTEGER, image TEXT, keeper INTEGER, score REAL, finder TEXT, status TEXT,
  moved_to TEXT, width INTEGER, height INTEGER, bytes INTEGER, PRIMARY KEY(group_id, image));
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v INTEGER);
CREATE TABLE IF NOT EXISTS scan_state(      -- folder mtimes seen by tools/scan.py (lets a rescan skip unchanged folders)
  dir TEXT PRIMARY KEY, mtime_ns INTEGER NOT NULL, subdirs TEXT NOT NULL);
INSERT OR IGNORE INTO meta VALUES('rev', 0);
INSERT OR IGNORE INTO meta VALUES('seq', 0);
"""


def loc_of(path: str) -> str:
    return path.split("/", 1)[0]


class LabelDB:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path or os.environ.get("FACE_LABELS_DB") or DEFAULT_DB)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.c = sqlite3.connect(self.path, timeout=30)
        self.c.row_factory = sqlite3.Row
        self.c.execute("PRAGMA journal_mode=WAL")
        self.c.execute("PRAGMA synchronous=NORMAL")
        self.c.executescript(SCHEMA)

    # ---- plumbing ----
    @contextmanager
    def tx(self):
        """One write transaction; bumps the revision counter so the GUI knows something changed."""
        self.c.execute("BEGIN IMMEDIATE")
        try:
            yield self.c
            self.c.execute("UPDATE meta SET v=v+1 WHERE k='rev'")
            self.c.execute("COMMIT")
        except BaseException:
            self.c.execute("ROLLBACK")
            raise

    def rev(self) -> int:
        return self.c.execute("SELECT v FROM meta WHERE k='rev'").fetchone()[0]

    # ---- images ----
    def add_images(self, rows: Iterable[Dict]) -> None:
        with self.tx() as c:
            c.executemany(
                "INSERT OR IGNORE INTO images(path,loc,source,legacy_path,legacy_task,legacy_split,legacy_label)"
                " VALUES(:path,:loc,:source,:legacy_path,:legacy_task,:legacy_split,:legacy_label)", list(rows))

    def known_paths(self, loc: str = "raw") -> set:
        return {r[0] for r in self.c.execute("SELECT path FROM images WHERE loc=?", (loc,))}

    def scan_state(self) -> Dict[str, tuple]:
        return {r[0]: (r[1], json.loads(r[2])) for r in self.c.execute("SELECT dir, mtime_ns, subdirs FROM scan_state")}

    def scan_commit(self, new_rows: List[Dict], state: Dict[str, tuple]) -> None:
        """Insert newly found images and remember the folder mtimes in ONE transaction."""
        with self.tx() as c:
            if new_rows:
                c.executemany(
                    "INSERT OR IGNORE INTO images(path,loc,source,legacy_path,legacy_task,legacy_split,legacy_label)"
                    " VALUES(:path,:loc,:source,:legacy_path,:legacy_task,:legacy_split,:legacy_label)", new_rows)
            c.executemany("INSERT OR REPLACE INTO scan_state VALUES(?,?,?)",
                          [(d, m, json.dumps(s)) for d, (m, s) in state.items()])

    def candidates(self, loc: str = "raw", only_unlabeled: bool = True) -> List[str]:
        q = "SELECT path FROM images WHERE loc=?" + (" AND (valid IS NULL OR valid=0)" if only_unlabeled else "")
        return [r[0] for r in self.c.execute(q + " ORDER BY path", (loc,))]

    def set_labels(self, rows: Iterable[Dict]) -> None:
        """rows: path, age, emotion, gender, overlay, valid (0/1), raw."""
        with self.tx() as c:
            for r in rows:
                seq = None
                if r["valid"]:
                    c.execute("UPDATE meta SET v=v+1 WHERE k='seq'")
                    seq = c.execute("SELECT v FROM meta WHERE k='seq'").fetchone()[0]
                c.execute("UPDATE images SET age=?,emotion=?,gender=?,overlay=?,valid=?,raw=?,seq=COALESCE(seq,?)"
                          " WHERE path=?", (r["age"], r["emotion"], r["gender"], r["overlay"], int(r["valid"]),
                                            r["raw"], seq, r["path"]))

    def reset_labels(self, loc: str = "raw") -> int:
        """Erase all VLM labels of the images in `loc` (back to 'not labeled yet'). Files are not touched."""
        with self.tx() as c:
            n = c.execute("UPDATE images SET age=NULL,emotion=NULL,gender=NULL,overlay=NULL,valid=NULL,raw=NULL,seq=NULL"
                          " WHERE loc=? AND valid IS NOT NULL", (loc,)).rowcount
        return n

    def unlabel(self, paths: List[str]) -> int:
        """Erase the labels of these images (back to 'not labeled yet'), e.g. to verify them again."""
        n = 0
        with self.tx() as c:
            for i in range(0, len(paths), 500):
                part = paths[i:i + 500]
                n += c.execute("UPDATE images SET age=NULL,emotion=NULL,gender=NULL,overlay=NULL,valid=NULL,raw=NULL,"
                               "seq=NULL WHERE path IN (%s)" % ",".join("?" * len(part)), part).rowcount
        return n

    def move_paths(self, mapping: Dict[str, str]) -> None:
        if not mapping:
            return
        with self.tx() as c:
            c.executemany("UPDATE images SET path=?, loc=? WHERE path=?",
                          [(new, loc_of(new), old) for old, new in mapping.items()])

    # ---- queries used by the GUI ----
    @staticmethod
    def _where(loc: str, f: Dict[str, str]):
        # kept: labeled images only. discarded: EVERYTHING in the discarded/ folder (labeled or not, dedup-moved too).
        if loc == "failed":                           # kept images the VLM could not label (valid=0), to review later
            sql, args = "loc='raw' AND valid=0", []
        else:
            sql, args = ("loc=? AND valid=1", [loc]) if loc != "discarded" else ("loc=?", [loc])
        for key in ("age", "emotion", "gender", "source", "overlay"):
            if f.get(key) and f[key] != "All":
                sql += f" AND {key}=?"; args.append(f[key])
        return sql, args

    def count(self, loc: str, f: Dict[str, str]) -> int:
        sql, args = self._where(loc, f)
        return self.c.execute(f"SELECT COUNT(*) FROM images WHERE {sql}", args).fetchone()[0]

    def page(self, loc: str, f: Dict[str, str], limit: int, offset: int) -> List[Dict]:
        sql, args = self._where(loc, f)
        q = f"SELECT * FROM images WHERE {sql} ORDER BY seq, path LIMIT ? OFFSET ?"
        return [dict(r) for r in self.c.execute(q, args + [limit, offset])]

    def sources(self, loc: str) -> List[str]:
        if loc == "failed":
            where, args = "loc='raw' AND valid=0", ()
        else:
            where, args = "loc=?" + (" AND valid=1" if loc != "discarded" else ""), (loc,)
        return [r[0] for r in self.c.execute(f"SELECT DISTINCT source FROM images WHERE {where} ORDER BY 1", args)]

    def distribution(self, field: str, kept_only: bool) -> Counter:
        where = "valid=1" + (" AND loc='raw'" if kept_only else "")
        return Counter({r[0]: r[1] for r in self.c.execute(f"SELECT {field}, COUNT(*) FROM images WHERE {where} GROUP BY 1")})

    def totals(self) -> Dict[str, int]:
        """Discarded images are forgotten: `total`, `labeled` and `unlabeled` only look at kept (raw/) images,
        so total = labeled + unlabeled. `discarded` is just the size of the discarded/ folder."""
        total, labeled, failed, disc = self.c.execute(
            """SELECT COALESCE(SUM(loc='raw'),0), COALESCE(SUM(loc='raw' AND valid=1),0),
                      COALESCE(SUM(loc='raw' AND valid=0),0), COALESCE(SUM(loc='discarded'),0) FROM images""").fetchone()
        return {"total": total, "labeled": labeled, "kept": labeled, "discarded": disc, "failed": failed,
                "unlabeled": total - labeled}

    def get(self, path: str) -> Optional[Dict]:
        r = self.c.execute("SELECT * FROM images WHERE path=?", (path,)).fetchone()
        return dict(r) if r else None

    # ---- duplicates report ----
    def dups_all(self) -> List[Dict[str, str]]:
        """Same shape the old duplicates.csv had (all values as strings)."""
        out = []
        for r in self.c.execute("SELECT * FROM dups ORDER BY group_id, keeper DESC, image"):
            d = {k: ("" if r[k] is None else str(r[k])) for k in r.keys()}
            d["score"] = f"{r['score']:.4f}"
            out.append(d)
        return out

    def dups_replace(self, rows: List[Dict]) -> None:
        with self.tx() as c:
            c.execute("DELETE FROM dups")
            c.executemany("INSERT INTO dups VALUES(?,?,?,?,?,?,?,?,?,?)", [
                (int(r["group_id"]), r["image"], int(r["keeper"]), float(r["score"]), r["finder"], r["status"],
                 r.get("moved_to", ""), int(r.get("width") or 0), int(r.get("height") or 0), int(r.get("bytes") or 0))
                for r in rows])

    def dups_pending_count(self) -> int:
        return self.c.execute("SELECT COUNT(*) FROM dups WHERE status='pending' AND keeper=0").fetchone()[0]

    # ---- CSV import / export ----
    def migrate_csv(self, data_dir: Path = DATA) -> Dict[str, int]:
        lab = data_dir / "labels"
        n = {"images": 0, "labels": 0, "dups": 0}
        with open(lab / "master.csv", newline="") as f:
            rows = [{"path": r["image"], "loc": loc_of(r["image"]), "source": r["source"],
                     "legacy_path": r.get("legacy_path", ""), "legacy_task": r["legacy_task"],
                     "legacy_split": r["legacy_split"], "legacy_label": r["legacy_label"]} for r in csv.DictReader(f)]
        self.add_images(rows); n["images"] = len(rows)
        if (lab / "vlm_labels.csv").exists():
            with open(lab / "vlm_labels.csv", newline="") as f:
                ls = [{"path": r["image"], "age": r["vlm_age"], "emotion": r["vlm_emotion"], "gender": r["vlm_gender"],
                       "overlay": r.get("vlm_overlay") or "none", "valid": r["valid"] == "1", "raw": r["raw"]}
                      for r in csv.DictReader(f)]
            self.set_labels(ls); n["labels"] = len(ls)
        if (lab / "duplicates.csv").exists():
            with open(lab / "duplicates.csv", newline="") as f:
                ds = list(csv.DictReader(f))
            self.dups_replace(ds); n["dups"] = len(ds)
        return n

    def export_csv(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, query in (("images.csv", "SELECT * FROM images ORDER BY path"), ("duplicates.csv", "SELECT * FROM dups")):
            cur = self.c.execute(query)
            with open(out_dir / name, "w", newline="") as f:
                w = csv.writer(f); w.writerow([d[0] for d in cur.description]); w.writerows(cur)


def open_db(path: Optional[Path] = None) -> LabelDB:
    return LabelDB(path)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "stats"
    db = LabelDB()
    if cmd == "migrate":
        print(db.migrate_csv(), "->", db.path)
    elif cmd == "export":
        db.export_csv(DATA / "labels" / "export"); print("written to", DATA / "labels" / "export")
    else:
        print(db.path, db.totals(), "pending dups:", db.dups_pending_count())
