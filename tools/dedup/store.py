"""The duplicates report (table `dups` in the label database) and the actions on it: apply / restore."""
import shutil
import sys
from pathlib import Path
from typing import Callable, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from labeldb import LabelDB  # noqa: E402

FIELDS = ["group_id", "image", "keeper", "score", "finder", "status", "moved_to", "width", "height", "bytes"]
# status: pending -> moved -> restored ;  keeper rows: pending -> kept


def read_report(data_dir: Path = None) -> List[Dict[str, str]]:
    return LabelDB().dups_all()


def write_report(data_dir: Path, rows: List[Dict[str, str]]) -> None:
    LabelDB().dups_replace(rows)


def set_keeper(data_dir: Path, group_id: str, image: str) -> None:
    rows = read_report()
    for r in rows:
        if r["group_id"] == group_id and r["status"] == "pending":
            r["keeper"] = "1" if r["image"] == image else "0"
    write_report(data_dir, rows)


def _sync_db(mapping: Dict[str, str]) -> None:
    LabelDB().move_paths(mapping)


def _move_files(data_dir: Path, rows: List[Dict[str, str]], pairs, done_status: str,
                progress: Callable[[str, int, int], None]) -> int:
    """pairs: list of (row, src_rel, dst_rel). Moves files, records status, syncs csvs even on failure."""
    mapping: Dict[str, str] = {}
    try:
        for n, (row, src, dst) in enumerate(pairs, 1):
            if (data_dir / src).exists():
                (data_dir / dst).parent.mkdir(parents=True, exist_ok=True)
                shutil.move(data_dir / src, data_dir / dst)
                mapping[src] = dst
                row["status"] = done_status
            else:
                row["status"] = "missing"
            if n % 25 == 0 or n == len(pairs):
                progress("move", n, len(pairs))
            if n % 500 == 0:
                _sync_db(mapping); write_report(data_dir, rows); mapping = {}
    finally:
        _sync_db(mapping)
        write_report(data_dir, rows)
    return len(pairs)


def apply(data_dir: Path, progress: Callable[[str, int, int], None]) -> int:
    """Move every pending non-keeper to discarded/<source>/<file>; keepers stay in raw/."""
    rows = read_report()
    pairs = []
    for r in rows:
        if r["status"] != "pending":
            continue
        if r["keeper"] == "1":
            r["status"] = "kept"
        else:
            new = "discarded/" + r["image"].split("/", 1)[1]
            r["moved_to"] = new
            pairs.append((r, r["image"], new))
    return _move_files(data_dir, rows, pairs, "moved", progress)


def restore(data_dir: Path, progress: Callable[[str, int, int], None]) -> int:
    """Put every duplicate moved by apply() back into raw/."""
    rows = read_report()
    pairs = [(r, r["moved_to"], r["image"]) for r in rows if r["status"] == "moved"]
    return _move_files(data_dir, rows, pairs, "restored", progress)
