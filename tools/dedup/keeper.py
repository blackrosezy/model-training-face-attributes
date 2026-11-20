"""Which image of a duplicate group survives. Swap `pick_keeper` for a different policy if you like."""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image


def describe(data_dir: Path, rel: str) -> Tuple[int, int, int]:
    """(width, height, bytes) - reads only the image header."""
    path = data_dir / rel
    try:
        with Image.open(path) as im:
            w, h = im.size
    except Exception:
        w = h = 0
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    return w, h, size


def pick_keeper(data_dir: Path, members: List[str], info: Optional[Dict[str, Tuple[int, int, int]]] = None) -> str:
    """Highest resolution, then largest file, then smallest name (deterministic).
    `info` = {path: (w, h, bytes)} if already read, so files are not opened twice."""
    def key(rel):
        w, h, size = info[rel] if info else describe(data_dir, rel)
        return (-(w * h), -size, rel)
    return sorted(members, key=key)[0]
