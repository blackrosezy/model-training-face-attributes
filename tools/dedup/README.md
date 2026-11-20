# dedup - pluggable near-duplicate finder

```
tools/dedup/
├── base.py        DuplicateFinder (the contract), DupGroup, Param
├── registry.py    @register + auto-discovery of finders/*.py
├── keeper.py      which image of a group survives (resolution -> file size -> name)
├── store.py       data/labels/duplicates.csv + apply (move to discarded/) / restore
├── pipeline.py    find -> choose keepers -> write the report   (back-end independent)
└── finders/
    └── faiss_embed.py   ResNet embeddings (GPU) + faiss cosine search
tools/dedup_cli.py       find | apply | restore | list      (the GUI calls this)
```

A finder only *finds groups*. Keeper choice, moving files, the report, the GUI page and the
CLI are shared, so every back-end behaves the same for the user.

## Add another back-end (fastdup, imagededup, ...)

1. Create `tools/dedup/finders/<anything>.py` (it is discovered automatically):

```python
from ..base import DupGroup, DuplicateFinder, Param
from ..registry import register

@register
class ImageDedupFinder(DuplicateFinder):
    name = "imagededup"                       # shown in the GUI / --finder
    description = "Perceptual hashing (CPU)."
    requires = ("imagededup",)                # modules that must be importable
    install_hint = "uv pip install imagededup"
    params = {"max_distance": Param(10, "Max Hamming distance.", 0, 64, 1)}   # becomes a GUI input

    def find(self, data_dir, images, progress):
        from imagededup.methods import PHash          # import heavy deps HERE, not at the top
        # ... run the library on [data_dir / i for i in images] ...
        progress("hash", done, total)                 # optional, drives the progress bar
        return [DupGroup(members=[...relative paths...], score=0.99), ...]
```

2. `python tools/dedup_cli.py list` should now show it. Pick it in the GUI "Method" box.

Rules: return paths exactly as given in `images` (relative to `data_dir`), only groups with 2+
members, and never move or delete files yourself.

## Notes on faiss_embed
- Embeddings are cached in `data/cache/`, so changing the threshold re-runs in seconds.
- ImageNet features rate *different* faces 0.92-0.96 cosine; real duplicates are >= 0.98. Do not go below ~0.97.
- Grouping is "star" clustering (each member compared with one anchor) to avoid chains of look-alikes.
