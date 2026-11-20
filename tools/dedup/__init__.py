"""Near-duplicate image detection with swappable back-ends ("finders").

    from dedup import registry
    finder = registry.create("faiss_embed", threshold=0.95)

See README.md in this folder for how to add another vendor (fastdup, imagededup, ...).
"""
from .base import DupGroup, DuplicateFinder, Param  # noqa: F401
