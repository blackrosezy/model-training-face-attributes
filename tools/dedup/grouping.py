"""Turn similar-pair edges into duplicate groups. Shared by every finder that works on pairs."""
from typing import Iterable, List, Tuple

from .base import DupGroup


def star_groups(paths: List[str], edges: Iterable[Tuple[int, int, float]]) -> List[DupGroup]:
    """Star clustering, NOT connected components: every member must be >= threshold similar to the
    group's anchor. Connected components chain A~B~C~... and merge unrelated faces into huge groups."""
    adj = {}
    for i, j, sim in edges:
        adj.setdefault(i, []).append((j, sim))
        adj.setdefault(j, []).append((i, sim))
    taken, groups = set(), []
    for anchor in sorted(adj, key=lambda n: (-len(adj[n]), n)):          # most-connected images anchor first
        if anchor in taken:
            continue
        near = [(n, s) for n, s in adj[anchor] if n not in taken]
        if not near:
            continue
        ids = [anchor] + [n for n, _ in near]
        taken.update(ids)
        groups.append(DupGroup(sorted(paths[n] for n in ids), min(s for _, s in near)))
    return groups
