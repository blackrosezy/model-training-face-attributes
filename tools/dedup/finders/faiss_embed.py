"""CNN embeddings (GPU) + exact cosine search with faiss (CPU build)."""
from pathlib import Path
from typing import List

from ..base import DupGroup, DuplicateFinder, Param, ProgressFn
from ..embeddings import embed_images
from ..grouping import star_groups
from ..registry import register

COMMON_PARAMS = {
    "threshold": Param(0.98, "Cosine similarity at or above this counts as a duplicate. ImageNet features rate "
                             "different faces 0.92-0.96, so keep this at 0.98+.", 0.90, 1.00, 0.005),
    "model": Param("resnet18", "torchvision backbone: resnet18 (fast) or resnet50 (more precise)."),
    "batch_size": Param(128, "Images per GPU batch.", 8, 1024, 8),
    "workers": Param(8, "Image-loading processes.", 0, 32, 1),
}


@register
class FaissEmbeddingFinder(DuplicateFinder):
    name = "faiss_embed"
    hidden = True                                  # GUI offers torch_gpu only; kept as a reference back-end
    description = "CNN embeddings on the GPU + faiss cosine search on the CPU."
    requires = ("faiss", "torch", "torchvision", "numpy", "PIL")
    install_hint = "Install with: uv pip install faiss-cpu"
    params = dict(COMMON_PARAMS)

    def find(self, data_dir: Path, images: List[str], progress: ProgressFn) -> List[DupGroup]:
        import faiss

        paths, X = embed_images(data_dir, images, self.p["model"], self.p["batch_size"], self.p["workers"], progress)
        index = faiss.IndexFlatIP(X.shape[1])
        index.add(X)
        thr, step, edges = float(self.p["threshold"]), 4096, []
        for s in range(0, len(X), step):
            lims, D, I = index.range_search(X[s:s + step], thr)
            for q in range(len(lims) - 1):
                i = s + q
                for k in range(lims[q], lims[q + 1]):
                    if int(I[k]) > i:
                        edges.append((i, int(I[k]), float(D[k])))
            progress("search", min(s + step, len(X)), len(X))
        return star_groups(paths, edges)
