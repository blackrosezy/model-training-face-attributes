"""Everything on the GPU: CNN embeddings AND the all-pairs cosine search (plain PyTorch, no faiss)."""
from pathlib import Path
from typing import List

from ..base import DupGroup, DuplicateFinder, Param, ProgressFn
from ..embeddings import embed_images
from ..grouping import star_groups
from ..registry import register
from .faiss_embed import COMMON_PARAMS

MAX_EDGES_PER_CHUNK = 5_000_000


@register
class TorchGpuFinder(DuplicateFinder):
    name = "torch_gpu"
    description = "CNN embeddings + all-pairs cosine search, both on the GPU (PyTorch only, fp16)."
    requires = ("torch", "torchvision", "numpy", "PIL")
    install_hint = "PyTorch with CUDA is required."
    params = {**COMMON_PARAMS,
              "chunk": Param(4096, "Rows compared per GPU step (lower it if you run out of VRAM).", 256, 32768, 256)}

    def find(self, data_dir: Path, images: List[str], progress: ProgressFn) -> List[DupGroup]:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("torch_gpu needs a CUDA GPU (torch.cuda.is_available() is False). Use faiss_embed instead.")
        paths, X = embed_images(data_dir, images, self.p["model"], self.p["batch_size"], self.p["workers"], progress)
        E = torch.from_numpy(X).to("cuda", dtype=torch.float16)            # [n, d], rows normalised
        n, thr, step, edges = len(E), float(self.p["threshold"]), int(self.p["chunk"]), []
        cols = torch.arange(n, device="cuda")
        for s in range(0, n, step):
            sim = E[s:s + step] @ E.T                                      # [chunk, n] cosine similarities
            rows = torch.arange(s, min(s + step, n), device="cuda").unsqueeze(1)
            hit = (sim >= thr) & (cols.unsqueeze(0) > rows)                # upper triangle only (j > i)
            idx = hit.nonzero()
            if len(idx) > MAX_EDGES_PER_CHUNK:
                raise RuntimeError(f"{len(idx):,} similar pairs in one step - the threshold {thr} is too low "
                                   f"for this embedding; raise it (try 0.98).")
            if len(idx):
                sims = sim[idx[:, 0], idx[:, 1]].float().cpu().numpy()
                ii = (idx[:, 0] + s).cpu().numpy()
                jj = idx[:, 1].cpu().numpy()
                edges.extend(zip(ii.tolist(), jj.tolist(), sims.tolist()))
            progress("search", min(s + step, n), n)
        return star_groups(paths, edges)
