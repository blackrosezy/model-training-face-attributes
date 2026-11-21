"""Same-PERSON finder for FRONTAL faces: InsightFace ArcFace identity embeddings (GPU).

Only frontal faces are compared. A person's frontal photos are grouped; from each group a few photos that look
different from each other (expression, glasses, lighting...) survive and the near-identical rest are duplicates.
Faces turned away (3/4 view, profile) and images with no detectable face are never grouped, so a duplicate at
another angle is always kept.

Embeddings + yaw are cached in data/cache/face_id2_<model>.npz, so changing a threshold re-runs in seconds."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

from ..base import DupGroup, DuplicateFinder, Param, ProgressFn
from ..grouping import star_groups
from ..registry import register

PAD = 80                     # black border so detectors see tight crops as a "whole" image
SAVE_EVERY = 5000
BATCH = 64                   # images decoded ahead of the GPU at any moment (memory bound)
READERS = 4                  # image-reading threads


def _yaw_ratio(kps) -> float:
    """Head-turn proxy from the 5 detector landmarks: nose offset from the eye midpoint / eye distance.
    ~0 = frontal; grows as the head turns. (Not degrees; see the `frontal_max` parameter.)"""
    (lx, _), (rx, _), (nx, _) = kps[0], kps[1], kps[2]
    d = float(rx - lx)
    return abs((float(nx) - (float(lx) + float(rx)) / 2) / d) if d > 1e-3 else 9.0


def _embed(data_dir: Path, images: List[str], model: str, det_size: int, progress: ProgressFn):
    import numpy as np
    import torch  # noqa: F401  (import first: puts torch's bundled CUDA libs on the loader path for onnxruntime)
    import cv2
    cv2.setNumThreads(1)                                   # don't let OpenCV spawn a thread per core on top of ours
    from insightface.app import FaceAnalysis

    cache = data_dir / "cache" / f"face_id2_{model}.npz"
    known = {}                                             # rel -> (vec512 float16, yaw_ratio); zero vec = no face
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        known = {p: (v, float(y)) for p, v, y in zip(z["paths"].tolist(), z["vecs"], z["yaw"])}
    todo = [i for i in images if i not in known]
    print(f"face_id: {len(images) - len(todo):,} cached, {len(todo):,} to compute", flush=True)

    def save():
        cache.parent.mkdir(parents=True, exist_ok=True)
        ks = list(known)
        tmp = cache.with_suffix(".tmp.npz")
        np.savez(tmp, paths=np.array(ks), vecs=np.stack([known[k][0] for k in ks]).astype("float16"),
                 yaw=np.array([known[k][1] for k in ks], dtype="float32"))
        tmp.replace(cache)

    if todo:
        app = FaceAnalysis(name=model, allowed_modules=["detection", "recognition"],
                           providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(det_size, det_size))

        def load(rel):
            im = cv2.imread(str(data_dir / rel))
            return rel, None if im is None else cv2.copyMakeBorder(im, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT)

        zero = np.zeros(512, dtype="float16")
        n = 0
        # Bounded read-ahead: ex.map(todo) would queue EVERY image and decoded frames pile up in RAM faster
        # than the GPU loop consumes them (tens of GB -> WSL hangs). Only BATCH images are ever in memory.
        with ThreadPoolExecutor(READERS) as ex:
            for b in range(0, len(todo), BATCH):
                for rel, im in ex.map(load, todo[b:b + BATCH]):
                    faces = app.get(im) if im is not None else []
                    if faces:
                        f = max(faces, key=lambda f: f.det_score)
                        known[rel] = (f.normed_embedding.astype("float16"), _yaw_ratio(f.kps))
                    else:
                        known[rel] = (zero, 9.0)
                    n += 1
                if n % SAVE_EVERY < BATCH:
                    save()
                progress("embed", n, len(todo))
        save()

    paths = [i for i in images if i in known and known[i][0].any()]
    print(f"face_id: {len(images) - len(paths):,} images without a detectable face are skipped", flush=True)
    return (paths, np.stack([known[p][0] for p in paths]).astype("float32"),
            np.array([known[p][1] for p in paths], dtype="float32"))


def _thumb_and_sharpness(path: Path):
    """48x48 grayscale thumbnail (for 'looks different') + sharpness (for 'best quality')."""
    import cv2
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None, 0.0
    g = cv2.resize(im, (96, 96), interpolation=cv2.INTER_AREA)
    return cv2.resize(g, (48, 48), interpolation=cv2.INTER_AREA).astype("float32").ravel() / 255.0, \
        float(cv2.Laplacian(g, cv2.CV_32F).var())


def pick_diverse(data_dir: Path, members: List[str], k: int) -> List[str]:
    """k survivors: the sharpest image first, then repeatedly the one least like those already kept.
    'Least alike' uses the picture itself (not the identity vector, which is built to ignore expression/glasses)."""
    import numpy as np
    thumbs, sharp, ok = [], [], []
    for m in members:
        t, s = _thumb_and_sharpness(data_dir / m)
        if t is not None:
            thumbs.append(t); sharp.append(s); ok.append(m)
    if len(ok) <= k:
        return ok
    X = np.stack(thumbs)
    kept = [int(np.argmax(sharp))]
    dist = np.linalg.norm(X - X[kept[0]], axis=1)                  # distance to the nearest kept image
    while len(kept) < k:
        nxt = int(np.argmax(dist))
        kept.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(X - X[nxt], axis=1))
    return [ok[i] for i in kept]


@register
class FaceIdFinder(DuplicateFinder):
    name = "face_id"
    slow = True                                    # ~35 img/s (rough): a full run over the dataset takes hours
    description = ("Same person, frontal faces only: ArcFace identity embeddings (InsightFace) + GPU search. "
                   "Keeps a few different-looking photos per person; other angles are never touched.")
    requires = ("insightface", "onnxruntime", "torch", "cv2", "numpy")
    install_hint = "uv pip install insightface onnxruntime-gpu"
    params = {
        "threshold": Param(0.6, "Cosine similarity at or above this = same person (frontal pairs score higher "
                                "than angled ones; 0.6 is a starting point, not yet checked on real groups).",
                           0.2, 0.95, 0.01),
        "keep": Param(3, "Photos kept per person (chosen to look different from each other); the rest are duplicates.",
                      1, 50, 1),
        "frontal_max": Param(0.25, "A face counts as frontal if its head-turn ratio (nose offset / eye distance) is "
                                   "at most this. Bigger = also compares slightly turned faces.", 0.05, 1.0, 0.05),
        "model": Param("buffalo_l", "InsightFace pack: buffalo_l (accurate), buffalo_s (faster)."),
        "det_size": Param(320, "Detector input size.", 160, 640, 32),
        "chunk": Param(4096, "Rows compared per GPU step.", 256, 32768, 256),
    }

    def find(self, data_dir: Path, images: List[str], progress: ProgressFn) -> List[DupGroup]:
        import numpy as np
        import torch

        paths, X, yaw = _embed(data_dir, images, self.p["model"], int(self.p["det_size"]), progress)
        front = yaw <= float(self.p["frontal_max"])
        paths, X = [p for p, f in zip(paths, front) if f], X[front]
        print(f"face_id: {len(paths):,} frontal faces compared; {int((~front).sum()):,} turned faces left alone",
              flush=True)
        if len(paths) < 2:
            return []

        E = torch.from_numpy(X).to("cuda", dtype=torch.float16)
        n, thr, step, edges = len(E), float(self.p["threshold"]), int(self.p["chunk"]), []
        cols = torch.arange(n, device="cuda")
        for s in range(0, n, step):
            sim = E[s:s + step] @ E.T
            rows = torch.arange(s, min(s + step, n), device="cuda").unsqueeze(1)
            idx = ((sim >= thr) & (cols.unsqueeze(0) > rows)).nonzero()
            if len(idx):
                sims = sim[idx[:, 0], idx[:, 1]].float().cpu().numpy()
                edges.extend(zip((idx[:, 0] + s).cpu().tolist(), idx[:, 1].cpu().tolist(), sims.tolist()))
            progress("search", min(s + step, n), n)

        k = int(self.p["keep"])
        groups = [g for g in star_groups(paths, edges) if len(g.members) > k]      # nothing to remove otherwise
        for i, g in enumerate(groups, 1):
            g.keep = pick_diverse(data_dir, g.members, k)
            if i % 20 == 0 or i == len(groups):
                progress("diversify", i, len(groups))
        return groups
