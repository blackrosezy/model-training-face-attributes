"""Image -> L2-normalised CNN embedding (GPU if available), cached on disk. Shared by embedding-based finders."""
import sys
from pathlib import Path
from typing import List, Tuple

from .base import ProgressFn


def embed_images(data_dir: Path, images: List[str], model: str, batch_size: int, workers: int,
                 progress: ProgressFn) -> Tuple[List[str], "np.ndarray"]:
    """Returns (paths that could be read, float32 array [n, dim], rows L2-normalised).

    Embeddings are cached in data/cache/embeddings_<model>.npz, so changing a threshold is instant.
    """
    import numpy as np
    import torch
    import torchvision.models as tvm
    import torchvision.transforms as T
    from PIL import Image

    cache = data_dir / "cache" / f"embeddings_{model}.npz"
    known = {}
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        known = dict(zip(z["paths"].tolist(), z["vecs"]))
    todo = [i for i in images if i not in known]
    print(f"embeddings: {len(images) - len(todo):,} cached, {len(todo):,} to compute", flush=True)

    if todo:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        weights = getattr(tvm, f"{model.replace('resnet', 'ResNet')}_Weights").IMAGENET1K_V1
        net = getattr(tvm, model)(weights=weights)
        net.fc = torch.nn.Identity()
        net = net.to(device).eval()
        tfm = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        class DS(torch.utils.data.Dataset):
            def __len__(self): return len(todo)
            def __getitem__(self, i):
                try:
                    with Image.open(data_dir / todo[i]) as im:
                        return tfm(im.convert("RGB")), True
                except Exception:
                    return torch.zeros(3, 224, 224), False

        dl = torch.utils.data.DataLoader(DS(), batch_size=batch_size, num_workers=workers,
                                         pin_memory=(device == "cuda"))
        done, bad = 0, 0
        with torch.no_grad():
            for x, ok in dl:
                with torch.autocast(device, enabled=(device == "cuda")):
                    f = net(x.to(device, non_blocking=True))
                f = torch.nn.functional.normalize(f.float(), dim=1).cpu().numpy()
                for j in range(len(f)):
                    if ok[j]:
                        known[todo[done + j]] = f[j].astype(np.float16)
                    else:
                        bad += 1
                        print(f"WARN cannot read {todo[done + j]}", file=sys.stderr, flush=True)
                done += len(f)
                progress("embed", done, len(todo))
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, paths=np.array(list(known)), vecs=np.stack(list(known.values())))
        if bad:
            print(f"{bad} unreadable image(s) skipped", flush=True)

    paths = [i for i in images if i in known]
    X = np.stack([known[i] for i in paths]).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True).clip(1e-9)
    return paths, X
