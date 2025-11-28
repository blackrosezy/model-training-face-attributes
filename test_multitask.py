import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import onnxruntime as ort
from PIL import Image
import torchvision.transforms as T
import time

START_TIME = time.perf_counter()


# Class names based on the provided folder structure; adjust if yours differ.
CLASS_NAMES: Dict[str, List[str]] = {
    "age": ["01-10", "11-20", "21-30", "31-40", "41-55", "56-65", "66-80", "80plus"],
    "emotion": ["angry", "fear", "happy", "neutral", "sad", "surprise"],
    "gender": ["female", "male"],
}


def get_session(model_path: Path) -> ort.InferenceSession:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    now = time.perf_counter()
    print(f"[debug] Starting session creation at {(now - START_TIME)*1000:.1f} ms since start")
    t0 = time.perf_counter()
    session = ort.InferenceSession(model_path.as_posix(), providers=providers)
    print(
        f"[debug] Loaded ONNX in {(time.perf_counter() - t0)*1000:.1f} ms "
        f"with providers={session.get_providers()} device={ort.get_device()}"
    )
    return session


def build_transform() -> T.Compose:
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            normalize,
        ]
    )


def preprocess(image_path: Path, transform: T.Compose) -> np.ndarray:
    t0 = time.perf_counter()
    img = Image.open(image_path).convert("RGB")
    orig_size = img.size
    tensor = transform(img).unsqueeze(0)  # (1, 3, 224, 224)
    print(
        f"[debug] Preprocess time: {(time.perf_counter() - t0)*1000:.1f} ms "
        f"(original size={orig_size}, resized->224x224)"
    )
    return tensor.numpy().astype(np.float32)


def warmup_session(session: ort.InferenceSession) -> None:
    dummy = np.zeros((1, 3, 224, 224), dtype=np.float32)
    t0 = time.perf_counter()
    session.run(None, {"input": dummy})
    print(f"[debug] Warmup inference time: {(time.perf_counter() - t0)*1000:.1f} ms")


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=-1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="ONNX inference REPL for age/emotion/gender.")
    parser.add_argument("--model", type=Path, default=Path("multitask_model.onnx"), help="Path to ONNX model.")
    args = parser.parse_args()

    print(f"[debug] Script start -> main entry: {(time.perf_counter() - START_TIME)*1000:.1f} ms")

    transform = build_transform()
    print(f"[debug] Transform built at {(time.perf_counter() - START_TIME)*1000:.1f} ms")
    session = get_session(args.model)
    warmup_session(session)

    while True:
        try:
            path_str = input("\nEnter image path (or Ctrl+C to exit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not path_str:
            print("No path provided, try again.")
            continue

        image_path = Path(path_str)
        if not image_path.exists():
            print(f"Path not found: {image_path}")
            continue

        inp = preprocess(image_path, transform)
        t0 = time.perf_counter()
        outputs = session.run(None, {"input": inp})
        print(f"[debug] Inference time: {(time.perf_counter() - t0)*1000:.1f} ms")

        output_names = [o.name for o in session.get_outputs()]
        results = dict(zip(output_names, outputs))

        for task in ("age", "emotion", "gender"):
            logits = results[f"{task}_logits"][0]
            probs = softmax(logits)
            top_idx = int(probs.argmax())
            class_name = CLASS_NAMES[task][top_idx] if task in CLASS_NAMES else str(top_idx)
            print(f"{task:7s}: {class_name:>8s} (p={probs[top_idx]:.3f})")


if __name__ == "__main__":
    main()
