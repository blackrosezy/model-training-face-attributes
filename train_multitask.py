import argparse
import copy
import itertools
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision import datasets, models

try:
    from tqdm.auto import tqdm
except ImportError as err:
    raise SystemExit("Missing dependency tqdm. Install with `pip install tqdm`.") from err

import wandb


# Fixed task definitions that match the provided directory layout.
TASKS = ("age", "emotion", "gender")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-task training (age, emotion, gender) with ONNX export."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Root data directory.")
    parser.add_argument("--batch-size", type=int, default=32, help="Per-task batch size.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for AdamW.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW.")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument(
        "--output-onnx", type=Path, default=Path("multitask_model.onnx"), help="ONNX export path."
    )
    parser.add_argument(
        "--save-checkpoint",
        type=Path,
        default=None,
        help="Optional path to store a PyTorch checkpoint (best val).",
    )
    parser.add_argument(
        "--wandb-project", type=str, default="web-ml-multitask", help="Weights & Biases project name."
    )
    parser.add_argument("--wandb-entity", type=str, default=None, help="Weights & Biases entity.")
    parser.add_argument("--wandb-run", type=str, default=None, help="Optional run name.")
    parser.add_argument(
        "--wandb-mode",
        type=str,
        choices=["online", "offline", "disabled"],
        default="online",
        help="W&B mode for logging progress.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_transforms() -> Tuple[T.Compose, T.Compose]:
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tfms = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(256),
            T.RandomResizedCrop(224, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
        ]
    )
    val_tfms = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalize])
    return train_tfms, val_tfms


def build_dataloaders(
    data_root: Path, batch_size: int, num_workers: int
) -> Tuple[Dict[str, DataLoader], Dict[str, DataLoader], Dict[str, List[str]]]:
    train_tfms, val_tfms = build_transforms()
    train_loaders: Dict[str, DataLoader] = {}
    val_loaders: Dict[str, DataLoader] = {}
    class_names: Dict[str, List[str]] = {}

    for task in TASKS:
        train_dir = data_root / task / "train"
        val_dir = data_root / task / "val"
        if not train_dir.is_dir() or not val_dir.is_dir():
            raise FileNotFoundError(f"Missing train/val folders for task '{task}' under {data_root}")

        train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
        val_ds = datasets.ImageFolder(val_dir, transform=val_tfms)
        class_names[task] = train_ds.classes

        train_loaders[task] = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
        )
        val_loaders[task] = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )

    return train_loaders, val_loaders, class_names


class MultiHeadResNet(nn.Module):
    def __init__(self, num_classes: Dict[str, int]):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.heads = nn.ModuleDict(
            {task: nn.Linear(in_features, classes) for task, classes in num_classes.items()}
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        return {task: head(features) for task, head in self.heads.items()}


def train_epoch(
    model: nn.Module,
    train_loaders: Dict[str, DataLoader],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    global_step: int,
) -> int:
    model.train()
    max_batches = max(len(dl) for dl in train_loaders.values())
    train_iters = {task: iter(dl) for task, dl in train_loaders.items()}
    total_steps = max_batches * len(train_loaders)
    progress = tqdm(total=total_steps, desc=f"Epoch {epoch} [train]", unit="step")

    for step in range(max_batches):
        for task, loader in train_loaders.items():
            try:
                images, labels = next(train_iters[task])
            except StopIteration:
                train_iters[task] = iter(loader)
                images, labels = next(train_iters[task])

            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)[task]
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            global_step += 1

            if wandb.run:
                wandb.log(
                    {f"{task}/train_loss": loss.item(), "lr": optimizer.param_groups[0]["lr"], "step": global_step},
                    step=global_step,
                )

            progress.update(1)

    progress.close()
    print(f"Epoch {epoch} finished: total steps={global_step}")
    return global_step


@torch.no_grad()
def evaluate(
    model: nn.Module, val_loaders: Dict[str, DataLoader], device: torch.device
) -> Tuple[float, Dict[str, Dict[str, float]]]:
    model.eval()
    per_task: Dict[str, Dict[str, float]] = {}
    total_loss = 0.0
    total_batches = 0
    total_steps = sum(len(dl) for dl in val_loaders.values())
    progress = tqdm(total=total_steps, desc="Validation", unit="batch")

    for task, loader in val_loaders.items():
        task_loss = 0.0
        correct = 0
        total = 0
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            logits = model(images)[task]
            loss = F.cross_entropy(logits, labels)
            task_loss += loss.item() * images.size(0)
            total_loss += loss.item()
            total_batches += 1

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            progress.update(1)

        avg_loss = task_loss / max(total, 1)
        acc = correct / max(total, 1)
        per_task[task] = {"loss": avg_loss, "accuracy": acc}

    mean_loss = total_loss / max(total_batches, 1)
    progress.close()
    return mean_loss, per_task


def export_onnx(model: nn.Module, onnx_path: Path, device: torch.device) -> None:
    try:
        import onnx  # noqa: F401
    except ImportError as err:
        raise SystemExit(
            "onnx package not found. Install with `pip install onnx` (or onnxruntime) to export ONNX."
        ) from err

    model = copy.deepcopy(model).to("cpu").eval()
    dummy_input = torch.randn(1, 3, 224, 224, device="cpu")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path.as_posix(),
        input_names=["input"],
        output_names=[f"{task}_logits" for task in TASKS],
        dynamic_axes={"input": {0: "batch"}, **{f"{task}_logits": {0: "batch"} for task in TASKS}},
        opset_version=17,
    )
    print(f"Exported ONNX model to {onnx_path}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loaders, val_loaders, class_names = build_dataloaders(args.data_root, args.batch_size, args.num_workers)
    num_classes = {task: len(names) for task, names in class_names.items()}

    model = MultiHeadResNet(num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    wandb_run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run,
        mode=args.wandb_mode,
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "num_classes": num_classes,
            "backbone": "resnet18_imagenet1k_v1",
        },
    )
    if wandb_run:
        wandb.watch(model, log_freq=50)

    best_val = float("inf")
    best_state = None
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        global_step = train_epoch(model, train_loaders, optimizer, device, epoch, global_step)
        val_loss, metrics = evaluate(model, val_loaders, device)
        scheduler.step()

        if wandb.run:
            log_dict = {f"{task}/val_loss": v["loss"] for task, v in metrics.items()}
            log_dict.update({f"{task}/val_acc": v["accuracy"] for task, v in metrics.items()})
            log_dict["val/mean_loss"] = val_loss
            log_dict["epoch"] = epoch
            log_dict["step"] = global_step
            wandb.log(log_dict, step=global_step)

        print(f"Epoch {epoch}: val_loss={val_loss:.4f} metrics={metrics}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())

    if best_state:
        model.load_state_dict(best_state)
        if args.save_checkpoint:
            args.save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state": best_state, "class_names": class_names}, args.save_checkpoint)
            print(f"Saved checkpoint to {args.save_checkpoint}")

    export_onnx(model, args.output_onnx, device)
    print("Done. ONNX model is ready for browser inference.")


if __name__ == "__main__":
    main()
