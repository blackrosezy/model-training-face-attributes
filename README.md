# Multi-Task Learning: Age, Emotion, and Gender Classification

Multi-task deep learning model that simultaneously predicts age range, emotion, and gender from facial images.

## Model Tasks

- **Age**: 8 classes (01-10, 11-20, 21-30, 31-40, 41-55, 56-65, 66-80, 80plus)
- **Emotion**: 6 classes (angry, fear, happy, neutral, sad, surprise)
- **Gender**: 2 classes (female, male)

## Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support
- Weights & Biases account

## Setup

1. **Build the Docker image:**
   ```bash
   docker-compose build
   ```

2. **Set your Weights & Biases API key:**
   ```bash
   export WANDB_API_KEY=your_api_key_here
   ```

## Data Structure

Organize your dataset as follows:
```
data/
├── age/
│   ├── train/
│   │   ├── 01-10/
│   │   ├── 11-20/
│   │   ├── 21-30/
│   │   ├── 31-40/
│   │   ├── 41-55/
│   │   ├── 56-65/
│   │   ├── 66-80/
│   │   └── 80plus/
│   └── val/
│       └── (same structure)
├── emotion/
│   ├── train/
│   │   ├── angry/
│   │   ├── fear/
│   │   ├── happy/
│   │   ├── neutral/
│   │   ├── sad/
│   │   └── surprise/
│   └── val/
│       └── (same structure)
└── gender/
    ├── train/
    │   ├── female/
    │   └── male/
    └── val/
        └── (same structure)
```

## Usage

### Training

```bash
docker-compose run --rm multitask python train_multitask.py \
    --data-root data \
    --epochs 10 \
    --batch-size 32 \
    --wandb-project multitask \
    --wandb-mode online \
    --output-onnx multitask_model.onnx
```

**Key Parameters:**
- `--data-root`: Root directory containing the data (default: `data`)
- `--epochs`: Number of training epochs (default: `10`)
- `--batch-size`: Batch size per task (default: `32`)
- `--wandb-project`: Weights & Biases project name
- `--wandb-mode`: `online`, `offline`, or `disabled` (default: `online`)
- `--output-onnx`: Path to save ONNX model (default: `multitask_model.onnx`)
- `--lr`: Learning rate (default: `3e-4`)
- `--save-checkpoint`: Optional path to save PyTorch checkpoint

### Testing/Inference

```bash
docker-compose run --rm multitask python test_multitask.py --model multitask_model.onnx
```

Interactive prompt for entering image paths. Example output:
```
age    : 21-30    (p=0.892)
emotion: happy    (p=0.756)
gender : female   (p=0.943)
```
