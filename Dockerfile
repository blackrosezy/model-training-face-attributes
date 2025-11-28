# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Install Python 3.10 and curl
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    curl \
    && ln -s /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv (lightning-fast pip/venv)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Workdir that will be bind-mounted
WORKDIR /app

# Create venv OUTSIDE /app so bind mount doesn't mask it
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
RUN uv venv "${VIRTUAL_ENV}"

# Install Python dependencies using uv
RUN uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN uv pip install onnxruntime-gpu pillow tqdm onnx wandb

# Copy application files
COPY train_multitask.py .
COPY test_multitask.py .

# Default command (can be overridden)
CMD ["python", "--help"]
