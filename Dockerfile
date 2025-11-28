# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Install Python 3.10 and curl
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    curl \
    && ln -s /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

# Install Python dependencies using uv
RUN uv pip install --system torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN uv pip install --system onnxruntime-gpu pillow tqdm onnx wandb

# Copy application files
COPY train_multitask.py .
COPY test_multitask.py .

# Default command (can be overridden)
CMD ["python", "--help"]
