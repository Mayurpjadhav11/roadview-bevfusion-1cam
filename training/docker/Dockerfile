
# BEVFusion training environment
# Machine: Roadview MIFCOM
# GPU: NVIDIA RTX 2080 Ti
# CUDA 11.3, Python 3.8, PyTorch 1.10.2

FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system tools
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3.8-venv \
    python3-pip \
    build-essential \
    ninja-build \
    git \
    wget \
    curl \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgtk2.0-dev \
    openmpi-bin \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create isolated Python environment
RUN python3.8 -m venv /opt/bevfusion-venv

ENV PATH="/opt/bevfusion-venv/bin:/usr/local/cuda/bin:${PATH}"
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="7.5"
ENV MAX_JOBS=2

# Install compatible Python packaging tools
RUN python -m pip install --no-cache-dir \
    "pip==23.3.2" \
    "setuptools==59.5.0" \
    "wheel==0.41.3"

# Install NumPy
RUN python -m pip install --no-cache-dir \
    "numpy==1.23.5"

# Install CUDA-enabled PyTorch
RUN python -m pip install --no-cache-dir \
    "torch==1.10.2+cu113" \
    "torchvision==0.11.3+cu113" \
    --extra-index-url https://download.pytorch.org/whl/cu113

WORKDIR /workspace

CMD ["/bin/bash"]
