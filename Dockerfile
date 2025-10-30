# Airbrush Painter - Containerized training and inference
#
# Base image: CUDA-compatible PyTorch for DGX Spark (Grace Hopper)
# Supports: Training, inference, GUI (optional), CI
#
# Build:
#   docker build -t airbrush_painter:latest .
#
# Run training:
#   docker run --gpus all -v $(pwd)/data:/workspace/data \
#              -v $(pwd)/outputs:/workspace/outputs \
#              airbrush_painter:latest python scripts/train.py --config configs/train.yaml
#
# Run inference:
#   docker run --gpus all -v $(pwd)/data:/workspace/data \
#              -v $(pwd)/gcode_output:/workspace/gcode_output \
#              airbrush_painter:latest python scripts/paint.py \
#              --checkpoint outputs/checkpoints/best.pth \
#              --target data/target_images/cmy_only/hard/sample.png \
#              --output gcode_output/sample/
#
# Run GUI:
#   docker run --gpus all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
#              airbrush_painter:latest python scripts/launch_gui.py

# Base image: Use ARG for architecture flexibility
ARG ARCH=aarch64
ARG CUDA_VERSION=12.4
ARG PYTORCH_VERSION=2.4.0

FROM nvcr.io/nvidia/pytorch:${PYTORCH_VERSION}-py3

# Metadata
LABEL maintainer="Airbrush Painter Team"
LABEL description="AI-powered robotic airbrush painting system"
LABEL version="2.3.0"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Build nvdiffrast from source (CUDA rasterizer)
RUN git clone https://github.com/NVlabs/nvdiffrast.git /tmp/nvdiffrast && \
    cd /tmp/nvdiffrast && \
    pip install --no-cache-dir . && \
    rm -rf /tmp/nvdiffrast

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . /workspace/

# Environment variables for DGX Spark defaults
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDNN_V8_API_ENABLED=1
ENV CUDA_LAUNCH_BLOCKING=0

# Expose ports (MLflow UI)
EXPOSE 5000

# Default entrypoint
CMD ["/bin/bash"]

