FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu24.04

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    python3-full \
    build-essential \
    cmake \
    git \
    wget \
    ninja-build \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create and activate a virtual environment to comply with PEP 668
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies in the virtual environment
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 && \
    pip install --no-cache-dir numpy

# Copy the source code
COPY . /app/

# Build the CUDA extension
RUN python3 setup.py build_ext --inplace

# Set the entrypoint to run the benchmark script
ENTRYPOINT ["python3", "benchmark_softmax.py"]