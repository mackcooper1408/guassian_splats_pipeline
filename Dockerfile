# Gaussian Splatting Environment Dockerfile
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install the version of cmake required by colmap
RUN apt-get update && apt-get install -y \
    gpg wget software-properties-common \
    && wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor -o /usr/share/keyrings/kitware-archive-keyring.gpg \
    && echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main' | tee /etc/apt/sources.list.d/kitware.list >/dev/null \
    && apt-get update && apt-get install -y \
    cmake

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev \
    python3.10 \
    python3-pip \
    python3-dev \
    ffmpeg \
    xvfb \
    wget \
    libopenimageio-dev \
    openimageio-tools \
    libopenexr-dev \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Install COLMAP
RUN git clone --branch 3.10 https://github.com/colmap/colmap.git /tmp/colmap && \
    cd /tmp/colmap && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89" && \
    make -j2 && \
    make install && \
    rm -rf /tmp/colmap

# Install GLOMAP (faster reconstruction than COLMAP)
RUN git clone https://github.com/colmap/glomap.git /tmp/glomap && \
    cd /tmp/glomap && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89" && \
    make -j2 && \
    make install && \
    rm -rf /tmp/glomap

# Set up Python environment
WORKDIR /workspace
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA support
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Clone and setup Gaussian Splatting repository
RUN git clone https://github.com/graphdeco-inria/gaussian-splatting /workspace/gaussian-splatting --recursive

# Install Gaussian Splatting dependencies
WORKDIR /workspace/gaussian-splatting
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# graphdeco CUDA submodules (diff-gaussian-rasterization, simple-knn) must be
# compiled on a machine with a real NVIDIA GPU + matching CUDA runtime.
# The entrypoint script compiles them on first launch (~2 min).
COPY entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

# Set working directory back to main workspace
WORKDIR /workspace/project

# Expose port for viewers if needed
EXPOSE 6006

ENTRYPOINT ["/workspace/entrypoint.sh"]
CMD ["/bin/bash"]
