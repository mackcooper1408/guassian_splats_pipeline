#!/bin/bash
# Compile graphdeco CUDA submodules on first launch.
# These require a real NVIDIA GPU + CUDA runtime, so they can't be built
# during `docker build` on a Mac or CI without a GPU.

MARKER="/workspace/gaussian-splatting/.submodules_installed"

if [ ! -f "$MARKER" ]; then
    echo "First launch: compiling CUDA submodules (diff-gaussian-rasterization, simple-knn)..."
    pip3 install /workspace/gaussian-splatting/submodules/diff-gaussian-rasterization \
                 /workspace/gaussian-splatting/submodules/simple-knn \
        && touch "$MARKER" \
        && echo "CUDA submodules compiled successfully." \
        || { echo "ERROR: Failed to compile CUDA submodules."; exit 1; }
fi

exec "$@"
