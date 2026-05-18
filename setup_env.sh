#!/bin/bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Run 'source setup_env.sh' — activation cannot propagate from a subshell."
    exit 1
fi

ENV_NAME="dt2119"
ENV_PATH="/home/jovyan/conda-envs/$ENV_NAME"
REQUIREMENTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/requirements.txt"

source /opt/conda/etc/profile.d/conda.sh

# Use the CUDA async allocator. The default PyTorch caching allocator queries NVML
# for memory info which fails (INTERNAL ASSERT) on MIG-partitioned GPUs.
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync

if conda env list | grep -q "$ENV_PATH"; then
    conda activate "$ENV_PATH"
    echo "Activated '$ENV_NAME'."
else
    echo "Creating conda env '$ENV_NAME' at $ENV_PATH..."
    conda create -y -p "$ENV_PATH" python=3.11

    conda activate "$ENV_PATH"

    echo "Installing PyTorch with CUDA 12.4 support..."
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

    echo "Installing remaining dependencies..."
    pip install -r "$REQUIREMENTS"

    echo "Done. Env '$ENV_NAME' is ready."
fi
