#!/bin/bash

ENV_NAME="dt2119"
ENV_PATH="/home/jovyan/conda-envs/$ENV_NAME"

if conda env list | grep -q "$ENV_PATH"; then
    echo "Conda env '$ENV_NAME' already exists, activating..."
else
    echo "Creating conda env '$ENV_NAME' at $ENV_PATH..."
    conda create -y -p "$ENV_PATH" python=3.11
fi

source activate "$ENV_PATH"

echo "Installing dependencies from requirements.txt..."
pip install -r "$(dirname "$0")/requirements.txt"

echo "Done. To activate the env in future sessions:"
echo "  source activate $ENV_PATH"
