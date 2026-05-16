#!/bin/bash

ENV_NAME="dt2119"
ENV_PATH="/home/jovyan/conda-envs/$ENV_NAME"
REQUIREMENTS="$(dirname "$0")/requirements.txt"

if command -v conda &> /dev/null; then
    if conda env list | grep -q "$ENV_PATH"; then
        echo "Conda env '$ENV_NAME' already exists, activating..."
    else
        echo "Creating conda env '$ENV_NAME' at $ENV_PATH..."
        conda create -y -p "$ENV_PATH" python=3.11
    fi

    source activate "$ENV_PATH"

    if [ $? -eq 0 ]; then
        echo "Installing dependencies into conda env..."
        pip install -r "$REQUIREMENTS"
        echo ""
        echo "Done. To activate in future sessions:"
        echo "  source activate $ENV_PATH"
        exit 0
    fi

    echo "Could not activate conda env, falling back to --user install..."
fi

echo "Installing dependencies with --user..."
pip install --user -r "$REQUIREMENTS"
echo ""
echo "Done."
