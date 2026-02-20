#!/bin/bash
set -e
REPO=${FLASHINFER_REPO:-/scratch/repo/flashinfer}

cd "$REPO"

# --no-deps: deps already in base env
# --no-build-isolation: build needs torch/cuda from current env
pip install -e . --no-deps --no-build-isolation

# Disable version mismatch check between Python package and cubin
export FLASHINFER_DISABLE_VERSION_CHECK=1
echo "export FLASHINFER_DISABLE_VERSION_CHECK=1  # add this to your shell if needed"

echo "FlashInfer install done."
pip list | grep flashinfer
