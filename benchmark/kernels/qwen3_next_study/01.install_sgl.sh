#!/bin/bash
set -e
REPO=${SGLANG_REPO:-/scratch/repo/sglang}

cd "$REPO"
git config --global --add safe.directory "*"
pip install -e python --no-deps

echo "SGLang install done."
