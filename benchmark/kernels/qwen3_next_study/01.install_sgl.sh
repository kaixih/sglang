#!/bin/bash
set -e
REPO=${SGLANG_REPO:-/scratch/repo/sglang}

cd "$REPO"
git config --global --add safe.directory "*"

# Build and install sgl-kernel (CUDA, needs --no-build-isolation)
if [[ "$1" == "kernel" || "$1" == "all" ]]; then
  export CCACHE_DIR=/scratch/cache/sglang
  export CCACHE_BACKEND=""
  export CCACHE_KEEP_LOCAL_STORAGE="TRUE"
  unset CCACHE_READONLY

  cd "$REPO/sgl-kernel"
  if [[ "$2" == "cleanup" ]]; then
    rm -rf build/ dist/sgl_kernel-*.whl
  fi
  CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) \
    python -m uv build --wheel -Cbuild-dir=build --color=always . --no-build-isolation
  pip install --force-reinstall --no-deps dist/sgl_kernel-*.whl
  cd "$REPO"
fi

# Install sglang python package (pure Python, no --no-build-isolation needed)
if [[ "$1" == "python" || "$1" == "all" ]]; then
  pip install -e "$REPO/python" --no-deps
fi

echo "SGLang install done."
