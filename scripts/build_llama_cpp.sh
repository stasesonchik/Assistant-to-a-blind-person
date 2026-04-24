#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_DIR="$ROOT/third_party/llama.cpp"
BACKEND="${1:-auto}"

if [[ ! -d "$LLAMA_DIR/.git" ]]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
fi

cmake_args=(-S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DCMAKE_BUILD_TYPE=Release)
case "$BACKEND" in
  auto)
    if [[ "$(uname -s)" == "Darwin" ]]; then
      cmake_args+=(-DGGML_METAL=ON)
    elif command -v nvidia-smi >/dev/null 2>&1; then
      cmake_args+=(-DGGML_CUDA=ON)
    fi
    ;;
  metal) cmake_args+=(-DGGML_METAL=ON) ;;
  cuda) cmake_args+=(-DGGML_CUDA=ON) ;;
  cpu) ;;
  *) echo "Unknown backend: $BACKEND" >&2; exit 2 ;;
esac

cmake "${cmake_args[@]}"
cmake --build "$LLAMA_DIR/build" --config Release -j "${JOBS:-4}"

echo "Built: $LLAMA_DIR/build/bin/llama-server"

