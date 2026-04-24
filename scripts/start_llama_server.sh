#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${LLAMA_SERVER_BIN:-$ROOT/third_party/llama.cpp/build/bin/llama-server}"
MODEL_REPO="${MODEL_REPO:-ggml-org/Qwen2.5-VL-3B-Instruct-GGUF:Q4_K_M}"
MODEL_DIR="${MODEL_DIR:-$ROOT/models/qwen2_5_vl_3b_gguf}"
MODEL_FILE="${MODEL_FILE:-$MODEL_DIR/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf}"
MMPROJ_FILE="${MMPROJ_FILE:-$MODEL_DIR/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
CTX_SIZE="${CTX_SIZE:-4096}"
GPU_LAYERS="${GPU_LAYERS:-auto}"
PARALLEL="${PARALLEL:-auto}"
SAFE_ON_DARWIN="${SAFE_ON_DARWIN:-1}"
FORCE_UNSAFE_METAL="${FORCE_UNSAFE_METAL:-0}"

if [[ ! -x "$BIN" ]]; then
  "$ROOT/scripts/build_llama_cpp.sh" auto
fi

common_args=(--host "$HOST" --port "$PORT" --ctx-size "$CTX_SIZE")
if [[ "$PARALLEL" != "auto" ]]; then
  common_args+=(--parallel "$PARALLEL")
fi

if [[ "$(uname -s)" == "Darwin" && "$SAFE_ON_DARWIN" == "1" && "$FORCE_UNSAFE_METAL" != "1" ]]; then
  echo "Using macOS safe mode for Qwen2.5-VL: CPU-only llama-server (Metal disabled for model decode)." >&2
  echo "Set FORCE_UNSAFE_METAL=1 to try Metal again." >&2
  common_args+=(--device none --n-gpu-layers 0 --parallel 1 --no-warmup)
else
  common_args+=(--n-gpu-layers "$GPU_LAYERS")
fi

if [[ -f "$MODEL_FILE" && -f "$MMPROJ_FILE" ]]; then
  exec "$BIN" -m "$MODEL_FILE" --mmproj "$MMPROJ_FILE" "${common_args[@]}"
fi

exec "$BIN" -hf "$MODEL_REPO" "${common_args[@]}"
