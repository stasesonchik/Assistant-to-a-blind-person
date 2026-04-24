#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

exec vllm serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --trust-remote-code \
  --max-model-len "$MAX_MODEL_LEN" \
  --limit-mm-per-prompt.image 1

