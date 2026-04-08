#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE_NAME="${LOCAL_LLM_DOCKER_BENCH_IMAGE:-local-llm-php-bench}"
MODEL_PATH="${LOCAL_LLM_DOCKER_MODEL_PATH:-}"
MODEL_NAME="${LOCAL_LLM_DOCKER_MODEL_NAME:-granite3.3:2b}"
RUNS="${LOCAL_LLM_DOCKER_RUNS:-3}"
WARMUPS="${LOCAL_LLM_DOCKER_WARMUPS:-1}"
GPU_LAYERS="${LOCAL_LLM_DOCKER_GPU_LAYERS:-0}"
OLLAMA_BASE_URL="${LOCAL_LLM_DOCKER_OLLAMA_URL:-http://host.docker.internal:11434}"

if [[ -z "$MODEL_PATH" ]]; then
  MODEL_PATH="$(ollama show --modelfile "$MODEL_NAME" 2>/dev/null | awk '/^FROM \//{sub(/^FROM /, ""); print; exit}')"
fi

if [[ -z "$MODEL_PATH" || ! -f "$MODEL_PATH" ]]; then
  echo "Could not resolve a local GGUF model path. Set LOCAL_LLM_DOCKER_MODEL_PATH or ensure ollama model '$MODEL_NAME' exists." >&2
  exit 1
fi

cd "$ROOT_DIR"

docker build --progress=plain -t "$IMAGE_NAME" -f docker/linux-benchmark.Dockerfile .

docker run --rm \
  --add-host host.docker.internal:host-gateway \
  -v "$MODEL_PATH:/models/model.gguf:ro" \
  "$IMAGE_NAME" \
  php -d ffi.enable=1 ./bin/benchmark suite \
    --library-path /app/var/native/llama.cpp/build/bin/libllama.so \
    --model-path /models/model.gguf \
    --ollama-model "$MODEL_NAME" \
    --ollama-base-url "$OLLAMA_BASE_URL" \
    --gpu-layers "$GPU_LAYERS" \
    --runs "$RUNS" \
    --warmups "$WARMUPS"
