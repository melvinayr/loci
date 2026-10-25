#!/usr/bin/env bash
# Fetches small GGUF fixtures for integration tests into Tests/Fixtures/models (git-ignored).
# Tests also accept LOCI_TEST_MODELS=<dir> to use models you already have.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
dest="${root}/Tests/Fixtures/models"
mkdir -p "$dest"

# name | url | sha256 (empty = unpinned)
models=(
  "bge-small-en-v1.5-q8_0.gguf|https://huggingface.co/ggml-org/bge-small-en-v1.5-Q8_0-GGUF/resolve/main/bge-small-en-v1.5-q8_0.gguf|f046db1dc724cf4f6f0a0c5917e922823b73eb1d27b8f9a9c2797f7866974804"
  "qwen3-reranker-0.6b-q8_0.gguf|https://huggingface.co/ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/resolve/main/qwen3-reranker-0.6b-q8_0.gguf|22c9979ce4fbcdc5acdc310c6641c32797eff1aa980b8f7a2db8a8ea23429a48"
  "qwen3-0.6b-q8_0.gguf|https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf|9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031"
  "bge-reranker-v2-m3-Q4_K_M.gguf|https://huggingface.co/gpustack/bge-reranker-v2-m3-GGUF/resolve/main/bge-reranker-v2-m3-Q4_K_M.gguf|e186a244ed455b4ab66ec64339ce7427a6ae13f5c0b5e544de96e50f0f8b3673"
)

for entry in "${models[@]}"; do
  IFS='|' read -r name url sha <<<"$entry"
  target="${dest}/${name}"
  if [[ ! -f "$target" ]]; then
    echo "Fetching ${name}"
    curl -fsSL -o "${target}.part" "$url"
    mv "${target}.part" "$target"
  fi
  if [[ -n "$sha" ]]; then
    echo "${sha}  ${target}" | shasum -a 256 -c -
  fi
done
echo "Fixtures in ${dest}"
