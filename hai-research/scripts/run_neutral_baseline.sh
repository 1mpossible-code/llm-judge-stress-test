#!/usr/bin/env bash
set -euo pipefail

configs=(
  configs/exp_neutral_groq_llama31_8b.yaml
  configs/exp_neutral_groq_qwen3_32b.yaml
  configs/exp_neutral_claude_haiku.yaml
  configs/exp_neutral_gpt54.yaml
  configs/exp_neutral_gemini_flash.yaml
)

for cfg in "${configs[@]}"; do
  echo "== Running $cfg =="
  uv run --extra groq python -m src.cli run --config "$cfg"
done
