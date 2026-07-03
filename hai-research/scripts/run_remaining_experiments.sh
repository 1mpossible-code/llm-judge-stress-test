#!/usr/bin/env bash
set -euo pipefail

# Run the remaining provider-backed experiments used for final paper tables.
# Existing run caches are preserved by each run directory; completed runs should
# be summarized after all commands finish.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

uv run --extra groq python -m src.cli run --config configs/exp_core_groq_llama31_8b.yaml
uv run --extra groq python -m src.cli run --config configs/exp_core_groq_qwen3_32b.yaml
uv run python -m src.cli run --config configs/exp_core_claude_haiku.yaml
uv run python -m src.cli run --config configs/exp_core_gpt54.yaml
uv run python -m src.cli run --config configs/exp_core_antigravity_gemini_flash.yaml
