# LLM Judge Stress Testing Pipeline

This directory contains the code used in the paper **"A Pilot Stress Test of LLM-as-a-Judge Robustness in Culturally Sensitive Annotation."**

The pipeline evaluates LLM judges on paired control/perturbation conditions for subjective offensiveness annotation. It supports repeated calls, parser repair, paired metrics, human perturbation validation, neutral-wrapper baselines, and paper-ready tables.

## Setup

```bash
uv sync
uv run pytest -q
```

Optional provider credentials can be placed in a local `.env` file. Start from:

```bash
cp .env.example .env
```

Do not commit `.env` or API keys.

## Important directories

```text
configs/                 Experiment YAML files
src/                     Core pipeline implementation
scripts/                 Analysis, repair, validation, and summary scripts
data/lewidi/             Selected 30-item paper sample and metadata
prompts/                 Judge prompt template
paper_artifacts/         Paper-ready tables, figures, and validation summaries
tests/                   Test suite
```

Raw `runs/` directories are intentionally ignored because they are large. The included `paper_artifacts/` directory contains the derived summaries used by the paper.

## Main paper configs

Core perturbation suite:

```bash
uv run --extra groq python -m src.cli run --config configs/exp_core_groq_llama31_8b.yaml
uv run --extra groq python -m src.cli run --config configs/exp_core_groq_qwen3_32b.yaml
uv run python -m src.cli run --config configs/exp_core_claude_haiku.yaml
uv run python -m src.cli run --config configs/exp_core_gpt54.yaml
uv run python -m src.cli run --config configs/exp_core_antigravity_gemini_flash.yaml
```

Neutral-wrapper baseline:

```bash
scripts/run_neutral_baseline.sh
```

## Regenerating paper artifacts

If raw run directories are available locally:

```bash
uv run python scripts/paper_robustness_analysis.py
uv run python scripts/summarize_neutral_baseline.py
uv run python scripts/human_perturbation_validation.py summarize --input-glob 'human_validation/*.jsonl'
```

Aggregate core run tables:

```bash
uv run python -m src.cli summarize \
  --runs runs/2026-06-25_204534_exp_core_groq_llama31_8b \
         runs/2026-06-25_212409_exp_core_groq_qwen3_32b \
         runs/2026-06-26_023709_exp_core_claude_haiku \
         runs/2026-06-25_220140_exp_core_gpt54 \
         runs/2026-06-25_212106_exp_core_antigravity_gemini_flash \
  --out paper_artifacts/tables_core
```

## Human validation

Interactive terminal validation:

```bash
uv run python scripts/human_perturbation_validation.py annotate --annotator-id rater01
```

Summarize collected ratings:

```bash
uv run python scripts/human_perturbation_validation.py summarize --input-glob 'human_validation/*.jsonl'
```

Raw human-validation files are not tracked by default; aggregated summaries are included.
