# A Pilot Stress Test of LLM-as-a-Judge Robustness in Culturally Sensitive Annotation

This repository contains the paper and reproducibility code for a pilot robustness study of LLM-as-a-Judge systems on culturally sensitive offensiveness annotation.

## Paper

- Source: [`paper.tex`](paper.tex)
- Compiled PDF: [`paper.pdf`](paper.pdf)

The paper reports a deliberately small paired stress test, not a population-level benchmark. It includes uncertainty intervals, paired tests, multiple-comparison correction, repeated-call self-consistency checks, human semantic validation of perturbations, and a neutral-wrapper baseline.

## Repository layout

```text
paper.tex                         # main paper source
paper.pdf                         # compiled paper
hai-research/
  configs/                        # experiment configs
  data/lewidi/                    # selected 30-item paper sample + metadata
  prompts/                        # judge prompt template
  src/                            # experiment pipeline
  scripts/                        # analysis, repair, validation, and summary scripts
  tests/                          # unit/integration tests
  paper_artifacts/                # paper-ready tables and validation summaries
```

Large raw run directories are intentionally not tracked in Git. Paper-ready derived artifacts are included under `hai-research/paper_artifacts/`.

## Setup

The code uses Python 3.11+ and [`uv`](https://docs.astral.sh/uv/) for reproducible execution.

```bash
cd hai-research
uv sync
```

For provider-backed reruns, create a local `.env` from the example and add your own keys. Do not commit secrets.

```bash
cp .env.example .env
```

## Tests

```bash
cd hai-research
uv run pytest -q
```

Expected result for the current repository state:

```text
22 passed
```

## Reproducing analysis tables from included artifacts

The paper-ready robustness, human-validation, and neutral-baseline summaries are stored in:

```text
hai-research/paper_artifacts/robustness/
hai-research/paper_artifacts/human_validation/
hai-research/paper_artifacts/neutral_baseline/
hai-research/paper_artifacts/tables_core/
```

If the raw run directories are available locally, regenerate the robustness and neutral-baseline summaries with:

```bash
cd hai-research
uv run python scripts/paper_robustness_analysis.py
uv run python scripts/summarize_neutral_baseline.py
uv run python scripts/human_perturbation_validation.py summarize --input-glob 'human_validation/*.jsonl'
```

## Compiling the paper

```bash
tectonic paper.tex
```

## Notes on data and ethics

The tracked data includes only the selected 30-item paper sample and metadata needed for the pilot analysis. Full LeWiDi data should be obtained from the original task release. Raw human-validation files are not tracked because they may contain reviewer identifiers and offensive text; aggregated validation summaries are included.

## License

Code is released under the MIT License. Dataset licensing remains governed by the original data providers.
