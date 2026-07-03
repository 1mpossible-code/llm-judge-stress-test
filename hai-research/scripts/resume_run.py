#!/usr/bin/env python3
"""Resume an incomplete run by appending only missing judgments.

This is useful for provider sessions that must be completed in multiple chunks.
It reconstructs the original experiment from meta.json, skips already recorded
(model, base_id, factor, level, trial_idx) cells, and appends missing judgments
to the same judgments.jsonl file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ExperimentConfig
from src.dataset.load_base import load_base_dataset
from src.dataset.normalize import normalize_example
from src.judging.parser import parse_judgment
from src.perturb.generators import generate_all_perturbations
from src.registry import get_backend
from src.run.runner import format_prompt, load_prompt
from src.schemas import JudgeRequest, JudgeResponse, Judgment, ModelSpec
from src.utils.io import append_jsonl, read_jsonl


def compute_prompt_hash(prompt_template: str) -> str:
    return hashlib.sha256(prompt_template.encode()).hexdigest()


def existing_keys(judgments_file: Path) -> set[tuple[str, str, str, str, int]]:
    if not judgments_file.exists():
        return set()
    keys = set()
    for row in read_jsonl(judgments_file):
        keys.add((row["model"], row["base_id"], row.get("factor"), row.get("level"), row["trial_idx"]))
    return keys


def resume_run(run_dir: Path, max_new: int | None = None) -> None:
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    config = ExperimentConfig(**meta["config"])
    run_id = meta["run_id"]
    prompt_template = load_prompt(config.judging.prompt_file)
    prompt_hash = meta.get("prompt_hash") or compute_prompt_hash(prompt_template)

    factor_levels = [
        (factor.name, level)
        for factor in config.perturbations.factors
        for level in factor.levels
    ]

    base_examples = list(load_base_dataset(config.dataset.path, config.dataset.languages))
    if config.dataset.limit:
        base_examples = base_examples[: config.dataset.limit]
    normalized_examples = [normalize_example(ex, config.dataset.allowed_labels) for ex in base_examples]
    base_id_to_text = {ex.id: ex.text for ex in base_examples}

    perturbations = []
    for ex in normalized_examples:
        perturbations.extend(generate_all_perturbations(ex.id, ex.text, ex.language, factor_levels))

    judgments_file = run_dir / "judgments.jsonl"
    seen = existing_keys(judgments_file)
    backends = {}
    appended = 0

    total_expected = len(config.judging.models) * len(perturbations) * config.judging.repeats
    print(f"Existing judgments: {len(seen)} / expected cells: {total_expected}")

    for model_spec in config.judging.models:
        if model_spec.backend not in backends:
            backends[model_spec.backend] = get_backend(model_spec.backend)
        backend = backends[model_spec.backend]

        for perturb in perturbations:
            for trial_idx in range(config.judging.repeats):
                key = (model_spec.name, perturb.base_id, perturb.factor, perturb.level, trial_idx)
                if key in seen:
                    continue
                if max_new is not None and appended >= max_new:
                    print(f"Reached --max-new={max_new}; appended {appended}")
                    return

                formatted_prompt = format_prompt(
                    prompt_template,
                    perturb.text,
                    config.dataset.allowed_labels,
                    language=perturb.language,
                )
                req = JudgeRequest(
                    text=perturb.text,
                    allowed_labels=config.dataset.allowed_labels,
                    prompt=formatted_prompt,
                    temperature=config.judging.temperature,
                    max_tokens=config.judging.max_tokens,
                )

                print(
                    f"Appending {appended + 1}: model={model_spec.name} "
                    f"base={perturb.base_id} factor={perturb.factor} level={perturb.level} trial={trial_idx}",
                    flush=True,
                )
                try:
                    response = backend.judge(model_spec, req)
                except Exception as exc:
                    response = JudgeResponse(
                        raw_output="",
                        usage=None,
                        status="error",
                        error=f"{type(exc).__name__}: {exc}",
                    )

                if response.status == "error":
                    judgment = Judgment(
                        run_id=run_id,
                        model=model_spec.name,
                        backend=model_spec.backend,
                        prompt_version=config.judging.prompt_version,
                        base_id=perturb.base_id,
                        perturb_id=perturb.perturb_id,
                        factor=perturb.factor,
                        level=perturb.level,
                        language=perturb.language,
                        trial_idx=trial_idx,
                        label=None,
                        confidence=None,
                        rationale=None,
                        raw_output=response.raw_output,
                        status="error",
                        error=response.error,
                        usage=response.usage,
                        input_text=base_id_to_text.get(perturb.base_id),
                        prompt_hash=prompt_hash,
                        perturbation_text_applied=perturb.text,
                    )
                else:
                    judgment = parse_judgment(
                        raw_output=response.raw_output,
                        allowed_labels=config.dataset.allowed_labels,
                        run_id=run_id,
                        model=model_spec.name,
                        backend=model_spec.backend,
                        prompt_version=config.judging.prompt_version,
                        base_id=perturb.base_id,
                        perturb_id=perturb.perturb_id,
                        factor=perturb.factor,
                        level=perturb.level,
                        language=perturb.language,
                        trial_idx=trial_idx,
                        usage=response.usage,
                        input_text=base_id_to_text.get(perturb.base_id),
                        prompt_hash=prompt_hash,
                        perturbation_text_applied=perturb.text,
                    )

                append_jsonl(judgments_file, judgment.to_dict())
                seen.add(key)
                appended += 1

    print(f"Run complete; appended {appended}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--max-new", type=int, default=None)
    args = parser.parse_args()
    resume_run(args.run, args.max_new)


if __name__ == "__main__":
    main()
