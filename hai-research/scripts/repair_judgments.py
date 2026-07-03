#!/usr/bin/env python3
"""Repair failed judgments in-place by rerunning only non-ok rows.

The script preserves a .bak copy of judgments.jsonl, calls the same configured
backend/model for selected failed rows, reparses the new raw output, and rewrites
judgments.jsonl. It does not rerun successful judgments.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.judging.parser import parse_judgment
from src.registry import get_backend
from src.run.runner import format_prompt, load_prompt
from src.schemas import JudgeRequest, ModelSpec


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def model_spec_for(row: dict, meta: dict) -> ModelSpec:
    model_versions = meta.get("model_versions", {})
    version_info = model_versions.get(row["model"], {})
    return ModelSpec(
        name=row["model"],
        backend=row["backend"],
        params=version_info.get("params", {}),
    )


def repair_run(run_dir: Path, statuses: set[str], max_repairs: int | None) -> None:
    judgments_path = run_dir / "judgments.jsonl"
    meta_path = run_dir / "meta.json"
    if not judgments_path.exists():
        raise FileNotFoundError(judgments_path)
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    config = meta["config"]
    allowed_labels = config["dataset"]["allowed_labels"]
    prompt_template = load_prompt(config["judging"]["prompt_file"])
    temperature = config["judging"]["temperature"]
    max_tokens = config["judging"]["max_tokens"]

    rows = load_rows(judgments_path)
    backup_path = judgments_path.with_suffix(judgments_path.suffix + ".repair.bak")
    if not backup_path.exists():
        write_rows(backup_path, rows)

    repaired = 0
    backends = {}
    for idx, row in enumerate(rows):
        if row.get("status") not in statuses:
            continue
        if max_repairs is not None and repaired >= max_repairs:
            break

        model_spec = model_spec_for(row, meta)
        if model_spec.backend not in backends:
            backends[model_spec.backend] = get_backend(model_spec.backend)
        backend = backends[model_spec.backend]

        text = row.get("perturbation_text_applied") or row.get("input_text") or ""
        prompt = format_prompt(prompt_template, text, allowed_labels, language=row.get("language") or "en")
        req = JudgeRequest(
            text=text,
            allowed_labels=allowed_labels,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        print(
            f"Repairing {idx + 1}/{len(rows)}: model={row['model']} "
            f"factor={row.get('factor')} level={row.get('level')} trial={row.get('trial_idx')}",
            flush=True,
        )
        response = backend.judge(model_spec, req)

        if response.status == "error":
            row.update({
                "raw_output": response.raw_output,
                "status": "error",
                "error": response.error or "Backend returned error status",
                "usage": response.usage,
                "label": None,
                "confidence": None,
                "rationale": None,
            })
        else:
            parsed = parse_judgment(
                raw_output=response.raw_output,
                allowed_labels=allowed_labels,
                run_id=row["run_id"],
                model=row["model"],
                backend=row["backend"],
                prompt_version=row["prompt_version"],
                base_id=row["base_id"],
                perturb_id=row["perturb_id"],
                factor=row.get("factor"),
                level=row.get("level"),
                language=row.get("language"),
                trial_idx=row.get("trial_idx", 0),
                usage=response.usage,
                input_text=row.get("input_text"),
                prompt_hash=row.get("prompt_hash"),
                perturbation_text_applied=row.get("perturbation_text_applied"),
            ).to_dict()
            row.update(parsed)

        rows[idx] = row
        repaired += 1
        write_rows(judgments_path, rows)

    print(f"Repaired attempts: {repaired}; backup={backup_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--statuses", nargs="+", default=["error", "parse_error"])
    parser.add_argument("--max-repairs", type=int, default=None)
    args = parser.parse_args()
    repair_run(args.run, set(args.statuses), args.max_repairs)


if __name__ == "__main__":
    main()
