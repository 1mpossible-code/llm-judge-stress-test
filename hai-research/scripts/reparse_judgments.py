#!/usr/bin/env python3
"""Reparse stored raw model outputs after parser improvements.

This does not call any model. It only updates label/confidence/rationale/status
from the raw outputs already stored in judgments.jsonl and keeps a .bak copy.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.judging.parser import parse_judgment


def load_allowed_labels(run_dir: Path) -> list[str]:
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    return meta["config"]["dataset"]["allowed_labels"]


def reparse_run(run_dir: Path) -> None:
    judgments_path = run_dir / "judgments.jsonl"
    if not judgments_path.exists():
        raise FileNotFoundError(judgments_path)

    allowed_labels = load_allowed_labels(run_dir)
    rows = [json.loads(line) for line in judgments_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    new_rows = []
    changed = 0

    for row in rows:
        # Preserve provider/runtime errors with no raw output.
        if row.get("status") == "error" and not row.get("raw_output"):
            new_rows.append(row)
            continue

        parsed = parse_judgment(
            raw_output=row.get("raw_output") or "",
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
            usage=row.get("usage"),
            input_text=row.get("input_text"),
            prompt_hash=row.get("prompt_hash"),
            perturbation_text_applied=row.get("perturbation_text_applied"),
        ).to_dict()

        if parsed.get("status") != row.get("status") or parsed.get("label") != row.get("label"):
            changed += 1
        new_rows.append(parsed)

    backup_path = judgments_path.with_suffix(judgments_path.suffix + ".bak")
    if not backup_path.exists():
        backup_path.write_text(judgments_path.read_text(encoding="utf-8"), encoding="utf-8")

    judgments_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in new_rows),
        encoding="utf-8",
    )
    print(f"{run_dir}: reparsed {len(rows)} judgments, changed {changed}; backup={backup_path}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: scripts/reparse_judgments.py RUN_DIR [RUN_DIR ...]")
        raise SystemExit(2)
    for arg in sys.argv[1:]:
        reparse_run(Path(arg))


if __name__ == "__main__":
    main()
