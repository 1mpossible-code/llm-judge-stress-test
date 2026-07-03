#!/usr/bin/env python3
"""Summarize judgment quality for one or more run directories."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open("r", encoding="utf-8") if line.strip()]


def summarize(run: Path) -> None:
    rows = load_jsonl(run / "judgments.jsonl")
    print(f"\n== {run} ==")
    print(f"judgments: {len(rows)}")
    if not rows:
        return

    print("status:", dict(Counter(row.get("status") for row in rows)))
    print("by model/status:", dict(Counter((row.get("model"), row.get("status")) for row in rows)))

    for model in sorted({row.get("model") for row in rows}):
        model_rows = [row for row in rows if row.get("model") == model]
        ok_rows = [row for row in model_rows if row.get("status") == "ok"]
        parse_rate = len(ok_rows) / len(model_rows) if model_rows else 0.0
        labels = Counter(row.get("label") for row in ok_rows)
        print(f"  {model}: ok={len(ok_rows)}/{len(model_rows)} ({parse_rate:.1%}), labels={dict(labels)}")

    bad = [row for row in rows if row.get("status") != "ok"][:5]
    for row in bad:
        print(
            "  non-ok:",
            row.get("model"),
            row.get("factor"),
            row.get("level"),
            row.get("error"),
            repr((row.get("raw_output") or "")[:120]),
        )

    metrics = run / "metrics" / "metrics.csv"
    paired = run / "paired_results.jsonl"
    print(f"metrics_csv: {metrics.exists()}")
    print(f"paired_results: {paired.exists()} lines={len(load_jsonl(paired)) if paired.exists() else 0}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: scripts/check_run_quality.py RUN_DIR [RUN_DIR ...]")
        raise SystemExit(2)
    for arg in sys.argv[1:]:
        summarize(Path(arg))


if __name__ == "__main__":
    main()
