#!/usr/bin/env python3
"""Print progress for expected experiment runs."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPECTED = [
    ("Llama 3.1 8B", "exp_core_groq_llama31_8b", 900),
    ("Qwen3 32B", "exp_core_groq_qwen3_32b", 900),
    ("Claude Haiku 4.5", "exp_core_claude_haiku", 900),
    ("GPT-5.4", "exp_core_gpt54", 900),
    ("Gemini 3.5 Flash", "exp_core_antigravity_gemini_flash", 900),
    ("Claude/GPT validation", "exp_cli_validation_claude_codex", 360),
]


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def latest_run(pattern: str) -> Path | None:
    runs = sorted(Path("runs").glob(f"*{pattern}"), key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


def main() -> None:
    for label, pattern, expected in EXPECTED:
        run = latest_run(pattern)
        if run is None:
            print(f"{label:28} not started")
            continue
        n = count_lines(run / "judgments.jsonl")
        pct = (n / expected * 100) if expected else 0.0
        status = "done" if n >= expected else "running/pending"
        print(f"{label:28} {n:4d}/{expected:<4d} {pct:6.1f}%  {status}  {run}")


if __name__ == "__main__":
    main()
