#!/usr/bin/env python3
"""Summarize neutral-wrapper baseline runs for the paper."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_artifacts" / "neutral_baseline"

RUNS = {
    "Claude Haiku 4.5": ROOT / "runs/2026-07-02_162455_exp_neutral_claude_haiku",
    "GPT-5.4": ROOT / "runs/2026-07-02_164317_exp_neutral_gpt54",
    "Gemini 3.5 Flash": ROOT / "runs/2026-07-02_180237_exp_neutral_gemini_flash",
    "Llama 3.1 8B": ROOT / "runs/2026-07-02_160850_exp_neutral_groq_llama31_8b",
    "Qwen3 32B": ROOT / "runs/2026-07-02_161542_exp_neutral_groq_qwen3_32b",
}


def read_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def pct(x: float) -> str:
    return f"{100*x:.1f}\\%"


def ci(lo: float, hi: float) -> str:
    return f"[{100*lo:.1f}, {100*hi:.1f}]"


def majority(labels: list[str]) -> str:
    counts = Counter(labels)
    return "offensive" if counts["offensive"] > counts["not_offensive"] else "not_offensive"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    core = {}
    with (ROOT / "paper_artifacts" / "robustness" / "majority_overall_asr.csv").open() as f:
        for row in csv.DictReader(f):
            core[row["model"]] = float(row["asr"])

    rows = []
    raw_rows = []
    for model, run in RUNS.items():
        groups = defaultdict(list)
        for j in read_jsonl(run / "judgments.jsonl"):
            groups[(j["base_id"], j["factor"], j["level"])].append(j["label"])

        flips = 0
        n = 0
        base_ids = sorted({key[0] for key in groups})
        for base_id in base_ids:
            control = majority(groups[(base_id, "control", "original")])
            neutral = majority(groups[(base_id, "neutral_context", "online_discussion")])
            flips += control != neutral
            n += 1
        lo, hi = wilson(flips, n)
        rows.append({
            "model": model,
            "n": n,
            "flips": flips,
            "asr": flips / n,
            "ci_low": lo,
            "ci_high": hi,
            "core_asr": core[model],
            "difference_core_minus_neutral": core[model] - flips / n,
        })

        paired = list(read_jsonl(run / "paired_results.jsonl"))
        rk = sum(bool(p["flipped"]) for p in paired)
        rn = len(paired)
        rlo, rhi = wilson(rk, rn)
        raw_rows.append({"model": model, "n": rn, "flips": rk, "asr": rk / rn, "ci_low": rlo, "ci_high": rhi})

    with (OUT / "neutral_majority_asr.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (OUT / "neutral_raw_asr.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
        writer.writeheader()
        writer.writerows(raw_rows)

    with (OUT / "neutral_majority_asr.tex").open("w") as f:
        f.write("\\begin{tabular}{lrrrrr}\n\\toprule\n")
        f.write("Model & $n$ & Neutral flips & Neutral ASR & 95\\% CI & Core ASR \\\\\n\\midrule\n")
        for r in rows:
            f.write(f"{r['model']} & {r['n']} & {r['flips']} & {pct(r['asr'])} & {ci(r['ci_low'], r['ci_high'])} & {pct(r['core_asr'])} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    print(f"Wrote neutral baseline summaries to {OUT}")


if __name__ == "__main__":
    main()
