#!/usr/bin/env python3
"""Interactive human validation for perturbation semantic preservation.

Friends/reviewers can run this locally to rate whether each perturbation preserves
meaning and merely adds framing. Outputs are JSONL files that can be summarized
into paper-ready CSV/LaTeX tables.

Example:
  uv run python scripts/human_perturbation_validation.py annotate --annotator-id alice --num-shards 3 --shard-index 0
  uv run python scripts/human_perturbation_validation.py summarize --input-glob 'human_validation/*.jsonl'
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import random
import sys
import textwrap
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import registers perturbation generators as a side effect.
from src.perturb.generators import generate_perturbation  # noqa: E402

CORE_FACTORS: list[tuple[str, str]] = [
    ("cultural_pride", "west_comparison"),
    ("cultural_pride", "ingroup_pride"),
    ("cultural_framing", "west_bias_medium"),
    ("cultural_framing", "ingroup_medium"),
    ("traditionalism", "modern_decay"),
    ("authority", "expert_consensus"),
    ("verbosity", "padded_fluent"),
    ("position_bias", "affirmative_medium"),
    ("position_bias", "negative_medium"),
]

RATING_HELP = """
Ratings use 1--5 scales:

1) Meaning preservation
   1 = changes original meaning heavily
   3 = somewhat changes/framing affects interpretation
   5 = preserves original meaning very well

2) Framing-only score
   1 = adds substantial new claim/content beyond framing
   3 = mixed: framing plus some new content
   5 = adds only context/framing around the same base text

3) Direct offensiveness-change score
   1 = does NOT directly change how offensive the text itself is
   3 = may somewhat change perceived offensiveness
   5 = directly makes the text more/less offensive

For the paper, high scores on (1) and (2), and low scores on (3), are best.
""".strip()


def load_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl_line(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def task_id(base_id: str, factor: str, level: str) -> str:
    return hashlib.sha256(f"{base_id}:{factor}:{level}".encode()).hexdigest()[:16]


def load_tasks(dataset_path: Path) -> list[dict]:
    tasks: list[dict] = []
    for row in load_jsonl(dataset_path):
        labels = row["human_labels"]
        off_votes = sum(1 for x in labels if x == "offensive")
        for factor, level in CORE_FACTORS:
            pert = generate_perturbation(
                base_id=row["id"],
                base_text=row["text"],
                language=row.get("language", "en"),
                factor=factor,
                level=level,
            )
            tasks.append(
                {
                    "task_id": task_id(row["id"], factor, level),
                    "base_id": row["id"],
                    "language": row.get("language", "en"),
                    "factor": factor,
                    "level": level,
                    "original_text": row["text"],
                    "perturbed_text": pert.text,
                    "original_text_hash": text_hash(row["text"]),
                    "perturbed_text_hash": text_hash(pert.text),
                    "human_offensive_votes": off_votes,
                    "human_total_votes": len(labels),
                    "human_offensive_rate": off_votes / len(labels),
                }
            )
    return tasks


def completed_instance_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    done = set()
    for row in load_jsonl(output_path):
        if not row.get("skipped"):
            done.add(row.get("instance_id", row["task_id"]))
    return done


def add_duplicate_checks(
    tasks: list[dict],
    rng: random.Random,
    duplicate_rate: float,
    min_gap: int,
) -> list[dict]:
    """Add delayed duplicate tasks for intra-rater reliability checks.

    Reviewers are not told which tasks are duplicates. Duplicates are placed at
    least ``min_gap`` positions away from the original when possible, so the
    second rating is less likely to be based on short-term memory.
    """
    scheduled = []
    for t in tasks:
        main = dict(t)
        main["instance_id"] = f"{t['task_id']}:main"
        main["is_duplicate"] = False
        main["duplicate_of"] = ""
        scheduled.append(main)

    k = int(round(len(tasks) * duplicate_rate))
    if k <= 0:
        return scheduled

    chosen = rng.sample(tasks, min(k, len(tasks)))
    for t in chosen:
        dup = dict(t)
        dup["instance_id"] = f"{t['task_id']}:dup:{hashlib.sha256((t['task_id'] + ':dup').encode()).hexdigest()[:8]}"
        dup["is_duplicate"] = True
        dup["duplicate_of"] = t["task_id"]

        original_positions = [i for i, x in enumerate(scheduled) if x["task_id"] == t["task_id"]]
        original_pos = original_positions[0] if original_positions else 0
        valid_positions = [
            i for i in range(len(scheduled) + 1)
            if abs(i - original_pos) >= min_gap
        ]
        if not valid_positions:
            valid_positions = list(range(len(scheduled) + 1))
        insert_at = rng.choice(valid_positions)
        scheduled.insert(insert_at, dup)

    for idx, t in enumerate(scheduled):
        t["sequence_position"] = idx
    return scheduled


def prompt_rating(prompt: str, allow_blank: bool = False) -> int | None:
    while True:
        val = input(prompt).strip()
        if allow_blank and val == "":
            return None
        if val in {"q", "quit", "exit"}:
            raise KeyboardInterrupt
        try:
            num = int(val)
        except ValueError:
            print("Please enter an integer from 1 to 5, or q to quit.")
            continue
        if 1 <= num <= 5:
            return num
        print("Please enter a value from 1 to 5.")


def prompt_yes_no(prompt: str, default: bool = False) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        val = input(f"{prompt} {suffix}: ").strip().lower()
        if val == "":
            return default
        if val in {"y", "yes"}:
            return True
        if val in {"n", "no"}:
            return False
        print("Please enter y or n.")


def wrapped(text: str, width: int = 92) -> str:
    return "\n".join(textwrap.wrap(text, width=width, replace_whitespace=False, drop_whitespace=False))


def command_annotate(args: argparse.Namespace) -> None:
    dataset_path = ROOT / args.dataset
    output_path = ROOT / args.output_dir / f"{args.annotator_id}.jsonl"
    tasks = load_tasks(dataset_path)

    rng = random.Random(args.seed)
    rng.shuffle(tasks)

    if args.num_shards > 1:
        tasks = [t for idx, t in enumerate(tasks) if idx % args.num_shards == args.shard_index]

    if args.max_tasks is not None:
        tasks = tasks[: args.max_tasks]

    tasks = add_duplicate_checks(tasks, rng, args.duplicate_rate, args.duplicate_min_gap)

    done = completed_instance_ids(output_path)
    pending = [t for t in tasks if t.get("instance_id", t["task_id"]) not in done]

    print("\nHuman perturbation validation")
    print("=" * 34)
    print(f"Annotator ID: {args.annotator_id}")
    print(f"Output file:  {display_path(output_path)}")
    print(f"Tasks total for this run: {len(tasks)}")
    print(f"Duplicate reliability checks: {sum(1 for t in tasks if t.get('is_duplicate'))}")
    print(f"Already completed: {len(tasks) - len(pending)}")
    print(f"Remaining: {len(pending)}")
    print("\n" + RATING_HELP + "\n")
    print("Controls: enter 1-5 for each rating; q quits safely; notes are optional.\n")

    if not pending:
        print("Nothing to annotate. You are done.")
        return

    try:
        for idx, task in enumerate(pending, start=1):
            print("\n" + "-" * 96)
            print(f"Task {idx}/{len(pending)}")
            if args.show_metadata:
                print(f"Metadata: item={task['base_id']} | {task['factor']} / {task['level']}")
                print(f"Human offensive votes: {task['human_offensive_votes']}/{task['human_total_votes']}")
            print("\nORIGINAL TEXT:\n")
            print(wrapped(task["original_text"]))
            print("\nPERTURBED TEXT:\n")
            print(wrapped(task["perturbed_text"]))
            print("\nRatings:")

            meaning = prompt_rating("  Meaning preservation (1=changed, 5=preserved): ")
            framing = prompt_rating("  Framing-only score   (1=new content, 5=only framing): ")
            offense = prompt_rating("  Direct offensiveness change (1=no direct change, 5=directly changes): ")
            problematic = prompt_yes_no("  Flag this perturbation as problematic?", default=False)
            notes = input("  Optional note (press Enter to skip): ").strip()

            result = {
                **task,
                "annotator_id": args.annotator_id,
                "annotated_at": datetime.now(timezone.utc).isoformat(),
                "meaning_preservation": meaning,
                "framing_only": framing,
                "direct_offensiveness_change": offense,
                "problematic": problematic,
                "notes": notes,
                "skipped": False,
            }
            write_jsonl_line(output_path, result)
            print("Saved.")
    except KeyboardInterrupt:
        print("\nStopped. Progress was saved; rerun the same command to resume.")


def command_export(args: argparse.Namespace) -> None:
    tasks = load_tasks(ROOT / args.dataset)
    rng = random.Random(args.seed)
    rng.shuffle(tasks)
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        fieldnames = [
            "task_id",
            "base_id",
            "factor",
            "level",
            "human_offensive_votes",
            "human_total_votes",
            "original_text",
            "perturbed_text",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in tasks:
            writer.writerow({k: t[k] for k in fieldnames})
    print(f"Wrote {len(tasks)} tasks to {display_path(out)}")


def fmt_mean_sd(vals: list[float]) -> str:
    if not vals:
        return "--"
    if len(vals) == 1:
        return f"{vals[0]:.2f}"
    return f"{mean(vals):.2f} ({pstdev(vals):.2f})"


def tex_escape(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )


def command_summarize(args: argparse.Namespace) -> None:
    paths = [Path(p) for p in glob.glob(str(ROOT / args.input_glob))]
    rows: list[dict] = []
    for path in paths:
        rows.extend([r for r in load_jsonl(path) if not r.get("skipped")])

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not rows:
        print("No annotation rows found.")
        return

    annotators = sorted({r["annotator_id"] for r in rows})
    duplicate_rows = [r for r in rows if r.get("is_duplicate")]
    analysis_rows = [r for r in rows if not r.get("is_duplicate")]
    print(f"Loaded {len(rows)} ratings from {len(annotators)} annotator(s): {', '.join(annotators)}")
    print(f"Primary ratings used for semantic summary: {len(analysis_rows)}")
    print(f"Blinded duplicate ratings used for reliability checks: {len(duplicate_rows)}")

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in analysis_rows:
        grouped[(r["factor"], r["level"])].append(r)

    summary = []
    for (factor, level), vals in sorted(grouped.items()):
        meaning_vals = [float(v["meaning_preservation"]) for v in vals]
        framing_vals = [float(v["framing_only"]) for v in vals]
        offense_vals = [float(v["direct_offensiveness_change"]) for v in vals]
        problematic_rate = sum(bool(v.get("problematic")) for v in vals) / len(vals)
        summary.append(
            {
                "factor": factor,
                "level": level,
                "n_ratings": len(vals),
                "n_annotators": len({v["annotator_id"] for v in vals}),
                "meaning_preservation_mean": mean(meaning_vals),
                "meaning_preservation_sd": pstdev(meaning_vals) if len(meaning_vals) > 1 else 0.0,
                "framing_only_mean": mean(framing_vals),
                "framing_only_sd": pstdev(framing_vals) if len(framing_vals) > 1 else 0.0,
                "direct_offensiveness_change_mean": mean(offense_vals),
                "direct_offensiveness_change_sd": pstdev(offense_vals) if len(offense_vals) > 1 else 0.0,
                "problematic_rate": problematic_rate,
                "flag_low_preservation": mean(meaning_vals) < args.min_meaning,
                "flag_direct_offense_change": mean(offense_vals) > args.max_direct_offense,
            }
        )

    csv_path = out_dir / "semantic_validation_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    raw_path = out_dir / "semantic_validation_raw_merged.jsonl"
    with raw_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    reliability = []
    by_annotator_task = defaultdict(list)
    for r in rows:
        by_annotator_task[(r["annotator_id"], r["task_id"])].append(r)
    for (annotator, tid), vals in by_annotator_task.items():
        main_vals = [v for v in vals if not v.get("is_duplicate")]
        dup_vals = [v for v in vals if v.get("is_duplicate")]
        if not main_vals or not dup_vals:
            continue
        main = main_vals[0]
        dup = dup_vals[0]
        reliability.append(
            {
                "annotator_id": annotator,
                "task_id": tid,
                "factor": main["factor"],
                "level": main["level"],
                "meaning_abs_diff": abs(float(main["meaning_preservation"]) - float(dup["meaning_preservation"])),
                "framing_abs_diff": abs(float(main["framing_only"]) - float(dup["framing_only"])),
                "direct_offense_abs_diff": abs(float(main["direct_offensiveness_change"]) - float(dup["direct_offensiveness_change"])),
            }
        )

    if reliability:
        rel_path = out_dir / "semantic_validation_duplicate_reliability.csv"
        with rel_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(reliability[0].keys()))
            writer.writeheader()
            writer.writerows(reliability)
    else:
        rel_path = None

    tex_path = out_dir / "semantic_validation_summary.tex"
    linebreak = r" \\" + "\n"
    with tex_path.open("w") as f:
        f.write(r"\begin{tabular}{llrrrr}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write("Factor & Level & $n$ & Meaning & Framing-only & Direct offense change" + linebreak)
        f.write(r"\midrule" + "\n")
        for r in summary:
            flag = " $^{\\dagger}$" if r["flag_low_preservation"] or r["flag_direct_offense_change"] else ""
            f.write(
                f"{tex_escape(r['factor'])} & {tex_escape(r['level'])} & {r['n_ratings']} & "
                f"{r['meaning_preservation_mean']:.2f} & {r['framing_only_mean']:.2f} & "
                f"{r['direct_offensiveness_change_mean']:.2f}{flag}" + linebreak
            )
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")

    print(f"Wrote summary CSV: {display_path(csv_path)}")
    print(f"Wrote merged raw JSONL: {display_path(raw_path)}")
    print(f"Wrote LaTeX table: {display_path(tex_path)}")
    if rel_path:
        print(f"Wrote duplicate reliability CSV: {display_path(rel_path)}")
        print(
            "Duplicate reliability mean absolute differences: "
            f"meaning={mean([r['meaning_abs_diff'] for r in reliability]):.2f}, "
            f"framing={mean([r['framing_abs_diff'] for r in reliability]):.2f}, "
            f"direct_offense={mean([r['direct_offense_abs_diff'] for r in reliability]):.2f}"
        )
    flagged = [r for r in summary if r["flag_low_preservation"] or r["flag_direct_offense_change"]]
    if flagged:
        print("\nFlagged perturbations:")
        for r in flagged:
            print(
                f"  {r['factor']}/{r['level']}: meaning={r['meaning_preservation_mean']:.2f}, "
                f"direct_offense_change={r['direct_offensiveness_change_mean']:.2f}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    ann = sub.add_parser("annotate", help="Run interactive annotation UI in the terminal")
    ann.add_argument("--annotator-id", required=True, help="Short anonymous ID, e.g. rater01")
    ann.add_argument("--dataset", default="data/lewidi/paper_sample_30_balanced.jsonl")
    ann.add_argument("--output-dir", default="human_validation")
    ann.add_argument("--seed", type=int, default=20260626)
    ann.add_argument("--num-shards", type=int, default=1, help="Split tasks across annotators")
    ann.add_argument("--shard-index", type=int, default=0, help="0-indexed shard to annotate")
    ann.add_argument("--max-tasks", type=int, default=None, help="Optional cap for pilot testing")
    ann.add_argument("--duplicate-rate", type=float, default=0.10, help="Fraction of tasks repeated later as blinded reliability checks")
    ann.add_argument("--duplicate-min-gap", type=int, default=25, help="Minimum task gap between an item and its duplicate when possible")
    ann.add_argument("--show-metadata", action="store_true", help="Show item IDs, perturbation labels, and human votes (hidden by default to reduce bias)")
    ann.set_defaults(func=command_annotate)

    exp = sub.add_parser("export-tasks", help="Export task CSV for spreadsheet/manual annotation")
    exp.add_argument("--dataset", default="data/lewidi/paper_sample_30_balanced.jsonl")
    exp.add_argument("--output", default="human_validation/tasks.csv")
    exp.add_argument("--seed", type=int, default=20260626)
    exp.set_defaults(func=command_export)

    summ = sub.add_parser("summarize", help="Summarize completed JSONL annotations")
    summ.add_argument("--input-glob", default="human_validation/*.jsonl")
    summ.add_argument("--out-dir", default="paper_artifacts/human_validation")
    summ.add_argument("--min-meaning", type=float, default=4.0)
    summ.add_argument("--max-direct-offense", type=float, default=2.5)
    summ.set_defaults(func=command_summarize)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "num_shards") and args.num_shards < 1:
        parser.error("--num-shards must be >= 1")
    if hasattr(args, "shard_index") and not (0 <= args.shard_index < args.num_shards):
        parser.error("--shard-index must satisfy 0 <= shard_index < num_shards")
    if hasattr(args, "duplicate_rate") and not (0 <= args.duplicate_rate <= 1):
        parser.error("--duplicate-rate must be between 0 and 1")
    args.func(args)


if __name__ == "__main__":
    main()
