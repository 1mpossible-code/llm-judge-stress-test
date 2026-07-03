"""Deterministic subset selection for paper experiments."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.dataset.load_base import load_base_dataset
from src.dataset.normalize import normalize_example
from src.utils.io import write_json, write_jsonl


def offensive_bucket(offensive_rate: float) -> str:
    """Return a stable bucket name for a binary offensive-label rate."""
    return f"offensive_rate_{offensive_rate:.1f}"


def select_balanced_binary_subset(
    input_path: str | Path,
    out_path: str | Path,
    n: int = 30,
    language: str = "en",
    seed: int = 1337,
    allowed_labels: list[str] | None = None,
    positive_label: str = "offensive",
) -> dict[str, Any]:
    """Select a deterministic subset balanced by human label distribution.

    For the LeWiDi offensiveness data each item has five annotations, so the
    positive-label rate falls into six buckets: 0.0, 0.2, ..., 1.0. Balancing
    across these buckets keeps both easy-consensus and high-disagreement cases
    in the paper sample instead of using the first N rows of the dataset.
    """
    if allowed_labels is None:
        allowed_labels = ["offensive", "not_offensive"]

    rng = random.Random(seed)
    buckets: dict[float, list[dict[str, Any]]] = defaultdict(list)

    for ex in load_base_dataset(input_path, [language]):
        norm = normalize_example(ex, allowed_labels)
        positive_rate = round(norm.human_label_dist.get(positive_label, 0.0), 1)
        item = ex.model_dump()
        item["selection_metadata"] = {
            "positive_label": positive_label,
            "positive_rate": positive_rate,
            "human_disagreement": norm.human_disagreement,
            "n_annotators": norm.n_annotators,
        }
        buckets[positive_rate].append(item)

    if not buckets:
        raise ValueError(f"No examples found for language={language} in {input_path}")

    bucket_keys = sorted(buckets)
    base_per_bucket = n // len(bucket_keys)
    remainder = n % len(bucket_keys)

    selected: list[dict[str, Any]] = []
    by_bucket: dict[str, int] = {}

    for idx, key in enumerate(bucket_keys):
        quota = base_per_bucket + (1 if idx < remainder else 0)
        candidates = list(buckets[key])
        rng.shuffle(candidates)
        take = min(quota, len(candidates))
        selected.extend(candidates[:take])
        by_bucket[offensive_bucket(key)] = take

    # If any bucket was short, fill remaining slots from all unused examples.
    if len(selected) < n:
        selected_ids = {item["id"] for item in selected}
        remaining = [item for key in bucket_keys for item in buckets[key] if item["id"] not in selected_ids]
        rng.shuffle(remaining)
        selected.extend(remaining[: n - len(selected)])

    # Stable output order independent of bucket iteration/shuffle internals.
    selected.sort(key=lambda item: item["id"])

    out_path = Path(out_path)
    write_jsonl(out_path, selected)

    metadata = {
        "input_path": str(input_path),
        "out_path": str(out_path),
        "n_requested": n,
        "n_selected": len(selected),
        "language": language,
        "seed": seed,
        "allowed_labels": allowed_labels,
        "positive_label": positive_label,
        "bucket_counts": by_bucket,
        "selected_ids": [item["id"] for item in selected],
    }
    write_json(out_path.parent / f"{out_path.stem}_meta.json", metadata)
    return metadata
