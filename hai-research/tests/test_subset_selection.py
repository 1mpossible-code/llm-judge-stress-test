"""Tests for deterministic subset selection."""

from pathlib import Path

from src.dataset.select_subset import select_balanced_binary_subset
from src.utils.io import read_jsonl


def test_select_balanced_subset_is_deterministic(tmp_path):
    out1 = tmp_path / "subset1.jsonl"
    out2 = tmp_path / "subset2.jsonl"

    meta1 = select_balanced_binary_subset(
        input_path="data/lewidi/offensiveness.jsonl",
        out_path=out1,
        n=12,
        language="en",
        seed=123,
    )
    meta2 = select_balanced_binary_subset(
        input_path="data/lewidi/offensiveness.jsonl",
        out_path=out2,
        n=12,
        language="en",
        seed=123,
    )

    assert meta1["selected_ids"] == meta2["selected_ids"]
    assert meta1["n_selected"] == 12
    assert len(list(read_jsonl(out1))) == 12


def test_select_balanced_subset_covers_multiple_buckets(tmp_path):
    out = tmp_path / "subset.jsonl"

    meta = select_balanced_binary_subset(
        input_path="data/lewidi/offensiveness.jsonl",
        out_path=out,
        n=30,
        language="en",
        seed=1337,
    )

    nonempty_buckets = [count for count in meta["bucket_counts"].values() if count > 0]
    assert len(nonempty_buckets) >= 5
    assert sum(nonempty_buckets) == 30
