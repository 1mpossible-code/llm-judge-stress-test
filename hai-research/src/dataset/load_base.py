"""Load base dataset from JSONL"""

from typing import Iterator
from pathlib import Path

from src.schemas import BaseExample
from src.utils.io import read_jsonl


def load_base_dataset(path: str | Path, languages: list[str] | None = None) -> Iterator[BaseExample]:
    """Load base dataset from JSONL file.
    
    Args:
        path: Path to JSONL file
        languages: Optional filter by languages
    """
    languages_set = set(languages) if languages else None
    
    for item in read_jsonl(path):
        example = BaseExample(**item)
        if languages_set is None or example.language in languages_set:
            yield example

