"""Deterministic hashing utilities"""

import hashlib
import json
from typing import Any


def hash_dict(obj: dict[str, Any]) -> str:
    """Create deterministic SHA256 hash from dict."""
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode()).hexdigest()


def hash_string(s: str) -> str:
    """Create deterministic SHA256 hash from string."""
    return hashlib.sha256(s.encode()).hexdigest()

