"""Deterministic caching system"""

from pathlib import Path
from typing import Any
from src.schemas import JudgeRequest, JudgeResponse, ModelSpec
from src.utils.hashing import hash_dict, hash_string
from src.utils.io import read_json, write_json


def make_cache_key(
    model_spec: ModelSpec,
    prompt_hash: str,
    perturb_id: str,
    trial_idx: int,
    temperature: float,
    max_tokens: int,
) -> str:
    """Create deterministic cache key."""
    cache_dict = {
        "model": model_spec.name,
        "backend": model_spec.backend,
        "params": model_spec.params,
        "prompt_hash": prompt_hash,
        "perturb_id": perturb_id,
        "trial_idx": trial_idx,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    return hash_dict(cache_dict)


def get_cache_path(run_dir: Path, cache_key: str) -> Path:
    """Get cache file path for a key."""
    return run_dir / "cache" / f"{cache_key}.json"


def load_cache(run_dir: Path, cache_key: str) -> JudgeResponse | None:
    """Load cached response if it exists."""
    cache_path = get_cache_path(run_dir, cache_key)
    if not cache_path.exists():
        return None
    
    try:
        data = read_json(cache_path)
        return JudgeResponse(**data)
    except Exception:
        return None


def save_cache(run_dir: Path, cache_key: str, response: JudgeResponse) -> None:
    """Save response to cache."""
    cache_path = get_cache_path(run_dir, cache_key)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(cache_path, response.model_dump())

