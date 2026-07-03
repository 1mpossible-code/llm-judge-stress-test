"""Perturbation generator registry with category support"""

from typing import Callable
from src.schemas import PerturbedExample


PerturbationGenerator = Callable[[str, str], str]  # (base_text, level) -> perturbed_text

_registry: dict[str, dict[str, PerturbationGenerator]] = {}
_category_map: dict[str, str] = {}  # factor -> category


def register_factor(
    factor_name: str, 
    level_name: str, 
    generator: PerturbationGenerator,
    category: str | None = None,
) -> None:
    """Register a perturbation generator.
    
    Args:
        factor_name: Factor identifier
        level_name: Level identifier  
        generator: Function that generates perturbed text
        category: Optional category (e.g., "authority_signal", "cultural_framing")
    """
    if factor_name not in _registry:
        _registry[factor_name] = {}
    _registry[factor_name][level_name] = generator
    
    if category:
        _category_map[factor_name] = category


def get_generator(factor_name: str, level_name: str) -> PerturbationGenerator | None:
    """Get perturbation generator for factor/level."""
    return _registry.get(factor_name, {}).get(level_name)


def get_category(factor_name: str) -> str | None:
    """Get category for a factor."""
    return _category_map.get(factor_name)


def list_factors() -> list[str]:
    """List all registered factor names."""
    return list(_registry.keys())


def get_perturbation_info(factor_name: str, level_name: str) -> dict:
    """Get metadata for a perturbation factor/level."""
    generator = get_generator(factor_name, level_name)
    category = get_category(factor_name)
    
    return {
        "factor": factor_name,
        "level": level_name,
        "category": category or "unknown",
        "registered": generator is not None,
    }

