"""Configuration system with YAML parsing and Pydantic validation"""

from pathlib import Path
from typing import Any
import yaml
from pydantic import BaseModel, Field

from src.schemas import ModelSpec


class DatasetConfig(BaseModel):
    """Dataset configuration."""
    path: str
    allowed_labels: list[str]
    languages: list[str] = Field(default_factory=lambda: ["en"])
    limit: int | None = Field(default=None, description="Limit number of examples to process (for testing)")


class PerturbationFactor(BaseModel):
    """Single perturbation factor configuration."""
    name: str
    levels: list[str]


class PerturbationsConfig(BaseModel):
    """Perturbations configuration."""
    factors: list[PerturbationFactor]
    seed: int = 1337


class JudgingConfig(BaseModel):
    """Judging configuration."""
    mode: str = "classification"
    prompt_file: str
    prompt_version: str
    repeats: int = Field(default=3, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=250, ge=1)
    max_workers: int = Field(default=5, ge=1, description="Maximum concurrent judge calls")
    models: list[ModelSpec]


class MetricsConfig(BaseModel):
    """Metrics configuration."""
    bootstrap_samples: int = Field(default=200, ge=1)
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)


class OutputConfig(BaseModel):
    """Output configuration."""
    runs_dir: str = "runs"
    save_figures: bool = True
    figure_format: str = "png"


class CacheConfig(BaseModel):
    """Cache configuration."""
    enabled: bool = True
    max_retries: int = Field(default=5, ge=1)
    backoff_base_seconds: float = Field(default=1.0, gt=0.0)


class ExperimentConfig(BaseModel):
    """Complete experiment configuration."""
    experiment_name: str
    dataset: DatasetConfig
    perturbations: PerturbationsConfig
    judging: JudgingConfig
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)


def load_config(config_path: str | Path) -> ExperimentConfig:
    """Load and validate YAML config file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    return ExperimentConfig(**data)

