"""Pydantic schemas for all data artifacts"""

from typing import Any
from pydantic import BaseModel, Field


# Dataset schemas
class BaseExample(BaseModel):
    """Base dataset example with human disagreement."""
    id: str
    language: str
    text: str
    human_labels: list[str] = Field(..., description="List of human annotation labels")


class NormalizedExample(BaseModel):
    """Normalized example with label distribution."""
    id: str
    language: str
    text: str
    human_label_dist: dict[str, float] = Field(..., description="Normalized label distribution")
    human_disagreement: float = Field(..., ge=0.0, le=1.0, description="Disagreement measure")
    n_annotators: int = Field(..., ge=1)


class PerturbedExample(BaseModel):
    """Generated perturbation."""
    perturb_id: str = Field(..., description="Deterministic perturbation ID")
    base_id: str
    language: str
    factor: str = Field(..., description="Perturbation factor name")
    level: str = Field(..., description="Perturbation level name")
    text: str = Field(..., description="Perturbed text")
    metadata: dict[str, Any] = Field(default_factory=dict)


# Model and judging schemas
class ModelSpec(BaseModel):
    """Model specification."""
    name: str = Field(..., description="Model name/identifier")
    backend: str = Field(..., description="Backend type: anthropic, openai, hf, mock")
    params: dict[str, Any] = Field(default_factory=dict, description="Backend-specific parameters")


class JudgeRequest(BaseModel):
    """Request for model judgment."""
    text: str
    allowed_labels: list[str]
    prompt: str
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=250, ge=1)


class JudgeResponse(BaseModel):
    """Response from model backend."""
    raw_output: str
    usage: dict[str, Any] | None = Field(default=None, description="Token usage info if available")
    status: str = Field(..., description="ok or error")
    error: str | None = Field(default=None)


class Judgment(BaseModel):
    """Final parsed judgment stored in JSONL."""
    run_id: str
    model: str
    backend: str
    prompt_version: str
    base_id: str
    perturb_id: str
    factor: str | None = Field(None, description="Perturbation factor name")
    level: str | None = Field(None, description="Perturbation level name")
    language: str | None = Field(None, description="Language of the text")
    trial_idx: int
    label: str | None = None
    confidence: float | None = Field(None, ge=0.0, le=1.0)
    rationale: str | None = None
    raw_output: str
    status: str = Field(..., description="ok, parse_error, or error")
    error: str | None = None
    usage: dict[str, Any] | None = None
    # Paper-grade logging fields
    input_text: str | None = Field(None, description="Original input text (before perturbation)")
    prompt_hash: str | None = Field(None, description="Hash of the prompt template used")
    perturbation_text_applied: str | None = Field(None, description="The perturbed text that was actually judged")
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSONL storage."""
        return self.model_dump()

