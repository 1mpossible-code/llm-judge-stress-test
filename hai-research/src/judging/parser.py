"""Strict output parser for judge responses"""

import re
from src.schemas import Judgment
from src.utils.text import extract_confidence


def parse_judgment(
    raw_output: str,
    allowed_labels: list[str],
    run_id: str,
    model: str,
    backend: str,
    prompt_version: str,
    base_id: str,
    perturb_id: str,
    factor: str | None = None,
    level: str | None = None,
    language: str | None = None,
    trial_idx: int = 0,
    usage: dict | None = None,
    input_text: str | None = None,
    prompt_hash: str | None = None,
    perturbation_text_applied: str | None = None,
) -> Judgment:
    """Parse model output into structured judgment.
    
    Expected format:
    LABEL: <label>
    CONFIDENCE: <float>
    RATIONALE: <text>
    """
    # Normalize whitespace
    text = " ".join(raw_output.split())
    
    # Extract label. Prefer the required LABEL: field, but tolerate common
    # provider deviations such as "not_offensive: 0.8" or a bare allowed label.
    label = None
    label_match = re.search(r"LABEL:\s*(\S+)", text, re.IGNORECASE)
    if not label_match:
        allowed_pattern = "|".join(re.escape(label) for label in sorted(allowed_labels, key=len, reverse=True))
        label_match = re.search(rf"(?:^|\b)({allowed_pattern})(?:\s*:|\b)", text, re.IGNORECASE)

    if label_match:
        candidate = label_match.group(1).strip()
        # Check if it matches an allowed label (case-insensitive)
        for allowed in allowed_labels:
            if candidate.lower() == allowed.lower():
                label = allowed
                break
    
    # Extract confidence
    confidence = extract_confidence(text)
    
    # Extract rationale
    rationale = None
    rationale_match = re.search(r"RATIONALE:\s*(.+?)(?:\n|$)", text, re.IGNORECASE | re.DOTALL)
    if rationale_match:
        rationale = rationale_match.group(1).strip()
        # Clean up if it spans multiple lines
        rationale = " ".join(rationale.split())
    
    # Determine status
    if label is None or confidence is None:
        status = "parse_error"
        error = "Failed to parse label or confidence"
    else:
        status = "ok"
        error = None
    
    return Judgment(
        run_id=run_id,
        model=model,
        backend=backend,
        prompt_version=prompt_version,
        base_id=base_id,
        perturb_id=perturb_id,
        factor=factor,
        level=level,
        language=language,
        trial_idx=trial_idx,
        label=label,
        confidence=confidence,
        rationale=rationale,
        raw_output=raw_output,
        status=status,
        error=error,
        usage=usage,
        input_text=input_text,
        prompt_hash=prompt_hash,
        perturbation_text_applied=perturbation_text_applied,
    )

