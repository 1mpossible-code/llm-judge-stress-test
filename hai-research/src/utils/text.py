"""Text processing utilities"""

import re
from typing import Any


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text."""
    return " ".join(text.split())


def extract_confidence(text: str) -> float | None:
    """Extract confidence value from text, handling percentages."""
    # Look for CONFIDENCE: pattern first (allows negative and decimal)
    confidence_pattern = r"CONFIDENCE:\s*(-?\d+\.?\d*)%?"
    match = re.search(confidence_pattern, text, re.IGNORECASE)
    if match:
        val = float(match.group(1))
        # If explicitly marked with %, treat as percentage
        if "%" in match.group(0):
            val = val / 100.0
        # If value > 1 and not marked with %, clip to 1.0 (treat as out-of-range)
        elif val > 1.0:
            val = 1.0
        return max(0.0, min(1.0, val))  # Clip to [0, 1]
    
    # Tolerate compact outputs such as "offensive: 1" or "not_offensive: 0.8".
    label_score_pattern = r"\b[A-Za-z_]+:\s*(0?\.\d+|0(?:\.0+)?|1(?:\.0+)?)\b"
    match = re.search(label_score_pattern, text, re.IGNORECASE)
    if match:
        val = float(match.group(1))
        return max(0.0, min(1.0, val))

    # Look for standalone percentages
    percent_pattern = r"(-?\d+\.?\d*)%"
    match = re.search(percent_pattern, text, re.IGNORECASE)
    if match:
        val = float(match.group(1))
        return max(0.0, min(1.0, val / 100.0))
    
    # Look for decimal in [0, 1] range (skip negatives in this fallback)
    decimal_pattern = r"\b(\d+\.\d+)\b"
    match = re.search(decimal_pattern, text)
    if match:
        val = float(match.group(1))
        if 0.0 <= val <= 1.0:
            return val
    
    return None

