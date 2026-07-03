"""Label normalization mappings for different tasks"""

# Standard label mappings for common tasks
TASK_LABEL_MAPPINGS: dict[str, dict[str, str]] = {
    "offensiveness": {
        # Raw LeWiDi labels -> normalized
        "0": "not_offensive",  # LeWiDi uses 0 for not offensive
        "1": "offensive",      # LeWiDi uses 1 for offensive
        "OFF": "offensive",
        "NOT_OFF": "not_offensive",
        "offensive": "offensive",
        "not_offensive": "not_offensive",
        "hateful": "offensive",
        "normal": "not_offensive",
        "abusive": "offensive",
        "neutral": "not_offensive",
    },
    "hate_speech": {
        "HATE": "offensive",
        "NOT_HATE": "not_offensive",
        "hateful": "offensive",
        "non-hateful": "not_offensive",
        "abusive": "offensive",
        "neutral": "not_offensive",
    },
    "sentiment": {
        "positive": "positive",
        "negative": "negative",
        "neutral": "neutral",
        "POS": "positive",
        "NEG": "negative",
        "NEU": "neutral",
    },
}


def get_label_mapping(task: str) -> dict[str, str] | None:
    """Get label mapping for a task."""
    return TASK_LABEL_MAPPINGS.get(task)


def normalize_labels(
    raw_labels: list[str],
    mapping: dict[str, str] | None = None,
    task: str | None = None,
) -> list[str]:
    """Normalize labels using mapping.
    
    Args:
        raw_labels: List of raw label strings
        mapping: Explicit label mapping dict (overrides task-based mapping)
        task: Task name to look up mapping if mapping not provided
    
    Returns:
        List of normalized labels
    """
    if mapping is None and task:
        mapping = get_label_mapping(task)
    
    if mapping:
        return [mapping.get(label, label) for label in raw_labels]
    
    return raw_labels


def validate_labels(
    labels: list[str],
    allowed_labels: list[str],
    task: str | None = None,
) -> tuple[list[str], list[str]]:
    """Validate labels against allowed set.
    
    Returns:
        (valid_labels, invalid_labels)
    """
    valid = [l for l in labels if l in allowed_labels]
    invalid = [l for l in labels if l not in allowed_labels]
    
    if invalid and task:
        # Try mapping invalid labels
        mapping = get_label_mapping(task)
        if mapping:
            mapped_valid = []
            for label in invalid:
                mapped = mapping.get(label)
                if mapped and mapped in allowed_labels:
                    mapped_valid.append(mapped)
                    valid.append(mapped)
                else:
                    # Keep as invalid
                    pass
    
    return valid, invalid

