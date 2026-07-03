"""Normalize base examples with label distributions"""

from collections import Counter
import numpy as np

from src.schemas import BaseExample, NormalizedExample


def compute_disagreement(labels: list[str]) -> float:
    """Compute normalized disagreement (entropy-based).
    
    Returns value in [0, 1] where 1 = maximum disagreement.
    """
    if len(labels) == 0:
        return 0.0
    
    counts = Counter(labels)
    n = len(labels)
    probs = [count / n for count in counts.values()]
    
    # Normalized entropy
    if len(probs) == 1:
        return 0.0
    
    entropy = -sum(p * np.log2(p) for p in probs)
    max_entropy = np.log2(len(counts))
    return entropy / max_entropy if max_entropy > 0 else 0.0


def normalize_example(example: BaseExample, allowed_labels: list[str] | None = None) -> NormalizedExample:
    """Normalize base example to compute label distribution and disagreement."""
    labels = example.human_labels
    
    if allowed_labels:
        labels = [l for l in labels if l in allowed_labels]
    
    counts = Counter(labels)
    n = len(labels)
    
    # Compute normalized distribution
    label_dist = {label: counts.get(label, 0) / n for label in (allowed_labels or list(counts.keys()))}
    
    # Compute disagreement
    disagreement = compute_disagreement(labels)
    
    return NormalizedExample(
        id=example.id,
        language=example.language,
        text=example.text,
        human_label_dist=label_dist,
        human_disagreement=disagreement,
        n_annotators=n,
    )

