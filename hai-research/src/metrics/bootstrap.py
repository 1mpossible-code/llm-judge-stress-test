"""Bootstrap confidence intervals"""

import numpy as np
from typing import Callable

from src.utils.stats import bootstrap_ci


def compute_bootstrap_ci(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    n_samples: int = 1000,
    confidence_level: float = 0.95,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for a statistic.
    
    Returns:
        (observed_value, lower_ci, upper_ci)
    """
    return bootstrap_ci(values, statistic, n_samples, confidence_level)

