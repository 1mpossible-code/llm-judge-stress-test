"""Statistical utilities"""

import numpy as np
from typing import Callable


def bootstrap_ci(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    n_samples: int = 1000,
    confidence_level: float = 0.95,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval.
    
    Returns:
        (statistic, lower_ci, upper_ci)
    """
    observed = statistic(data)
    n = len(data)
    bootstrap_values = []
    
    for _ in range(n_samples):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_values.append(statistic(sample))
    
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_values, 100 * alpha / 2)
    upper = np.percentile(bootstrap_values, 100 * (1 - alpha / 2))
    
    return observed, lower, upper

