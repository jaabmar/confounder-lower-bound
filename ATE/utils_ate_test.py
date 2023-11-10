from concurrent.futures import ProcessPoolExecutor
from typing import Tuple

import numpy as np


def compute_group_phi(
    y_rct: np.ndarray,
    t_rct: np.ndarray,
    s_rct: np.ndarray,
    p_vector: np.ndarray,
) -> float:
    """
    Compute the group phi based on the given inputs.

    Args:
        y_rct: An array of outcome values.
        t_rct: An array of treatment indicators (1 or 0).
        s_rct: An array of group indicators.
        p_vector: An array of probabilities associated with each group.

    Returns:
        The computed group phi value.

    """
    unique_s = np.unique(s_rct)
    n_test = np.sum(s_rct[t_rct == 1, np.newaxis] == unique_s, axis=0)
    y_test = np.sum(
        [
            np.sum(y_rct[(t_rct == 1) & (s_rct == group)]) * p_vector[idx] / n_test[idx]
            for idx, group in enumerate(unique_s)
        ]
    )

    return y_test


def compute_group_variance(
    y_rct: np.ndarray,
    t_rct: np.ndarray,
    s_rct: np.ndarray,
    group_probs: np.ndarray,
    n_resamples: int = 1000,
) -> float:
    """
    Compute the total group variance.

    Args:
        y_rct: An array of outcome values.
        t_rct: An array of treatment indicators (1 or 0).
        s_rct: An array of group indicators.
        group_probs: An array of probabilities associated with each group.
        n_resamples: The number of resamples for bootstrap sampling.

    Returns:
        The computed total group variance.

    """

    def compute_group_bootstrap_means(args: Tuple[int, np.ndarray]) -> float:
        idx, y_rct_group = args
        bootstrap_y = y_rct_group[idx]
        group_mean = np.mean(bootstrap_y)
        return group_mean

    y1_rct = y_rct[t_rct == 1]
    s1_rct = s_rct[t_rct == 1]

    group_variances = []

    for test_group in np.unique(s_rct):
        y1_group_rct = y1_rct[s1_rct == test_group]

        bootstrap_samples = [
            (
                np.random.choice(len(y1_group_rct), size=len(y1_group_rct), replace=True),
                y1_group_rct,
            )
            for _ in range(n_resamples)
        ]
        with ProcessPoolExecutor() as executor:
            group_bootstrap_means = list(
                executor.map(compute_group_bootstrap_means, bootstrap_samples)
            )

        group_variances.append(np.var(group_bootstrap_means))

    total_group_variance = np.sum(np.multiply(group_variances, group_probs**2))

    return total_group_variance
