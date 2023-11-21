from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, Union

import numpy as np

from CATE.BLearner.models.blearner.BLearner import (
    BinaryCATEBLearner,
    BinaryPhiBLearner,
    BLearner,
    PhiBLearner,
)


def compute_bootstrap_variance(
    Y: np.ndarray, T: np.ndarray, n_bootstraps: int, arm: Optional[int] = None
) -> float:
    """
    Computes the bootstrap variance estimate of the mean of Y where T == arm or the ATE.

    Args:
        Y (numpy.ndarray): The array of outputs.
        T (numpy.ndarray): The array of treatments.
        n_bootstraps (int): The number of bootstrap resampling iterations.
        arm (Optional[int]): Compute variance for the treated (1) or control (0) group. None for the CATE.

    Returns:
        float: The bootstrap variance estimate.

    Raises:
        ValueError: If Y and T have different lengths or if n_bootstraps is less than 1.
    """
    # Perform input validation
    if len(Y) != len(T):
        raise ValueError("Y and T must have the same length.")
    if n_bootstraps < 1:
        raise ValueError("n_bootstraps must be greater than or equal to 1.")

    # Perform bootstrap resampling using parallelization
    with ThreadPoolExecutor() as executor:
        bootstrap_means = list(
            executor.map(
                resample_and_calculate_mean,
                [Y] * n_bootstraps,
                [T] * n_bootstraps,
                [arm] * n_bootstraps,
            )
        )

    # Calculate bootstrap variance estimate
    bootstrap_variance = float(np.var(bootstrap_means))

    return bootstrap_variance


def compute_bootstrap_variances_cate_bounds(
    bounds_est: Union[PhiBLearner, BinaryPhiBLearner, BLearner, BinaryCATEBLearner],
    x_rct: np.ndarray,
    n_bootstraps: int,
) -> Tuple[float, float]:
    """
    Estimates the bootstrap variances of upper and lower bounds means.

    Args:
        bounds_est (callable): Function that calculates upper and lower bounds for a given dataset.
        x_rct (np.ndarray): The RCT dataset.
        n_bootstraps (int): Number of bootstrap samples to generate.

    Returns:
        Tuple[float, float]: The bootstrap variances of the upper and lower bounds means.
    """

    n_samples, _ = x_rct.shape

    # Perform bootstrapping
    upper_bounds_mean_list = []
    lower_bounds_mean_list = []
    for _ in range(n_bootstraps):
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        lower_bounds, upper_bounds = bounds_est.effect(x_rct[bootstrap_indices])
        upper_bounds_mean_list.append(np.mean(upper_bounds))
        lower_bounds_mean_list.append(np.mean(lower_bounds))

    # Calculate the bootstrap variances of the upper and lower bounds means
    upper_mean_variance = np.var(upper_bounds_mean_list)
    lower_mean_variance = np.var(lower_bounds_mean_list)
    lower_quantile = np.min(lower_bounds_mean_list)
    upper_quantile = np.max(upper_bounds_mean_list)

    return float(lower_mean_variance), float(upper_mean_variance), lower_quantile, upper_quantile


def resample_and_calculate_mean(Y: np.ndarray, T: np.ndarray, arm: Optional[int] = None) -> float:
    """
    Resamples the data arrays Y and T and calculates the mean of Y where T == arm or the ATE.

    Args:
        Y (numpy.ndarray): The array of outputs.
        T (numpy.ndarray): The array of treatments.
        arm (Optional[int]): Compute variance for the treated (1) or control (0) group. None for the CATE.

    Returns:
        float: The mean of Y where T == arm or the ATE for the resampled data.
    """
    # Generate random indices with replacement
    n = len(Y)
    indices = np.random.choice(range(n), size=n, replace=True)

    # Resample Y and T using the generated indices
    Y_resampled = Y[indices]
    T_resampled = T[indices]

    # Calculate mean
    if arm is not None:
        mean_resampled = float(np.mean(Y_resampled[T_resampled == arm]))
    else:
        mean_resampled = float(
            np.mean(Y_resampled[T_resampled == 1]) - np.mean(Y_resampled[T_resampled == 0])
        )

    return mean_resampled
