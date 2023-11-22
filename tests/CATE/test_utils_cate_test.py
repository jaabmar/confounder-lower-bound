from unittest.mock import Mock

import numpy as np
import pytest

from test_confounding.CATE.utils_cate_test import (
    compute_bootstrap_variance,
    compute_bootstrap_variances_cate_bounds,
    resample_and_calculate_mean,
)


def test_compute_bootstrap_variance_valid_input():
    Y = np.random.randn(100)
    T = np.random.binomial(1, 0.5, 100)
    n_bootstraps = 10
    arm = 1

    variance = compute_bootstrap_variance(Y, T, n_bootstraps, arm)

    assert isinstance(variance, float)
    assert variance >= 0


def test_compute_bootstrap_variance_invalid_input_lengths():
    Y = np.random.randn(100)
    T = np.random.binomial(1, 0.5, 101)  # Different length
    n_bootstraps = 10

    with pytest.raises(ValueError):
        compute_bootstrap_variance(Y, T, n_bootstraps)


def test_compute_bootstrap_variance_invalid_bootstraps():
    Y = np.random.randn(100)
    T = np.random.binomial(1, 0.5, 100)
    n_bootstraps = 0

    with pytest.raises(ValueError):
        compute_bootstrap_variance(Y, T, n_bootstraps)


def test_compute_bootstrap_variances_cate_bounds():
    bounds_est = Mock()
    bounds_est.effect.side_effect = lambda x: (np.array([0.1]), np.array([0.2]))  # Mocked bounds
    x_rct = np.random.randn(100, 5)
    n_bootstraps = 10

    (
        lower_variance,
        upper_variance,
        lower_quantile,
        upper_quantile,
    ) = compute_bootstrap_variances_cate_bounds(bounds_est, x_rct, n_bootstraps)

    assert isinstance(lower_variance, float)
    assert isinstance(upper_variance, float)
    assert lower_variance >= 0
    assert upper_variance >= 0
    assert lower_quantile <= upper_quantile


def test_resample_and_calculate_mean_treated():
    Y = np.random.randn(100)
    T = np.random.binomial(1, 0.5, 100)
    arm = 1

    mean_resampled = resample_and_calculate_mean(Y, T, arm)

    assert isinstance(mean_resampled, float)


def test_resample_and_calculate_mean_control():
    Y = np.random.randn(100)
    T = np.random.binomial(1, 0.5, 100)
    arm = 0

    mean_resampled = resample_and_calculate_mean(Y, T, arm)

    assert isinstance(mean_resampled, float)


def test_resample_and_calculate_mean_cate():
    Y = np.random.randn(100)
    T = np.random.binomial(1, 0.5, 100)
    arm = None

    mean_resampled = resample_and_calculate_mean(Y, T, arm)

    assert isinstance(mean_resampled, float)
