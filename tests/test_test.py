from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from test_confounding.cate_bounds.cate_bounds import MultipleCATEBoundEstimators
from test_confounding.test import (
    construct_ate_test_statistic,
    construct_cate_test_statistic,
    hypothesis_test,
    run_multiple_ate_hypothesis_test,
    run_multiple_cate_hypothesis_test,
)


@pytest.fixture
def mock_bounds_estimator():
    mock_estimator = MultipleCATEBoundEstimators(
        gammas=[5.0],
        n_bootstrap=100,
    )
    mock_estimator.dict_bound_estimators = {"5.0": MagicMock()}
    return mock_estimator


def test_construct_ate_test_statistic():
    mean_rct = 2.0
    mean_ub = 3.0
    mean_lb = 1.0
    var_rct = 1.0
    var_ub = 5.0
    var_lb = 5.0
    gamma = 5.0

    stat, results = construct_ate_test_statistic(
        mean_rct, mean_ub, mean_lb, var_rct, var_ub, var_lb, gamma
    )

    assert isinstance(stat, float)
    assert isinstance(results, dict)
    assert results["gamma"] == gamma
    assert results["mean_rct"] == mean_rct
    assert results["var_rct"] == var_rct
    assert results["mean_ub"] == mean_ub
    assert results["var_ub"] == var_ub
    assert results["mean_lb"] == mean_lb
    assert results["var_lb"] == var_lb
    assert "test_statistic_min" in results
    assert "test_statistic_plus" in results


def test_construct_cate_test_statistic():
    ate = 2.0
    ate_variance = 1.0
    ate_ub = 3.0
    ate_lb = 1.0
    ate_variance_ub = 5.0
    ate_variance_lb = 5.0
    gamma = 5.0

    stat, results = construct_cate_test_statistic(
        ate, ate_variance, ate_ub, ate_lb, ate_variance_ub, ate_variance_lb, gamma
    )

    assert isinstance(stat, float)
    assert isinstance(results, dict)
    assert results["gamma"] == gamma
    assert results["mean_rct"] == ate
    assert results["var_rct"] == ate_variance
    assert results["mean_ub"] == ate_ub
    assert results["var_ub"] == ate_variance_ub
    assert results["mean_lb"] == ate_lb
    assert results["var_lb"] == ate_variance_lb
    assert "test_statistic_min" in results
    assert "test_statistic_plus" in results


def test_run_multiple_ate_hypothesis_test():
    mean_rct = np.array([2.0])
    var_rct = 1.0
    bounds_dist = {"5.0": (np.array([1]), np.array([3]))}  # Example data
    alpha = 0.05
    user_conf = [5.0]

    with patch("test_confounding.test.construct_ate_test_statistic") as mock_ate, patch(
        "test_confounding.test.hypothesis_test"
    ) as mock_hypothesis:
        mock_ate.return_value = (1.0, {})
        mock_hypothesis.return_value = 1
        results = run_multiple_ate_hypothesis_test(mean_rct, var_rct, bounds_dist, alpha, user_conf)

    assert isinstance(results, dict)
    assert "critical_gamma" in results
    assert mock_ate.called
    assert mock_hypothesis.called


def test_run_multiple_cate_hypothesis_test(mock_bounds_estimator):
    ate = 2.0
    ate_variance = 1.0
    alpha = 0.05
    x_rct = np.array([1, 2, 3])  # Example data
    user_conf = [5.0]

    # Mocking predict_bounds and estimate_bootstrap_variances methods
    mock_bounds_estimator.dict_bound_estimators["5.0"].predict_bounds.return_value = (
        np.array([1]),
        np.array([3]),
    )
    mock_bounds_estimator.dict_bound_estimators["5.0"].estimate_bootstrap_variances.return_value = (
        5.0,
        5.0,
        5.0,
        5.0,
    )

    with patch("test_confounding.test.construct_cate_test_statistic") as mock_cate, patch(
        "test_confounding.test.hypothesis_test"
    ) as mock_hypothesis:
        mock_cate.return_value = (1.0, {})
        mock_hypothesis.return_value = 1
        results = run_multiple_cate_hypothesis_test(
            bounds_estimator=mock_bounds_estimator,
            ate=ate,
            ate_variance=ate_variance,
            alpha=alpha,
            x_rct=x_rct,
            user_conf=user_conf,
            verbose=True,
        )

    assert isinstance(results, dict)
    assert "critical_gamma" in results
    assert mock_cate.called
    assert mock_hypothesis.called
    assert mock_bounds_estimator.dict_bound_estimators["5.0"].predict_bounds.called
    assert mock_bounds_estimator.dict_bound_estimators["5.0"].estimate_bootstrap_variances.called


def test_hypothesis_test():
    test_statistic = 1.5
    alpha = 0.05

    result = hypothesis_test(test_statistic, alpha)

    assert result in [0, 1]
