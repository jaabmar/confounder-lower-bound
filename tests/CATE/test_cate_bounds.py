import numpy as np

from test_confounding.CATE.cate_bounds import (
    CATEBoundsEstimator,
    MultipleCATEBoundEstimators,
)


def test_multiple_cate_bound_estimators_init():
    gammas = [1.0, 5.0, 10.0]
    n_bootstrap = 100
    cate_estimators = MultipleCATEBoundEstimators(gammas, n_bootstrap)

    assert len(cate_estimators.dict_bound_estimators) == len(gammas)
    for gamma in gammas:
        assert str(gamma) in cate_estimators.dict_bound_estimators


def test_multiple_cate_bound_estimators_fit():
    gammas = [1.0, 5.0, 10.0]
    n_bootstrap = 100
    cate_estimators = MultipleCATEBoundEstimators(gammas, n_bootstrap)

    x_obs = np.random.randn(100, 5)
    t_obs = np.random.binomial(1, 0.5, 100)
    y_obs = np.random.randn(100)
    sample_weight = np.random.rand(100)

    cate_estimators.fit(x_obs, t_obs, y_obs, sample_weight)


def test_cate_bounds_estimator_methods():
    cate_estimator = CATEBoundsEstimator(
        binary=False,
        user_conf=1.0,
        n_bootstrap=100,
        # Additional arguments as needed
    )

    x_obs = np.random.randn(100, 5)
    t_obs = np.random.binomial(1, 0.5, 100)
    y_obs = np.random.randn(100)

    cate_estimator.fit(x_obs, t_obs, y_obs)

    # Mock data
    x_rct = np.random.randn(100, 5)

    # Test predict_bounds
    lower_bounds, upper_bounds = cate_estimator.predict_bounds(x_rct)
    assert isinstance(lower_bounds, np.ndarray)
    assert isinstance(upper_bounds, np.ndarray)

    # Test compute_ate_bounds
    mean_lb, mean_ub = cate_estimator.compute_ate_bounds(x_rct)
    assert isinstance(mean_lb, float)
    assert isinstance(mean_ub, float)

    # Test estimate_bootstrap_variances
    var_lb, var_ub, quantile_lb, quantile_ub = cate_estimator.estimate_bootstrap_variances(x_rct)
    assert isinstance(var_lb, float)
    assert isinstance(var_ub, float)
    assert isinstance(quantile_lb, float)
    assert isinstance(quantile_ub, float)
