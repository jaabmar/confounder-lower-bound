import numpy as np

from test_confounding.ATE.ate_bounds import BootstrapSensitivityAnalysis


def test_bootstrap_sensitivity_analysis_init():
    inputs = np.random.randn(100, 5)
    treatment = np.random.binomial(1, 0.5, 100)
    outcome = np.random.randn(100)
    gammas = [1.0, 1.5, 2.0]
    sa = BootstrapSensitivityAnalysis("SA", inputs, treatment, outcome, gammas)

    assert sa.sa_name == "SA"
    assert np.array_equal(sa.obs_inputs, inputs)
    assert np.array_equal(sa.obs_treatment, treatment)
    assert np.array_equal(sa.obs_outcome, outcome)
    assert sa.gammas == gammas
    assert sa.seed == 50
    assert sa.e_x_func is None
    assert sa.bounds_dist is None


def test_bootstrap_method():
    inputs = np.random.randn(100, 5)
    treatment = np.random.binomial(1, 0.5, 100)
    outcome = np.random.randn(100)
    gammas = [1.0, 1.5, 2.0]
    sa = BootstrapSensitivityAnalysis("QB", inputs, treatment, outcome, gammas)

    num_samples = 10
    sample_size = 50

    bounds_dist = sa.bootstrap(num_samples, sample_size)

    assert isinstance(bounds_dist, dict)
    for gamma in gammas:
        assert str(gamma) in bounds_dist
        assert len(bounds_dist[str(gamma)][0]) == num_samples
        assert len(bounds_dist[str(gamma)][1]) == num_samples


def test_solve_sample_bounds():
    inputs = np.random.randn(100, 5)
    treatment = np.random.binomial(1, 0.5, 100)
    outcome = np.random.randn(100)
    gammas = [1.0, 1.5, 2.0]
    sa = BootstrapSensitivityAnalysis("ZSB", inputs, treatment, outcome, gammas)

    sample_indices = np.random.choice(len(inputs), size=len(inputs), replace=True)
    args = (sample_indices, inputs, treatment, outcome)

    lower_bounds, upper_bounds = sa.solve_sample_bounds(args)

    assert isinstance(lower_bounds, list)
    assert isinstance(upper_bounds, list)
    assert len(lower_bounds) == len(gammas)
    assert len(upper_bounds) == len(gammas)


def test_get_all_quantile_models():
    inputs = np.random.randn(100, 5)
    treatment = np.random.binomial(1, 0.5, 100)
    outcome = np.random.randn(100)
    gammas = [1.0, 1.5, 2.0]
    sa = BootstrapSensitivityAnalysis("QB", inputs, treatment, outcome, gammas)
    fast_quantile = True

    treated_models_dict, control_models_dict = sa.get_all_quantile_models(fast_quantile)

    assert isinstance(treated_models_dict, dict)
    assert isinstance(control_models_dict, dict)
    for gamma in gammas:
        assert gamma in treated_models_dict
        assert gamma in control_models_dict
        assert "ub_qr_func" in treated_models_dict[gamma]
        assert "lb_qr_func" in treated_models_dict[gamma]
        assert "ub_qr_func" in control_models_dict[gamma]
        assert "lb_qr_func" in control_models_dict[gamma]
