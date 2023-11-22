import numpy as np

from test_confounding.ATE.methods.QB import QBSensitivityAnalysis


def test_qb_sensitivity_analysis_init():
    obs_inputs = np.random.randn(100, 5)
    obs_treatment = np.random.binomial(1, 0.5, 100)
    obs_outcome = np.random.randn(100)
    gamma = 1.5
    arm = 1

    sa = QBSensitivityAnalysis(obs_inputs, obs_treatment, obs_outcome, gamma, arm)

    assert np.array_equal(sa.obs_inputs, obs_inputs)
    assert np.array_equal(sa.obs_treatment, obs_treatment)
    assert np.array_equal(sa.obs_outcome, obs_outcome)
    assert sa.gamma == gamma
    assert sa.arm == arm
    assert sa.e_x_func is None or callable(sa.e_x_func)
    assert sa.binary is False or sa.binary is True


def test_propensity_score_calculation():
    obs_inputs = np.random.randn(100, 5)
    obs_treatment = np.random.binomial(1, 0.5, 100)
    obs_outcome = np.random.randn(100)
    gamma = 1.5
    arm = 1

    sa = QBSensitivityAnalysis(obs_inputs, obs_treatment, obs_outcome, gamma, arm)

    assert hasattr(sa, "e_x")
    assert len(sa.e_x) == len(obs_inputs)


def test_solve_bounds():
    obs_inputs = np.random.randn(100, 5)
    obs_treatment = np.random.binomial(1, 0.5, 100)
    obs_outcome = np.random.randn(100)
    gamma = 1.5
    arm = 1
    binary = False

    sa = QBSensitivityAnalysis(obs_inputs, obs_treatment, obs_outcome, gamma, arm, binary)

    lb, ub = sa.solve_bounds()

    assert isinstance(lb, float)
    assert isinstance(ub, float)
    assert lb <= ub
