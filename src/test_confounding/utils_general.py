import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from test_confounding.ate_bounds.utils_ate_bounds import get_quantile_regressor
from test_confounding.datasets.synthetic import alpha_fn, beta_fn


def estimate_ATE_HT(x_obs, t_obs, y_obs):
    """
    Estimate the Average Treatment Effect (ATE) using the Horvitz-Thompson estimator.

    Parameters:
    x_obs (array-like): Covariates.
    t_obs (array-like): Treatment indicators (1 for treated, 0 for control).
    y_obs (array-like): Observed outcomes.

    Returns:
    float: The estimated Average Treatment Effect using the Horvitz-Thompson estimator.
    """

    # 1. Estimate propensity scores
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(x_obs, t_obs)
    prop_scores = log_reg.predict_proba(x_obs)[:, 1]

    # 2. Compute ATE using the Horvitz-Thompson estimator
    N = len(t_obs)
    weights_treated = t_obs / prop_scores
    weights_control = (1 - t_obs) / (1 - prop_scores)
    ATE_HT = (1 / N) * np.sum(weights_treated * y_obs - weights_control * y_obs)

    return ATE_HT


def e_x_func(x):
    return (1 + np.exp(-(x * 0.75 + 0.5))) ** -1


def adv_propensity_plus(
    nominal_e_x: float, x: np.ndarray, y: np.ndarray, t: np.ndarray, gamma: float
) -> np.ndarray:
    """
    Adversarial propensity score calculation using quantile regression.

    Args:
        nominal_e_x (float): Nominal propensity score.
        x (np.ndarray): Feature matrix for the quantile regression.
        y (np.ndarray): Target variable for the quantile regression.
        t (np.ndarray): Treatment indicator.
        gamma (float): Confounding strength.

    Returns:
        np.ndarray: Adversarial propensity scores.
    """
    alpha = alpha_fn(nominal_e_x, gamma)
    beta = beta_fn(nominal_e_x, gamma)
    a = 1 / beta
    b = 1 / alpha
    qr = get_quantile_regressor(x[t == 1], y[t == 1], gamma / (gamma + 1), fast_solver=True)
    if qr is not None:
        ind = y > qr.predict(x)
        return a * ind + b * (1 - ind)


class CATEEstimator:
    """
    Causal inference model for estimating the Conditional Average Treatment Effect (CATE).

    Attributes:
        propensity_estimator (LogisticRegression): Model for estimating propensity scores.
        outcome0_rf (RandomForestRegressor): Random forest model for outcome estimation when treatment is not applied.
        outcome1_rf (RandomForestRegressor): Random forest model for outcome estimation when treatment is applied.
    """

    def __init__(self, seed=50):
        self.propensity_estimator = LogisticRegression(
            C=1,
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.7,
            max_iter=10000,
            random_state=seed,
        )
        self.outcome0_rf = RandomForestRegressor(
            n_estimators=200, min_samples_leaf=0.01, max_depth=6, random_state=seed
        )
        self.outcome1_rf = RandomForestRegressor(
            n_estimators=200, min_samples_leaf=0.01, max_depth=6, random_state=seed
        )

    def fit(self, data):
        outcome0_data = data[data["T"] == 0]
        outcome1_data = data[data["T"] == 1]

        X_propensity = data.drop(columns=["Y", "T"])
        self.propensity_estimator.fit(X_propensity, data["T"])

        X_outcome0 = outcome0_data.drop(columns=["Y", "T"])
        y_outcome0 = outcome0_data["Y"]
        self.outcome0_rf.fit(X_outcome0, y_outcome0)

        X_outcome1 = outcome1_data.drop(columns=["Y", "T"])
        y_outcome1 = outcome1_data["Y"]
        self.outcome1_rf.fit(X_outcome1, y_outcome1)

    def predict(self, data):
        data = data.drop(columns=["Y", "T"], errors="ignore")
        propensity_scores = self.propensity_estimator.predict_proba(data)[:, 1]
        outcome0_predictions = self.outcome0_rf.predict(data)
        outcome1_predictions = self.outcome1_rf.predict(data)

        ipw_outcome0 = outcome0_predictions / (1 - propensity_scores)
        ipw_outcome1 = outcome1_predictions / propensity_scores

        cate_predictions = ipw_outcome1 - ipw_outcome0

        return propensity_scores, outcome0_predictions, outcome1_predictions, cate_predictions
