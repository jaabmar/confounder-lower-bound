from typing import Dict, Optional, Union

import numpy as np
from mlinsights.mlmodel import QuantileLinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, QuantileRegressor
from sklearn.metrics import make_scorer, mean_pinball_loss
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
from statsmodels.api import QuantReg

from test_confounding.datasets.synthetic import alpha_fn, beta_fn

PARAMS_DICT_LGBM = {
    "min_samples_leaf": [1, 5, 10, 20],
    "min_samples_split": [5, 10, 20, 30],
    "n_estimators": [100, 200, 300, 500],
}

PARAMS_DICT_QR = {"alpha": [0], "fit_intercept": [True, False]}


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


def get_quantile_regressor(
    X: np.ndarray,
    y: np.ndarray,
    tau: float,
    weights: Optional[np.ndarray] = None,
    cv_folds: int = 5,
    lgbm_param_dist: Optional[Dict[str, Union[float, int]]] = None,
    quantile_param_dist: Optional[Dict[str, Union[float, int]]] = None,
    fast_solver: bool = False,
) -> Optional[
    Union[QuantReg, QuantileRegressor, GradientBoostingRegressor, QuantileLinearRegression]
]:
    """
    Trains a quantile regressor model on given data and returns the best model based on test scores.
    The function creates a cross-validation object to split the data into train and validation sets with n_splits = cv_folds.
    It further creates a quantile loss scorer using make_scorer() from sklearn.metrics.

    Args:
        X (np.ndarray): Input feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Target variable vector of shape (n_samples,).
        tau (float): Quantile level to estimate.
        weights (Optional[np.ndarray], optional): Sample weights of shape (n_samples,). Default is None.
        cv_folds (int, optional): Number of cross-validation folds. Defaults to 5.
        lgbm_param_dist (Dict[str, Union[float, int]], optional): Parameter distribution
            for LightGBM randomized search. Defaults to PARAMS_DICT_LGBM.
        quantile_param_dist (Dict[str, Union[float, int]], optional): Parameter distribution
            for Quantile Regressor randomized search. Defaults to PARAMS_DICT_QR.
        fast_solver (bool, optional): Use a faster solver for linear quantile regression. Default is False.

    Returns:
        Optional[Union[QuantReg, QuantileRegressor, GradientBoostingRegressor, QuantileLinearRegression]]:
            Best model object based on test scores.
    """

    if quantile_param_dist is None:
        quantile_param_dist = PARAMS_DICT_QR

    if lgbm_param_dist is None:
        lgbm_param_dist = PARAMS_DICT_LGBM

    # Solve linear quantile regression approximately (no linear program) --> much faster and scales to 100k+ datapoints
    if fast_solver:
        quant_reg = QuantileLinearRegression(quantile=tau, max_iter=500)
        quant_reg.fit(X, y)
        return quant_reg

    # Create cross-validation object
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # Create quantile loss scorer
    neg_mean_pinball_loss_scorer = make_scorer(
        mean_pinball_loss, alpha=tau, greater_is_better=False  # maximize the negative loss
    )

    # Create model dictionary
    models = {}
    lgbm_model = GradientBoostingRegressor(
        loss="quantile", alpha=tau
    )  # Include sklearn.ensemble.HistGradientBoostingRegressor for large datasets
    models["LightGBM"] = (RandomizedSearchCV, lgbm_model, lgbm_param_dist)

    quantile_model = QuantileRegressor(quantile=tau, solver="highs")
    models["Quantile Regressor"] = (GridSearchCV, quantile_model, quantile_param_dist)

    # Perform search and select best model
    best_score = np.inf
    best_model = None

    for name, (search, model, param_dist) in models.items():
        hyperparam_search = search(
            model,
            param_dist,
            cv=cv,
            scoring=neg_mean_pinball_loss_scorer,
            n_jobs=1,
        ).fit(X, y, sample_weight=weights)

        print(f"Best Parameters ({name}): {hyperparam_search.best_params_}")
        print(f"Cross Validation Score ({name}): {abs(hyperparam_search.best_score_)}")

        if abs(hyperparam_search.best_score_) < best_score:
            best_score = abs(hyperparam_search.best_score_)
            best_model = hyperparam_search.best_estimator_

    return best_model


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
