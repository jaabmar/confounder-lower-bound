from typing import Dict, Optional, Union

import numpy as np
# from mlinsights.mlmodel import QuantileLinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import make_scorer, mean_pinball_loss
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV

PARAMS_DICT_LGBM = {
    "min_samples_leaf": [1, 5, 10, 20],
    "min_samples_split": [5, 10, 20, 30],
    "n_estimators": [100, 200, 300, 500],
}

PARAMS_DICT_QR = {"alpha": [0], "fit_intercept": [True, False]}


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
    Union[QuantileRegressor, GradientBoostingRegressor]
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
        Optional[Union[QuantileRegressor, GradientBoostingRegressor]]:
            Best model object based on test scores.
    """

    if quantile_param_dist is None:
        quantile_param_dist = PARAMS_DICT_QR

    if lgbm_param_dist is None:
        lgbm_param_dist = PARAMS_DICT_LGBM

    # Solve linear quantile regression approximately (no linear program) --> much faster and scales to 100k+ datapoints
    if fast_solver:
        # quant_reg = QuantileLinearRegression(quantile=tau, max_iter=500)
        quant_reg = QuantileRegressor(quantile=tau, max_iter=500)
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
