from typing import Dict, Optional, Union

import numpy as np
from matplotlib import pyplot as plt
from mlinsights.mlmodel import QuantileLinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, QuantileRegressor
from sklearn.metrics import make_scorer, mean_pinball_loss
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV

# from sklearn.preprocessing import OneHotEncoder
from statsmodels.api import QuantReg

from datasets.synthetic import alpha_fn, beta_fn

PARAMS_DICT_LGBM = {
    "min_samples_leaf": [1, 5, 10, 20],
    "min_samples_split": [5, 10, 20, 30],
    "n_estimators": [100, 200, 300, 500],
}

PARAMS_DICT_QR = {"alpha": [0], "fit_intercept": [True, False]}


def estimate_ATE_HT(x_obs, t_obs, y_obs):
    """
    Estimate the Average Treatment Effect using the Horvitz-Thompson estimator.

    Parameters:
    - t_obs: treatment indicators (1 for treated, 0 for control)
    - y_obs: observed outcomes
    - x_obs: covariates

    Returns:
    - ATE_HT: Average Treatment Effect using the Horvitz-Thompson estimator
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


def adv_propensity_plus(nominal_e_x, x, y, t, gamma):
    alpha = alpha_fn(nominal_e_x, gamma)
    beta = beta_fn(nominal_e_x, gamma)
    a = 1 / beta
    b = 1 / alpha
    qr = get_quantile_regressor(x[t == 1], y[t == 1], gamma / (gamma + 1), fast_solver=True)
    if qr is not None:
        ind = y > qr.predict(x)
        return a * ind + b * (1 - ind)


def plot_bootstrap_dist(bootstrap_dist):
    _, ax = plt.subplots()
    ax.hist(bootstrap_dist.bootstrap_distribution, bins=25)
    ax.set_title("Bootstrap Distribution")
    ax.set_xlabel("statistic value")
    ax.set_ylabel("frequency")
    plt.show()


def calibrate_confound_strength(
    x_obs, t_obs, confounder_col, wrt_confounder=False, ratio_nominal_prop=1.0, seed=50
):
    if wrt_confounder:
        model = LogisticRegression(
            C=1,
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.7,
            max_iter=10000,
            # class_weight="balanced",
            random_state=seed,
        )
        confounder = np.array(confounder_col).reshape(-1, 1)
        model.fit(confounder, t_obs)
        probabilities = model.predict_proba(confounder)

        ratio_conf = probabilities[:, 1] / probabilities[:, 0]
        ratio_nominal = ratio_nominal_prop

    else:
        features_no_YTC = np.array(x_obs)

        # One-hot encode the confounder column
        # encoder = OneHotEncoder(sparse_output=False)
        # confounder_encoded = encoder.fit_transform(np.array(confounder_col).reshape(-1, 1))

        # Concatenate x_obs and the one-hot encoded confounder column
        confounder = np.array(confounder_col).reshape(-1, 1)
        features_no_YT = np.concatenate((features_no_YTC, confounder), axis=1)

        model_no_YTC = LogisticRegression(
            C=1,
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.7,
            max_iter=10000,
            # class_weight="balanced",
            random_state=seed,
        )
        model_no_YTC.fit(features_no_YTC, t_obs)
        probabilities_no_YTC = model_no_YTC.predict_proba(features_no_YTC)

        model_no_YT = LogisticRegression(
            C=1,
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.7,
            max_iter=10000,
            # class_weight="balanced",
            random_state=seed,
        )
        model_no_YT.fit(features_no_YT, t_obs)
        probabilities_no_YT = model_no_YT.predict_proba(features_no_YT)

        ratio_conf = probabilities_no_YT[:, 1] / probabilities_no_YT[:, 0]
        ratio_nominal = probabilities_no_YTC[:, 1] / probabilities_no_YTC[:, 0]

    indiv_conf_strng = ratio_conf / ratio_nominal
    max_conf_strng = max(max(indiv_conf_strng), max(1 / indiv_conf_strng))

    return max_conf_strng


def calculate_all_conf_strength_combinations(confounder, features, df_loader, seed=50, **kwargs):
    df = df_loader(conf_var=confounder, **kwargs)
    df_no_YTC = df.drop(["Y", "T", "C"], axis=1)
    conf_model = LogisticRegression(
        C=1,
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.7,
        max_iter=10000,
        # class_weight="balanced",
        random_state=seed,
    )
    conf_model.fit(df_no_YTC, df["T"])
    conf_probabilities = conf_model.predict_proba(df_no_YTC)
    ratio_nominal_conf = conf_probabilities[:, 1] / conf_probabilities[:, 0]

    df_no_YT = df.drop(["Y", "T"], axis=1)
    full_model = LogisticRegression(
        C=1,
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.7,
        max_iter=10000,
        # class_weight="balanced",
        random_state=seed,
    )
    full_model.fit(df_no_YT, df["T"])
    full_probabilities = full_model.predict_proba(df_no_YT)
    ratio_nominal = full_probabilities[:, 1] / full_probabilities[:, 0]

    true_indiv_conf_strng = ratio_nominal / ratio_nominal_conf
    true_conf_strng = max(max(true_indiv_conf_strng), max(1 / true_indiv_conf_strng))

    conf_strengths = {}
    conf_strengths[confounder] = true_conf_strng

    features_excluding_confounder = [feature for feature in features if feature != confounder]

    for column in features_excluding_confounder:
        X = df_no_YTC.drop(column, axis=1)

        model = LogisticRegression(
            C=1,
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.7,
            max_iter=10000,
            # class_weight="balanced",
            random_state=seed,
        )
        model.fit(X, df["T"])
        probabilities = model.predict_proba(X)
        ratio_conf = probabilities[:, 1] / probabilities[:, 0]
        indiv_conf_strng = ratio_nominal_conf / ratio_conf
        max_conf_strng = max(max(indiv_conf_strng), max(1 / indiv_conf_strng))

        conf_strengths[column] = max_conf_strng

    features_larger_than_conf_strng = [
        feature for feature, value in conf_strengths.items() if value > conf_strengths[confounder]
    ]
    num_larger_strngs_confounder = len(features_larger_than_conf_strng)
    proportion_features_larger_than_conf_strng = num_larger_strngs_confounder / len(
        features_excluding_confounder
    )  # type: ignore

    print(
        f"For {confounder}, proportion of features with larger conf_strengths: {proportion_features_larger_than_conf_strng}"
    )
    print(f"Features with larger conf_strengths: {features_larger_than_conf_strng}")

    return conf_strengths, proportion_features_larger_than_conf_strng


def calculate_conf_strength(features, df_loader, seed=50, **kwargs):
    conf_strengths = {}

    for confounder in features:
        df = df_loader(conf_var=confounder, **kwargs)
        df_no_YTC = df.drop(["Y", "T", "C"], axis=1)
        conf_model = LogisticRegression(
            C=1,
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.7,
            max_iter=10000,
            # class_weight="balanced",
            random_state=seed,
        )
        conf_model.fit(df_no_YTC, df["T"])
        conf_probabilities = conf_model.predict_proba(df_no_YTC)
        ratio_nominal_conf = conf_probabilities[:, 1] / conf_probabilities[:, 0]

        df_no_YT = df.drop(["Y", "T"], axis=1)
        full_model = LogisticRegression(
            C=1,
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.7,
            max_iter=10000,
            # class_weight="balanced",
            random_state=seed,
        )
        full_model.fit(df_no_YT, df["T"])
        full_probabilities = full_model.predict_proba(df_no_YT)
        ratio_nominal = full_probabilities[:, 1] / full_probabilities[:, 0]

        true_indiv_conf_strng = ratio_nominal / ratio_nominal_conf
        true_conf_strng = max(max(true_indiv_conf_strng), max(1 / true_indiv_conf_strng))

        conf_strengths[confounder] = true_conf_strng

    return conf_strengths


class CATEEstimator:
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
