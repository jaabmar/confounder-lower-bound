import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Union

import numpy as np
from quantile_forest import RandomForestQuantileRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, XGBRegressor

from test_confounding.cate_bounds.blearner.BLearner import (
    BinaryCATEBLearner,
    BinaryPhiBLearner,
    BLearner,
    PhiBLearner,
)
from test_confounding.cate_bounds.utils_cate_bounds import (
    compute_bootstrap_variances_cate_bounds,
)


class MultipleCATEBoundEstimators:
    """
    A class to create and manage multiple Conditional Average Treatment Effect (CATE) bounds estimators.

    This class allows for the instantiation and parallel fitting of multiple CATE bounds estimators, each with different
    user-specified confounding strengths (gammas).

    Args:
        gammas (List[float]): A list of confounding strength values (gamma) to use for each CATE estimator.
        n_bootstrap (int): Number of bootstrap samples for each estimator.
        binary (bool): Indicates whether the outcome is binary. Defaults to False.
        cv (int): Number of cross-validation folds. Defaults to 1.
        seed (int): Random seed for reproducibility. Defaults to 50.
        mu (Optional[Callable]): Pre-specified model for mean outcomes. If None, a default model is used in estimators.
        quantile_upper (Optional[Callable]): Pre-specified model for upper quantile. If None, a default model is used.
        quantile_lower (Optional[Callable]): Pre-specified model for lower quantile. If None, a default model is used.
        bounds (Optional[Callable]): Pre-specified model for bounds estimation. If None, a default model is used.

    Attributes:
        dict_bound_estimators (Dict[str, CATEBoundsEstimator]): Dictionary of CATE bounds estimators indexed by gamma value.
    """

    def __init__(
        self,
        gammas: List[float],
        n_bootstrap: int,
        binary: bool = False,
        cv: int = 1,
        seed: int = 50,
        mu: Optional[Callable] = None,
        quantile_upper: Optional[Callable] = None,
        quantile_lower: Optional[Callable] = None,
        bounds: Optional[Callable] = None,
    ):
        self.dict_bound_estimators: Dict[str, CATEBoundsEstimator] = {}

        def create_estimator(gamma_value: float) -> (str, CATEBoundsEstimator):
            return str(gamma_value), CATEBoundsEstimator(
                binary=binary,
                user_conf=gamma_value,
                n_bootstrap=n_bootstrap,
                cv=cv,
                seed=seed,
                mu=mu,
                quantile_upper=quantile_upper,
                quantile_lower=quantile_lower,
                bounds=bounds,
            )

        with ThreadPoolExecutor() as executor:
            self.dict_bound_estimators = dict(executor.map(create_estimator, gammas))
        print("All CATE estimators are now instantiated.")

    def fit(
        self,
        x_obs: np.ndarray,
        t_obs: np.ndarray,
        y_obs: np.ndarray,
        sample_weight: Optional[bool] = False,
    ) -> None:
        """
        Fit all CATE bounds estimators in parallel.

        Args:
            x_obs (np.ndarray): The observed covariates.
            t_obs (np.ndarray): The observed treatment status.
            y_obs (np.ndarray): The observed outcomes.
            sample_weight (Optional[bool]): Use sample weights for imbalance data. Defaults to False.
        """

        def fit_estimator(cate_estimator: CATEBoundsEstimator):
            cate_estimator.fit(x_obs, t_obs, y_obs, sample_weight)

        start_time = time.time()

        with ThreadPoolExecutor() as executor:
            executor.map(fit_estimator, self.dict_bound_estimators.values())

        end_time = time.time()
        elapsed_time = end_time - start_time

        print(
            f"All CATE bounds estimators are now trained. Elapsed time: {elapsed_time:.2f} seconds"
        )


class _BaseBoundsEstimator:
    def __init__(
        self,
        bounds_est: Union[PhiBLearner, BinaryPhiBLearner, BLearner, BinaryCATEBLearner],
        user_conf: float,
        n_bootstrap: int,
    ):
        self.bounds_est = bounds_est
        self.user_conf = user_conf
        self.n_bootstrap = n_bootstrap

    def _compute_tau(self) -> float:
        """
        Compute the tau value based on the user_conf.

        Returns:
            float: The computed tau value.
        """
        return self.user_conf / (1 + self.user_conf)

    def fit(
        self, x_obs: np.ndarray, t_obs: np.ndarray, y_obs: np.ndarray, sample_weight: bool = False
    ):
        """
        Fit the bounds estimator model on the provided data.

        Args:
            x_obs (np.ndarray): The observational covariate data.
            t_obs (np.ndarray): The treatment data.
            y_obs (np.ndarray): The outcome data.
        """

        self.bounds_est.fit(x_obs, t_obs, y_obs, sample_weight)

    def predict_bounds(self, x_rct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict the bounds for the provided covariate data.

        Args:
            x_rct (np.ndarray): The covariate data.

        Returns:
            tuple[np.ndarray, np.ndarray]: The lower and upper bounds of the CATE.
        """
        lower_bounds, upper_bounds = self.bounds_est.effect(x_rct)
        return lower_bounds, upper_bounds

    def compute_ate_bounds(self, x_rct: np.ndarray) -> tuple[float, float]:
        """
        Compute the ATE bounds for the provided covariate data.

        Args:
            x_rct (np.ndarray): The covariate data.

        Returns:
            tuple[float, float]: The lower and upper bounds of the ATE.
        """
        lower_bounds, upper_bounds = self.predict_bounds(x_rct)
        mean_lb, mean_ub = float(np.mean(lower_bounds)), float(np.mean(upper_bounds))
        return mean_lb, mean_ub

    def estimate_bootstrap_variances(
        self, x_rct: np.ndarray, n_bootstrap: Optional[int] = None
    ) -> tuple[float, float]:
        """
        Estimate the bootstrap variances for the provided covariate data.

        Args:
            x_rct (np.ndarray): The covariate data.
            n_bootstrap (int): The number of bootstrap iterations.

        Returns:
            tuple[float, float]: The bootstrap variances of the bounds
        """
        n_boots = n_bootstrap if n_bootstrap is not None else self.n_bootstrap
        var_lb, var_ub, quantile_lb, quantile_ub = compute_bootstrap_variances_cate_bounds(
            self.bounds_est, x_rct, n_boots
        )
        return var_lb, var_ub, quantile_lb, quantile_ub


class CATEBoundsEstimator(_BaseBoundsEstimator):
    """
    Estimator for Conditional Average Treatment Effect (CATE) bounds.

    This class estimates bounds for the CATE under potential outcomes framework using a variety of models.
    It supports both binary and continuous outcomes.

    Args:
        binary (bool): Indicates whether the outcome is binary. Defaults to False.
        user_conf (float): User-specified confounding level. Defaults to 1.0.
        n_bootstrap (int): Number of bootstrap samples to use. Defaults to 1000.
        cv (int): Number of cross-validation folds. Defaults to 5.
        seed (int): Random seed for reproducibility. Defaults to 50.
        mu (Optional[Callable]): Pre-specified model for mean outcomes. If None, a default model is used.
        quantile_upper (Optional[Callable]): Pre-specified model for upper quantile. If None, a default model is used.
        quantile_lower (Optional[Callable]): Pre-specified model for lower quantile. If None, a default model is used.
        bounds (Optional[Callable]): Pre-specified model for bounds estimation. If None, a default model is used.
    """

    def __init__(
        self,
        binary: bool = False,
        user_conf: float = 1.0,
        n_bootstrap: int = 1000,
        cv: int = 5,
        seed: int = 50,
        mu: Optional[Callable] = None,
        quantile_upper: Optional[Callable] = None,
        quantile_lower: Optional[Callable] = None,
        bounds: Optional[Callable] = None,
    ):
        self.user_conf = user_conf
        self.n_bootstrap = n_bootstrap
        self.cv = cv
        self.seed = seed

        propensity_model = LogisticRegression(
            C=1,
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.7,
            max_iter=10000,
            random_state=seed,
        )

        if not binary:
            mu_model = (
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=6,
                    min_samples_leaf=0.01,
                    n_jobs=-2,
                    random_state=seed,
                )
                if mu is None
                else mu
            )
            quantile_model_upper = (
                RandomForestQuantileRegressor(
                    n_estimators=300,
                    max_depth=6,
                    min_samples_leaf=0.01,
                    n_jobs=-2,
                    default_quantiles=[self._compute_tau()],
                    random_state=seed,
                )
                if quantile_upper is None
                else quantile_upper
            )

            quantile_model_lower = (
                RandomForestQuantileRegressor(
                    n_estimators=300,
                    max_depth=6,
                    min_samples_leaf=0.01,
                    n_jobs=-2,
                    default_quantiles=[1 - self._compute_tau()],
                    random_state=seed,
                )
                if quantile_lower is None
                else quantile_lower
            )

            bounds_model = (
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=6,
                    n_jobs=-2,
                    random_state=seed,
                )
                if bounds is None
                else bounds
            )

            self.phi_bounds_est = BLearner(
                propensity_model=propensity_model,
                quantile_plus_model=quantile_model_upper,
                quantile_minus_model=quantile_model_lower,
                mu_model=mu_model,
                cate_bounds_model=bounds_model,
                use_rho=True,
                gamma=self.user_conf,
                random_state=seed,
                cv=cv,
            )

        else:
            mu_model = (
                XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    n_jobs=-2,
                    random_state=seed,
                )
                if mu is None
                else mu
            )

            bounds_model = (
                XGBRegressor(
                    n_estimators=300,
                    max_depth=6,
                    n_jobs=-2,
                    random_state=seed,
                )
                if bounds is None
                else bounds
            )

            self.phi_bounds_est = BinaryCATEBLearner(
                propensity_model=propensity_model,
                mu_model=mu_model,
                cate_bounds_model=bounds_model,
                gamma=self.user_conf,
                random_state=seed,
                cv=cv,
            )

        super().__init__(
            bounds_est=self.phi_bounds_est, user_conf=user_conf, n_bootstrap=n_bootstrap
        )


class PhiBoundsEstimator(_BaseBoundsEstimator):
    """
    Estimator for potential outcome regression bounds for treated or control group.

    Args:
        binary (bool): Indicates whether the outcome is binary. Defaults to False.
        arm (int): Indicator for treatment (1) or control (0) group.
        user_conf (float): User-specified confounding level. Defaults to 1.0.
        n_bootstrap (int): Number of bootstrap samples to use. Defaults to 1000.
        cv (int): Number of cross-validation folds. Defaults to 5.
        seed (int): Random seed for reproducibility. Defaults to 50.
        mu (Optional[Callable]): Pre-specified model for mean outcomes. If None, a default model is used.
        bounds (Optional[Callable]): Pre-specified model for bounds estimation. If None, a default model is used.
        quantile_upper (Optional[Callable]): Pre-specified model for upper quantile. If None, a default model is used.
        quantile_lower (Optional[Callable]): Pre-specified model for lower quantile. If None, a default model is used.
    """

    def __init__(
        self,
        binary: bool = False,
        arm: int = 1,
        user_conf: float = 1.0,
        n_bootstrap: int = 1000,
        cv: int = 5,
        seed: int = 50,
        mu: Optional[Callable] = None,
        bounds: Optional[Callable] = None,
        quantile_upper: Optional[Callable] = None,
        quantile_lower: Optional[Callable] = None,
    ):
        self.arm = arm
        self.user_conf = user_conf
        self.n_bootstrap = n_bootstrap
        self.cv = cv
        self.seed = seed
        propensity_model = LogisticRegression(
            C=1,
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.7,
            max_iter=10000,
            random_state=seed,
        )

        if not binary:
            mu_model = (
                XGBRegressor(
                    n_estimators=300,
                    max_depth=6,
                    n_jobs=-2,
                    random_state=seed,
                )
                if mu is None
                else mu
            )
            quantile_model_upper = (
                RandomForestQuantileRegressor(
                    n_estimators=300,
                    max_depth=6,
                    min_samples_leaf=0.01,
                    n_jobs=-2,
                    default_quantiles=[self._compute_tau()],
                    random_state=seed,
                )
                if quantile_upper is None
                else quantile_upper
            )

            quantile_model_lower = (
                RandomForestQuantileRegressor(
                    n_estimators=300,
                    max_depth=6,
                    min_samples_leaf=0.01,
                    n_jobs=-2,
                    default_quantiles=[1 - self._compute_tau()],
                    random_state=seed,
                )
                if quantile_lower is None
                else quantile_lower
            )

            bounds_model = (
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=6,
                    min_samples_leaf=0.01,
                    n_jobs=-2,
                    random_state=seed,
                )
                if bounds is None
                else bounds
            )

            phi_bounds_est = PhiBLearner(
                propensity_model=propensity_model,
                quantile_plus_model=quantile_model_upper,
                quantile_minus_model=quantile_model_lower,
                mu_model=mu_model,
                cate_bounds_model=bounds_model,
                use_rho=True,
                gamma=self.user_conf,
                random_state=seed,
                cv=cv,
                arm=arm,
            )

        else:
            mu_model = (
                XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    min_samples_leaf=0.01,
                    n_jobs=-2,
                    random_state=seed,
                )
                if mu is None
                else mu
            )

            bounds_model = (
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=6,
                    min_samples_leaf=0.01,
                    n_jobs=-2,
                    random_state=seed,
                )
                if bounds is None
                else bounds
            )

            phi_bounds_est = BinaryPhiBLearner(
                propensity_model=propensity_model,
                mu_model=mu_model,
                cate_bounds_model=bounds_model,
                gamma=self.user_conf,
                random_state=seed,
                cv=cv,
                arm=arm,
            )

        super().__init__(bounds_est=phi_bounds_est, user_conf=user_conf, n_bootstrap=n_bootstrap)
