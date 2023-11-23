"Code adapted from https://github.com/CausalML/BLearner"

import numpy as np
from sklearn import clone
from sklearn.model_selection import KFold

from .utils import (
    Binary_CATE_Nuisance_Model,
    Binary_Phi_Nuisance_Model,
    CATE_Nuisance_Model,
    Phi_Nuisance_Model,
    _crossfit,
)


class _BaseBLearner:
    """Base class for BLearner estimators."""

    def __init__(
        self,
        nuisance_model,
        cate_bounds_model,
        use_rho=False,
        gamma=1.0,
        cv=5,
        random_state=None,
    ):
        self.gamma = gamma
        self.tau = self.gamma / (1 + self.gamma)
        self.use_rho = use_rho
        self.cate_upper_model = clone(cate_bounds_model, safe=False)
        self.cate_lower_model = clone(cate_bounds_model, safe=False)
        self.nuisance_model = nuisance_model
        self.cv = cv
        self.random_state = random_state

    def fit(self, X, A, Y, weighting):
        if self.cv > 1:
            folds = list(
                KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state).split(X)
            )
            nuisances, self.nuisance_models = _crossfit(
                self.nuisance_model, folds, X, A, Y, weighting
            )
        else:
            self.nuisance_model.fit(X, A, Y, weighting)
            nuisances = self.nuisance_model.predict(X)
        self._fit_with_nuisances(X, A, Y, nuisances)
        return self

    def effect(self, X):
        return (
            self.cate_lower_model.predict(X).flatten(),
            self.cate_upper_model.predict(X).flatten(),
        )

    def _fit_with_nuisances(self, X, A, Y, nuisances):
        phi_plus_1 = self._get_pseudo_outcome_plus_1(A, Y, nuisances)
        phi_minus_0 = self._get_pseudo_outcome_minus_0(A, Y, nuisances)
        phi_plus = phi_plus_1 - phi_minus_0

        phi_minus_1 = self._get_pseudo_outcome_minus_1(A, Y, nuisances)
        phi_plus_0 = self._get_pseudo_outcome_plus_0(A, Y, nuisances)
        phi_minus = phi_minus_1 - phi_plus_0

        self.cate_upper_model.fit(X, phi_plus)
        self.cate_lower_model.fit(X, phi_minus)

    def _get_pseudo_outcome_plus_1(self, A, Y, nuisances):
        e = nuisances[:, 0]
        q_tau_1 = nuisances[:, 2]
        if self.use_rho:
            rho_plus_1 = nuisances[:, 8]
        else:
            mu_1 = nuisances[:, 6]
            cvar_tau_1 = nuisances[:, 8]
            rho_plus_1 = (1 / self.gamma) * mu_1 + (1 - (1 / self.gamma)) * cvar_tau_1
        R_plus_1 = (1 / self.gamma) * Y + (1 - (1 / self.gamma)) * (
            q_tau_1 + (1 / (1 - self.tau)) * np.maximum(Y - q_tau_1, 0)
        )
        phi = rho_plus_1 + (A / e) * (R_plus_1 - rho_plus_1) + A * Y - A * R_plus_1
        return phi

    def _get_pseudo_outcome_minus_1(self, A, Y, nuisances):
        e = nuisances[:, 0]
        q_tau_1 = nuisances[:, 4]
        if self.use_rho:
            rho_minus_1 = nuisances[:, 10]
        else:
            mu_1 = nuisances[:, 6]
            cvar_tau_1 = nuisances[:, 10]
            rho_minus_1 = (1 / self.gamma) * mu_1 + (1 - (1 / self.gamma)) * cvar_tau_1
        R_minus_1 = (1 / self.gamma) * Y + (1 - (1 / self.gamma)) * (
            q_tau_1 + (1 / (1 - self.tau)) * np.minimum(Y - q_tau_1, 0)
        )
        phi = rho_minus_1 + (A / e) * (R_minus_1 - rho_minus_1) + A * Y - A * R_minus_1

        return phi

    def _get_pseudo_outcome_plus_0(self, A, Y, nuisances):
        e = nuisances[:, 0]
        q_tau_0 = nuisances[:, 1]
        if self.use_rho:
            rho_plus_0 = nuisances[:, 7]
        else:
            mu_0 = nuisances[:, 5]
            cvar_tau_0 = nuisances[:, 7]
            rho_plus_0 = (1 / self.gamma) * mu_0 + (1 - (1 / self.gamma)) * cvar_tau_0
        R_plus_0 = (1 / self.gamma) * Y + (1 - (1 / self.gamma)) * (
            q_tau_0 + (1 / (1 - self.tau)) * np.maximum(Y - q_tau_0, 0)
        )

        phi = (
            rho_plus_0
            + (1 - A) / (1 - e) * (R_plus_0 - rho_plus_0)
            + (1 - A) * Y
            - (1 - A) * R_plus_0
        )
        return phi

    def _get_pseudo_outcome_minus_0(self, A, Y, nuisances):
        e = nuisances[:, 0]
        q_tau_0 = nuisances[:, 3]
        if self.use_rho:
            rho_minus_0 = nuisances[:, 9]
        else:
            mu_0 = nuisances[:, 5]
            cvar_tau_0 = nuisances[:, 9]
            rho_minus_0 = (1 / self.gamma) * mu_0 + (1 - (1 / self.gamma)) * cvar_tau_0
        R_minus_0 = (1 / self.gamma) * Y + (1 - (1 / self.gamma)) * (
            q_tau_0 + (1 / (1 - self.tau)) * np.minimum(Y - q_tau_0, 0)
        )
        phi = (
            rho_minus_0
            + (1 - A) / (1 - e) * (R_minus_0 - rho_minus_0)
            + (1 - A) * Y
            - (1 - A) * R_minus_0
        )
        return phi


class BLearner(_BaseBLearner):
    """Estimator for CATE sharp bounds that uses doubly-robust correction techniques.

    Parameters
    ----------
    propensity_model : classification model (scikit-learn or other)
        Estimator for Pr[A=1 | X=x].  Must implement `fit` and `predict_proba` methods.
    quantile_plus_model : quantile regression model (e.g. RandomForestQuantileRegressor)
        Estimator for the 1-tau conditional quantile. Must implement `fit` and `predict` methods.
    quantile_minus_model : quantile regression model (e.g. RandomForestQuantileRegressor)
        Estimator for the tau conditional quantile. Must implement `fit` and `predict` methods.
    mu_model : regression model (scikit-learn or other)
        Estimator for the conditional outcome E[Y | X=x, A=a] when `use_rho=False` or for the modified
        conditional outcome rho_+(x, a)=E[Gamma^{-1}Y+(1-Gamma^{-1}){q + 1/1(1-tau)*max{Y-q, 0}} | X=x, A=a]
        Must implement `fit` and `predict` methods.
    cvar_plus_model : superquantile model (default=None)
        Estimator for the conditional right tau tail CVaR when `use_rho=False`. Must implement `fit` and `predict` methods.
        Only used when `use_rho=False`.
     cvar_minus_model : superquantile model (default=None)
        Estimator for the conditional left tau tail CVaR when `use_rho=False`. Must implement `fit` and `predict` methods.
        Only used when `use_rho=False`.
    use_rho :  bool (default=False)
        Whether to construct rho using a direct regression with plug-in quantiles (`use_rho=True`) or to estimate rho by
        estimating the conditional outcome and conditional CVaR models separately (`use_rho=False`).
    gamma : float, >=1
        Sensitivity model parameter. Must be greater than 1.
    cv : int, (default=5)
        The number of folds to use for K-fold cross-validation.
    random_state : int (default=None)
        Controls the randomness of the estimator.
    """

    def __init__(
        self,
        propensity_model,
        quantile_plus_model,
        quantile_minus_model,
        mu_model,
        cate_bounds_model,
        cvar_plus_model=None,
        cvar_minus_model=None,
        use_rho=False,
        gamma=1.0,
        cv=5,
        random_state=None,
    ):
        if not use_rho and (cvar_plus_model is None or cvar_minus_model is None):
            raise ValueError("'cvar_model' parameter cannot be None when use_rho=False.")
        nuisance_model = CATE_Nuisance_Model(
            propensity_model,
            quantile_plus_model,
            quantile_minus_model,
            mu_model,
            cvar_plus_model,
            cvar_minus_model,
            use_rho=use_rho,
            gamma=gamma,
        )
        super().__init__(
            nuisance_model=nuisance_model,
            cate_bounds_model=cate_bounds_model,
            use_rho=use_rho,
            gamma=gamma,
            cv=cv,
            random_state=random_state,
        )


class _BinaryBaseBLearner:
    def __init__(
        self,
        nuisance_model,
        cate_bounds_model,
        gamma=1.0,
        cv=5,
        random_state=None,
    ):
        self.gamma = gamma
        self.tau = self.gamma / (1 + self.gamma)
        self.cate_lower_model = clone(cate_bounds_model, safe=False)
        self.cate_upper_model = clone(cate_bounds_model, safe=False)
        self.nuisance_model = nuisance_model
        self.cv = cv
        self.random_state = random_state

    def fit(self, X, A, Y, weighting=False):
        if self.cv > 1:
            folds = list(
                KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state).split(X)
            )
            nuisances, self.nuisance_models = _crossfit(
                self.nuisance_model, folds, X, A, Y, weighting
            )
        else:
            self.nuisance_model.fit(X, A, Y, weighting)
            nuisances = self.nuisance_model.predict(X)
        self._fit_with_nuisances(X, A, Y, nuisances)
        return self

    def effect(self, X):
        return (
            self.cate_lower_model.predict(X).flatten(),
            self.cate_upper_model.predict(X).flatten(),
        )

    def _fit_with_nuisances(self, X, A, Y, nuisances):
        phi_plus_1 = self._get_pseudo_outcome_plus_1(A, Y, nuisances)
        phi_minus_0 = self._get_pseudo_outcome_minus_0(A, Y, nuisances)
        phi_plus = phi_plus_1 - phi_minus_0

        phi_minus_1 = self._get_pseudo_outcome_minus_1(A, Y, nuisances)
        phi_plus_0 = self._get_pseudo_outcome_plus_0(A, Y, nuisances)
        phi_minus = phi_minus_1 - phi_plus_0

        self.cate_upper_model.fit(X, phi_plus)
        self.cate_lower_model.fit(X, phi_minus)

    def _get_pseudo_outcome_plus_1(self, A, Y, nuisances):
        e = nuisances[:, 0]
        q_tau_1 = nuisances[:, 2]
        rho_plus_1 = nuisances[:, 8]
        R_plus_1 = (1 / self.gamma) * Y + (1 - (1 / self.gamma)) * (
            q_tau_1 + (1 / (1 - self.tau)) * np.maximum(Y - q_tau_1, 0)
        )
        phi = rho_plus_1 + (A / e) * (R_plus_1 - rho_plus_1) + A * Y - A * R_plus_1

        phi = rho_plus_1 * (1 - A) + A * Y
        return phi

    def _get_pseudo_outcome_minus_1(self, A, Y, nuisances):
        e = nuisances[:, 0]
        q_tau_1 = nuisances[:, 4]
        rho_minus_1 = nuisances[:, 10]
        R_minus_1 = (1 / self.gamma) * Y + (1 - (1 / self.gamma)) * (
            q_tau_1 + (1 / (1 - self.tau)) * np.minimum(Y - q_tau_1, 0)
        )
        phi = rho_minus_1 + (A / e) * (R_minus_1 - rho_minus_1) + A * Y - A * R_minus_1
        phi = (1 - A) * rho_minus_1 + (A) * Y
        return phi

    def _get_pseudo_outcome_plus_0(self, A, Y, nuisances):
        e = nuisances[:, 0]
        q_tau_0 = nuisances[:, 1]
        rho_plus_0 = nuisances[:, 7]
        R_plus_0 = (1 / self.gamma) * Y + (1 - (1 / self.gamma)) * (
            q_tau_0 + (1 / (1 - self.tau)) * np.maximum(Y - q_tau_0, 0)
        )

        phi = (
            rho_plus_0
            + (1 - A) / (1 - e) * (R_plus_0 - rho_plus_0)
            + (1 - A) * Y
            - (1 - A) * R_plus_0
        )
        phi = (A) * rho_plus_0 + Y * (1 - A)
        return phi

    def _get_pseudo_outcome_minus_0(self, A, Y, nuisances):
        e = nuisances[:, 0]
        q_tau_0 = nuisances[:, 3]
        rho_minus_0 = nuisances[:, 9]
        R_minus_0 = (1 / self.gamma) * Y + (1 - (1 / self.gamma)) * (
            q_tau_0 + (1 / (1 - self.tau)) * np.minimum(Y - q_tau_0, 0)
        )
        phi = (
            rho_minus_0
            + (1 - A) / (1 - e) * (R_minus_0 - rho_minus_0)
            + (1 - A) * Y
            - (1 - A) * R_minus_0
        )
        phi = rho_minus_0 * A + (1 - A) * Y
        return phi


class BinaryCATEBLearner(_BinaryBaseBLearner):
    def __init__(
        self,
        propensity_model,
        mu_model,
        cate_bounds_model,
        gamma=1.0,
        cv=5,
        random_state=None,
    ):
        nuisance_model = Binary_CATE_Nuisance_Model(
            propensity_model=propensity_model,
            mu_model=mu_model,
            gamma=gamma,
        )
        super().__init__(
            nuisance_model=nuisance_model,
            cate_bounds_model=cate_bounds_model,
            gamma=gamma,
            cv=cv,
            random_state=random_state,
        )


class _BasePhiBLearner:
    """Base class for BLearner estimators."""

    def __init__(
        self,
        nuisance_model,
        cate_bounds_model,
        arm,
        use_rho=True,
        gamma=1.0,
        cv=5,
        random_state=None,
    ):
        self.gamma = gamma
        self.tau = self.gamma / (1 + self.gamma)
        self.use_rho = use_rho
        self.phi_upper_model = clone(cate_bounds_model, safe=False)
        self.phi_lower_model = clone(cate_bounds_model, safe=False)
        self.nuisance_model = nuisance_model
        self.cv = cv
        self.random_state = random_state
        self.arm = arm

    def fit(self, X, A, Y, weighting=False):
        if self.cv > 1:
            folds = list(
                KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state).split(X)
            )
            nuisances, self.nuisance_models = _crossfit(
                self.nuisance_model, folds, X, A, Y, weighting
            )
        else:
            self.nuisance_model.fit(X, A, Y, weighting)
            nuisances = self.nuisance_model.predict(X)
        self._fit_with_nuisances(X, A, Y, nuisances)
        return self

    def effect(self, X):
        return (
            self.phi_lower_model.predict(X).flatten(),
            self.phi_upper_model.predict(X).flatten(),
        )

    def _fit_with_nuisances(self, X, A, Y, nuisances):
        phi_plus = self._get_pseudo_outcome_plus(A, Y, nuisances)
        phi_minus = self._get_pseudo_outcome_minus(A, Y, nuisances)

        # Fit final regression models
        self.phi_upper_model.fit(X, phi_plus)
        self.phi_lower_model.fit(X, phi_minus)

    def _get_pseudo_outcome_plus(self, A, Y, nuisances):
        e = nuisances[:, 0]
        q_tau = nuisances[:, 1]
        if self.use_rho:
            rho_plus = nuisances[:, 4]
        else:
            mu = nuisances[:, 3]
            cvar_tau = nuisances[:, 4]
            rho_plus = (1 / self.gamma) * mu + (1 - (1 / self.gamma)) * cvar_tau
        R_plus = (1 / self.gamma) * Y + (1 - (1 / self.gamma)) * (
            q_tau + (1 / (1 - self.tau)) * np.maximum(Y - q_tau, 0)
        )
        if self.arm == 1:
            phi = rho_plus + (A / e) * (R_plus - rho_plus) + A * Y - A * R_plus
        else:
            phi = (
                rho_plus + (1 - A) / (1 - e) * (R_plus - rho_plus) + (1 - A) * Y - (1 - A) * R_plus
            )

        return phi

    def _get_pseudo_outcome_minus(self, A, Y, nuisances):
        e = nuisances[:, 0]
        q_tau = nuisances[:, 2]
        if self.use_rho:
            rho_minus = nuisances[:, 5]
        else:
            mu = nuisances[:, 3]
            cvar_tau = nuisances[:, 5]
            rho_minus = (1 / self.gamma) * mu + (1 - (1 / self.gamma)) * cvar_tau
        R_minus = (1 / self.gamma) * Y + (1 - (1 / self.gamma)) * (
            q_tau + (1 / (1 - self.tau)) * np.minimum(Y - q_tau, 0)
        )
        if self.arm == 1:
            phi = rho_minus + (A / e) * (R_minus - rho_minus) + A * Y - A * R_minus
        else:
            phi = (
                rho_minus
                + (1 - A) / (1 - e) * (R_minus - rho_minus)
                + (1 - A) * Y
                - (1 - A) * R_minus
            )

        return phi


class PhiBLearner(_BasePhiBLearner):
    def __init__(
        self,
        propensity_model,
        quantile_plus_model,
        quantile_minus_model,
        mu_model,
        cate_bounds_model,
        arm,
        cvar_plus_model=None,
        cvar_minus_model=None,
        use_rho=False,
        gamma=1.0,
        cv=5,
        random_state=None,
    ):
        if not use_rho and (cvar_plus_model is None or cvar_minus_model is None):
            raise ValueError("'cvar_model' parameter cannot be None when use_rho=False.")
        nuisance_model = Phi_Nuisance_Model(
            propensity_model=propensity_model,
            quantile_plus_model=quantile_plus_model,
            quantile_minus_model=quantile_minus_model,
            mu_model=mu_model,
            cvar_plus_model=cvar_plus_model,
            cvar_minus_model=cvar_minus_model,
            use_rho=use_rho,
            gamma=gamma,
            arm=arm,
        )
        super().__init__(
            nuisance_model=nuisance_model,
            cate_bounds_model=cate_bounds_model,
            use_rho=use_rho,
            gamma=gamma,
            cv=cv,
            random_state=random_state,
            arm=arm,
        )


class _BinaryBasePhiBLearner:
    def __init__(
        self,
        nuisance_model,
        cate_bounds_model,
        arm,
        gamma=1.0,
        cv=5,
        random_state=None,
    ):
        self.gamma = gamma
        self.tau = self.gamma / (1 + self.gamma)
        self.phi_upper_model = clone(cate_bounds_model, safe=False)
        self.phi_lower_model = clone(cate_bounds_model, safe=False)
        self.nuisance_model = nuisance_model
        self.cv = cv
        self.random_state = random_state
        self.arm = arm

    def fit(self, X, A, Y, weighting=False):
        if self.cv > 1:
            folds = list(
                KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state).split(X)
            )
            nuisances, self.nuisance_models = _crossfit(
                self.nuisance_model, folds, X, A, Y, weighting
            )
        else:
            self.nuisance_model.fit(X, A, Y, weighting)
            nuisances = self.nuisance_model.predict(X)
        self._fit_with_nuisances(X, A, Y, nuisances)
        return self

    def effect(self, X):
        return (
            self.phi_lower_model.predict(X).flatten(),
            self.phi_upper_model.predict(X).flatten(),
        )

    def _fit_with_nuisances(self, X, A, Y, nuisances):
        phi_plus = self._get_pseudo_outcome_plus(A, Y, nuisances)
        phi_minus = self._get_pseudo_outcome_minus(A, Y, nuisances)

        # Fit final regression models
        self.phi_upper_model.fit(X, phi_plus)
        self.phi_lower_model.fit(X, phi_minus)

    def _get_pseudo_outcome_plus(self, A, Y, nuisances):
        e = nuisances[:, 0]
        q_tau = nuisances[:, 1]
        rho_plus = nuisances[:, 4]
        R_plus = (1 / self.gamma) * Y + (1 - (1 / self.gamma)) * (
            q_tau + (1 / (1 - self.tau)) * np.maximum(Y - q_tau, 0)
        )
        if self.arm == 1:
            phi = rho_plus + (A / e) * (R_plus - rho_plus) + A * Y - A * R_plus
            phi = rho_plus - A / e * rho_plus
        else:
            phi = (
                rho_plus + (1 - A) / (1 - e) * (R_plus - rho_plus) + (1 - A) * Y - (1 - A) * R_plus
            )
            phi = rho_plus - (1 - A) / (1 - e) * rho_plus

        return phi

    def _get_pseudo_outcome_minus(self, A, Y, nuisances):
        e = nuisances[:, 0]
        q_tau = nuisances[:, 2]
        rho_minus = nuisances[:, 5]
        R_minus = (1 / self.gamma) * Y + (1 - (1 / self.gamma)) * (
            q_tau + (1 / (1 - self.tau)) * np.minimum(Y - q_tau, 0)
        )
        if self.arm == 1:
            phi = rho_minus + (A / e) * (R_minus - rho_minus) + A * Y - A * R_minus
            phi = rho_minus - A / e * rho_minus
        else:
            phi = (
                rho_minus
                + (1 - A) / (1 - e) * (-rho_minus)
                + (1 - A) / (1 - e) * (R_minus - rho_minus)
                + (1 - A) * Y
                - (1 - A) * R_minus
            )

        return phi


class BinaryPhiBLearner(_BinaryBasePhiBLearner):
    def __init__(
        self,
        propensity_model,
        mu_model,
        cate_bounds_model,
        arm,
        gamma=1.0,
        cv=5,
        random_state=None,
    ):
        nuisance_model = Binary_Phi_Nuisance_Model(
            propensity_model=propensity_model, mu_model=mu_model, gamma=gamma, arm=arm
        )
        super().__init__(
            nuisance_model=nuisance_model,
            cate_bounds_model=cate_bounds_model,
            gamma=gamma,
            cv=cv,
            random_state=random_state,
            arm=arm,
        )
