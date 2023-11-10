import numpy as np
from sklearn import clone
from sklearn.utils import compute_sample_weight

##################
# Wrapper models #
##################


def _crossfit(model, folds, X, A, Y, weighting=False):
    model_list = []
    fitted_inds = []
    nuisances = []

    for idx, (train_idxs, test_idxs) in enumerate(folds):
        model_list.append(clone(model, safe=False))
        fitted_inds = np.concatenate((fitted_inds, test_idxs))
        model_list[idx].fit(
            X[train_idxs],
            A[train_idxs],
            Y[train_idxs],
            weighting=weighting,
        )
        nuisance = model_list[idx].predict(X[test_idxs])
        if idx == 0:
            nuisances = np.full((X.shape[0], nuisance.shape[1]), np.nan)
        nuisances[test_idxs] = nuisance
    return nuisances, model_list


class CATE_Nuisance_Model:
    def __init__(
        self,
        propensity_model,
        quantile_plus_model,
        quantile_minus_model,
        mu_model,
        cvar_plus_model,
        cvar_minus_model,
        gamma=1.0,
        use_rho=False,
    ):
        self.use_rho = use_rho
        self.gamma = gamma
        self.tau = self.gamma / (1 + self.gamma)
        self.propensity_model = clone(propensity_model, safe=False)
        self.quantile_plus_models = [
            clone(quantile_plus_model, safe=False),
            clone(quantile_plus_model, safe=False),
        ]
        self.quantile_minus_models = [
            clone(quantile_minus_model, safe=False),
            clone(quantile_minus_model, safe=False),
        ]
        if self.use_rho:
            self.mu_models = [clone(mu_model, safe=False), clone(mu_model, safe=False)]
            self.rho_plus_models = [clone(mu_model, safe=False), clone(mu_model, safe=False)]
            self.rho_minus_models = [clone(mu_model, safe=False), clone(mu_model, safe=False)]
        else:
            self.mu_models = [clone(mu_model, safe=False), clone(mu_model, safe=False)]
            self.cvar_minus_models = [
                clone(cvar_minus_model, safe=False),
                clone(cvar_minus_model, safe=False),
            ]
            self.cvar_plus_models = [
                clone(cvar_plus_model, safe=False),
                clone(cvar_plus_model, safe=False),
            ]

    def fit(self, X, A, Y, weighting=None):
        if weighting:
            weights_0 = compute_sample_weight(class_weight="balanced", y=Y[A == 0])
            weights_1 = compute_sample_weight(class_weight="balanced", y=Y[A == 1])
        else:
            weights_0 = None
            weights_1 = None
        self.propensity_model.fit(X, A)
        self.quantile_plus_models[0].fit(X[A == 0], Y[A == 0])
        self.quantile_plus_models[1].fit(X[A == 1], Y[A == 1])
        self.quantile_minus_models[0].fit(X[A == 0], Y[A == 0])
        self.quantile_minus_models[1].fit(X[A == 1], Y[A == 1])
        self.mu_models[0].fit(X[A == 0], Y[A == 0], sample_weight=weights_0)
        self.mu_models[1].fit(X[A == 1], Y[A == 1], sample_weight=weights_1)
        if self.use_rho:
            # rho_plus_1
            q_tau_1 = self.quantile_plus_models[1].predict(X[A == 1])
            R_plus_1 = (1 / self.gamma) * Y[A == 1] + (1 - (1 / self.gamma)) * (
                q_tau_1 + (1 / (1 - self.tau)) * np.maximum(Y[A == 1] - q_tau_1, 0)
            )
            self.rho_plus_models[1].fit(X[A == 1], R_plus_1)
            # rho_minus_1
            q_tau_1 = self.quantile_minus_models[1].predict(X[A == 1])
            R_minus_1 = (1 / self.gamma) * Y[A == 1] + (1 - (1 / self.gamma)) * (
                q_tau_1 + (1 / (1 - self.tau)) * np.minimum(Y[A == 1] - q_tau_1, 0)
            )
            self.rho_minus_models[1].fit(X[A == 1], R_minus_1)
            # rho_plus_0
            q_tau_0 = self.quantile_plus_models[0].predict(X[A == 0])
            R_plus_0 = (1 / self.gamma) * Y[A == 0] + (1 - (1 / self.gamma)) * (
                q_tau_0 + (1 / (1 - self.tau)) * np.maximum(Y[A == 0] - q_tau_0, 0)
            )
            self.rho_plus_models[0].fit(X[A == 0], R_plus_0)
            # rho_minus_0
            q_tau_0 = self.quantile_minus_models[0].predict(X[A == 0])
            R_minus_0 = (1 / self.gamma) * Y[A == 0] + (1 - (1 / self.gamma)) * (
                q_tau_0 + (1 / (1 - self.tau)) * np.minimum(Y[A == 0] - q_tau_0, 0)
            )
            self.rho_minus_models[0].fit(X[A == 0], R_minus_0)

        else:
            self.cvar_plus_models[0].fit(X[A == 0], Y[A == 0])
            self.cvar_plus_models[1].fit(X[A == 1], Y[A == 1])
            self.cvar_minus_models[0].fit(X[A == 0], Y[A == 0])
            self.cvar_minus_models[1].fit(X[A == 1], Y[A == 1])

    def predict(self, X):
        if self.use_rho:
            predictions = np.hstack(
                (
                    self.propensity_model.predict_proba(X)[:, [1]],
                    self.quantile_plus_models[0].predict(X).reshape(-1, 1),
                    self.quantile_plus_models[1].predict(X).reshape(-1, 1),
                    self.quantile_minus_models[0].predict(X).reshape(-1, 1),
                    self.quantile_minus_models[1].predict(X).reshape(-1, 1),
                    self.mu_models[0].predict(X).reshape(-1, 1),
                    self.mu_models[1].predict(X).reshape(-1, 1),
                    self.rho_plus_models[0].predict(X).reshape(-1, 1),
                    self.rho_plus_models[1].predict(X).reshape(-1, 1),
                    self.rho_minus_models[0].predict(X).reshape(-1, 1),
                    self.rho_minus_models[1].predict(X).reshape(-1, 1),
                )
            )
        else:
            predictions = np.hstack(
                (
                    self.propensity_model.predict_proba(X)[:, [1]],
                    self.quantile_plus_models[0].predict(X).reshape(-1, 1),
                    self.quantile_plus_models[1].predict(X).reshape(-1, 1),
                    self.quantile_minus_models[0].predict(X).reshape(-1, 1),
                    self.quantile_minus_models[1].predict(X).reshape(-1, 1),
                    self.mu_models[0].predict(X).reshape(-1, 1),
                    self.mu_models[1].predict(X).reshape(-1, 1),
                    self.cvar_plus_models[0].predict(X).reshape(-1, 1),
                    self.cvar_plus_models[1].predict(X).reshape(-1, 1),
                    self.cvar_minus_models[0].predict(X).reshape(-1, 1),
                    self.cvar_minus_models[1].predict(X).reshape(-1, 1),
                )
            )
        return predictions


class Binary_CATE_Nuisance_Model:
    def __init__(
        self,
        propensity_model,
        mu_model,
        gamma=1.0,
    ):
        self.gamma = gamma
        self.tau = self.gamma / (1 + self.gamma)
        self.propensity_model = clone(propensity_model, safe=False)
        self.mu_models = [clone(mu_model, safe=False), clone(mu_model, safe=False)]

    def fit(self, X, A, Y, weighting=False):
        if weighting:
            weights_0 = compute_sample_weight(class_weight="balanced", y=Y[A == 0])
            weights_1 = compute_sample_weight(class_weight="balanced", y=Y[A == 1])
        else:
            weights_0 = None
            weights_1 = None
        self.propensity_model.fit(X, A)
        self.mu_models[0].fit(X[A == 0], Y[A == 0], sample_weight=weights_0)
        self.mu_models[1].fit(X[A == 1], Y[A == 1], sample_weight=weights_1)

    def quantile_plus(self, X, arm):
        if hasattr(self.mu_models[arm], 'predict_proba'):
            mu = self.mu_models[arm].predict_proba(X)[:,1]
        else:
            mu = self.mu_models[arm].predict(X)
        Q_plus = np.zeros_like(mu)
        Q_plus[mu > (1 - self.tau)] = 1
        return Q_plus

    def quantile_minus(self, X, arm):
        if hasattr(self.mu_models[arm], 'predict_proba'):
            mu = self.mu_models[arm].predict_proba(X)[:,1]
        else:
            mu = self.mu_models[arm].predict(X)
        Q_minus = np.zeros_like(mu)
        Q_minus[mu > (self.tau)] = 1
        return Q_minus

    def rho_plus(self, X, arm):
        if hasattr(self.mu_models[arm], 'predict_proba'):
            mu = self.mu_models[arm].predict_proba(X)[:,1]
        else:
            mu = self.mu_models[arm].predict(X)
        rho_plus = np.minimum(1 - 1 / self.gamma + mu / self.gamma, mu * self.gamma)
        return rho_plus

    def rho_minus(self, X, arm):
        if hasattr(self.mu_models[arm], 'predict_proba'):
            mu = self.mu_models[arm].predict_proba(X)[:,1]
        else:
            mu = self.mu_models[arm].predict(X)
        rho_minus = np.maximum(1 - self.gamma + mu * self.gamma, mu / self.gamma)
        return rho_minus

    def predict(self, X):
        predictions = np.hstack(
            (
                self.propensity_model.predict_proba(X)[:, [1]],
                self.quantile_plus(X, arm=0).reshape(-1, 1),
                self.quantile_plus(X, arm=1).reshape(-1, 1),
                self.quantile_minus(X, arm=0).reshape(-1, 1),
                self.quantile_minus(X, arm=1).reshape(-1, 1),
                self.mu_models[0].predict(X).reshape(-1, 1),
                self.mu_models[1].predict(X).reshape(-1, 1),
                self.rho_plus(X, arm=0).reshape(-1, 1),
                self.rho_plus(X, arm=1).reshape(-1, 1),
                self.rho_minus(X, arm=0).reshape(-1, 1),
                self.rho_minus(X, arm=1).reshape(-1, 1),
            )
        )

        return predictions


class Phi_Nuisance_Model:
    def __init__(
        self,
        propensity_model,
        quantile_plus_model,
        quantile_minus_model,
        mu_model,
        cvar_plus_model,
        cvar_minus_model,
        arm,
        gamma=1.0,
        use_rho=False,
    ):
        self.use_rho = use_rho
        self.gamma = gamma
        self.tau = self.gamma / (1 + self.gamma)
        self.propensity_model = clone(propensity_model, safe=False)
        self.quantile_plus_model = clone(quantile_plus_model, safe=False)
        self.quantile_minus_model = clone(quantile_minus_model, safe=False)
        self.mu_model = clone(mu_model, safe=False)
        self.arm = arm

        if self.use_rho:
            self.rho_plus_model = clone(mu_model, safe=False)
            self.rho_minus_model = clone(mu_model, safe=False)
        else:
            self.cvar_minus_model = clone(cvar_minus_model, safe=False)
            self.cvar_plus_model = clone(cvar_plus_model, safe=False)

    def fit(self, X, A, Y, weighting=False):
        if weighting:
            weights = compute_sample_weight(class_weight="balanced", y=Y[A == self.arm])
        else:
            weights = None
        self.propensity_model.fit(X, A)
        self.mu_model.fit(X[A == self.arm], Y[A == self.arm], sample_weight=weights)
        self.quantile_plus_model.fit(X[A == self.arm], Y[A == self.arm])
        self.quantile_minus_model.fit(X[A == self.arm], Y[A == self.arm])

        if self.use_rho:
            # rho_plus_1
            q_tau = self.quantile_plus_model.predict(X[A == self.arm])
            R_plus = (1 / self.gamma) * Y[A == self.arm] + (1 - (1 / self.gamma)) * (
                q_tau + (1 / (1 - self.tau)) * np.maximum(Y[A == self.arm] - q_tau, 0)
            )

            # rho_minus_1
            q_tau = self.quantile_minus_model.predict(X[A == self.arm])
            R_minus = (1 / self.gamma) * Y[A == self.arm] + (1 - (1 / self.gamma)) * (
                q_tau + (1 / (1 - self.tau)) * np.minimum(Y[A == self.arm] - q_tau, 0)
            )

            self.rho_plus_model.fit(X[A == self.arm], R_plus)
            self.rho_minus_model.fit(X[A == self.arm], R_minus)

        else:
            self.cvar_plus_model.fit(X[A == self.arm], Y[A == self.arm])
            self.cvar_minus_model.fit(X[A == self.arm], Y[A == self.arm])

    def predict(self, X):
        if self.use_rho:
            predictions = np.hstack(
                (
                    self.propensity_model.predict_proba(X)[:, [1]],
                    self.quantile_plus_model.predict(X).reshape(-1, 1),
                    self.quantile_minus_model.predict(X).reshape(-1, 1),
                    self.mu_model.predict(X).reshape(-1, 1),
                    self.rho_plus_model.predict(X).reshape(-1, 1),
                    self.rho_minus_model.predict(X).reshape(-1, 1),
                )
            )
        else:
            predictions = np.hstack(
                (
                    self.propensity_model.predict_proba(X)[:, [1]],
                    self.quantile_plus_model.predict(X).reshape(-1, 1),
                    self.quantile_minus_model.predict(X).reshape(-1, 1),
                    self.mu_model.predict(X).reshape(-1, 1),
                    self.cvar_plus_model.predict(X).reshape(-1, 1),
                    self.cvar_minus_model.predict(X).reshape(-1, 1),
                )
            )
        return predictions


class Binary_Phi_Nuisance_Model:
    def __init__(
        self,
        propensity_model,
        mu_model,
        arm,
        gamma=1.0,
    ):
        self.gamma = gamma
        self.tau = self.gamma / (1 + self.gamma)
        self.propensity_model = clone(propensity_model, safe=False)
        self.mu_model = clone(mu_model, safe=False)
        self.arm = arm

    def fit(self, X, A, Y, weighting=False):
        if weighting:
            weights = compute_sample_weight(class_weight="balanced", y=Y[A == self.arm])
        else:
            weights = None
        self.propensity_model.fit(X, A)
        self.mu_model.fit(X[A == self.arm], Y[A == self.arm], sample_weight=weights)

    def quantile_plus(self, X):
        mu = self.mu_model.predict_proba(X)[:, [1]]
        Q_plus = np.zeros_like(mu)
        Q_plus[mu > (1 - self.tau)] = 1
        return Q_plus

    def quantile_minus(self, X):
        mu = self.mu_model.predict_proba(X)[:, [1]]
        Q_minus = np.zeros_like(mu)
        Q_minus[mu > (self.tau)] = 1
        return Q_minus

    def rho_plus(self, X):
        mu = self.mu_model.predict_proba(X)[:, [1]]
        rho_plus = np.minimum(1 - 1 / self.gamma + mu / self.gamma, mu * self.gamma)
        return rho_plus

    def rho_minus(self, X):
        mu = self.mu_model.predict_proba(X)[:, [1]]
        rho_minus = np.maximum(1 - self.gamma + mu * self.gamma, mu / self.gamma)
        return rho_minus

    def predict(self, X):
        predictions = np.hstack(
            (
                self.propensity_model.predict_proba(X)[:, [1]],
                self.quantile_plus(X).reshape(-1, 1),
                self.quantile_minus(X).reshape(-1, 1),
                self.mu_model.predict_proba(X)[:, [1]],
                self.rho_plus(X).reshape(-1, 1),
                self.rho_minus(X).reshape(-1, 1),
            )
        )

        return predictions
