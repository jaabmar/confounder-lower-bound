import numpy as np
from sklearn.gaussian_process.kernels import RBF


#######################
# Quantile Regressors #
#######################
class KernelQuantileRegressor:
    def __init__(self, kernel, tau):
        self.kernel = kernel
        self.tau = tau

    def fit(self, X, Y):
        self.sorted_Y_idx = np.argsort(Y)
        self.sorted_Y = Y[self.sorted_Y_idx]
        self.kernel.fit(X[self.sorted_Y_idx], Y[self.sorted_Y_idx])
        return self

    def predict(self, X):
        preds = np.empty(X.shape[0])
        sorted_weights = self.kernel.predict(X)
        for i, _ in enumerate(X):
            quantile_idx = np.where((np.cumsum(sorted_weights[i]) >= self.tau) is True)[0][0]
            preds[i] = self.sorted_Y[quantile_idx]
        return preds


##################
# Kernel Methods #
##################
class RFKernel:
    def __init__(self, rf):
        self.rf = rf

    def fit(self, X, Y):
        self.rf.fit(X, Y)
        self.train_leaf_map = self.rf.apply(X)

    def predict(self, X):
        weights = np.empty((X.shape[0], self.train_leaf_map.shape[0]))
        leaf_map = self.rf.apply(X)
        for i, _ in enumerate(X):
            P = self.train_leaf_map == leaf_map[[i]]
            weights[i] = (1.0 * P / P.sum(axis=0)).mean(axis=1)
        return weights


class RBFKernel:
    def __init__(self, scale=1):
        self.kernel = RBF(length_scale=scale)

    def fit(self, X, Y):
        self.X_train = X
        self.Y_train = Y
        return self

    def predict(self, X):
        weights = self.kernel(X, self.X_train)
        # Normalize weights
        norm_weights = weights / weights.sum(axis=1).reshape(-1, 1)  # type: ignore
        return norm_weights @ self.Y_train


############################
# Superquantile regressors #
############################
class KernelSuperquantileRegressor:
    def __init__(self, kernel, tau, tail="left"):
        self.kernel = kernel
        self.tau = tau
        if tail not in ["left", "right"]:
            raise ValueError(
                f"The 'tail' parameter can only take values in ['left', 'right']. Got '{tail}' instead."
            )
        self.tail = tail

    def fit(self, X, Y):
        self.sorted_Y_idx = np.argsort(Y)
        self.sorted_Y = Y[self.sorted_Y_idx]
        self.kernel.fit(X[self.sorted_Y_idx], Y[self.sorted_Y_idx])
        return self

    def predict(self, X):
        preds = np.empty(X.shape[0])
        sorted_weights = self.kernel.predict(X)
        for i, _ in enumerate(X):
            if self.tail == "right":
                idx_tail = np.where((np.cumsum(sorted_weights[i]) >= self.tau) is True)[0]
                preds[i] = np.sum(self.sorted_Y[idx_tail] * sorted_weights[i][idx_tail]) / (
                    1 - self.tau
                )
            else:
                idx_tail = np.where((np.cumsum(sorted_weights[i]) <= self.tau) is True)[0]
                preds[i] = np.sum(self.sorted_Y[idx_tail] * sorted_weights[i][idx_tail]) / self.tau
        return preds
