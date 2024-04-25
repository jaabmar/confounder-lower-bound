from typing import Callable, Optional, Tuple

import numpy as np
# from mlinsights.mlmodel import QuantileLinearRegression
from sklearn.linear_model import LogisticRegression, QuantileRegressor

from test_confounding.ate_bounds.utils_ate_bounds import get_quantile_regressor


class QBSensitivityAnalysis:
    """
    Class for conducting sharp sensitivity analysis for Inverse Probability Weighting via quantile balancing (Dorn et al.).

    This class analyzes the sensitivity of causal effect estimates to unobserved confounding. It uses quantile regression
    to estimate bounds on the potential outcomes under different levels of unmeasured confounding.

    Args:
        obs_inputs (np.ndarray): Array containing the input data.
        obs_treatment (np.ndarray): Array containing the treatment status for each observation.
        obs_outcome (np.ndarray): Array containing the outcomes for each observed individual.
        gamma (float): Confounding strength.
        arm (int): Indicator for treatment (1) or control (0) group.
        binary (bool): Indicator for binary outcome. Defaults to False.
        lb_qr_func (Optional[Callable[..., np.ndarray]]): Quantile function for the lower bound.
        ub_qr_func (Optional[Callable[..., np.ndarray]]): Quantile function for the upper bound.
        e_x_func (Optional[Callable[..., np.ndarray]]): True propensity score function.
        outcome_func_dict (Optional[Dict[int, Callable]]): Dictionary of functions to compute outcomes for binary cases.

    Methods:
        solve_bounds(): Solves for the lower and upper bounds.
        compute_closed_form_bounds(quant_reg, target): Computes closed form bounds.
    """

    def __init__(
        self,
        obs_inputs: np.ndarray,
        obs_treatment: np.ndarray,
        obs_outcome: np.ndarray,
        gamma: float,
        arm: int,
        binary: bool = False,
        lb_qr_func: Optional[Callable[..., np.ndarray]] = None,
        ub_qr_func: Optional[Callable[..., np.ndarray]] = None,
        e_x_func: Optional[Callable[..., np.ndarray]] = None,
        outcome_func_dict: Optional[dict] = None,
    ) -> None:
        self.obs_inputs = obs_inputs
        self.obs_treatment = obs_treatment
        self.obs_outcome = obs_outcome
        self.e_x_func = e_x_func
        self.gamma = gamma
        self.tau = gamma / (gamma + 1)
        self.binary = binary
        if ub_qr_func is None and not binary:
            self.ub_qr_func = get_quantile_regressor(
                obs_inputs[obs_treatment == arm],
                obs_outcome[obs_treatment == arm],
                self.tau,
                fast_solver=False,
            )
            self.lb_qr_func = get_quantile_regressor(
                obs_inputs[obs_treatment == arm],
                obs_outcome[obs_treatment == arm],
                1 - self.tau,
                fast_solver=False,
            )
        else:
            self.ub_qr_func = ub_qr_func
            self.lb_qr_func = lb_qr_func

        self.arm = arm

        # Compute propensity score if not given
        if e_x_func is None:
            logreg = LogisticRegression(
                C=1,
                penalty="elasticnet",
                solver="saga",
                l1_ratio=0.7,
                max_iter=10000,
            )
            logreg.fit(obs_inputs, obs_treatment)
            self.e_x = logreg.predict_proba(obs_inputs)[:, 1]
        else:
            self.e_x = e_x_func(obs_inputs).reshape(-1)

        if self.arm == 0:
            self.e_x = 1 - self.e_x

        if not binary:
            self.lb_g_x = self.lb_qr_func.predict(obs_inputs[obs_treatment == self.arm]).reshape(
                -1, 1
            )
            self.ub_g_x = self.ub_qr_func.predict(obs_inputs[obs_treatment == self.arm]).reshape(
                -1, 1
            )
            self.weights = (1 - self.e_x[obs_treatment == self.arm]) / self.e_x[
                obs_treatment == self.arm
            ]
        else:
            self.outcome_func = outcome_func_dict[self.arm]

        self.name = "QB"

    def solve_bounds(self) -> Tuple:
        """
        Solves for the lower and upper bounds.

        Returns:
            List containing the lower bound and upper bound.
        """

        if self.binary:
            q_min = (self.outcome_func.predict(self.obs_inputs) > self.tau).astype(int)
            q_plus = (self.outcome_func.predict(self.obs_inputs) > 1 - self.tau).astype(int)
            self.obs_outcome = self.obs_outcome.astype(int)

            if self.arm == 1:
                ub = (
                    q_plus * self.obs_treatment / self.e_x
                    + (self.obs_outcome - q_plus)
                    * self.obs_treatment
                    * (
                        1
                        + (1 - self.e_x)
                        / self.e_x
                        * self.gamma ** (np.sign(self.obs_outcome - q_plus))
                    )
                ).mean()
                lb = (
                    q_min * self.obs_treatment / self.e_x
                    + (self.obs_outcome - q_min)
                    * self.obs_treatment
                    * (
                        1
                        + (1 - self.e_x)
                        / self.e_x
                        * self.gamma ** (-np.sign(self.obs_outcome - q_min))
                    )
                ).mean()

            else:
                ub = (
                    q_plus * (1 - self.obs_treatment) / self.e_x
                    + (self.obs_outcome - q_plus)
                    * (1 - self.obs_treatment)
                    * (
                        1
                        + (1 - self.e_x)
                        / self.e_x
                        * self.gamma ** (np.sign(self.obs_outcome - q_plus))
                    )
                ).mean()
                lb = (
                    q_min * (1 - self.obs_treatment) / self.e_x
                    + (self.obs_outcome - q_min)
                    * (1 - self.obs_treatment)
                    * (
                        1
                        + (1 - self.e_x)
                        / self.e_x
                        * self.gamma ** (-np.sign(self.obs_outcome - q_min))
                    )
                ).mean()

        else:
            # Compute bounds
            lb = -self.compute_closed_form_bounds(
                self.lb_g_x, -self.obs_outcome[self.obs_treatment == self.arm]
            )

            ub = self.compute_closed_form_bounds(
                self.ub_g_x, self.obs_outcome[self.obs_treatment == self.arm]
            )

        return (lb, ub)

    def compute_closed_form_bounds(self, g_x: np.ndarray, target: np.ndarray) -> float:
        """Computes closed form bounds.

        Args:
            g_x (np.ndarray): balancing function, i.e., conditional quantiles.
            target (np.ndarray): outcomes Y.

        Returns:
            The computed lower/upper bound.
        """
        # quant_reg = QuantileLinearRegression(quantile=self.tau, max_iter=1000)
        quant_reg = QuantileRegressor(quantile=self.tau)

        # Fit  quantile regressor
        quant_reg.fit(g_x, target, sample_weight=self.weights)
        quantile_pred = quant_reg.predict(g_x)
        residual = target - quantile_pred
        sign_residual = np.sign(residual)

        term_1 = (residual * (1 + np.power(self.gamma, sign_residual) * self.weights)).mean()

        term_2 = (quantile_pred / self.e_x[self.obs_treatment == self.arm]).mean()

        term_3 = (1 / self.e_x[self.obs_treatment == self.arm]).mean()

        bound = (term_1 + term_2) / term_3

        return bound
