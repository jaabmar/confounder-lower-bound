from typing import Callable, Optional, Tuple

import numpy as np
from cvxpy.expressions.variable import Variable
from cvxpy.problems.objective import Maximize, Minimize
from cvxpy.problems.problem import Problem
from sklearn.linear_model import LogisticRegression


class ZSBSensitivityAnalysis:
    """
    Class for conducting sensitivity analysis using Convex Optimization (Zhao et al.)

    Args:
        obs_inputs (np.ndarray): array containing the input data.
        obs_treatment (np.ndarray): array containing the treatment status for each observation.
        obs_outcome (np.ndarray): array containing the outcome for each observed individual.
        gamma (float): confounding strength.
        arm (int): indicator for treatment (arm = 1) or control (arm = 0) group.
        e_x_func (Optional[Callable[..., np.ndarray]]): true propensity score function.

    Methods:
        solve_bounds(): solves for the lower and upper bounds.
    """

    def __init__(
        self,
        obs_inputs: np.ndarray,
        obs_treatment: np.ndarray,
        obs_outcome: np.ndarray,
        gamma: float,
        arm: int,
        e_x_func: Optional[Callable[..., np.ndarray]] = None,
    ) -> None:
        self.obs_inputs = obs_inputs
        self.obs_treatment = obs_treatment
        self.obs_outcome = obs_outcome
        self.e_x_func = e_x_func
        self.arm = arm
        # Compute propensity score if not given
        if e_x_func is None:
            clf = LogisticRegression(
                C=1,
                penalty="elasticnet",
                solver="saga",
                l1_ratio=0.7,
                max_iter=10000,
            )
            clf.fit(obs_inputs, obs_treatment)
            self.e_x = clf.predict_proba(obs_inputs)[:, 1]
        else:
            self.e_x = e_x_func(obs_inputs).reshape(-1)

        self.gamma = gamma

        self.n_obs_t = (obs_treatment == self.arm).sum()
        if self.arm == 1:
            self.p = (1 - self.e_x[obs_treatment == self.arm]) / self.e_x[obs_treatment == self.arm]
        else:
            self.p = self.e_x[obs_treatment == self.arm] / (1 - self.e_x[obs_treatment == self.arm])

        self.k = self.p * obs_outcome[obs_treatment == self.arm]
        self.o = obs_outcome[obs_treatment == self.arm].sum()
        self.name = "ZSB"

    def solve_bounds(self) -> Tuple:
        """
        Solves for the lower and upper bounds.

        Returns:
        - Tuple containing the lower bound and upper bound
        """
        # Initialize variables
        t = Variable(nonneg=True)
        z = Variable((self.n_obs_t))

        # Create objective functions
        obj_lb = Minimize(self.k @ z + t * self.o)
        obj_ub = Maximize(self.k @ z + t * self.o)

        # Initialize constraints (must be a list)
        constraints = [
            t / self.gamma <= z,
            z <= self.gamma * t,
            self.p @ z + t * self.n_obs_t == 1,
        ]

        # Create our problems
        problem_lb = Problem(obj_lb, constraints)
        problem_ub = Problem(obj_ub, constraints)

        # Solve the lower and upper bound problems
        lb = problem_lb.solve()
        ub = problem_ub.solve()

        return lb, ub
