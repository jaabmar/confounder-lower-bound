import concurrent.futures
import time
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import xgboost as xgb

from ATE.methods.QB import QBSensitivityAnalysis
from ATE.methods.ZSB import ZSBSensitivityAnalysis
from utils_evaluate import get_quantile_regressor


class BootstrapSensitivityAnalysis:
    """
    A class that performs bootstrap sensitivity analysis. It applies bootstrap methodology
    to a SensitivityAnalysis object to obtain distributions for upper and lower bounds of
    sensitivity measures.

    Attributes:
        sa_name (str): Name of the sensitivity analysis.
        obs_inputs (np.ndarray): Array of observational inputs.
        obs_treatment (np.ndarray): Array of treatment indicators.
        obs_outcome (np.ndarray): Array of outcomes.
        gammas (list): List of gamma values for sensitivity analysis.
        binary (bool): Indicates if the outcome is binary. Defaults to False.
        seed (int): Random seed for reproducibility. Defaults to 50.
        e_x_func (Optional[Callable[..., np.ndarray]]): Function for propensity score.
        bounds_dist (Optional[Dict[str, Tuple]]): Distribution of bounds. Initialized as None.
        kwargs_sa_control (Dict): Keyword arguments for control group in sensitivity analysis.
        kwargs_sa_treated (Dict): Keyword arguments for treated group in sensitivity analysis.
        outcome_func_dict (Optional[Dict]): Dictionary of outcome functions. Initialized as None.
    """

    def __init__(
        self,
        sa_name: str,
        inputs: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
        gammas: list,
        binary: bool = False,
        seed: int = 50,
        e_x_func: Optional[Callable[..., np.ndarray]] = None,
    ):
        self.sa_name = sa_name
        self.obs_inputs = inputs
        self.obs_treatment = treatment
        self.obs_outcome = outcome
        self.gammas = gammas
        self.seed = seed
        self.e_x_func = e_x_func
        self.bounds_dist = None
        self.binary = binary
        self.kwargs_sa_control = {gamma: {} for gamma in self.gammas}
        self.kwargs_sa_treated = {gamma: {} for gamma in self.gammas}
        self.outcome_func_dict = None

    def bootstrap(
        self,
        num_samples: int,
        sample_size: Optional[int] = None,
        fast_quantile: bool = False,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Performs bootstrap analysis to obtain distributions for upper and lower bounds.

        Args:
            num_samples (int): Number of bootstrap samples to generate.
            sample_size (Optional[int]): Size of each bootstrap sample. Defaults to the size of obs_inputs.
            fast_quantile (bool): Indicates if fast quantile computation is used.

        Returns:
            Dict[str, Tuple[np.ndarray, np.ndarray]]: Dictionary mapping gamma values to tuples
            of arrays containing lower and upper bound distributions, respectively.
        """
        if sample_size is None:
            sample_size = len(self.obs_inputs)

        start_time = time.time()

        # Save a copy of obs_inputs, obs_treatment, obs_outcome
        obs_inputs = self.obs_inputs
        obs_treatment = self.obs_treatment
        obs_outcome = self.obs_outcome
        if self.sa_name == "QB" and (not self.binary):
            self.kwargs_sa_treated, self.kwargs_sa_control = self.get_all_quantile_models(
                fast=fast_quantile
            )
            print("Quantile functions are now trained for QB. Starting bootstrap.")

        elif self.sa_name == "QB" and self.binary:
            outcome_model_control = xgb.XGBRegressor()
            outcome_model_treatment = xgb.XGBRegressor()

            outcome_model_control.fit(
                obs_inputs[obs_treatment == 0], obs_outcome[obs_treatment == 0]
            )
            outcome_model_treatment.fit(
                obs_inputs[obs_treatment == 1], obs_outcome[obs_treatment == 1]
            )
            self.outcome_func_dict = [outcome_model_control, outcome_model_treatment]
            print("Outcome functions are now trained for binary QB. Starting bootstrap.")

        # Generate bootstrap samples using random sampling with replacement
        bootstrap_samples = [
            (
                np.random.choice(len(obs_inputs), size=sample_size, replace=True),
                obs_inputs,
                obs_treatment,
                obs_outcome,
            )
            for _ in range(num_samples)
        ]

        # Solve for upper and lower bounds for each bootstrap sample in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            bounds = list(executor.map(self.solve_sample_bounds, bootstrap_samples))

        bounds_dist = {str(gamma): ([], []) for gamma in self.gammas}
        for gamma in self.gammas:
            for bag in bounds:
                lb = [i[1] for i in filter(lambda x, gamma=gamma: x[0] == gamma, bag[0])][0]
                ub = [i[1] for i in filter(lambda x, gamma=gamma: x[0] == gamma, bag[1])][0]
                bounds_dist[str(gamma)][0].append(lb)
                bounds_dist[str(gamma)][1].append(ub)

        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"Elapsed time for {num_samples} bootstrap samples: {elapsed_time:.2f} seconds")

        self.bounds_dist = bounds_dist

        return self.bounds_dist

    def solve_sample_bounds(self, args: Tuple) -> Tuple[list, list]:
        """
        Solves for lower and upper bounds using a bootstrap sample.

        Args:
            args (Tuple): A tuple containing sample indices, obs_inputs, obs_treatment, and obs_outcome.

        Returns:
            Tuple[list, list]: A tuple of two lists containing lower bounds and upper bounds, respectively.
        """

        # Unpack arguments
        sample_indices, obs_inputs, obs_treatment, obs_outcome = args
        sample_obs_inputs = obs_inputs[sample_indices]
        sample_obs_treatment = obs_treatment[sample_indices]
        sample_obs_outcome = obs_outcome[sample_indices]

        # Check if object type is supported
        sa_class_map = {
            "ZSB": ZSBSensitivityAnalysis,
            "QB": QBSensitivityAnalysis,
        }
        try:
            sa_class = sa_class_map[self.sa_name]
        except KeyError as exc:
            raise ValueError("Sensitivity analysis type is not supported") from exc

        lower_bounds = []
        upper_bounds = []

        for gamma in self.gammas:
            if self.sa_name == "ZSB":
                sa_new_treatment = sa_class(
                    obs_inputs=sample_obs_inputs,
                    obs_treatment=sample_obs_treatment,
                    obs_outcome=sample_obs_outcome,
                    gamma=gamma,
                    arm=1,
                    e_x_func=self.e_x_func,
                )
                sa_new_control = sa_class(
                    obs_inputs=sample_obs_inputs,
                    obs_treatment=sample_obs_treatment,
                    obs_outcome=sample_obs_outcome,
                    gamma=gamma,
                    arm=0,
                    e_x_func=self.e_x_func,
                )
            elif self.sa_name == "QB" and self.binary:
                sa_new_treatment = sa_class(
                    obs_inputs=sample_obs_inputs,
                    obs_treatment=sample_obs_treatment,
                    obs_outcome=sample_obs_outcome,
                    gamma=gamma,
                    arm=1,
                    e_x_func=self.e_x_func,
                    binary=self.binary,
                    outcome_func_dict=self.outcome_func_dict,
                )
                sa_new_control = sa_class(
                    obs_inputs=sample_obs_inputs,
                    obs_treatment=sample_obs_treatment,
                    obs_outcome=sample_obs_outcome,
                    gamma=gamma,
                    arm=0,
                    e_x_func=self.e_x_func,
                    binary=self.binary,
                    outcome_func_dict=self.outcome_func_dict,
                )
            else:  # QB not binary
                sa_new_treatment = sa_class(
                    obs_inputs=sample_obs_inputs,
                    obs_treatment=sample_obs_treatment,
                    obs_outcome=sample_obs_outcome,
                    gamma=gamma,
                    arm=1,
                    e_x_func=self.e_x_func,
                    binary=self.binary,
                    lb_qr_func=self.kwargs_sa_treated[gamma]["lb_qr_func"],
                    ub_qr_func=self.kwargs_sa_treated[gamma]["ub_qr_func"],
                )
                sa_new_control = sa_class(
                    obs_inputs=sample_obs_inputs,
                    obs_treatment=sample_obs_treatment,
                    obs_outcome=sample_obs_outcome,
                    gamma=gamma,
                    arm=0,
                    e_x_func=self.e_x_func,
                    binary=self.binary,
                    lb_qr_func=self.kwargs_sa_control[gamma]["lb_qr_func"],
                    ub_qr_func=self.kwargs_sa_control[gamma]["ub_qr_func"],
                )
            lb_new_control, ub_new_control = sa_new_control.solve_bounds()
            lb_new_treatment, ub_new_treatment = sa_new_treatment.solve_bounds()
            upper_bounds.append((gamma, ub_new_treatment - lb_new_control))
            lower_bounds.append((gamma, lb_new_treatment - ub_new_control))

        return lower_bounds, upper_bounds

    def get_all_quantile_models(
        self, fast: bool = False
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """
        Initializes and fits quantile regression models for both treated and control groups.

        Args:
            fast (bool): Indicates if a fast solver should be used for quantile regression.

        Returns:
            Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]: A tuple of two dictionaries,
            one for treated and one for control groups, containing the upper and lower bound quantile
            regression functions.
        """

        # Initializing dictionaries to store the models for treated and control groups.
        treated_models_dict = {}
        control_models_dict = {}

        # Iterate over gamma values.
        for gamma_value in self.gammas:
            # Calculate tau.
            tau = gamma_value / (1 + gamma_value)

            # Initialize and fit models for control group.
            ub_qr_func_control = get_quantile_regressor(
                self.obs_inputs[self.obs_treatment == 0],
                self.obs_outcome[self.obs_treatment == 0],
                tau,
                fast_solver=fast,
            )
            lb_qr_func_control = get_quantile_regressor(
                self.obs_inputs[self.obs_treatment == 0],
                self.obs_outcome[self.obs_treatment == 0],
                1 - tau,
                fast_solver=fast,
            )

            # Store the models in control_models_dict.
            control_models_dict[gamma_value] = {
                "ub_qr_func": ub_qr_func_control,
                "lb_qr_func": lb_qr_func_control,
            }

            # Initialize and fit models for treated group.
            ub_qr_func_treated = get_quantile_regressor(
                self.obs_inputs[self.obs_treatment == 1],
                self.obs_outcome[self.obs_treatment == 1],
                tau,
                fast_solver=fast,
            )

            lb_qr_func_treated = get_quantile_regressor(
                self.obs_inputs[self.obs_treatment == 1],
                self.obs_outcome[self.obs_treatment == 1],
                1 - tau,
                fast_solver=fast,
            )

            # Store the models in treated_models_dict.
            treated_models_dict[gamma_value] = {
                "ub_qr_func": ub_qr_func_treated,
                "lb_qr_func": lb_qr_func_treated,
            }

        return treated_models_dict, control_models_dict
