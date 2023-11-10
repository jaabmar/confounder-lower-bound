import concurrent.futures
import time
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import xgboost as xgb

from ATE.methods.QB import QBSensitivityAnalysis
from ATE.methods.ZSB import ZSBSensitivityAnalysis

from utils_evaluate import get_quantile_regressor
<<<<<<< HEAD
=======
import xgboost as xgb
from quantile_forest import RandomForestQuantileRegressor
from sklearn.ensemble import RandomForestRegressor
from CATE.cate_bounds import  MultipleCATEBoundEstimators
>>>>>>> 1177b35e2f1510c613324413f19c5fd8728b96fb


class BootstrapSensitivityAnalysis:
    """
    A class that performs bootstrap on a SensitivityAnalysis object and obtains distributions for the upper bounds and
    lower bounds
    """

    def __init__(
        self,
        sa_name,
        inputs,
        treatment,
        outcome,
        gammas,
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
        fast_quantile: Optional[bool] = False,
    ) -> Dict[str, Tuple]:
        """
        Perform bootstrap on the SensitivityAnalysis object and obtain distributions for the upper bounds and lower bounds
        :param num_samples: number of bootstrap samples to generate
        :param sample_size: size of each bootstrap sample
        :return: tuple of lists containing distribution values for lower bounds and upper bounds respectively
        """
        if sample_size is None:
            sample_size = len(self.obs_inputs)

        start_time = time.time()

        # Save a copy of obs_inputs, obs_treatment, obs_outcome
        obs_inputs = self.obs_inputs
        obs_treatment = self.obs_treatment
        obs_outcome = self.obs_outcome
<<<<<<< HEAD
        if self.sa_name == "QB" and (not self.binary):
            self.kwargs_sa_treated, self.kwargs_sa_control = self.get_all_quantile_models(
                fast=fast_quantile
            )
            print("Quantile functions are now trained for QB. Starting bootstrap.")

        elif self.binary:
            outcome_model_control = xgb.XGBRegressor()
            outcome_model_treatment = xgb.XGBRegressor()

            outcome_model_control.fit(
                obs_inputs[obs_treatment == 0], obs_outcome[obs_treatment == 0]
            )
            outcome_model_treatment.fit(
                obs_inputs[obs_treatment == 1], obs_outcome[obs_treatment == 1]
            )
            self.outcome_func_dict = [outcome_model_control, outcome_model_treatment]
            print("Outcome functions are now trained. Starting bootstrap.")
=======
        #if self.sa_name == "QB" and (not self.binary):
          #  self.kwargs_sa_treated, self.kwargs_sa_control = self.get_all_quantile_models()
           # print("Quantile functions are now trained for QB. Starting bootstrap.")\
        self.kwargs = {}
        if self.sa_name == "QB" and not self.binary:
            self.qr_func_control = RandomForestQuantileRegressor(
                    n_estimators=200,
                    max_depth=6,
                    min_samples_leaf=0.01,
                    n_jobs=-2)
            self.qr_func_treatment = RandomForestQuantileRegressor(n_estimators=200,
                    max_depth=6,
                    min_samples_leaf=0.01,
                    n_jobs=-2)

            self.qr_func_control.fit(obs_inputs[obs_treatment == 0], obs_outcome[obs_treatment == 0])
            self.qr_func_treatment.fit(obs_inputs[obs_treatment == 1], obs_outcome[obs_treatment == 1])

            self.kwargs= {'qr_funcs':[self.qr_func_control,self.qr_func_treatment]}
            

        elif self.sa_name == "QB" and self.binary:
            outcome_model_control = LogisticRegression()
            outcome_model_treatment = LogisticRegression()
            
            outcome_model_control.fit(obs_inputs[obs_treatment == 0], obs_outcome[obs_treatment == 0])
            outcome_model_treatment.fit(obs_inputs[obs_treatment == 1], obs_outcome[obs_treatment == 1])
            self.outcome_func_dict = [outcome_model_control, outcome_model_treatment]
            print("Outcome functions are now trained for QB. Starting bootstrap.")


            
            
            
>>>>>>> 1177b35e2f1510c613324413f19c5fd8728b96fb

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

    def solve_sample_bounds(self, args):
        """
        Helper function to solve for lower and upper bounds using a bootstrap sample.
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
            if self.binary:
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
            elif not self.binary and self.sa_name == "QB":
                sa_new_treatment = sa_class(
                    obs_inputs=sample_obs_inputs,
                    obs_treatment=sample_obs_treatment,
                    obs_outcome=sample_obs_outcome,
                    gamma=gamma,
                    arm=1,
                    e_x_func=self.e_x_func,
                    lb_qr_func=self.kwargs_sa_treated[gamma]["lb_qr_func"],
                    ub_qr_func=self.kwargs_sa_treated[gamma]["ub_qr_func"],
                    binary=self.binary,
                )
                sa_new_control = sa_class(
                    obs_inputs=sample_obs_inputs,
                    obs_treatment=sample_obs_treatment,
                    obs_outcome=sample_obs_outcome,
                    gamma=gamma,
                    arm=0,
                    e_x_func=self.e_x_func,
                    lb_qr_func=self.kwargs_sa_control[gamma]["lb_qr_func"],
                    ub_qr_func=self.kwargs_sa_control[gamma]["ub_qr_func"],
                    binary=self.binary,
                )
            else:
                sa_new_treatment = sa_class(
                    obs_inputs=sample_obs_inputs,
                    obs_treatment=sample_obs_treatment,
                    obs_outcome=sample_obs_outcome,
                    gamma=gamma,
                    arm=1,
                    e_x_func=self.e_x_func,
<<<<<<< HEAD
=======
                    binary=self.binary,
                    **self.kwargs
>>>>>>> 1177b35e2f1510c613324413f19c5fd8728b96fb
                )
                sa_new_control = sa_class(
                    obs_inputs=sample_obs_inputs,
                    obs_treatment=sample_obs_treatment,
                    obs_outcome=sample_obs_outcome,
                    gamma=gamma,
                    arm=0,
                    e_x_func=self.e_x_func,
<<<<<<< HEAD
=======
                    binary=self.binary,
                    **self.kwargs
>>>>>>> 1177b35e2f1510c613324413f19c5fd8728b96fb
                )
            lb_new_control, ub_new_control = sa_new_control.solve_bounds()
            lb_new_treatment, ub_new_treatment = sa_new_treatment.solve_bounds()
            upper_bounds.append((gamma, ub_new_treatment - lb_new_control))
            lower_bounds.append(((gamma, lb_new_treatment - ub_new_control)))

        return lower_bounds, upper_bounds

    def get_all_quantile_models(self, fast=False):
        # Initializing dictionaries to store the models for treated and control groups.
        treated_models_dict = {}
        control_models_dict = {}

        # Define model hyperparameters.
        # kwargs = {
        #     "n_estimators": 200,
        #     "max_depth": 6,
        #     "min_samples_leaf": 0.01,
        #     "n_jobs": 1,
        #     "random_state": self.seed,
        # }

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
