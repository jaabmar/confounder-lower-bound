from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import fsolve


def solve_quantile_search(z, x, y):
    return y - (1 / (1 + (1 - y) * (1 / x) / y) * z + 1 / (1 + (1 - y) * x / y) * (1 - z))


def alpha_fn(pi, lambda_):
    return (pi * lambda_) ** -1 + 1.0 - lambda_**-1


def beta_fn(pi, lambda_):
    return lambda_ * (pi) ** -1 + 1.0 - lambda_


def rejection_sampler(
    data: pd.DataFrame,
    weights: np.ndarray,
    rng: np.random.Generator,
    M: int,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, np.ndarray]]:
    """
    This function performs the rejection sampling method.

    Args:
        data (pd.DataFrame): The input data.
        weights (np.ndarray): The weights for rejection sampling.
        rng (np.random.Generator): Random number generator.
        M (int): Constant used in rejection sampling.

    Returns:
        pd.DataFrame: The resampled data.
    """
    uniform_vector = rng.uniform(0, 1, len(data))
    accepted_rows = uniform_vector < (weights / M)
    data_resampled = data[accepted_rows].reset_index(drop=True)
    return data_resampled


def weights_for_rejection_sampler(
    data: pd.DataFrame, confound_func_params: Dict[str, Union[str, float]] = {}
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function calculates the weights for rejection sampling.

    Args:
        data (pd.DataFrame): The input data.
        confound_func_params (Dict[str, Union[str, float]]): Parameters for the confounding function.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: The weights, p_TC, and pT for rejection sampling.
    """

    pT = np.mean(data["T"]) * np.ones(len(data))
    pT[data["T"] == 0] = 1 - pT[data["T"] == 0]
    p_TC = researcher_specified_function_for_confounding(
        data, confound_func_params=confound_func_params
    )
    p_TC[data["T"] == 0] = 1 - p_TC[data["T"] == 0]
    weights = p_TC / pT
    return weights, p_TC, pT


def researcher_specified_function_for_confounding(
    data: pd.DataFrame, confound_func_params: Dict = {}
) -> np.ndarray:
    """
    This function implements the researcher-specified function for generating confounding.

    Args:
        data (pd.DataFrame): The input data.
        confound_func_params (Dict): Parameters for the confounding function.

    Returns:
        np.ndarray: p_TC.
    """

    assert confound_func_params.get("para_form") is not None

    if confound_func_params["para_form"] == "adversarial_with_conf":
        true_gamma = confound_func_params["true_gamma"]
        marginal_treatment = np.mean(data["T"])
        weight = (1 - marginal_treatment) / marginal_treatment
        adv_prop_up = 1 / (1 + weight * true_gamma)
        adv_prop_down = 1 / (1 + weight / true_gamma)
        if len(np.unique(data["C"])) == 2:
            p_TC = np.array([adv_prop_up if c == 1.0 else adv_prop_down for c in data["C"]])

        else:
            quantile = fsolve(solve_quantile_search, 0.5, args=(true_gamma, marginal_treatment))[0]
            quantile_c = np.quantile(data["C"], q=quantile)
            p_TC = np.array([adv_prop_up if c > quantile_c else adv_prop_down for c in data["C"]])
    elif confound_func_params["para_form"] == "adversarial_with_conf_inv":
        true_gamma = confound_func_params["true_gamma"]
        marginal_treatment = np.mean(data["T"])
        weight = (1 - marginal_treatment) / marginal_treatment
        adv_prop_up = 1 / (1 + weight * true_gamma)
        adv_prop_down = 1 / (1 + weight / true_gamma)
        if len(np.unique(data["C"])) == 2:
            p_TC = np.array([adv_prop_up if c == 0.0 else adv_prop_down for c in data["C"]])
        else:
            quantile = fsolve(solve_quantile_search, 0.5, args=(true_gamma, marginal_treatment))[0]
            quantile_c = np.quantile(data["C"], q=1 - quantile)
            p_TC = np.array([adv_prop_up if c < quantile_c else adv_prop_down for c in data["C"]])
    elif confound_func_params["para_form"] == "adversarial":
        true_gamma = confound_func_params["true_gamma"]
        marginal_treatment = np.mean(data["T"])
        weight = (1 - marginal_treatment) / marginal_treatment
        adv_prop_up = 1 / (1 + weight * true_gamma)
        adv_prop_down = 1 / (1 + weight / true_gamma)
        if len(np.unique(data["Y"])) == 2:
            p_TC = np.array([adv_prop_up if y == 1.0 else adv_prop_down for y in data["Y"]])
        else:
            quantile = fsolve(solve_quantile_search, 0.5, args=(true_gamma, marginal_treatment))[0]
            quantile_y = np.quantile(data["Y"], q=quantile)
            p_TC = np.array([adv_prop_up if y > quantile_y else adv_prop_down for y in data["Y"]])
    elif confound_func_params["para_form"] == "adversarial_inv":
        true_gamma = confound_func_params["true_gamma"]
        marginal_treatment = np.mean(data["T"])
        weight = (1 - marginal_treatment) / marginal_treatment
        adv_prop_up = 1 / (1 + weight * true_gamma)
        adv_prop_down = 1 / (1 + weight / true_gamma)
        if len(np.unique(data["Y"])) == 2:
            p_TC = np.array([adv_prop_up if y == 0.0 else adv_prop_down for y in data["Y"]])
        else:
            quantile = fsolve(solve_quantile_search, 0.5, args=(true_gamma, marginal_treatment))[0]
            quantile_y = np.quantile(data["Y"], q=1 - quantile)
            p_TC = np.array([adv_prop_up if y < quantile_y else adv_prop_down for y in data["Y"]])
    elif confound_func_params["para_form"] == "multiple":
        unique_values = np.unique(data["C"])
        p_TC = np.zeros_like(data["C"], dtype=np.float32)
        for i, val in enumerate(unique_values):
            mask = data["C"] == val  # Create a boolean mask for elements equal to val
            p_TC[mask] = confound_func_params[
                f"zeta{i}"
            ]  # Assign corresponding value from confound_func_params
    else:
        assert confound_func_params["para_form"] == "piecewise"
        assert confound_func_params.get("zeta0") is not None
        assert confound_func_params.get("zeta1") is not None
        p_TC = np.array(
            [
                confound_func_params["zeta1"] if c == 1 else confound_func_params["zeta0"]
                for c in data["C"]
            ]
        )
    return p_TC


def resample_data_with_confound_func(
    df: pd.DataFrame,
    confound_func_params: Optional[Dict] = None,
    seed: int = 42,
    M: Optional[float] = None,
) -> pd.DataFrame:
    """
    Resamples a DataFrame based on confounder function parameters using a rejection sampler.

    This function adjusts the dataset based on specified confounder function parameters to
    create a new, resampled dataset.

    Args:
        df (pd.DataFrame): The DataFrame to resample.
        confound_func_params (Optional[Dict]): Parameters for the confounder function.
            Defaults to {"para_form": "piecewise", "zeta0": 0.15, "zeta1": 0.85} if None.
        seed (int): Seed for the random number generator. Default is 42.
        M (Optional[float]): A constant used in rejection sampling. If None, it is calculated
            based on the data.

    Returns:
        pd.DataFrame: A new DataFrame that has been resampled based on the provided confounder
        function parameters.
    """
    if confound_func_params is None:
        # Default parameter values for confounder function
        confound_func_params = {"para_form": "piecewise", "zeta0": 0.15, "zeta1": 0.85}

    # Run rejection sampler
    weights, p_TC, pT = weights_for_rejection_sampler(
        data=df, confound_func_params=confound_func_params
    )
    if M is None:
        M = np.max(p_TC) / np.min(pT)
    rng = np.random.default_rng(seed)
    data_resampled = rejection_sampler(data=df, weights=weights, rng=rng, M=M)

    print("Original data num. examples = ", len(df))
    print("Downsampled data num. examples = ", len(data_resampled))

    return pd.DataFrame(data_resampled)


def subsample_df(
    true_gamma: float,
    obs_data_pre_conf: pd.DataFrame,
    seed: int = 42,
    adversarial: bool = False,
    inv: bool = False,
) -> pd.DataFrame:
    """
    This function aims to create a subsampled dataset from the provided observational data
    by considering treatment effect (gamma) and confounding variables.

    Args:
        true_gamma (float): The true confounding.
        obs_data_pre_conf (pd.DataFrame): The observational DataFrame.
        seed (int): Seed for the random number generator. Default is 42.
        adversarial (bool): If True, subsamples for Y. Default is False.
        inv (bool): If True, confounder is inversely correlated with outcome. Default is False.

    Returns:
        pd.DataFrame: subsampled DataFrame.
    """
    marginal_treatment = np.mean(obs_data_pre_conf["T"])

    weight = (1 - marginal_treatment) / marginal_treatment
    adv_prop_up = 1 / (1 + weight * true_gamma)
    adv_prop_down = 1 / (1 + weight / true_gamma)

    if inv:
        desired_prob_conf = (marginal_treatment - adv_prop_up) / (adv_prop_down - adv_prop_up)
    else:
        desired_prob_conf = (adv_prop_down - marginal_treatment) / (adv_prop_down - adv_prop_up)

    total_samples = len(obs_data_pre_conf)
    desired_samples_c1 = int(desired_prob_conf * total_samples)
    desired_samples_c0 = total_samples - desired_samples_c1

    if adversarial:
        df_c0 = obs_data_pre_conf[obs_data_pre_conf["Y"] == 0]
        df_c1 = obs_data_pre_conf[obs_data_pre_conf["Y"] == 1]
    else:
        df_c0 = obs_data_pre_conf[obs_data_pre_conf["C"] == 0]
        df_c1 = obs_data_pre_conf[obs_data_pre_conf["C"] == 1]

    # Separate into treated and untreated within c0 and c1 groups
    df_c0_treated = df_c0[df_c0["T"] == 1]
    df_c0_untreated = df_c0[df_c0["T"] == 0]
    df_c1_treated = df_c1[df_c1["T"] == 1]
    df_c1_untreated = df_c1[df_c1["T"] == 0]

    # Calculate the number of treated and untreated samples required from each group
    num_c0_treated = int(desired_samples_c0 * marginal_treatment)
    num_c0_untreated = desired_samples_c0 - num_c0_treated
    num_c1_treated = int(desired_samples_c1 * marginal_treatment)
    num_c1_untreated = desired_samples_c1 - num_c1_treated

    # Sample the required number of treated and untreated from each group
    sampled_c0_treated = df_c0_treated.sample(n=num_c0_treated, random_state=seed, replace=True)
    sampled_c0_untreated = df_c0_untreated.sample(
        n=num_c0_untreated, random_state=seed, replace=True
    )
    sampled_c1_treated = df_c1_treated.sample(n=num_c1_treated, random_state=seed, replace=True)
    sampled_c1_untreated = df_c1_untreated.sample(
        n=num_c1_untreated, random_state=seed, replace=True
    )

    # Concatenate all sampled dataframes
    subsampled_df = pd.concat(
        [sampled_c0_treated, sampled_c0_untreated, sampled_c1_treated, sampled_c1_untreated]
    )
    subsampled_df = subsampled_df.sample(frac=1, random_state=seed)

    return subsampled_df
