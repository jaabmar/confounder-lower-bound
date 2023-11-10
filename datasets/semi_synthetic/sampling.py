from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from scipy.special import expit


def solve_quantile_search(z, x, y):
    return y - (1 / (1 + (1 - y) * (1 / x) / y) * z + 1 / (1 + (1 - y) * x / y) * (1 - z))


# from sklearn.linear_model import LogisticRegression
# from sklearn_quantile import RandomForestQuantileRegressor


def alpha_fn(pi, lambda_):
    return (pi * lambda_) ** -1 + 1.0 - lambda_**-1


def beta_fn(pi, lambda_):
    return lambda_ * (pi) ** -1 + 1.0 - lambda_


def osrct_algorithm(
    data: pd.DataFrame,
    rng: np.random.Generator,
    confound_func_params: Dict[str, Union[str, float]] = {},
) -> pd.DataFrame:
    """
    This function implements the OSRCT algorithm.

    Args:
        data (pd.DataFrame): The input data.
        rng (np.random.Generator): Random number generator.
        confound_func_params (Dict[str, str], optional): Parameters for the confounding function.

    Returns:
        pd.DataFrame: The resampled data based on OSRCT algorithm.
    """

    p_SC = researcher_specified_function_for_confounding(
        data, confound_func_params=confound_func_params
    )

    bernoulli = rng.binomial(1, p_SC)

    data_resampled = data[bernoulli == data["T"]].reset_index(drop=True)
    return data_resampled


def rejection_sampler(
    data: pd.DataFrame,
    weights: np.ndarray,
    rng: np.random.Generator,
    M: int = 2,
    return_accepted_rows: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, np.ndarray]]:
    """
    This function performs the rejection sampling method.

    Args:
        data (pd.DataFrame): The input data.
        weights (np.ndarray): The weights for rejection sampling.
        rng (np.random.Generator): Random number generator.
        M (int, optional): Constant used in rejection sampling. Defaults to 2.
        return_accepted_rows (bool, optional): Whether to return the accepted rows. Defaults to False.

    Returns:
        Union[pd.DataFrame, Tuple[pd.DataFrame, np.ndarray]]: The resampled data and optionally, the accepted rows.
    """
    uniform_vector = rng.uniform(0, 1, len(data))
    accepted_rows = uniform_vector < (weights / M)
    data_resampled = data[accepted_rows].reset_index(drop=True)

    if return_accepted_rows:
        return data_resampled, accepted_rows
    return data_resampled


def weights_for_rejection_sampler(
    data: pd.DataFrame, confound_func_params: Dict[str, Union[str, float]] = {}
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function calculates the weights for rejection sampling.

    Args:
        data (pd.DataFrame): The input data.
        confound_func_params (Dict[str, str], optional): Parameters for the confounding function.

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
        confound_func_params (Dict[str, str], optional): Parameters for the confounding function.

    Returns:
        np.ndarray: The generated confounding variables.
    """

    assert confound_func_params.get("para_form") is not None

    # if confound_func_params["para_form"] == "adversarial":
    #     gamma = confound_func_params["gamma"]
    #     tau = gamma / (1 + gamma)
    #     treated_data = data[data["T"] == 1]
    #     Y_treated = treated_data["Y"]
    #     features_treated = treated_data.drop(["Y", "T"], axis=1)
    #     quantile_func_treated = RandomForestQuantileRegressor(
    #         n_estimators=200,
    #         max_depth=6,
    #         min_samples_leaf=0.01,
    #         n_jobs=-2,
    #         q=tau,
    #     )
    #     quantile_func_treated.fit(features_treated, Y_treated)
    #     y_quantiles_treated = quantile_func_treated.predict(features_treated)

    #     control_data = data[data["T"] == 0]
    #     Y_control = control_data["Y"]
    #     features_control = control_data.drop(["Y", "T"], axis=1)
    #     quantile_func_control = RandomForestQuantileRegressor(
    #         n_estimators=200,
    #         max_depth=6,
    #         min_samples_leaf=0.01,
    #         n_jobs=-2,
    #         q=tau,
    #     )
    #     quantile_func_control.fit(features_control, Y_control)
    #     y_quantiles_control = quantile_func_control.predict(features_control)

    #     data.loc[data["T"] == 1, "quantiles"] = y_quantiles_treated
    #     data.loc[data["T"] == 0, "quantiles"] = y_quantiles_control

    #     features = data.drop(["Y", "T"], axis=1)
    #     prop_score = LogisticRegression(
    #         C=1, penalty="elasticnet", solver="saga", l1_ratio=0.7, max_iter=10000
    #     )
    #     prop_score.fit(features, data["T"])
    #     pi = prop_score.predict_proba(features)[:, 1]
    #     pdb.set_trace()
    #     alpha = alpha_fn(pi, gamma)
    #     beta = beta_fn(pi, gamma)

    #     p_TC = np.where(data["Y"] > data["quantiles"], 1 / alpha, 1 / beta)
    #     data.drop(["quantiles"], axis=1)
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
        _, counts = np.unique(data["T"], return_counts=True)
        marginal_treatment = counts[1] / counts.sum()
        weight = (1 - marginal_treatment) / marginal_treatment
        adv_prop_up = 1 / (1 + weight * true_gamma)
        adv_prop_down = 1 / (1 + weight / true_gamma)
        if len(np.unique(data["Y"])) == 2:
            p_TC = np.array([adv_prop_up if y == 1.0 else adv_prop_down for y in data["Y"]])
        else:
            quantile = fsolve(solve_quantile_search, 0.5, args=(true_gamma, marginal_treatment))[0]
            quantile_y = np.quantile(data["Y"], q=quantile)
            p_TC = np.array([adv_prop_up if y > quantile_y else adv_prop_down for y in data["Y"]])
    elif confound_func_params["para_form"] == "linear":
        standar_conf = (data["C"] - data["C"].min()) / (data["C"].max() - data["C"].min())
        p_TC = expit(confound_func_params["beta"] + confound_func_params["alpha"] * standar_conf)
    elif confound_func_params["para_form"] == "linear_inv":
        standar_conf = (data["C"] - data["C"].min()) / (data["C"].max() - data["C"].min())
        p_TC = expit(
            -(confound_func_params["beta"] + confound_func_params["alpha"] * standar_conf)
        )
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
    df, confound_func_params=None, seed=42, M=None
) -> pd.DataFrame:
    if confound_func_params is None:
        # Default parameter values for confounder function
        confound_func_params = {"para_form": "piecewise", "zeta0": 0.15, "zeta1": 0.85}

    # Run rejection sampler
    weights, p_TC, pT = weights_for_rejection_sampler(
        df, confound_func_params=confound_func_params
    )
    if M is None:
        M = np.max(p_TC) / np.min(pT)
    rng = np.random.default_rng(seed)
    data_resampled = rejection_sampler(df, weights, rng, M=M)

    print("Original data num. examples = ", len(df))
    print("Downsampled data num. examples = ", len(data_resampled))

    return pd.DataFrame(data_resampled)


def subsample_df(true_gamma, obs_data_pre_conf, seed=42, adversarial=False, inv=False):
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

    sampled_c0 = df_c0.sample(n=desired_samples_c0, random_state=seed, replace=True)
    sampled_c1 = df_c1.sample(n=desired_samples_c1, random_state=seed, replace=True)

    subsampled_df = pd.concat([sampled_c0, sampled_c1])

    subsampled_df = subsampled_df.sample(frac=1, random_state=seed)

    return subsampled_df


def subsample_df_ext(true_gamma, obs_data_pre_conf, seed=42, adversarial=False, inv=False):
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
