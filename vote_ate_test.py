import argparse
import ast
import warnings
from test import run_multiple_ate_hypothesis_test

import numpy as np
import pandas as pd
from scipy.stats import bootstrap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

import wandb
from ATE.ate_bounds import BootstrapSensitivityAnalysis
from datasets.semi_synthetic.sampling import (
    resample_data_with_confound_func,
    subsample_df,
    subsample_df_ext,
)
from datasets.semi_synthetic.vote import CAT_COVAR_VOTE, NUM_COVAR_VOTE, load_vote_data

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)


def run_vote_ate_test(args):
    confound_func = ast.literal_eval(args.confound_func)
    if args.wandb == 1:
        wandb.init(
            project="vote_strong_small",
            entity="sml-eth",
            settings=wandb.Settings(start_method="fork"),
        )
        wandb.config.update(args)

    support_var = "hh_size"
    support_feature_values = [1, 2, 3]

    if confound_func["para_form"] == "adversarial":
        assert args.conf_var == ""

    obs_data_pre_conf, rct_data = load_vote_data(
        conf_var=args.conf_var,
        support_var=support_var,
        split_data=True,
        support_feature_values=support_feature_values,
        proportion_full_support=args.proportion_full_support,
        seed=72,
    )

    if (
        confound_func["para_form"] == "adversarial_with_conf"
        and len(np.unique(obs_data_pre_conf["C"])) == 2
    ):
        true_gamma = confound_func["true_gamma"]
        obs_data_pre_conf = subsample_df(
            true_gamma=true_gamma, obs_data_pre_conf=obs_data_pre_conf, seed=args.seed
        )
    elif confound_func["para_form"] == "adversarial":
        true_gamma = confound_func["true_gamma"]
        obs_data_pre_conf = subsample_df_ext(
            true_gamma=true_gamma,
            obs_data_pre_conf=obs_data_pre_conf,
            seed=args.seed,
            adversarial=True,
            inv=False,
        )

    obs_data = resample_data_with_confound_func(
        obs_data_pre_conf, confound_func_params=confound_func, seed=args.seed
    )

    obs_data = obs_data.sample(frac=0.3, random_state=args.seed)

    numeric_covariates = NUM_COVAR_VOTE
    categorical_covariates = CAT_COVAR_VOTE

    # Define transformer for encoding and normalization
    transformer = ColumnTransformer(
        [
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                [col for col in categorical_covariates if col != args.conf_var],
            ),
            (
                "normalizer",
                MinMaxScaler(),
                [col for col in numeric_covariates if col != args.conf_var],
            ),
        ]
    )

    if not isinstance(rct_data, pd.DataFrame):
        raise TypeError("rct_data is not a pd.DataFrame")

    print("RCT num. samples: ", rct_data.shape[0])  # type: ignore
    print()

    if confound_func["para_form"] == "adversarial":
        x_obs_raw = obs_data.drop(["Y", "T"], axis=1)
        x_rct_raw = rct_data.drop(["Y", "T"], axis=1)
    else:
        x_obs_raw = obs_data.drop(["Y", "T", "C"], axis=1)
        x_rct_raw = rct_data.drop(["Y", "T", "C"], axis=1)

    # Transform the features into one-hot encoding
    x_obs_encoded = transformer.fit_transform(x_obs_raw)
    x_rct_encoded = transformer.transform(x_rct_raw)

    t_rct, y_rct, x_rct = (
        np.array(rct_data["T"].values),
        np.array(rct_data["Y"].values),
        x_rct_encoded,
    )
    t_obs, y_obs, x_obs = (
        np.array(obs_data["T"].values),
        np.array(obs_data["Y"].values),
        x_obs_encoded,
    )
    n_rct = x_rct.shape[0]
    n_obs = x_obs.shape[0]
    alpha_trim = 0.00001
    x_all = np.concatenate((x_rct, x_obs), axis=0)  # type: ignore
    s_all = np.concatenate((np.ones(n_rct), np.zeros(n_obs)))
    clf = RandomForestClassifier(random_state=args.seed)
    clf.fit(x_all, s_all)
    pi_s = clf.predict_proba(x_all)[:, 1]
    O_idx = pi_s > alpha_trim

    x_obs_trimm, t_obs_trimm, y_obs_trimm = (
        x_obs[O_idx[n_rct:]],  # type: ignore
        t_obs[O_idx[n_rct:]],
        y_obs[O_idx[n_rct:]],
    )
    _, t_rct_trimm, y_rct_trimm = (
        x_rct[O_idx[:n_rct]],  # type: ignore
        t_rct[O_idx[:n_rct]],
        y_rct[O_idx[:n_rct]],
    )

    mask = np.logical_and(O_idx, s_all)
    rct_to_obs_ratio = s_all[O_idx].sum() / (s_all[O_idx].size - s_all[O_idx].sum())
    ys = (
        2
        * (y_rct_trimm * t_rct_trimm - y_rct_trimm * (1 - t_rct_trimm))
        * (1 - pi_s[mask])
        / pi_s[mask]
    )
    bootstrap_rct = bootstrap((ys,), np.mean, n_resamples=args.n_bootstrap, axis=0)
    std_rct = bootstrap_rct.standard_error
    var_rct = np.power(std_rct, 2) * (rct_to_obs_ratio**2)
    mean_rct = rct_to_obs_ratio * ys.mean()

    bootstrap_sa = BootstrapSensitivityAnalysis(
        sa_name=args.sa_bounds,
        inputs=x_obs_trimm,
        treatment=t_obs_trimm,
        outcome=y_obs_trimm,
        gammas=args.user_conf,
        seed=args.seed,
        binary=True,
    )
    bounds_dist = bootstrap_sa.bootstrap(num_samples=args.n_bootstrap)

    run_multiple_ate_hypothesis_test(
        mean_rct=mean_rct,
        var_rct=var_rct,
        bounds_dist=bounds_dist,
        alpha=args.alpha,
        gammas=args.user_conf,
    )

    if confound_func["para_form"] in ["piecewise", "multiple"]:
        _, conf_counts = np.unique(obs_data_pre_conf["C"], return_counts=True)  # type: ignore
        total_counts = conf_counts.sum()
        conf_marginals = conf_counts / total_counts  # p(C=0), p(C=1) ...

        treatment_cond_conf = []
        for i, marginal in enumerate(conf_marginals):
            zeta_key = f"zeta{i}"
            zeta_value = confound_func.get(zeta_key, 0.0)
            treatment_cond_conf.append(
                zeta_value * marginal
            )  # p(T=1,C=0) = p(T=1|C=0) * p(C=0) ...

        treatment_marginal = np.sum(treatment_cond_conf)  # p(T=1) = p(T=1|X) = cte

        OR_not_confounder = treatment_marginal / (1 - treatment_marginal)

        prob_max = np.max(list(confound_func.values())[1:])  # max p(T=1|C)
        prob_min = np.min(list(confound_func.values())[1:])  # min p(T=1|C)

        OR_conf_max = prob_max / (1 - prob_max)
        true_strength_max = OR_conf_max / OR_not_confounder

        OR_conf_min = (1 - prob_min) / prob_min
        true_strength_min = OR_conf_min * OR_not_confounder

        true_strength = np.max([true_strength_max, true_strength_min])

        print("True strength: ", true_strength)

    elif confound_func["para_form"] in ["adversarial", "adversarial_with_conf"]:
        t_marginals = np.mean(obs_data["T"])
        OR_not_confounder = t_marginals / (1 - t_marginals)

        true_gamma = confound_func["true_gamma"]
        marginal_treatment = np.mean(obs_data_pre_conf["T"])
        weight = (1 - marginal_treatment) / marginal_treatment
        prob_min = 1 / (1 + weight * true_gamma)
        prob_max = 1 / (1 + weight / true_gamma)

        OR_conf_max = prob_max / (1 - prob_max)
        true_strength_max = OR_conf_max / OR_not_confounder

        OR_conf_min = (1 - prob_min) / prob_min
        true_strength_min = OR_conf_min * OR_not_confounder

        true_strength = np.max([true_strength_max, true_strength_min])

    else:
        true_strength = 0

    if args.wandb == 1:
        wandb.log(
            {
                "true strength": true_strength,
            }
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # optimizer settings
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--user_conf", type=list, default=np.linspace(1.0, 9.0, 30))
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--sa_bounds", type=str, default="QB")
    parser.add_argument("--proportion_full_support", type=float, default=0.8)
    parser.add_argument("--conf_var", type=str, default="")
    parser.add_argument("--confound_func", type=str, default="{}")
    parser.add_argument("--wandb", type=int, default=0)
    parser.add_argument("--test", type=str)

    argmuments = parser.parse_args()
    run_vote_ate_test(argmuments)
