import argparse
import ast
import warnings
from test import run_multiple_cate_hypothesis_test

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

import wandb
from CATE.cate_bounds import MultipleCATEBoundEstimators
from CATE.utils_cate_test import compute_bootstrap_variance
from datasets.semi_synthetic.fetch_hillstrom import (
    CAT_COVAR_HILLSTROM,
    NUM_COVAR_HILLSTROM,
    load_fetch_hillstrom_data,
)
from datasets.semi_synthetic.sampling import (
    resample_data_with_confound_func,
    subsample_df,
)

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)


def run_hillstrom_cate_test(args):
    confound_func = ast.literal_eval(args.confound_func)
    if args.wandb == 1:
        wandb.init(
            project="hillstrom_strong_small_sample",
            entity="sml-eth",
            settings=wandb.Settings(start_method="fork"),
        )
        wandb.config.update(args)

    support_var = "zip_code"
    support_feature_values = [0.0, 1.0]

    obs_data_pre_conf, rct_data = load_fetch_hillstrom_data(
        conf_var=args.conf_var,
        support_var=support_var,
        split_data=True,
        support_feature_values=support_feature_values,
        proportion_full_support=args.proportion_full_support,
        seed=52,
        target_col=args.target_col,
    )
    if (
        confound_func["para_form"] == "adversarial_with_conf"
        and len(np.unique(obs_data_pre_conf["C"])) == 2
    ):
        print("Weak conf")
        true_gamma = confound_func["true_gamma"]
        obs_data_pre_conf = subsample_df(
            true_gamma=true_gamma, obs_data_pre_conf=obs_data_pre_conf, seed=args.seed
        )

    obs_data = resample_data_with_confound_func(
        obs_data_pre_conf,
        confound_func_params=confound_func,
        seed=args.seed,
    )

    obs_data = obs_data.sample(frac=0.3, random_state=args.seed)

    numeric_covariates = NUM_COVAR_HILLSTROM
    categorical_covariates = CAT_COVAR_HILLSTROM

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

    ate = y_rct[t_rct == 1].mean() - y_rct[t_rct == 0].mean()
    ate_variance = compute_bootstrap_variance(y_rct, t_rct, args.n_bootstrap, arm=None)

    multiple_cate_bounds_estimators = MultipleCATEBoundEstimators(
        gammas=args.user_conf,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        binary=False if args.target_col == "spend" else True,
        cv=1,
    )

    multiple_cate_bounds_estimators.fit(
        x_obs,
        t_obs,
        y_obs,
        sample_weight=False,
    )
    run_multiple_cate_hypothesis_test(
        multiple_cate_bounds_estimators, ate, ate_variance, args.alpha, x_rct, args.user_conf
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
    parser.add_argument("--user_conf", type=list, default=np.linspace(1.00, 5.0, 20))
    # parser.add_argument("--max_user_conf", type=float)
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--proportion_full_support", type=float, default=0.8)
    parser.add_argument("--conf_var", type=str, default="")
    parser.add_argument("--confound_func", type=str, default="{}")
    parser.add_argument("--target_col", type=str, default="spend")
    parser.add_argument("--wandb", type=int, default=0)
    parser.add_argument("--test", type=str)

    argmuments = parser.parse_args()
    run_hillstrom_cate_test(argmuments)
