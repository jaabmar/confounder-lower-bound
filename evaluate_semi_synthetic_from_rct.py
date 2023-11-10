import argparse
import ast
import warnings
from test import run_cate_hypothesis_test

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

import wandb
from CATE.cate_bounds import CATEBoundsEstimator
from CATE.utils_cate_test import compute_bootstrap_variance
from datasets.semi_synthetic.fetch_hillstrom import (
    CAT_COVAR_HILLSTROM,
    NUM_COVAR_HILLSTROM,
    load_fetch_hillstrom_data,
)
from datasets.semi_synthetic.sampling import resample_data_with_confound_func
from datasets.semi_synthetic.star import CAT_COVAR_STAR, NUM_COVAR_STAR, load_star_data
from datasets.semi_synthetic.vote import CAT_COVAR_VOTE, NUM_COVAR_VOTE, load_vote_data
from utils_evaluate import calibrate_confound_strength

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # optimizer settings
    parser.add_argument("--dataset", type=str, default="hillstrom")
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--user_conf", type=float, default=1.2)
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--proportion_full_support", type=float, default=0.8)
    parser.add_argument("--conf_var", type=str, default="")
    parser.add_argument("--confound_func", type=str, default="{}")
    parser.add_argument("--target_col", type=str, default="spend")
    parser.add_argument("--wandb", type=int, default=0)
    args = parser.parse_args()
    confound_func = ast.literal_eval(args.confound_func)[0]
    if args.wandb == 1:
        wandb.init(
            project="hillstrom",
            entity="sml-eth",
            settings=wandb.Settings(start_method="fork"),
        )
        wandb.config.update(args)

    if args.dataset == "STAR":
        support_var = "g1surban"
        conf_var = "g1freelunch" if args.conf_var == "" else args.conf_var
        support_feature_values = [1, 2]
        obs_data_pre_conf, rct_data = load_star_data(
            conf_var=conf_var,
            support_var=support_var,
            split_data=True,
            support_feature_values=support_feature_values,
            proportion_full_support=args.proportion_full_support,
            seed=args.seed,
        )
        if not isinstance(rct_data, pd.DataFrame):
            raise TypeError("rct_data is not a pd.DataFrame")
        confound_func_params = (
            {
                "para_form": "adversarial",
                "quantile": 0.5,
                "zeta0": 0.30,
                "zeta1": 0.80,
            }
            if not confound_func
            else confound_func
        )
        obs_data = resample_data_with_confound_func(
            obs_data_pre_conf,
            confound_func_params=confound_func_params,
            seed=args.seed,
        )

        numeric_covariates = NUM_COVAR_STAR
        categorical_covariates = CAT_COVAR_STAR

    elif args.dataset == "VOTE":
        conf_var = "voted_before" if args.conf_var == "" else args.conf_var
        support_var = "hh_size"
        support_feature_values = [1, 2, 3]
        obs_data_pre_conf, rct_data = load_vote_data(
            conf_var=conf_var,
            support_var=support_var,
            split_data=True,
            support_feature_values=support_feature_values,
            proportion_full_support=args.proportion_full_support,
            seed=args.seed,
        )
        confound_func_params = (
            {
                "para_form": "multiple",
                "zeta0": 0.05,
                "zeta1": 0.50,
                "zeta2": 0.20,
                "zeta3": 0.05,
                "zeta4": 0.1,
                "zeta5": 0.05,
            }
            if not confound_func
            else confound_func
        )
        obs_data = resample_data_with_confound_func(
            obs_data_pre_conf, confound_func_params=confound_func_params, seed=args.seed, M=1
        )

        numeric_covariates = NUM_COVAR_VOTE
        categorical_covariates = CAT_COVAR_VOTE

    else:
        assert args.dataset == "hillstrom"
        conf_var = args.conf_var
        support_var = "zip_code"
        support_feature_values = [0.0, 1.0]
        obs_data_pre_conf, rct_data = load_fetch_hillstrom_data(
            conf_var=conf_var,
            support_var=support_var,
            split_data=True,
            support_feature_values=support_feature_values,
            proportion_full_support=args.proportion_full_support,
            seed=args.seed,
            target_col=args.target_col,
        )
        confound_func_params = confound_func

        obs_data = resample_data_with_confound_func(
            obs_data_pre_conf,
            confound_func_params=confound_func_params,
            seed=args.seed,
        )

        numeric_covariates = NUM_COVAR_HILLSTROM
        categorical_covariates = CAT_COVAR_HILLSTROM

    # Define transformer for encoding and normalization
    transformer = ColumnTransformer(
        [
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                [col for col in categorical_covariates if col != conf_var],
            ),
            (
                "normalizer",
                MinMaxScaler(),
                [col for col in numeric_covariates if col != conf_var],
            ),
        ]
    )
    if conf_var in numeric_covariates:
        confounder_scalar = MinMaxScaler()
        confounder = confounder_scalar.fit_transform(obs_data["C"].values.reshape(-1, 1))  # type: ignore
    else:
        confounder = obs_data["C"]

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
    print("ATE RCT: ", ate)
    print("ATE OBS (unweighted)", y_obs[t_obs == 1].mean() - y_obs[t_obs == 0].mean())
    ate_variance = compute_bootstrap_variance(y_rct, t_rct, args.n_bootstrap, arm=None)
    print("ATE variance: ", ate_variance)
    print()

    cate_bounds_estimator = CATEBoundsEstimator(
        user_conf=args.user_conf,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        binary=True if args.dataset in ["VOTE"] else False,
    )

    cate_bounds_estimator.fit(
        x_obs,  # type: ignore
        t_obs,
        y_obs,
        sample_weight=False if args.dataset in ["VOTE"] else True,
    )

    reject = run_cate_hypothesis_test(
        ate=float(ate),
        ate_variance=ate_variance,
        bounds_estimator=cate_bounds_estimator,
        x_rct=x_rct,  # type: ignore
        alpha=args.alpha,
        store_wandb=args.wandb,
    )

    calibrated_strength = calibrate_confound_strength(
        x_obs=x_obs,
        t_obs=obs_data["T"],
        confounder_col=confounder,
        wrt_confounder=False,
    )

    print("Confounder strength: ", calibrated_strength)

    if confound_func["para_form"] in ["piecewise", "multiple"]:
        _, conf_counts = np.unique(obs_data["C"], return_counts=True)
        total_counts = conf_counts.sum()
        conf_marginals = conf_counts / total_counts  # p(C=0), p(C=1) ...

        treatment_cond_conf = []
        for i, marginal in enumerate(conf_marginals):
            zeta_key = f"zeta{i}"
            zeta_value = confound_func_params.get(zeta_key, 0.0)
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

    else:
        true_strength = 0

    if args.wandb == 1:
        wandb.log(
            {
                "reject": reject,
                "calibrated strength": calibrated_strength,
                "true strength": true_strength,
            }
        )
