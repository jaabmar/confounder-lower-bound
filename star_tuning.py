import argparse
import ast
import warnings
from test import run_cate_hypothesis_test

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn_quantile import RandomForestQuantileRegressor

import wandb
from CATE.cate_bounds import CATEBoundsEstimator
from CATE.utils_cate_test import compute_bootstrap_variance
from datasets.semi_synthetic.sampling import resample_data_with_confound_func
from datasets.semi_synthetic.star import CAT_COVAR_STAR, load_star_data
from hyperparameters import HP_CON
from utils_evaluate import calibrate_confound_strength

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # optimizer settings
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--user_conf", type=float)
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--proportion_full_support", type=float, default=0.8)
    parser.add_argument("--conf_var", type=str, default="")
    parser.add_argument("--confound_func", type=str, default="{}")
    parser.add_argument("--wandb", type=int, default=0)

    args = parser.parse_args()
    confound_func = ast.literal_eval(args.confound_func)[0]

    support_var = "g1surban"
    support_feature_values = [1, 2]

    obs_data_pre_conf, rct_data = load_star_data(
        conf_var=args.conf_var,
        support_var=support_var,
        split_data=True,
        support_feature_values=support_feature_values,
        proportion_full_support=args.proportion_full_support,
        seed=args.seed,
    )

    obs_data = resample_data_with_confound_func(
        obs_data_pre_conf,
        confound_func_params=confound_func,
        seed=args.seed,
    )

    categorical_covariates = CAT_COVAR_STAR

    # Define transformer for encoding and normalization
    transformer = ColumnTransformer(
        [
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                [col for col in categorical_covariates if col != args.conf_var],
            ),
        ]
    )
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
    ate_variance = compute_bootstrap_variance(y_rct, t_rct, args.n_bootstrap, arm=None)

    hyperparameters = HP_CON

    strength_calibrated = False
    calibrated_strength = None
    true_strength = None

    for hp_config in hyperparameters:
        run = None
        if args.wandb == 1:
            run = wandb.init(
                project="star_tuning",
                entity="sml-eth",
                settings=wandb.Settings(start_method="fork"),
            )
            wandb.config.update(args)

        tau = args.user_conf / (1 + args.user_conf)

        mu_estimator = hp_config["mu_estimator_type"]
        mu_estimator_kwargs = hp_config["mu_estimator_kwargs"]
        bounds_estimator = hp_config["bounds_estimator_type"]
        bounds_estimator_kwargs = hp_config["bounds_estimator_kwargs"]

        mu_model = mu_estimator(random_state=args.seed, n_jobs=-2, **mu_estimator_kwargs)
        bounds_model = bounds_estimator(
            random_state=args.seed, n_jobs=-2, **bounds_estimator_kwargs
        )

        quantile_upper_model = RandomForestQuantileRegressor(
            **mu_estimator_kwargs,
            random_state=args.seed,
            n_jobs=-2,
            q=tau,
        )

        quantile_lower_model = RandomForestQuantileRegressor(
            **mu_estimator_kwargs,
            random_state=args.seed,
            n_jobs=-2,
            q=1 - tau,
        )

        cate_bounds_estimator = CATEBoundsEstimator(
            mu=mu_model,
            bounds=bounds_model,
            quantile_upper=quantile_upper_model,
            quantile_lower=quantile_lower_model,
            user_conf=args.user_conf,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            binary=False,
        )

        cate_bounds_estimator.fit(
            x_obs,  # type: ignore
            t_obs,
            y_obs,
            sample_weight=True,
        )

        reject = run_cate_hypothesis_test(
            ate=float(ate),
            ate_variance=ate_variance,
            bounds_estimator=cate_bounds_estimator,
            x_rct=x_rct,  # type: ignore
            alpha=args.alpha,
            store_wandb=args.wandb,
        )

        if not strength_calibrated:
            calibrated_strength = calibrate_confound_strength(
                x_obs=x_obs,
                t_obs=obs_data["T"],
                confounder_col=confounder,
                wrt_confounder=False,
            )

            strength_calibrated = True

        if args.wandb == 1:
            wandb.log(
                {
                    "hp_config": hp_config,
                    "reject": reject,
                    "calibrated strength": calibrated_strength,
                }
            )
            if run is not None:
                run.finish()
