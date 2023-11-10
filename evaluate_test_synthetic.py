import argparse
from test import run_multiple_cate_hypothesis_test, run_multiple_ate_hypothesis_test

import numpy as np
from scipy.stats import bootstrap
from sklearn.ensemble import RandomForestClassifier

import wandb
from ATE.ate_bounds import BootstrapSensitivityAnalysis
from CATE.cate_bounds import MultipleCATEBoundEstimators
from CATE.utils_cate_test import compute_bootstrap_variance
from datasets import synthetic
from utils_evaluate import e_x_func
import pdb




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # optimizer settingsq
    parser.add_argument("--n_obs", type=int, default=100000)
    parser.add_argument("--n_rct", type=int, default=5000)
    parser.add_argument("--true_conf", type=float, default=4)
    parser.add_argument("--effective_conf", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--user_conf", type=list, default=np.linspace(1.01, 4.0, 25))
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--n_bootstrap", type=int, default=10)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--sa_bounds", type=str, default="QB")
    parser.add_argument("--estimate_ps", type=int, default=1)
    parser.add_argument("--num_groups", type=int, default=6)
    parser.add_argument("--test_type", type=str, default="cate")
    parser.add_argument("--sigma_y", type=float, default=np.random.uniform())

    args = parser.parse_args()
    if args.debug:
        wandb.init(mode="disabled")
    else:
        wandb.init(
            project="falsification_test_synthetic_ablation_nobs",
            entity="sml-eth",
            settings=wandb.Settings(start_method="fork"),
        )
        wandb.config.update(args)
    
    data = synthetic.Synthetic(
        args.n_obs,
        args.n_rct,
        args.true_conf,
        args.effective_conf,
        num_groups=args.num_groups,
        seed=args.seed,
        sigma_y=args.sigma_y
    )

    rho_u_y1 = np.abs(np.corrcoef(np.stack((data.u.reshape(-1), data.y1), axis=0))[0][1])
    rho_u_y0 = np.abs(np.corrcoef(np.stack((data.u.reshape(-1), data.y0), axis=0))[0][1])
    wandb.log({"rho_u_y1": rho_u_y1, "rho_u_y0": rho_u_y0})

    x_rct, t_rct, y_rct, s_rct = (
        data.rct.x,
        data.rct.t,
        data.rct.y,
        data.rct.s,
    )  # assume support RCT is included in support obs

    if args.estimate_ps:
        e_x = None
    else:
        e_x = e_x_func

    if args.test_type == "ate":
        alpha_trim = 0.001
        x = np.concatenate((data.rct.x.reshape(-1), data.x.reshape(-1))).reshape(-1, 1)
        s = np.concatenate((np.ones(data.rct.x.size), np.zeros(data.x.size)))

        #clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
        # hidden_layer_sizes=(5, 2), random_state=1)
        clf = RandomForestClassifier(max_depth=5, random_state=0)
        clf.fit(x, s)
        pi_s = clf.predict_proba(x)[:, 1]
        O_idx = pi_s > alpha_trim

        s_obs, t_obs, y_obs, x_obs = (
            data.s[O_idx[args.n_rct :]],
            data.t[O_idx[args.n_rct :]],
            data.y[O_idx[args.n_rct :]],
            data.x[O_idx[args.n_rct :]],
        )
        x_rct, t_rct, y_rct, s_rct = (
            data.rct.x[O_idx[: args.n_rct]],
            data.rct.t[O_idx[: args.n_rct]],
            data.rct.y[O_idx[: args.n_rct]],
            data.rct.s[O_idx[: args.n_rct]],
        )




        mask = np.logical_and(O_idx, s)  # \pi_S over RCT and \OO
        rct_to_obs_ratio = s[O_idx].sum() / (s[O_idx].size - s[O_idx].sum())
        ys = 2 * (y_rct * t_rct - y_rct * (1 - t_rct)) * (1 - pi_s[mask]) / pi_s[mask]
        bootstrap_rct = bootstrap((ys,), np.mean, n_resamples=args.n_bootstrap, axis=0)
        std_rct = bootstrap_rct.standard_error
        var_rct = np.power(std_rct, 2) * (rct_to_obs_ratio**2)
        mean_rct = rct_to_obs_ratio * ys.mean()
        bootstrap_sa = BootstrapSensitivityAnalysis(
            args.sa_bounds, x_obs, t_obs, y_obs, args.user_conf,e_x_func=e_x)
        bounds_dist = bootstrap_sa.bootstrap(num_samples=args.n_bootstrap)

        # run the hypothesis test
        run_multiple_ate_hypothesis_test(
            mean_rct=mean_rct,
            var_rct=var_rct,
            bounds_dist=bounds_dist,
            alpha=args.alpha,
            gammas=args.user_conf,
        )


    elif args.test_type == "cate":
        s_obs, t_obs, y_obs, x_obs = (
            data.s,
            data.t,
            data.y,
            data.x,
        )

        ate = y_rct[t_rct == 1].mean() - y_rct[t_rct == 0].mean()
        ate_variance = compute_bootstrap_variance(y_rct, t_rct, args.n_bootstrap, arm=None)

        bounds_estimator = MultipleCATEBoundEstimators(
            gammas=args.user_conf, n_bootstrap=args.n_bootstrap
        )

        bounds_estimator.fit(x_obs, t_obs, y_obs, sample_weight=False)
        run_multiple_cate_hypothesis_test(
                bounds_estimator,
                ate,
                ate_variance,
                args.alpha,
                x_rct,args.user_conf)
   
        

        



