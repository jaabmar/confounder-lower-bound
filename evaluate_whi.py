from scipy.stats import bootstrap
import numpy as np
from ATE.ate_bounds import BootstrapSensitivityAnalysis
from CATE.cate_bounds import  MultipleCATEBoundEstimators
from test import  run_multiple_ate_hypothesis_test
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import wandb
import argparse
import pdb
from CATE.utils_cate_test import compute_bootstrap_variance
from test import run_multiple_cate_hypothesis_test, run_multiple_ate_hypothesis_test
from test import construct_ate_test_statistic, hypothesis_test
from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
 
    # optimizer settings
    parser.add_argument("--outcome", type=str, default="CHD")
    parser.add_argument("--test_type", type=str, default="cate")
    parser.add_argument("--alpha_trim", type=float, default=0.01)
    parser.add_argument("--alpha_test", type=float, default=10)

    parser.add_argument("--n_bootstrap", type=int, default=100)
    parser.add_argument("--user_conf", type=list, default=np.linspace(1.0, 1.25, 30))
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--start_time", type=int, default=10)
    parser.add_argument("--sa_bounds", type=str, default="QB")

    args = parser.parse_args()

    if args.debug:
        wandb.init(mode="disabled")
    else:
        wandb.init(
            project="falsification_test_WHI",
            entity="sml-eth",
            settings=wandb.Settings(start_method="fork"),
        )
        wandb.config.update(args)


    # prepare dataset, remove shift in time since treatment

    df_ctos = pd.read_csv('datasets/whi_processed/ctos_table.csv')
    df_ctos['TOTPTIME'].fillna(0, inplace=True)
    
    df_merged  = pd.read_csv('datasets/whi_processed/merged.csv', low_memory = False)
    #df_merged['EVENT'] = (df_merged[args.outcome + '_DY'] <= 7*365)*df_merged[args.outcome + '_E']
    df_merged['EVENT'] = df_merged[args.outcome + '_E']

    df_rct = df_merged[df_merged['OS'] == 0]
    df_obs = df_merged[(df_ctos['OS']==1) & (df_ctos['TOTPTIME'] <= args.start_time)]
    df_merged = df_merged[ (df_ctos['OS']==0) | (df_ctos['TOTPTIME']<= args.start_time)]

    covariates = ['AGE','ETHNIC_White (not of Hispanic origin)', 
    'BMI','SMOKING_Past Smoker','SMOKING_Current Smoker','EDUC_x_College graduate or Baccalaureate Degree',
    'EDUC_x_Some post-graduate or professional', 'MENO', 'PHYSFUN']

    others = ["EVENT",'HRTARM']
    df_rct = df_rct[covariates + others]
    df_obs = df_obs[covariates + others]
    df_merged_covariates = df_merged[covariates]
    df_merged = df_merged[covariates + others + ['OS']]


    # RCT classic estimate
    y1_rct = df_rct[df_rct['HRTARM'] == 1]["EVENT"].mean()
    y0_rct = df_rct[df_rct['HRTARM'] == 0]["EVENT"].mean()
    wandb.log({'ate_rct':y1_rct-y0_rct})


    if args.test_type == 'ate':
        # compute the reweighted ATE 
        x = df_merged_covariates.to_numpy()
        s = df_merged['OS'].to_numpy()

        clf_pi =  RandomForestClassifier(max_depth=15, random_state=0)
        clf_pi.fit(x, s)
        pi_s = 1-clf_pi.predict_proba(x)[:,1]
        O_idx =  np.logical_and(pi_s > args.alpha_trim, pi_s < 1- args.alpha_trim)
        df_overlap = df_merged[O_idx]
        df_overlap_obs = df_overlap[df_overlap['OS']==1] 
        df_overlap_rct = df_overlap[df_overlap['OS']==0] 

        y_rct = df_overlap_rct["EVENT"].to_numpy()
        y_obs = df_overlap_obs["EVENT"].to_numpy()

        x_rct = df_overlap_rct.iloc[:,:-3].to_numpy()
        x_obs = df_overlap_obs.iloc[:,:-3].to_numpy()
        t_obs = df_overlap_obs['HRTARM'].to_numpy()
        t_rct= df_overlap_rct['HRTARM'].to_numpy()

        mask = np.logical_and(O_idx, 1-s)  # \pi_S over RCT and \OO
        rct_to_obs_ratio = (s[O_idx].sum() / (s[O_idx].size - s[O_idx].sum()))**-1
        ys = 2 * (y_rct * t_rct - y_rct * (1 - t_rct)) * (1 - pi_s[mask]) / pi_s[mask]
        bootstrap_rct = bootstrap((ys,), np.mean, n_resamples=args.n_bootstrap, axis=0)
        std_rct = bootstrap_rct.standard_error
        var_rct = np.power(std_rct, 2) * (rct_to_obs_ratio**2)
        mean_rct = rct_to_obs_ratio * ys.mean()

        wandb.log({'mean_rct_weighted':mean_rct, 'var_rct_weighted':var_rct})

        # compute OBS sensitivity analysis

        bootstrap_sa = BootstrapSensitivityAnalysis(args.sa_bounds, x_obs, t_obs, y_obs, args.user_conf, e_x_func=None, binary=True)
        
        bounds_dist = bootstrap_sa.bootstrap(num_samples=args.n_bootstrap)
        

        # run the hypothesis test
        reject = run_multiple_ate_hypothesis_test(
            mean_rct=mean_rct,
            var_rct=var_rct,
            bounds_dist=bounds_dist,
            alpha=args.alpha_test,
            gammas=args.user_conf,
        )
    
    elif args.test_type == 'cate':
        y_rct = df_rct["EVENT"].to_numpy()
        y_obs = df_obs["EVENT"].to_numpy()
        x_rct = df_rct.iloc[:,:-2].to_numpy()
        x_obs = df_obs.iloc[:,:-2].to_numpy()
        t_obs = df_obs['HRTARM'].to_numpy()
        t_rct= df_rct['HRTARM'].to_numpy()
        ate = y_rct[t_rct == 1].mean() - y_rct[t_rct == 0].mean()
        ate_variance = compute_bootstrap_variance(y_rct, t_rct, args.n_bootstrap, arm=None)

        bounds_estimator = MultipleCATEBoundEstimators(gammas=args.user_conf, n_bootstrap=args.n_bootstrap, binary=True, 
                                                       mu=LogisticRegression())
        bounds_estimator.fit(x_obs, t_obs, y_obs, sample_weight=False)

        run_multiple_cate_hypothesis_test(
                bounds_estimator,
                ate,
                ate_variance,
                args.alpha_test,
                x_rct,args.user_conf)
        
    elif args.test_type == 'ate_dvds':
        # compute the reweighted ATE 
        x = df_merged_covariates.to_numpy()
        s = df_merged['OS'].to_numpy()

        #clf_pi =  RandomForestClassifier(max_depth=15, random_state=0)
        clf_pi = LogisticRegression()
        clf_pi.fit(x, s)
        pi_s = 1-clf_pi.predict_proba(x)[:,1]
        O_idx =  np.logical_and(pi_s > args.alpha_trim, pi_s < 1- args.alpha_trim)
        df_overlap = df_merged[O_idx]
        df_overlap_obs = df_overlap[df_overlap['OS']==1] 
        df_overlap_rct = df_overlap[df_overlap['OS']==0] 

        y_rct = df_overlap_rct["EVENT"].to_numpy()
        y_obs = df_overlap_obs["EVENT"].to_numpy()

        x_rct = df_overlap_rct.iloc[:,:-3].to_numpy()
        x_obs = df_overlap_obs.iloc[:,:-3].to_numpy()
        t_obs = df_overlap_obs['HRTARM'].to_numpy()
        t_rct= df_overlap_rct['HRTARM'].to_numpy()

        mask = np.logical_and(O_idx, 1-s)  # \pi_S over RCT and \OO
        rct_to_obs_ratio = (s[O_idx].sum() / (s[O_idx].size - s[O_idx].sum()))**-1
        ys = 2 * (y_rct * t_rct - y_rct * (1 - t_rct)) * (1 - pi_s[mask]) / pi_s[mask]
        bootstrap_rct = bootstrap((ys,), np.mean, n_resamples=args.n_bootstrap, axis=0)
        std_rct = bootstrap_rct.standard_error
        var_rct = np.power(std_rct, 2) * (rct_to_obs_ratio**2)
        mean_rct = rct_to_obs_ratio * ys.mean()

        wandb.log({'mean_rct':mean_rct, 'var_rct':var_rct})

        
        bounds_estimator = MultipleCATEBoundEstimators(gammas=args.user_conf, n_bootstrap=args.n_bootstrap, 
                                                       binary=True,  mu=LogisticRegression()) #TODO implemenet DVDS properly, uncertainty in e(x)
        bounds_estimator.fit(x_obs, t_obs, y_obs, sample_weight=False)

        dictionary_bounds_estimators = bounds_estimator.dict_bound_estimators
        critical_gamma = 0.0
        end_of_test = False

        for gamma in args.user_conf:
            ate_lb, ate_ub = dictionary_bounds_estimators[str(gamma)].compute_ate_bounds(x_obs)
            var_lb, var_ub, quantile_lb, quantile_ub = dictionary_bounds_estimators[str(gamma)].estimate_bootstrap_variances(x_obs)
            if not end_of_test:
                test_statistic = construct_ate_test_statistic(
                    mean_rct=mean_rct,
                    var_rct=var_rct,
                    mean_ub=ate_ub,
                    mean_lb=ate_lb,
                    var_ub=var_ub,
                    var_lb=var_lb,
                    gamma=gamma,
                )
                # run the hypothesis test
                reject = hypothesis_test(
                    test_statistic=test_statistic,
                    alpha=args.alpha_test,
                )
                wandb.log({ "test_statistic": test_statistic,"gamma": gamma,"reject": reject})


                if reject == 0:
                    end_of_test = True
                    wandb.log({ "gamma_effective": gamma if gamma > args.user_conf[0] else 1.0})


            else:
                reject = 0
                wandb.log({ "mean_ub": ate_ub, "mean_lb":ate_lb,"gamma": gamma,"reject": reject})

            if critical_gamma < 1 and np.sign(quantile_lb) != np.sign(quantile_ub):
                critical_gamma = gamma


        wandb.log({"critical_gamma": critical_gamma})
        