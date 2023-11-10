import numpy as np
from scipy.stats import norm
import wandb
<<<<<<< HEAD
from CATE.cate_bounds import MultipleCATEBoundEstimators


=======
from CATE.cate_bounds import MultipleCATEBoundEstimators, _BaseBoundsEstimator
import pdb

>>>>>>> 1177b35e2f1510c613324413f19c5fd8728b96fb
def construct_ate_test_statistic(
    mean_rct: np.ndarray,
    mean_ub: float,
    mean_lb: float,
    var_rct: float,
    var_ub: float,
    var_lb: float,
    gamma: float,
) -> float:
    test_statistic_plus = (mean_ub - mean_rct) / np.sqrt(var_rct + var_ub)
    test_statistic_min = (mean_rct - mean_lb) / np.sqrt(var_rct + var_lb)
    wandb.log(
        {
            "gamma": gamma,
            "mean_rct": mean_rct,
            "var_rct": var_rct,
            "mean_ub": mean_ub,
            "var_ub": var_ub,
            "mean_lb": mean_lb,
            "var_lb": var_lb,
            "test_statistic_min": test_statistic_min,
            "test_statistic_plus": test_statistic_plus,
        }
    )
    return min(test_statistic_min, test_statistic_plus)


def construct_cate_test_statistic(
    ate: float,
    ate_variance: float,
    ate_ub: float,
    ate_lb: float,
    ate_variance_ub: float,
    ate_variance_lb: float,
    gamma: float,
) -> float:
    test_statistic_plus = (ate_ub - ate) / np.sqrt(
        ate_variance + ate_variance_ub + 2 * np.sqrt(ate_variance * ate_variance_ub)
    )
    test_statistic_min = (ate - ate_lb) / np.sqrt(
        ate_variance + ate_variance_lb + 2 * np.sqrt(ate_variance * ate_variance_lb)
    )
    wandb.log(
        {
            "mean_rct": ate,
            "var_rct": ate_variance,
            "mean_ub": ate_ub,
<<<<<<< HEAD
            "var_ub": ate_variance_ub,
            "mean_lb": ate_lb,
            "var_lb": ate_variance_lb,
=======
            "variance_ub": ate_variance_ub,
            "mean_lb": ate_lb,
            "variance_lb": ate_variance_lb,
>>>>>>> 1177b35e2f1510c613324413f19c5fd8728b96fb
            "test_statistic_min": test_statistic_min,
            "test_statistic_plus": test_statistic_plus,
            "gamma": gamma,
        }
    )
    return min(test_statistic_min, test_statistic_plus)


<<<<<<< HEAD
=======



>>>>>>> 1177b35e2f1510c613324413f19c5fd8728b96fb
def run_multiple_ate_hypothesis_test(
    mean_rct: np.ndarray,
    var_rct: float,
    bounds_dist: dict,
    alpha: float,
    gammas: float,
):
    critical_gamma = 0.0
    end_of_test = False
    for gamma in gammas:
        if not end_of_test:
            # compute mean for upper and lower bounds, no bootstrap
            mean_lb = np.mean(bounds_dist[str(gamma)][0])
            mean_ub = np.mean(bounds_dist[str(gamma)][1])

            var_lb = np.var(bounds_dist[str(gamma)][0])
            var_ub = np.var(bounds_dist[str(gamma)][1])

<<<<<<< HEAD
            test_statistic = construct_ate_test_statistic(
                mean_rct, mean_ub, mean_lb, var_rct, var_ub, var_lb, gamma
            )
            reject = hypothesis_test(test_statistic=test_statistic, alpha=alpha)
            if reject == 0:
                end_of_test = True
                wandb.log({"gamma_effective": gamma})

        else:
            reject = 0

        if critical_gamma < 1 and np.sign(
            np.quantile(bounds_dist[str(gamma)][0], alpha / 200)
        ) != np.sign(np.quantile(bounds_dist[str(gamma)][1], 1 - alpha / 200)):
            critical_gamma = gamma

        wandb.log({"test_statistic": test_statistic, "gamma": gamma, "reject": reject})

    wandb.log({"critical_gamma": critical_gamma})
    return
=======
            test_statistic = construct_ate_test_statistic(mean_rct, mean_ub, mean_lb, var_rct, var_ub, var_lb, gamma)
            reject = hypothesis_test(test_statistic=test_statistic, alpha=alpha)

            wandb.log({"test_statistic": test_statistic, "gamma": gamma, "reject": reject})

            if reject == 0:
                end_of_test = True
                wandb.log({ "gamma_effective":gamma if gamma > gammas[0] else 1.0})

        else:
            reject = 0
            mean_lb = np.mean(bounds_dist[str(gamma)][0])
            mean_ub = np.mean(bounds_dist[str(gamma)][1])

            var_lb = np.var(bounds_dist[str(gamma)][0])
            var_ub = np.var(bounds_dist[str(gamma)][1])
            wandb.log({"mean_lb": mean_lb,"mean_ub":mean_ub, "var_lb":var_lb, "var_ub":var_ub, "gamma": gamma, "reject": reject})
        
        quantile_lb = np.min(bounds_dist[str(gamma)][0])
        quantile_ub = np.max(bounds_dist[str(gamma)][1])
        if critical_gamma < 1 and np.sign(quantile_lb) != np.sign(quantile_ub):

            critical_gamma = gamma

    
    wandb.log({"critical_gamma": critical_gamma})
    return 
>>>>>>> 1177b35e2f1510c613324413f19c5fd8728b96fb


def run_multiple_cate_hypothesis_test(
    bounds_estimator: MultipleCATEBoundEstimators,
    ate: float,
    ate_variance: float,
    alpha: float,
    x_rct: np.ndarray,
    user_conf: list,
):
    dictionary_bounds_estimators = bounds_estimator.dict_bound_estimators
    critical_gamma = 0.0
    end_of_test = False

    for gamma in user_conf:
        lower_bounds, upper_bounds = dictionary_bounds_estimators[str(gamma)].predict_bounds(x_rct)
        if not end_of_test:
<<<<<<< HEAD
            ate_lb, ate_ub = float(np.mean(lower_bounds)), float(np.mean(upper_bounds))
            var_lb, var_ub = dictionary_bounds_estimators[str(gamma)].estimate_bootstrap_variances(
                x_rct
            )
=======
            var_lb, var_ub, quantile_lb, quantile_ub = dictionary_bounds_estimators[str(gamma)].estimate_bootstrap_variances(x_rct)
>>>>>>> 1177b35e2f1510c613324413f19c5fd8728b96fb
            test_statistic = construct_cate_test_statistic(
                ate=ate,
                ate_variance=ate_variance,
                ate_ub=ate_ub,
                ate_lb=ate_lb,
                ate_variance_ub=var_ub,
                ate_variance_lb=var_lb,
                gamma=gamma,
            )
            # run the hypothesis test
            reject = hypothesis_test(
                test_statistic=test_statistic,
                alpha=alpha,
            )
            wandb.log({ "test_statistic": test_statistic,"gamma": gamma,"reject": reject})


            if reject == 0:
                end_of_test = True
<<<<<<< HEAD
                wandb.log({"gamma_effective": gamma})
=======
                wandb.log({ "gamma_effective": gamma if gamma > user_conf[0] else 1.0})

>>>>>>> 1177b35e2f1510c613324413f19c5fd8728b96fb

        else:
            reject = 0
            wandb.log({ "mean_ub": ate_ub, "mean_lb":ate_lb,"gamma": gamma,"reject": reject})

<<<<<<< HEAD
        if critical_gamma < 1 and np.sign(np.quantile(lower_bounds, alpha / 200)) != np.sign(
            np.quantile(upper_bounds, 1 - alpha / 200)
        ):
=======
        if critical_gamma < 1 and np.sign(quantile_lb) != np.sign(quantile_ub):
>>>>>>> 1177b35e2f1510c613324413f19c5fd8728b96fb
            critical_gamma = gamma
            wandb.log({"critical_gamma": critical_gamma})

<<<<<<< HEAD
        wandb.log({"test_statistic": test_statistic, "gamma": gamma, "reject": reject})
=======
>>>>>>> 1177b35e2f1510c613324413f19c5fd8728b96fb

    return


def hypothesis_test(test_statistic: float, alpha: float) -> int:
    std_dist = norm(loc=0, scale=1)
    alpha = alpha / (2 * 100)
    z_test = -std_dist.isf(alpha)
    if test_statistic < z_test:
        print(f"Reject null hypothesis: {test_statistic}<{z_test}")
        return 1
    else:
        print(f"Accept null hypothesis: {test_statistic}>{z_test}")
        return 0
