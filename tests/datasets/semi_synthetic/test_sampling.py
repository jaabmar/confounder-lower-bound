import numpy as np
import pandas as pd

from test_confounding.datasets.semi_synthetic.sampling import (
    rejection_sampler,
    resample_data_with_confound_func,
    researcher_specified_function_for_confounding,
    subsample_df,
    weights_for_rejection_sampler,
)


def test_rejection_sampler():
    data = pd.DataFrame({"A": range(10)})
    weights = np.array([0.5] * 10)
    rng = np.random.default_rng(seed=42)
    M = 1

    sampled_data = rejection_sampler(data, weights, rng, M)

    assert isinstance(sampled_data, pd.DataFrame)
    assert not sampled_data.empty
    assert len(sampled_data) <= len(data)


def test_weights_for_rejection_sampler():
    data = pd.DataFrame(
        {"T": np.random.binomial(1, 0.5, 100), "C": np.random.binomial(1, 0.5, 100)}
    )
    confound_func_params = {"para_form": "adversarial_with_conf", "true_gamma": 5.0}
    weights, p_TC, pT = weights_for_rejection_sampler(data, confound_func_params)

    assert isinstance(weights, np.ndarray)
    assert isinstance(p_TC, np.ndarray)
    assert isinstance(pT, np.ndarray)
    assert len(weights) == len(data)
    assert len(p_TC) == len(data)
    assert len(pT) == len(data)


def test_researcher_specified_function_for_confounding():
    data = pd.DataFrame(
        {"T": np.random.binomial(1, 0.5, 100), "C": np.random.binomial(1, 0.5, 100)}
    )
    confound_func_params = {"para_form": "adversarial_with_conf", "true_gamma": 5.0}

    p_TC = researcher_specified_function_for_confounding(data, confound_func_params)

    assert isinstance(p_TC, np.ndarray)
    assert len(p_TC) == len(data)


def test_resample_data_with_confound_func():
    df = pd.DataFrame({"T": np.random.binomial(1, 0.5, 100), "C": np.random.randn(100)})
    confound_func_params = {"para_form": "adversarial_with_conf", "true_gamma": 5.0}
    seed = 42

    resampled_df = resample_data_with_confound_func(df, confound_func_params, seed)

    assert isinstance(resampled_df, pd.DataFrame)
    assert not resampled_df.empty
    assert len(resampled_df) <= len(df)


def test_subsample_df():
    true_gamma = 5.0
    obs_data_pre_conf = pd.DataFrame(
        {
            "T": np.random.binomial(1, 0.5, 100),
            "C": np.random.binomial(1, 0.5, 100),
            "Y": np.random.randn(100),
        }
    )
    seed = 42

    subsampled_data = subsample_df(true_gamma, obs_data_pre_conf, seed)

    assert isinstance(subsampled_data, pd.DataFrame)
    assert not subsampled_data.empty
    assert len(subsampled_data) <= len(obs_data_pre_conf)
