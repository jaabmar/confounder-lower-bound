from copy import copy
from typing import Optional, Union

import pandas as pd

from datasets.semi_synthetic.utils import analyze_data, dataset_checks, split_dataset

CAT_COVAR_VOTE = [
    "gender",
    "voted_before",
    "hh_size",
]
NUM_COVAR_VOTE = ["age", "neighbors_voted_before"]


def load_vote_data(
    conf_var: Optional[str] = "voted_before",
    support_var: str = "hh_size",
    analyze_dataset: bool = True,
    split_data: bool = True,
    support_feature_values: Optional[Union[list, tuple]] = None,
    proportion_full_support: float = 0.5,
    seed: int = 42,
):
    vote_data = pd.read_csv("datasets/semi_synthetic/data/preprocessed_VOTE.csv")
    cat_covar_columns = copy(CAT_COVAR_VOTE)
    num_covar_columns = copy(NUM_COVAR_VOTE)
    all_covar_columns = cat_covar_columns + num_covar_columns
    if conf_var is not None:
        if conf_var in all_covar_columns:
            all_covar_columns.remove(conf_var)
        if conf_var in num_covar_columns:
            num_covar_columns.remove(conf_var)
        elif conf_var in cat_covar_columns:
            cat_covar_columns.remove(conf_var)

    X_all = vote_data[all_covar_columns].fillna(-1).values
    if conf_var is not None:
        confounder = vote_data[conf_var].values
    else:
        confounder = None

    Y_all = vote_data["voted"].values
    T_all = vote_data["treatment"].values

    if confounder is not None:
        data = {
            "Y": Y_all,
            "T": T_all,
            "C": confounder,
        }
    else:
        data = {
            "Y": Y_all,
            "T": T_all,
        }

    X_df = pd.DataFrame(X_all, columns=all_covar_columns)
    data.update(X_df)  # type: ignore
    df = pd.DataFrame(data)
    if analyze_dataset:
        if confounder is not None:
            dataset_checks(df)
        analyze_data(df, fit_xgboost=True)
        print()

    if split_data:
        if conf_var == support_var:
            raise ValueError("conf_var and support_var cannot have the same value")
        assert support_var in cat_covar_columns
        return split_dataset(
            df=df,
            support_feature=support_var,
            support_feature_values=support_feature_values,
            proportion_full_support=proportion_full_support,
            seed=seed,
        )

    return df
