from copy import copy
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

from datasets.semi_synthetic.utils import analyze_data, dataset_checks, split_dataset

CAT_COVAR_STAR = [
    "gender",
    "race",
    "birthmonth",
    "birthday",
    "birthyear",
    # "gkfreelunch",
    "g1freelunch",
    "g1tchid",
    "g1surban",
]

NUM_COVAR_STAR = []


def load_star_data(
    conf_var: str = "g1surban",
    support_var: str = "g1surban",
    analyze_dataset: bool = True,
    split_data: bool = True,
    support_feature_values: Optional[Union[list, tuple]] = None,
    proportion_full_support: float = 0.5,
    seed: int = 42,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    # Read STAR.csv into star_data DataFrame
    star_data = pd.read_csv("datasets/semi_synthetic/data/STAR.csv")

    treatment_filter = np.isfinite(star_data.g1classtype)
    outcome_filter = np.isfinite(star_data.g1tlistss + star_data.g1treadss + star_data.g1tmathss)
    outcome_and_treatment_filter = treatment_filter & outcome_filter

    cat_covar_columns = copy(CAT_COVAR_STAR)
    if conf_var != "":
        assert conf_var in cat_covar_columns
        cat_covar_columns.remove(conf_var)
    Y_columns = ["g1treadss", "g1tmathss", "g1tlistss"]

    T_all = star_data.g1classtype[outcome_and_treatment_filter].values
    filtered_indices = T_all != 3
    X_all = (
        star_data[cat_covar_columns][outcome_and_treatment_filter]
        .fillna(0)
        .values[filtered_indices]
    )

    Y_cols = star_data[Y_columns][outcome_and_treatment_filter].values[filtered_indices]

    T_all = T_all[filtered_indices]
    T_all[T_all == 2] = 0

    Y_all = np.sum(Y_cols, axis=1) / 3

    if conf_var != "":
        confounder = (
            star_data[conf_var][outcome_and_treatment_filter].fillna(0).values[filtered_indices]
        )
        if conf_var == "g1surban":
            confounder[np.logical_or(confounder == 1, confounder == 3)] = 0
            confounder[np.logical_or(confounder == 2, confounder == 4)] = 1
        if conf_var == "g1freelunch":
            confounder[confounder == 1] = 0
            confounder[confounder == 2] = 1

        data = {"Y": Y_all, "T": T_all, "C": confounder}
    else:
        data = {"Y": Y_all, "T": T_all}

    X_df = pd.DataFrame(X_all, columns=cat_covar_columns[: X_all.shape[1]])
    data.update(X_df)

    df = pd.DataFrame(data)
    if analyze_dataset and conf_var != "":
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
