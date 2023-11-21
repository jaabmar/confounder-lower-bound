"""
This file contains helper functions for analysis of results
"""
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import f_oneway, pearsonr
from sklearn.model_selection import train_test_split


def odds_ratio(Y: np.ndarray, C: np.ndarray):
    assert C.shape == (len(Y),)
    assert set(Y) == set([0, 1])
    assert set(C) == set([0, 1])

    Y1C1 = np.count_nonzero(Y[C == 1] == 1)
    Y0C1 = np.count_nonzero(Y[C == 1] == 0)
    Y1C0 = np.count_nonzero(Y[C == 0] == 1)
    Y0C0 = np.count_nonzero(Y[C == 0] == 0)

    # check that the denominators are not equal to 0
    if Y0C1 == 0 or Y0C0 == 0 or (Y1C0 / Y0C0) == 0:
        return np.nan

    return (Y1C1 / Y0C1) / (Y1C0 / Y0C0)


def dataset_checks(data: pd.DataFrame):
    Y = data["Y"].to_numpy()
    C = data["C"].to_numpy()
    T = data["T"].to_numpy()

    # Checks
    print("==== DATASET CHECKS ====")
    print("Num observations = ", len(Y))

    if len(np.unique(C)) == 2:
        print("P(C=1) = ", np.mean(C))
    print("P(T=1) = ", np.mean(T))

    if len(np.unique(Y)) == 2:  # binary Y
        if len(np.unique(C)) == 2:  # binary C
            oratio = odds_ratio(Y, C)
            print("OR(Y, C) = ", oratio)
    else:  # continuous Y
        if len(np.unique(C)) == 2:  # binary C
            pearson, p_value = pearsonr(C, Y)
            print("Corr(C, Y) = ", pearson)
            print("P-value = ", p_value)
        else:  # categorical C
            grouped_Y = [Y[C == category] for category in np.unique(C)]
            f_statistic, p_value = f_oneway(*grouped_Y)
            print("F-statistic(C, Y) = ", f_statistic)
            print("P-value: = ", p_value)

    ate = np.mean(Y[T == 1]) - np.mean(Y[T == 0])
    print("RCT_ATE =", ate)
    print("Mean(Y) = ", np.mean(Y))


def analyze_data(df: pd.DataFrame, fit_xgboost: bool = False, return_top_k_features: Optional[int] = None):
    columns = df.columns
    T_column = "T"
    other_columns = [col_name for col_name in columns if col_name != "Y" and col_name != T_column]
    Y = df["Y"]
    if not fit_xgboost:
        for col_name in other_columns:
            X = df[col_name]

            if len(np.unique(Y)) == 2:  # binary Y
                if len(np.unique(X)) == 2:  # binary X
                    oratio = odds_ratio(X, Y)
                    print(f"Comparing {col_name} with Y: OR(X, Y) =", oratio)

            else:  # continuous Y
                if len(np.unique(X)) == 2:  # binary X
                    pearson, p_value = pearsonr(X, Y)
                    print(f"Comparing {col_name} with Y: Corr(X, Y) =", pearson, "P-value =", p_value)

                else:  # categorical X
                    grouped_Y = [Y[X == category] for category in np.unique(X)]
                    f_statistic, p_value = f_oneway(*grouped_Y)
                    print(
                        f"Comparing {col_name} with Y: F-statistic(X, Y) =",
                        f_statistic,
                        "P-value =",
                        p_value,
                    )

    else:
        x_train = df.loc[df[T_column] == 1, other_columns]  # Training data (T=1)
        y_train = df.loc[df[T_column] == 1, "Y"]  # Target variable when T=1

        xgb_model = xgb.XGBRegressor()
        xgb_model.fit(x_train, y_train)
        feature_importances = xgb_model.feature_importances_

        if return_top_k_features is not None:
            top_k_indices = np.argsort(feature_importances)[::-1][:return_top_k_features]
            top_k_features = x_train.columns[top_k_indices]
            return list(top_k_features)

        print()
        print("Feature importance for each feature:")
        for feature, importance in zip(x_train.columns, feature_importances):
            print(feature, "=", importance)


def to_categorical(X):
    assert isinstance(X, type(np.ones((1,))))
    return pd.get_dummies(pd.DataFrame(X.astype(str)), dummy_na=True).values.astype(int)


def split_dataset(
    df: pd.DataFrame,
    support_feature: str = "g1surban",
    support_feature_values: Optional[List] = None,
    proportion_full_support: float = 0.5,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if support_feature_values is None:
        support_feature_values = [1, 2]

    # Perform train-test split with probability q
    full_support_df, reduced_support_df = train_test_split(df, test_size=1 - proportion_full_support, random_state=seed)

    # Keep only the points with specific values of support_feature_values in reduced_support_df
    final_reduced_support_df = reduced_support_df[reduced_support_df[support_feature].isin(support_feature_values)]
    excluded_datapoints = reduced_support_df[~reduced_support_df[support_feature].isin(support_feature_values)]

    final_full_support_df = pd.concat([full_support_df, excluded_datapoints])

    return final_full_support_df.reset_index(drop=True), final_reduced_support_df.reset_index(drop=True)
