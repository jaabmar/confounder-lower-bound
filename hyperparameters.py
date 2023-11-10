from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

HP_CON = [
    {
        "mu_estimator_type": RandomForestRegressor,
        "mu_estimator_kwargs": {
            "n_estimators": 200,
            "max_depth": 6,
            "min_samples_leaf": 0.01,
        },
        "bounds_estimator_type": RandomForestRegressor,
        "bounds_estimator_kwargs": {
            "n_estimators": 200,
            "max_depth": 6,
            "min_samples_leaf": 0.01,
        },
    },
    {
        "mu_estimator_type": RandomForestRegressor,
        "mu_estimator_kwargs": {
            "n_estimators": 150,
            "max_depth": 8,
            "min_samples_leaf": 0.02,
        },
        "bounds_estimator_type": RandomForestRegressor,
        "bounds_estimator_kwargs": {
            "n_estimators": 150,
            "max_depth": 8,
            "min_samples_leaf": 0.02,
        },
    },
    {
        "mu_estimator_type": RandomForestRegressor,
        "mu_estimator_kwargs": {
            "n_estimators": 200,
            "max_depth": 8,
            "min_samples_leaf": 0.02,
        },
        "bounds_estimator_type": RandomForestRegressor,
        "bounds_estimator_kwargs": {
            "n_estimators": 200,
            "max_depth": 8,
            "min_samples_leaf": 0.02,
        },
    },
    {
        "mu_estimator_type": RandomForestRegressor,
        "mu_estimator_kwargs": {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_leaf": 1,
        },
        "bounds_estimator_type": RandomForestRegressor,
        "bounds_estimator_kwargs": {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_leaf": 1,
        },
    },
    {
        "mu_estimator_type": RandomForestRegressor,
        "mu_estimator_kwargs": {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_leaf": 0.01,
        },
        "bounds_estimator_type": RandomForestRegressor,
        "bounds_estimator_kwargs": {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_leaf": 0.01,
        },
    },
    {
        "mu_estimator_type": RandomForestRegressor,
        "mu_estimator_kwargs": {
            "n_estimators": 150,
            "max_depth": None,
            "min_samples_leaf": 1,
        },
        "bounds_estimator_type": RandomForestRegressor,
        "bounds_estimator_kwargs": {
            "n_estimators": 150,
            "max_depth": None,
            "min_samples_leaf": 1,
        },
    },
    {
        "mu_estimator_type": RandomForestRegressor,
        "mu_estimator_kwargs": {
            "n_estimators": 200,
            "max_depth": None,
            "min_samples_leaf": 1,
        },
        "bounds_estimator_type": RandomForestRegressor,
        "bounds_estimator_kwargs": {
            "n_estimators": 200,
            "max_depth": None,
            "min_samples_leaf": 1,
        },
    },
    {
        "mu_estimator_type": RandomForestRegressor,
        "mu_estimator_kwargs": {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_leaf": 1,
        },
        "bounds_estimator_type": RandomForestRegressor,
        "bounds_estimator_kwargs": {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_leaf": 1,
        },
    },
]


HP_BIN = [
    {
        "mu_estimator_type": RandomForestClassifier,
        "mu_estimator_kwargs": {
            "n_estimators": 200,
            "max_depth": 6,
            "min_samples_leaf": 0.01,
        },
        "bounds_estimator_type": RandomForestRegressor,
        "bounds_estimator_kwargs": {
            "n_estimators": 200,
            "max_depth": 6,
            "min_samples_leaf": 0.01,
        },
    },
    {
        "mu_estimator_type": RandomForestClassifier,
        "mu_estimator_kwargs": {
            "n_estimators": 150,
            "max_depth": 8,
            "min_samples_leaf": 0.02,
        },
        "bounds_estimator_type": RandomForestRegressor,
        "bounds_estimator_kwargs": {
            "n_estimators": 150,
            "max_depth": 8,
            "min_samples_leaf": 0.02,
        },
    },
    {
        "mu_estimator_type": RandomForestClassifier,
        "mu_estimator_kwargs": {
            "n_estimators": 200,
            "max_depth": 8,
            "min_samples_leaf": 0.02,
        },
        "bounds_estimator_type": RandomForestRegressor,
        "bounds_estimator_kwargs": {
            "n_estimators": 200,
            "max_depth": 8,
            "min_samples_leaf": 0.02,
        },
    },
    {
        "mu_estimator_type": RandomForestClassifier,
        "mu_estimator_kwargs": {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_leaf": 1,
        },
        "bounds_estimator_type": RandomForestRegressor,
        "bounds_estimator_kwargs": {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_leaf": 1,
        },
    },
    {
        "mu_estimator_type": RandomForestClassifier,
        "mu_estimator_kwargs": {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_leaf": 0.01,
        },
        "bounds_estimator_type": RandomForestRegressor,
        "bounds_estimator_kwargs": {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_leaf": 0.01,
        },
    },
    {
        "mu_estimator_type": XGBClassifier,
        "mu_estimator_kwargs": {
            "n_estimators": 100,
            "max_depth": None,
        },
        "bounds_estimator_type": XGBRegressor,
        "bounds_estimator_kwargs": {
            "n_estimators": 100,
            "max_depth": None,
        },
    },
    {
        "mu_estimator_type": XGBClassifier,
        "mu_estimator_kwargs": {
            "n_estimators": 200,
            "max_depth": None,
        },
        "bounds_estimator_type": XGBRegressor,
        "bounds_estimator_kwargs": {
            "n_estimators": 200,
            "max_depth": None,
        },
    },
    {
        "mu_estimator_type": XGBClassifier,
        "mu_estimator_kwargs": {
            "n_estimators": 200,
            "max_depth": 6,
        },
        "bounds_estimator_type": XGBRegressor,
        "bounds_estimator_kwargs": {
            "n_estimators": 200,
            "max_depth": 6,
        },
    },
    {
        "mu_estimator_type": XGBClassifier,
        "mu_estimator_kwargs": {},
        "bounds_estimator_type": XGBRegressor,
        "bounds_estimator_kwargs": {},
    },
]
