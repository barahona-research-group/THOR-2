import datetime, json, os, glob, logging
from collections import defaultdict
import copy

import joblib
import pandas as pd
import numpy as np
import scipy

import dask.dataframe as dd

from pythor.utils.linalg import cross_correlation_mtx
from pythor.utils.stats import update_cumulative_stats
from pythor.utils.data import timeseries_dataloader, tabular_dataloader

from .constants import NUMERAI_SUNSHINE_JSON

"""
Implementation of Data Loaders 

Use Model States to store metadata of features from Numerai

1. Standard Numerai Data Loader 
    Given dataset and feature set and target columns load data from the data/era folder 
    we pass the feature/target/group names



"""


from .constants import (
    NUMERAI_SUNSHINE_TARGETS,
    NUMERAI_SUNSHINE_FEATURES_V2,
    NUMERAI_SUNSHINE_FEATURES_V4,
    NUMERAI_SUNSHINE_FEATURES_LOOKBACK,
    SMART_BETAS,
)

feature_set_mapping = {
    "v2": NUMERAI_SUNSHINE_FEATURES_V2,
    "v4.1": NUMERAI_SUNSHINE_FEATURES_V4,
}


def numerai_data_loader_tabular(
    model_states,
    **kwargs,
):
    ## Set Default feature and target columns
    dataset = model_states.get("dataset", "v4.1")
    featureset = model_states.get("feature_set", "v4.1")

    if dataset == "v4.1":
        feature_cols = feature_set_mapping.get(featureset, NUMERAI_SUNSHINE_FEATURES_V2)
    model_states["feature_cols_all"] = feature_cols

    ## Set target columns
    if not "target_cols" in model_states.keys():
        model_states["target_cols"] = NUMERAI_SUNSHINE_TARGETS
        model_states["target_cols"] = [
            "target_nomi_v4_20",
            "target_nomi_v4_60",
        ]
    model_states["era_col"] = ["era"]
    return model_states


"""

2. Time Series Numerai Data Loader 

    Given era number and model states, load feature.targets,groups and prediction files from the relevant folders


For Time Series Features
    - Use pandas to Dimensionality Reduce to 430 features


"""


@timeseries_dataloader
def numerai_data_loader_timeseries(era, model_states):
    era_str = f"{era:04d}"
    folder = model_states.get("folder", "data/era/numerai-v4.1")
    filename = f"{folder}_{era_str}.parquet"
    if model_states["prediction_layer"] < 2:
        (
            features,
            targets,
            groups,
        ) = load_numerai_timeseries_features(filename)
        data = {
            "features": features,
            "target": targets,
            "groups": groups,
        }
        ## Default we will calculate time series features for all targets
        #### as it does not take much overhead to do a multiple variable regression
        model_states["target_cols"] = list(targets.columns)
        return data
    else:
        (
            targets,
            groups,
        ) = load_numerai_timeseries_targets(filename)
        data = {
            "target": targets,
            "groups": groups,
        }
        ## Default we will calculate time series features for all targets
        #### as it does not take much overhead to do a multiple variable regression
        model_states["target_cols"] = list(targets.columns)
        return data


def load_numerai_timeseries_features(filename, format="pandas"):
    df = pd.read_parquet(filename)
    ## Get features in right order
    feature_cols = NUMERAI_SUNSHINE_FEATURES_V4
    target_cols = NUMERAI_SUNSHINE_TARGETS
    assert df[feature_cols].shape[1] == 1586
    raw_features = df[feature_cols]
    ## Split into groups
    smart_betas = raw_features.iloc[:, 1040:1181]
    group_V4_val = np.median(
        raw_features.iloc[:, :1040].values.reshape(df.shape[0], 5, 208), axis=1
    )
    group_sunshine_val = np.median(
        raw_features.iloc[:, 1181:].values.reshape(df.shape[0], 81, 5), axis=-1
    )
    group_cols = [f"Group_{i}_Median" for i in range(1, 290)]
    group_df = pd.DataFrame(
        np.concatenate([group_V4_val, group_sunshine_val], axis=1),
        columns=group_cols,
        index=df.index,
    )
    features = pd.concat([smart_betas, group_df], axis=1)
    return (
        features,
        df[target_cols],
        df["era"],
    )


def load_numerai_timeseries_targets(filename, format="pandas"):
    df = pd.read_parquet(filename)
    ## Get features in right order
    feature_cols = NUMERAI_SUNSHINE_FEATURES_V4
    target_cols = NUMERAI_SUNSHINE_TARGETS
    return (
        df[target_cols],
        df["era"],
    )


"""
Helper Functions to convert Numerai Era and Datetime 


"""

## Convert Daily Round Numbers into weekly data eras
def get_live_data_era(live_era):
    ## Round 449-453 is the week where Friday is 24 March 2023, in the dataset it is era 1056
    numerai_daily_rounds = int((live_era - 449) / 5)
    return numerai_daily_rounds + 1056


## Convert datetime into Numerai eras
def convert_datetime_to_era(sample_date):
    baseline = datetime.datetime(year=2003, month=1, day=3)
    differences = datetime.datetime.strptime(sample_date, "%Y-%m-%d") - baseline
    new_era = str(differences.days // 7 + 1)
    while len(new_era) < 4:
        new_era = "0" + new_era
    return new_era


def convert_era_to_datetime(era):
    baseline = datetime.datetime(year=2003, month=1, day=3)
    new_datetime = baseline + datetime.timedelta(days=7 * (int(era) - 1))
    return new_datetime
