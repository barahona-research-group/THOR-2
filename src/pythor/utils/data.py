from functools import wraps, partial

import copy, os, json, logging, joblib, glob, time
from joblib import Parallel, delayed


import pandas as pd
import numpy as np
import scipy

import dask.dataframe as dd

from pythor.constants import TABULAR_FORMATS, TIMESERIES_FORMATS
from .linalg import ridge_solve, cross_correlation_mtx


"""
Check File Status

"""


def check_file_status(filename, default=7):
    if os.path.exists(filename):
        days_since_updated = (time.time() - os.path.getmtime(filename)) / (3600 * 24)
    else:
        days_since_updated = default
    return days_since_updated


"""
 
Data Formats stored on disk (parquet file formats)


- Tabular Features: 
    Index using the unique ID by provider
    Both Features and Targets should be in integer bins, zero-centered. We load clearned features for training directly.

    features: DataFrame(N,M) where N is number of observations in era, M is number of features. 
    target: DataFrame(N,K) where N is number of observations in era, K is number of targets
    groups: DataFrame(N,1) which is the group label 


- Time Series Features:

    In each era, we calculate feature performances as follows
        pd.DataFrame(K,X) where X is the number of features and K is the number of targets 
    We then add two new columns, era and target_names for data grouping 



Creating Data Loaders for iterative/incremental learning data loaders 


Assumption 1: We only have less than 10k eras, as we name the files in the format 0001,0002, up to 9999. If you have more 10k data, then 
you should use a database 

Assumption 2: For Time Series DataLoader, we only calculate feature importance with respect to a single target, given by "target_name" in the config

Assumption 3: For Tabular Dataloader, we can specify "target_names" attribute, which be used by tabular ML models during data loading


Implementations 

TimeSeries Data Loader:
    Data is loaded by era using pandas and then calculate feature performances using the torch optimised tensor calcs
    We only need to use data from previous layers to calculate factor timing portfolios 
    However, for Dynamic Feature Neutralisation, we would load the original features and the model predictions separately
    Suitable for factor timing models where calculations are performed on each era independently 

Tabular Data Loader: 
    Data is loaded with pandas for small datasets or dask for big tabular datasets with row sampling and feature selection (by a given func which specifiy the rules)
    We will use predictions from the previous layer, along with the original given features to train tabular models
    Data Loading is postponed until data sampling during CV
    Suitable for LightGBM and XGBoost, also other distributed ML packages. 

PyTorch Data Loader: 
    A custome PyTorch dataset for converting dask dataframe to batches for training. Using Lazy loading means the tensors are created only when a sample batch is called
    ToDo: Use a cache to save the tensors if size not big
        


"""


def timeseries_dataloader(func):
    @wraps(func)
    def dataloader_iterated(
        era,
        model_state,
    ):
        output = dict()
        pred_files = model_state.get("previous_layer_files", list())
        calc_feature_performances = model_state.get("calc_feature_performance", True)

        if not calc_feature_performances:
            # logging.info("Skip Factor Timing Calcs for Tabular Model Ensembles")
            output = dict()
            return output

        if len(pred_files) > 0:
            predictions_df = list()
            for file_stem in pred_files:
                filename = f"{file_stem}_preds_{era:04d}.parquet"
                if os.path.exists(filename):
                    predictions_df.append(pd.read_parquet(filename))
                    loaded_predictions = True
            if len(predictions_df) > 0:
                predictions = pd.concat(predictions_df, axis=1)
                original_data = func(era, model_state)
            else:
                ## We do not have predictions for this era, so we skip the calculation
                output = dict()
                output["feature_performances"] = pd.DataFrame()
                return output
        else:
            if calc_feature_performances:
                original_data = func(era, model_state)
                loaded_predictions = False
            else:
                output = dict()
                return output

        ## Determine whether we are feature timing or portfolio timing
        DFN = model_state.get("run_DFN", False)
        if not loaded_predictions or DFN:
            features = original_data["features"]
            model_state["positive_weights_only"] = False
            if DFN:
                output["model_predictions"] = predictions
        else:
            ## Perform Factor Timing on previous predictions so use positive weights only
            features = predictions
            model_state["positive_weights_only"] = True

        output["features"] = features
        output["target"] = original_data["target"]
        output["groups"] = original_data["groups"]

        calc_feature_performances = model_state.get("calc_feature_performance", True)
        if calc_feature_performances:
            target = original_data["target"]
            ## Feature(Model) Performances is a dataframe (1,no_features)
            calc_method = model_state.get("calc_method", "ridge")
            ridge_alphas = model_state.get("ridge_alpha", [0.001])
            if calc_method == "ridge":
                coefs = (
                    ridge_solve(
                        np.array(features).astype(float),
                        np.array(target).astype(float),
                        alphas=ridge_alphas,
                    )
                    .mean(axis=0)
                    .transpose()
                )
            else:
                coefs = cross_correlation_mtx(
                    np.array(features).astype(float), np.array(target).astype(float)
                ).transpose()

            coefs_df = pd.DataFrame(coefs, columns=features.columns)
            coefs_df["era"] = era
            coefs_df["target_names"] = target.columns
            output["feature_performances"] = coefs_df
            return output
        else:
            return output

    return dataloader_iterated


"""

Data Loaders for tabular datasets 


Strategy 1: Load files by pandas and joblib.Parallel 

Strategy 2: Load from Dask Dataframes By Era 
    Pros: Always within memory
    Cons: 10 times slower than loading a single file by pandas, unless dask delay is used 

Strategy 1 is the default way to load data, code for strategt 2 is not tested.


"""


def resample_pandas_dataframe_single(
    era,
    folder,
    batch_size,
    random_state=0,
    columns=None,
):
    filename = f"{folder}_{era:04d}.parquet"
    df = pd.read_parquet(filename, columns=columns)
    if batch_size > 0:
        sampled_df = df.sample(batch_size, random_state=random_state)
    else:
        sampled_df = df
    return sampled_df


def resample_pandas_dataframe(
    model_state,
    folder,
    columns=None,
):
    start_era = model_state.get("start_era", 1)
    end_era = model_state.get("end_era", 101)
    subsample_freq = model_state.get("subsample_freq", 1)
    batch_size = model_state.get("batch_size", -1)
    random_state = model_state.get("data_random_state", 0)
    ## For shifting the start during various era sampling
    data_shift = random_state % subsample_freq
    pandas_load = partial(
        resample_pandas_dataframe_single,
        folder=folder,
        batch_size=batch_size,
        random_state=random_state,
        columns=columns,
    )
    ## These are optimal settings for a machine with 20 CPU cores and a single GPU
    if end_era - start_era > 10:
        max_jobs = 10
    else:
        max_jobs = 1
    ## Do not include end_era
    adjusted_start = min(start_era + data_shift, end_era - 1)
    df_collections = Parallel(n_jobs=max_jobs)(
        delayed(pandas_load)(i) for i in range(adjusted_start, end_era, subsample_freq)
    )
    return pd.concat(df_collections, axis=0)


def resample_features(model_states):
    ## model_states['feature_cols'] has the list of features to be sampled from
    feature_random_state = model_states.get("feature_random_state", 0)
    feature_subsample_ratio = model_states.get("feature_subsample_ratio", 1)
    if "feature_cols" not in model_states.keys():
        logging.info("Perform Global Feature Subsampling")
        randomiser = np.random.RandomState(feature_random_state)
        no_features_given = len(model_states["feature_cols_all"])
        no_features_sampled = int(feature_subsample_ratio * no_features_given)
        sampled_index = randomiser.permutation(no_features_given)[
            :no_features_sampled
        ].astype(int)
        model_states["feature_cols"] = list(
            np.array(model_states["feature_cols_all"])[sampled_index]
        )
    return model_states


def tabular_dataloader_pandas(tabular_load_func, model_state):

    folder = model_state.get("folder", "data/era/numerai-v4.1")
    previous_layer_files = model_state.get("previous_layer_files", list())
    use_given_features = model_state.get("use_given_features", False)

    ## Update the features/targets to be loaded for each model in different layers
    model_state = tabular_load_func(model_state)
    ## Perform Feature Subsampling
    model_state = resample_features(model_state)

    all_features = list()
    ## Sample once on the group sizes to obtain the index
    groups_df = resample_pandas_dataframe(
        model_state,
        folder,
        model_state["era_col"],
    )
    required_index = groups_df.index

    logging.info(f"Index Sampling done we have {required_index.shape} records")

    ## Disable Sampling for the rest of data loading process
    model_state["batch_size"] = -1
    ## Get Targets
    targets_df = resample_pandas_dataframe(
        model_state,
        folder,
        model_state["target_cols"],
    ).loc[required_index]

    if len(previous_layer_files) < 1 or use_given_features:
        features_df = resample_pandas_dataframe(
            model_state,
            folder,
            model_state["feature_cols"],
        ).loc[required_index]
        all_features.append(features_df)
        logging.info(f"Use the given features")

    ## Get Previous Layer files, which is merged into a single one now
    ## ToDo: Do we use predictions from all previous layers or just the latest one?
    for f in previous_layer_files:
        ## Target Neutralisation by subtracting previous factor timing/metamodel/benchmark model predictions
        target_projection_strength = model_state.get("target_projection_strength", 0)
        # logging.info(f"Use previous layers predictions {f}")
        generated_df = resample_pandas_dataframe(
            model_state,
            f"{f}_preds",
            None,
        ).loc[required_index]
        if target_projection_strength > 0:
            if generated_df.shape[1] == targets_df.shape[1]:
                targets_df = targets_df - target_projection_strength * generated_df
            else:
                average_preds = generated_df.mean(axis=1).values.reshape(-1, 1)
                targets_df = targets_df - target_projection_strength * average_preds

        ## For Boosting do not reuse predictions from previous layers
        if target_projection_strength <= 0:
            all_features.append(generated_df)

    ## ToDo: Create new targets based on difference of the actual targets and predictions from previous layers
    output = dict()
    output["features"] = pd.concat(all_features, axis=1)
    output["target"] = targets_df
    output["groups"] = groups_df
    return output


def tabular_dataloader(tabular_load_func, model_state):
    output = tabular_dataloader_pandas(tabular_load_func, model_state)
    return output


"""
Create/Update Data Loader for Tabular/Time Series features 

Tabular:
    Save the features/target metadata and other info for loading tabular data 

Time Series:
    Create/Update a data object given its path and the live_era 
    If the dataloader does not exists, then create a new one, otherwise update the TS features up to live era and embargo



"""


def update_dataloader_tabular(
    dataloader_path_stem,
    dataloader_func=None,
    dataloader_states=dict(),
    backtest_env_config=dict(),
):
    filepath = f"{dataloader_path_stem}.dataloader"
    dataloader_states["features_names"] = backtest_env_config["hyperparameters"][
        "features_names"
    ]
    dataloader_states["target_names"] = backtest_env_config["hyperparameters"][
        "target_names"
    ]
    ## Save the Update Data Object
    data_loader_object = {
        "dataloader_states": dataloader_states,
        "dataloader_func": dataloader_func,
    }
    joblib.dump(data_loader_object, filepath)


def update_dataloader_timeseries(
    dataloader_path_stem,
    live_era,
    embargo_size=5,
    dataloader_func=None,
    dataloader_states=dict(),
):

    filepath = f"{dataloader_path_stem}.dataloader"
    if os.path.exists(filepath):
        data_loader_object = joblib.load(filepath)
        dataloader_func = data_loader_object["dataloader_func"]
        dataloader_states = data_loader_object["dataloader_states"]
        ## Quick Solution, simply check the first key in TIMESERIES_FORMATS
        try:
            quick_key = TIMESERIES_FORMATS[0]
            if quick_key in dataloader_states["data_buffer"]:
                quick_ts_check = dataloader_states["data_buffer"][quick_key]
                ref_era = quick_ts_check["era"].max()
                logging.info(f"Load Data Loader end at {ref_era}")
        except:
            ref_era = 0
    else:
        ## ToDo: Update Data Buffer Start era for feature loaders starting at later levels
        dataloader_states["data_buffer"] = dict()
        ref_era = 0

    ## Load Live Data that is not in buffer
    live_data_buffer = dict()
    for key in TIMESERIES_FORMATS:
        live_data_buffer[key] = list()
    for era in range(
        ref_era + 1,
        live_era - embargo_size,
    ):
        ## For the given features, we have data since Era 1, However for predictions, we do not have data at the satrt
        try:
            data_era = dataloader_func(era, dataloader_states)
            for key in TIMESERIES_FORMATS:
                if key in data_era.keys():
                    live_data_buffer[key].append(data_era[key])
            # logging.info(f"Data Loader {dataloader_path_stem} Updating for Era {era}")
        except FileNotFoundError:
            pass
            # logging.info(f"Data Loader Skip Updating for Era {era}")

    ## Concatenate Each Data Era into DataFrames
    for key in TIMESERIES_FORMATS:
        if len(live_data_buffer[key]) > 0:
            live_data_buffer[key] = pd.concat(live_data_buffer[key], axis=0)
        else:
            live_data_buffer[key] = pd.DataFrame()
        ## Join Data from Buffer and Live Data
        current_data_buffer = dataloader_states["data_buffer"].get(key, None)
        if current_data_buffer is None:
            dataloader_states["data_buffer"][key] = live_data_buffer[key]
        else:
            df = pd.concat([current_data_buffer, live_data_buffer[key]], axis=0)
            dataloader_states["data_buffer"][key] = df.drop_duplicates(
                subset=["era", "target_names"]
            )
            logging.info(
                f"Merged New Data with Data Loader and removed duplicates for {key}"
            )

    dataloader_states["feature_names"] = dict()
    for key in TIMESERIES_FORMATS:
        dataloader_states["feature_names"][key] = list(
            dataloader_states["data_buffer"][key].columns
        )

    ## Save the Update Data Object
    data_loader_object = {
        "dataloader_states": copy.deepcopy(dataloader_states),
        "dataloader_func": dataloader_func,
    }
    joblib.dump(data_loader_object, filepath)
