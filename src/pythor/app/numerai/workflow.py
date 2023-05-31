import os, glob, json, gc, joblib, logging
import time

import pandas as pd
import numpy as np

import dask.dataframe as dd

from requests.exceptions import ConnectionError
from numerapi import NumerAPI

from .util import (
    get_live_data_era,
)
from pythor.utils.data import check_file_status
from pythor.base.workflow import iterative_tabular_benchmark_base


"""
Update Data Folders given the current live round 


1. Download Raw Data Files from Numerai 
    - Merge MetaModel (or anything with lags) with targets, Merge Benchmark Model (or anything without lag) as features
2. Process Raw Data 
    - Convert Targets into integer bins 
    - Convert Features into integer bins
    - Convert Era into format for integer eras starting with 1 
    - Impute Missing Data 

3. Perform Health Check on the data 
    - Get current era number
    - Calculated the expected last resolved data
    - Check the resolved eras data are not missing significnatly 
    - 
"""


def process_numerai_rawdata(i, df, version_name="numerai-v4.1"):
    ## Set Era as Integer
    try:
        df.drop("data_type", axis=1, inplace=True)
    except KeyError:
        pass
    df["era"] = df["era"].astype(int)
    feature_cols = [i for i in df.columns if i.startswith("feature")]
    target_cols = [i for i in df.columns if i.startswith("target")]
    ## Impute Features to (-2 to 2)
    feature_median = 2
    df[feature_cols] = df[feature_cols].fillna(feature_median) - feature_median
    ## Impute Targets to (0 to 1)
    for t in target_cols:
        df[t] = df[t].fillna(0.5)
    df.to_parquet(f"data/era/{version_name}_{i:04d}.parquet")


def numerai_data_download(version_name="numerai-v4.1"):

    napi = NumerAPI()
    train_file = f"data/{version_name}_train.parquet"
    validation_file = f"data/{version_name}_validation.parquet"
    batch_file = f"data/{version_name}_all.parquet"
    live_file = f"data/{version_name}_live.parquet"
    metamodel_file = f"data/numerai_meta_model.parquet"

    days_since_updated = check_file_status(validation_file, 7)
    weekend_rounds = napi.get_current_round() % 5 == 4
    logging.info(
        f"Days Since data update {days_since_updated} Is Weekend Round {weekend_rounds}"
    )

    if days_since_updated > 2 and weekend_rounds:
        if os.path.exists(validation_file):
            os.remove(validation_file)
        napi.download_dataset("v4.1/validation_int8.parquet", validation_file)
        if os.path.exists(metamodel_file):
            os.remove(metamodel_file)
        napi.download_dataset("v4.1/meta_model.parquet", metamodel_file)
        downloaded_rawdata = True
    else:
        downloaded_rawdata = False

    if downloaded_rawdata:
        validate_data = pd.read_parquet(validation_file)
        validate_data["era"] = validate_data["era"].astype(int)
        metamodel = pd.read_parquet(metamodel_file)
        metamodel["era"] = metamodel["era"].astype(int)
        validate_data["target_metamodel"] = metamodel["numerai_meta_model"]
        validate_data.to_parquet(validation_file)
        logging.info("Merged MetaModel with Validation Data")
        ## ToDo: Merge benchmark model with Features,
        # ## adding prefix to the columns names :feature_{benchmark_modelname}

    ## Download Live Data
    live_era = get_live_data_era(napi.get_current_round())
    days_since_updated = check_file_status(live_file, 1)
    if days_since_updated > 0.01:
        if os.path.exists(live_file):
            os.remove(live_file)
        napi.download_dataset("v4.1/live_int8.parquet", live_file)
        logging.info("Downloaded Numerai Live Data")

    ## Replace the latest era in the folder with live data
    live_data = pd.read_parquet(live_file)
    ## Assign the implied era calculated to replace the default X from Numerai
    live_data["era"] = live_era
    process_numerai_rawdata(live_era, live_data)

    if downloaded_rawdata:
        recent_data = validate_data[validate_data["era"] >= live_era - 20]
        for i, df in recent_data.groupby("era"):
            newdf = process_numerai_rawdata(i, df)
        logging.info("Updated Numerai Data folder")

    return live_era


"""

Run Live Predictions for Numerai Tasks 

Assumption: All the data files are correctly loaded by the above 


Run Iterative Modelling Pipeline with live submission settings 

No matter in weekday or weekend, we will overwrite the files in 
    - Data/era folder which have the given features/targets/... 
    (As we will always rewrtite the last 20 eras during weekly data download, it is not a problem if the live era data get replaced since it will be fixed later)
    - {ensemble_model}/predictions/ folder which has the individual files of predictions from layers
    (As we will also update the predictions in the most recent 10 eras or so during incremental learning updates)
    - {ensemble_model}/feaures/ folder which has the combined predictions from layers 


"""


def robust_upload_predictions(client, df, model_id):
    for max_retry in range(2):
        try:
            client.upload_predictions(
                df=df,
                model_id=model_id,
            )
            break
        except:
            time.sleep(5)


def update_live_submission(ensemble_folder, live_era, live_submission):
    ## Load Hyper-Parameters of a pipline
    pipeline_path = f"{ensemble_folder}/pipeline.joblib"
    pipeline = joblib.load(pipeline_path)

    pipeline["base_incremental_learning_params"]["live_era"] = live_era

    ## Run Iterative Modelling pipeline
    prediction_folders = iterative_tabular_benchmark_base(
        pipeline["tabular_dataloader_func"],
        pipeline["timeseries_dataloader_func"],
        base_tabular_dataloader_states=pipeline["base_tabular_dataloader_states"],
        base_timeseries_dataloader_states=pipeline["base_timeseries_dataloader_states"],
        base_incremental_learning_params=pipeline["base_incremental_learning_params"],
        learners_hyperparameters=pipeline["learners_hyperparameters"],
        learners_tabular_dataloader_states=pipeline[
            "learners_tabular_dataloader_states"
        ],
        learners_incremental_learning_params=pipeline[
            "learners_incremental_learning_params"
        ],
        ensemble_folder=ensemble_folder,
        debug=True,
        live_submission=live_submission,
    )
    ## Get Predictions
    combined_predictions = list()
    for prediction_path_stem in prediction_folders:
        prediction_file = f"{prediction_path_stem}_preds_{live_era:04d}.parquet"
        combined_predictions.append(pd.read_parquet(prediction_file))

    return pd.concat(combined_predictions, axis=1)


def numerai_live_submission(
    api_keys_path="secrets.json",
    live_submission=True,  ## Set this to False to update the models
    numerai_model_name="thomas51",
    ensemble_folder1="Base",
    ensemble_folder2="Hedge",
    hedge_weight=0.2,
    model1_col=0,
    model2_col=0,
):

    ## Download Data from Numerai and check quality
    live_era = numerai_data_download()

    logging.info(f"Live Era {live_era}")

    ## Run For Pipeline (Base Model and Hedge Model)
    combined_predictions1 = update_live_submission(
        ensemble_folder1,
        live_era,
        live_submission,
    ).dropna()
    if ensemble_folder2 != ensemble_folder1:
        combined_predictions2 = update_live_submission(
            ensemble_folder2,
            live_era,
            live_submission,
        ).dropna()
    else:
        combined_predictions2 = combined_predictions1

    ## Get API secrets
    try:
        if api_keys_path is None:
            api_keys_path = "secrets.json"
        with open(api_keys_path, "r") as f:
            API_KEYS = json.load(f)
        have_keys = True
    except:
        have_keys = False
        logging.info("API key not found, will not upload predictions to server")

    if have_keys and live_submission:
        ## Retrive/Process Live Submission Files
        ## Assume get first columns from prediction file as prediction, index is by id
        preds1 = combined_predictions1.iloc[:, model1_col]
        preds2 = combined_predictions2.iloc[:, model2_col]
        ## Combine predictions
        # numerai_submission = preds1 * (1 - hedge_weight) + preds2 * hedge_weight
        numerai_submission = preds1
        numerai_submission = pd.DataFrame(numerai_submission.rank(pct=True))
        numerai_submission.reset_index(inplace=True)
        numerai_submission.columns = [
            "id",
            "prediction",
        ]
        ## Call the submission API
        client = NumerAPI(API_KEYS["public_id"], API_KEYS["secret_key"])
        model_id = client.get_models()[numerai_model_name]
        robust_upload_predictions(client, numerai_submission, model_id)
        logging.info(f"Model {numerai_model_name} {model_id} submitted")

        numerai_submission.to_csv("temp_numerai_submission.csv")
