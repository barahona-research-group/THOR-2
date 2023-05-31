"""

Standard Ensemble Incremental Learners for Tabular Tasks 

    1. Gradient Boosting Ensembles
        - Gradient Boosting for L2-loss func

        
        

"""
import copy
import os, json, joblib, logging
import itertools

import pandas as pd
import numpy as np
import scipy

import torch


from pythor.constants import TABULAR_MODELS
from .snapshot import ModelSnapshotEnsemble


ARRAY_PACKAGE = np


def save_ensemble_model(model, model_type, outputpath):
    if model_type in TABULAR_MODELS["Ensemble"]:
        model.save_model(outputpath)
    return None


def load_ensemble_model(model_type, outputpath):
    if model_type == "ensemble-snapshot-tabular":
        reg = ModelSnapshotEnsemble()
    ## Assumption: any string given is considered as a path to the pickle object
    ## and anything else given is considered as the loaded python dictionary of objects
    if type(outputpath) == str:
        reg.load_model(outputpath)
    else:
        reg.load_model_parameters(outputpath)
    return reg


## Offline Batch Training using Incremental Leraning Models
def train_model_batch_ensemble(
    ml_model_name,
    ml_model_params_batch,
    tabular_dataloader_func,
    train_tabular_dataloader_state,
    validate_tabular_dataloader_state,
):

    ## Gradient Boosting
    if ml_model_name == "ensemble-snapshot-tabular":
        ml_model = ModelSnapshotEnsemble(ml_model_params_batch)
        ml_model.train(
            tabular_dataloader_func,
            train_tabular_dataloader_state,
            validate_tabular_dataloader_state,
        )

    return ml_model


## Continuous training of models
def train_model_live_ensemble(
    ml_model_name,
    ml_model_params_live,
    ml_model,
    tabular_dataloader_func,
    train_tabular_dataloader_state,
    validate_tabular_dataloader_state,
):
    ## Snapshot
    if ml_model_name == "ensemble-snapshot-tabular":
        ml_model.update(
            tabular_dataloader_func,
            train_tabular_dataloader_state,
            validate_tabular_dataloader_state,
            ml_model_params_live,
        )


## Prediction is done in numpy array, where dask arrays are casted into numpy ones
def get_model_pred_ensemble(tabular_data_era, ml_model_name, ml_model):

    matrix_loader = lambda x: ARRAY_PACKAGE.array(x)
    features = matrix_loader(tabular_data_era["features"])
    preds_names = [str(x) for x in tabular_data_era["target"].columns]

    ## Boosting Ensemble
    if ml_model_name == "ensemble-snapshot-tabular":
        predict_params = dict()
        predictions_raw = ml_model.predict(features, predict_params)
        snapshots = ml_model.num_boosters
        no_targets = int(predictions_raw.shape[1] / snapshots)
        cols = [
            f"{j}-Snapshot-{i}"
            for j in preds_names[:no_targets]
            for i in range(1, snapshots + 1)
        ]

    if predictions_raw.shape == 1:
        predictions_raw = predictions_raw.reshape(-1, 1)
    return predictions_raw, cols
