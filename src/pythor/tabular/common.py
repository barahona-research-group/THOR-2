"""

Integrated API for training a single Incremental Learner from PyTorch and LightGBM 

    - save_ML_model
    - load_ML_model 
    - train_model_batch_single
    - train_model_live_single
    - get_single_model_pred


Standard Incremental Learners for Tabular Tasks 

    1. Gradient Boosting Decision Trees 
        - LightGBM 
    2. Deep Learning Models 
        - MLP (PyTorch)
        - Sparse MLP (PyTorch)
    3. Gradient Boosting Ensembles
        - Gradient Boosting for L2-loss func
        - Model Snapshots 
    4. Iterative Boosting Ensembles 
        - 

"""
import copy
import os, json, joblib, logging

import pandas as pd
import numpy as np
import scipy

import torch


from pythor.utils.linalg import matrix_project

from .trees.lgbst import LightGBMIncrementalRegressor

from .trees.xgbst import XGBoostIncrementalRegressor
from .trees.cgbst import CatBoostIncrementalRegressor
from .pytorch.base import PyTorchIncrementalTabularModel
from .pytorch.nets import MLP, SparseMLP


from pythor.constants import TABULAR_MODELS


ARRAY_PACKAGE = np


def save_tabular_model(model, model_type, outputpath):
    if model_type in TABULAR_MODELS["XGBoost"]:
        model.save_model(outputpath)
    if model_type in TABULAR_MODELS["CatBoost"]:
        model.save_model(outputpath)
    if model_type in TABULAR_MODELS["PyTorch"]:
        model.save_model(outputpath)
    ## To be deprectaed soon
    if model_type in TABULAR_MODELS["LightGBM"]:
        model.save_model(outputpath)
    return None


def load_tabular_model(model_type, outputpath):
    if model_type in TABULAR_MODELS["XGBoost"]:
        reg = XGBoostIncrementalRegressor(dict())
    if model_type in TABULAR_MODELS["CatBoost"]:
        reg = CatBoostIncrementalRegressor(dict())
    if model_type in TABULAR_MODELS["PyTorch"]:
        if model_type == "torch-MLP-tabular":
            reg = PyTorchIncrementalTabularModel(MLP, dict())
        if model_type == "torch-SparseMLP-tabular":
            reg = PyTorchIncrementalTabularModel(SparseMLP, dict())
    ## To be deprecated soon
    if model_type in TABULAR_MODELS["LightGBM"]:
        reg = LightGBMIncrementalRegressor(dict())
    ## Assumption: any string given is considered as a path to the pickle object
    ## and anything else given is considered as the loaded python dictionary of objects
    if type(outputpath) == str:
        reg.load_model(outputpath)
    else:
        reg.load_model_parameters(outputpath)
    return reg


## Offline Batch Training using Incremental Leraning Models
def train_model_batch_tabular(
    ml_model_name,
    ml_model_params_batch,
    tabular_dataloader_func,
    train_tabular_dataloader_state,
    validate_tabular_dataloader_state,
):
    if ml_model_name.startswith("equal_weighted"):
        ml_model = None
        return ml_model

    ## XGBoost
    if ml_model_name.startswith("xgboost"):
        if ml_model_name == "xgboost-regression-tabular":
            ml_model = XGBoostIncrementalRegressor(ml_model_params_batch)

    ## CatBoost
    if ml_model_name.startswith("catboost"):
        if ml_model_name == "catboost-regression-tabular":
            ml_model = CatBoostIncrementalRegressor(ml_model_params_batch)

    ## MLP
    if ml_model_name.startswith("torch"):
        if ml_model_name == "torch-MLP-tabular":
            ml_model = PyTorchIncrementalTabularModel(MLP, config=ml_model_params_batch)
        if ml_model_name == "torch-SparseMLP-tabular":
            ml_model = PyTorchIncrementalTabularModel(
                SparseMLP, config=ml_model_params_batch
            )

    ## LightGBM (to be deprecated)
    if ml_model_name.startswith("lightgbm"):
        if ml_model_name == "lightgbm-regression-tabular":
            ml_model = LightGBMIncrementalRegressor(ml_model_params_batch)

    ml_model.train(
        tabular_dataloader_func,
        train_tabular_dataloader_state,
        validate_tabular_dataloader_state,
    )

    return ml_model


## Continuous training of models
def train_model_live_tabular(
    ml_model_name,
    ml_model_params_live,
    ml_model,
    tabular_dataloader_func,
    train_tabular_dataloader_state,
    validate_tabular_dataloader_state,
):
    ## ToDo: For CatBoost/XGBoost/PyTorch models, how to continue training without reloading data

    ## Assuming all models have the same signature for updating
    ml_model.update(
        tabular_dataloader_func,
        train_tabular_dataloader_state,
        validate_tabular_dataloader_state,
        ml_model_params_live,
    )


## We will cast any input arrays into numpy arrays for prediction
def get_model_pred_tabular(tabular_data_era, ml_model_name, ml_model):

    matrix_loader = lambda x: ARRAY_PACKAGE.array(x)
    features = matrix_loader(tabular_data_era["features"])
    cols = list()

    ## GBDT
    if (
        ml_model_name.startswith("catboost")
        or ml_model_name.startswith("xgboost")
        or ml_model_name.startswith("lightgbm")
    ):

        ## Model Snapshots Start with 0 and end with different number of trees
        use_snapshots = ml_model.config.get("use_snapshots", False)
        no_snapshots = ml_model.config.get("no_snapshots", 10)
        if use_snapshots:
            tree_ends = [i / no_snapshots for i in range(1, no_snapshots + 1)]
            predictions_raw = ARRAY_PACKAGE.zeros((features.shape[0], len(tree_ends)))
            for t in range(len(tree_ends)):
                predict_params = {
                    "start_iteration_ratio": 0,
                    "end_iteration_ratio": tree_ends[t],
                }
                predictions_raw[:, t] = ml_model.predict(features, predict_params)
                cols.append(f"snapshot-{t}")
        else:
            predictions_raw = ARRAY_PACKAGE.zeros((features.shape[0], 1))
            predict_params = {
                "start_iteration_ratio": 0,
                "end_iteration_ratio": 0,
            }
            tree_preds = ml_model.predict(features, predict_params).astype(float)
            predictions_raw[:, 0] = tree_preds
            cols.append(f"best")
            if False:
                predictions_raw[:, 1] = matrix_project(
                    tree_preds, features.astype(float), proportion=1
                )
                cols.append(f"best-FN")

    ## PyTorch
    if ml_model_name.startswith("torch"):
        predict_params = {}
        predictions_raw = ml_model.predict(features, predict_params)
        ## Multiple Targets are supported by PyTorch models
        torch_cols = [f"target-{i}" for i in range(1, predictions_raw.shape[1] + 1)]
        cols.extend(torch_cols)
        ## Fix Target Names if possible
        target_cols = tabular_data_era["target"].columns
        if len(target_cols) == len(cols):
            cols = target_cols

    ## Equal Weighted
    if ml_model_name.startswith("equal_weighted"):
        predictions_raw = ARRAY_PACKAGE.mean(features, axis=1)
        cols = ["average"]

    ## Fix Predictions arrays if we only have a single output only
    if predictions_raw.shape == 1:
        predictions_raw = predictions_raw.reshape(-1, 1)

    return predictions_raw, cols
