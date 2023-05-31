"""
Incremental Learning of Temporal Tabular Datasets 


Input: Temporal Tabular Data

Output: Temporal Tabular Data 

Score: Ranking Correlations 


"""

import copy
import os, json, joblib, logging

import pandas as pd
import numpy as np
import scipy

import torch


from pythor.utils.linalg import tail_corr, normalise_preds
from pythor.utils.stats import update_cumulative_stats

from .common import (
    load_tabular_model,
    save_tabular_model,
    train_model_batch_tabular,
    train_model_live_tabular,
    get_model_pred_tabular,
)

from pythor.base.backtest import TabularBase


"""
Incremental Learning: 

Train models regularly as create walk forward predictions. Monitor model performances in continuous manner. 

 
Data Formats 

Tabular Features: Index using the unique ID by provider 
Time Series Features: Index using the era number (integers) starting with 1, 

The DataLoader will data for each era 
    - Tabular Features: pd.DataFrame(N,M) where N is number of observations in era, M is number of features
    - Time Series Features pd.DataFrame(1,M) where M is the number of features with a single index indicated by era number 



BaseClass for Loading Data for Tabular Datasets


1. Temporal Tabular Env 

Run Standard Incremental Learners for Tabular Datasets 

2. Ensemble Tabular Env 

Run Ensemble Incremental Learners for Tabular Datasets 


"""


class TemporalTabular(TabularBase):
    def __init__(
        self,
        ml_model_name="lightgbm-regression-tabular",
        ml_model_params_batch=dict(),
        ml_model_params_live=dict(),
        model_path_stem="lightgbm-regression-tabular_0",
        prediction_path_stem="lightgbm-regression-tabular",
        **kwargs,
    ):

        ## Machine Learning Models
        self.ml_model_params_batch = ml_model_params_batch
        self.ml_model_params_live = ml_model_params_live
        self.ml_model_name = ml_model_name

        self.model_path_stem = model_path_stem
        self.prediction_path_stem = prediction_path_stem
        super().__init__(**kwargs)

    """
    Methods to be used in backtest loop
    
    """

    def train_model_batch(self):
        (
            train_tabular_dataloader_state,
            validate_tabular_dataloader_state,
        ) = self.get_data_splits_tabular()
        ml_model = train_model_batch_tabular(
            self.ml_model_name,
            self.ml_model_params_batch,
            self.tabular_dataloader_func,
            train_tabular_dataloader_state,
            validate_tabular_dataloader_state,
        )
        self.ml_model = ml_model

        if not self.ml_model_name.startswith("equal_weighted"):
            self.tabular_dataloader_states[
                "feature_cols"
            ] = train_tabular_dataloader_state["feature_cols"]

    def train_model_live(self):
        (
            train_tabular_dataloader_state,
            validate_tabular_dataloader_state,
        ) = self.get_data_splits_tabular()
        train_model_live_tabular(
            self.ml_model_name,
            self.ml_model_params_live,
            self.ml_model,
            self.tabular_dataloader_func,
            train_tabular_dataloader_state,
            validate_tabular_dataloader_state,
        )

    def get_model_predictions(
        self,
        tabular_data_era,
        timeseries_data_era,
    ):
        predictions_raw, cols = get_model_pred_tabular(
            tabular_data_era, self.ml_model_name, self.ml_model
        )
        preds_era_rank = normalise_preds(predictions_raw, axis=0, cutoff=0)
        predictions_era = pd.DataFrame(
            preds_era_rank,
            columns=cols,
            index=tabular_data_era["target"].index,
        )
        ## Add Model Names to prefix
        model_prefix = self.model_path_stem.split("/")[-1] + "-"
        return predictions_era.add_prefix(model_prefix)
