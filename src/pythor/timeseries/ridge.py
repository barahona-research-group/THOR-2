"""
TimeSeries Benchmark Workflow 

1. Run Clustering on Multi-Variate Time-Series 

2. Run ML Models 
    A. Feature Enginering and then Tabular Models 
    B. Use Deep Learning Models for Sequences 

3. Score Predictions on Validation and Test Data 

"""


import json, gc
import numpy as np
import pandas as pd

import torch

from .features import (
    RandomReLUBase,
    RandomFourierBase,
    SignatureTransformerBase,
    StatisticalBase,
)


from pythor.utils.linalg import ridge_solve
from pythor.constants import FACTOR_TIMING_MODELS

RIDGE_MODELS = FACTOR_TIMING_MODELS["RIDGE_MODELS"]

"""
Tabular Models for Multi-variate Time Series Prediction Tasks 

1, Ridge Base 
2. Time Series Ridge 


"""


class PyTorchRidgeBase:
    def __init__(self, alphas, add_constant=True, **args):
        self.alphas = alphas
        self.add_constant = add_constant
        self.device = torch.cuda.current_device()

    def _add_constant(self, X_train):
        X = torch.concatenate(
            [torch.ones((X_train.shape[0], 1), device=self.device), X_train], axis=1
        )
        return X

    def train(self, X_train, y_train):
        if self.add_constant:
            X = self._add_constant(X_train).float()
        else:
            X = X_train.astype(float)
        Y = torch.tensor(y_train, device=self.device).float().to(self.device)
        self.betas = ridge_solve(X, Y, self.alphas).to(self.device)

    def predict(self, X_test):
        if self.add_constant:
            X = self._add_constant(X_test).float().to(self.device)
        else:
            X = X_test.float().to(self.device)
        pred = (X @ self.betas).mean(axis=1)
        return pred


class TimeSeriesRidge:
    def __init__(
        self,
        ml_model_name="TimeSeriesRidge-Ensemble",
        data_embargo=6,
        train_size=200,
        lookback=50,
        feature_complexity=1,
        alphas=[
            1.0,
        ],
        seed=0,
        **kwargs,
    ):

        self.data_embargo = data_embargo
        self.lookback = lookback
        ## Machine Learning Parameters
        self.ml_model_name = ml_model_name
        self.alphas = alphas
        self.feature_complexity = feature_complexity
        self.seed = seed

    ## Train a new model at each era
    def train(self, TS_hist):

        ## Process Time Series
        self.timeseries_raw = np.array(TS_hist)
        self.train_size = TS_hist.shape[0] - self.data_embargo

        ## Feature Engineering on the whole TS Slice
        if self.ml_model_name == "TimeSeriesRidge-Ensemble":
            relu_transformer = RandomReLUBase(
                no_rcb_feature_sets=int(self.feature_complexity * self.train_size / 2),
                maxlookback=self.lookback,
                no_lookbacks=1,
                seed=self.seed,
            )
            four_transformer = RandomFourierBase(
                no_rfb_feature_sets=int(self.feature_complexity * self.train_size / 28),
                maxlookback=self.lookback,
                no_lookbacks=1,
                seed=self.seed,
            )
            relu_features = relu_transformer.transform(self.timeseries_raw)
            four_features = four_transformer.transform(self.timeseries_raw)
            self.extracted_features = torch.concat(
                [
                    relu_features,
                    four_features,
                ],
                axis=1,
            )

        if self.ml_model_name == "TimeSeriesRidge-ReLU":
            transformer = RandomReLUBase(
                no_rcb_feature_sets=int(self.feature_complexity * self.train_size),
                maxlookback=self.lookback,
                no_lookbacks=1,
                seed=self.seed,
            )
            self.extracted_features = transformer.transform(self.timeseries_raw)

        if self.ml_model_name == "TimeSeriesRidge-Fourier":
            transformer = RandomFourierBase(
                no_rfb_feature_sets=int(self.feature_complexity * self.train_size / 14),
                maxlookback=self.lookback,
                no_lookbacks=1,
                seed=self.seed,
            )
            self.extracted_features = transformer.transform(self.timeseries_raw)

        if self.ml_model_name == "TimeSeriesRidge-Signature":
            transformer = SignatureTransformerBase(
                no_features=int(self.feature_complexity * self.train_size),
                maxlookback=self.lookback,
                no_lookbacks=1,
                seed=self.seed,
                signature_level=2,
                max_path_width=4,
            )
            self.extracted_features = transformer.transform(self.timeseries_raw)

        ## Create features and targets on the historical TS slice
        # train_start = max( 0, self.extracted_features.shape[0] - self.data_embargo - self.train_size)
        train_start = 0
        train_end = -1 * self.data_embargo
        features = self.extracted_features[train_start:train_end]
        targets = self.timeseries_raw[train_start + self.data_embargo :]
        self.live_data = self.extracted_features[-2:-1, :]
        ## Train Ridge Models
        self.model = PyTorchRidgeBase(
            alphas=self.alphas,
        )
        self.model.train(features, targets)

    def predict(
        self,
    ):
        pred = self.model.predict(self.live_data)
        return pred

    def get_prediction_names(self):
        return [f"alpha-{a}" for a in self.alphas]
