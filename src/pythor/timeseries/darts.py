import json, gc
import numpy as np
import pandas as pd


import sklearn
import torch
import darts


ARRAY_PACKAGE = np

"""
Time Series Sequence Models 

1. DARTS 
    Wrapper class for DARTS to provide 

    __init__(self,model_class,config)
        Create the suitable DART models with config 
    trai
        convert dataframe/numpy into DART TS class
        run the training 
    predict
        run the prediction 
    save_model
    load_model 



"""


import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


from darts import TimeSeries
from darts.models import (
    TransformerModel,
    NBEATSModel,
    TCNModel,
    RNNModel,
    BlockRNNModel,
)


"""
Walk Forward Prediction from dart models

The input X_all is a time-series that is already embargoed in data loading process


"""


class TimeSeriesDARTS:
    def __init__(
        self,
        model_name="DARTS-TransformerModel",
        data_embargo=6,
        train_size=200,
        lookback=50,
        model_config=dict(),
        additional_hyper=dict(),
        **kwargs,
    ):

        self.data_embargo = data_embargo
        self.train_size = train_size
        self.lookback = lookback
        self.model_config = model_config
        self.additional_hyper = additional_hyper
        self.model_name = model_name
        self.model_config = model_config
        self.model_config["input_chunk_length"] = self.lookback
        self.model_config["output_chunk_length"] = self.data_embargo
        self.model_config["random_state"] = self.model_config.get("seed", 0)

        ## Prevent Saturation by stopping the training process when TRAINING_LOSS drops slows down
        train_stopper = EarlyStopping(
            monitor="train_loss",
            patience=20,
            min_delta=0.001,
            mode="min",
        )

        pl_trainer_kwargs = {
            "accelerator": "gpu",
            "devices": "auto",
            "callbacks": [train_stopper],
        }
        ## ToDo: Set learning rate from additional_hyper
        optimizer_cls = torch.optim.Adam
        optimizer_kwargs = {"lr": 1e-3}
        ## ToDo: How to adjust learning rate for LSTM-like models
        # lr_scheduler_cls = torch.optim.lr_scheduler.StepLR
        # lr_scheduler_kwargs = {"gamma": 0.95, "step_size": 10}
        lr_scheduler_cls = None
        lr_scheduler_kwargs = None

        if self.model_name == "DARTS-TransformerModel":
            self.model = TransformerModel(
                pl_trainer_kwargs=pl_trainer_kwargs,
                optimizer_cls=optimizer_cls,
                optimizer_kwargs=optimizer_kwargs,
                lr_scheduler_cls=lr_scheduler_cls,
                lr_scheduler_kwargs=lr_scheduler_kwargs,
                **model_config,
            )
        if self.model_name == "DARTS-TCNModel":
            self.model = TCNModel(
                pl_trainer_kwargs=pl_trainer_kwargs,
                optimizer_cls=optimizer_cls,
                optimizer_kwargs=optimizer_kwargs,
                lr_scheduler_cls=lr_scheduler_cls,
                lr_scheduler_kwargs=lr_scheduler_kwargs,
                **model_config,
            )
        if self.model_name == "DARTS-BlockRNNModel":
            self.model = BlockRNNModel(
                pl_trainer_kwargs=pl_trainer_kwargs,
                optimizer_cls=optimizer_cls,
                optimizer_kwargs=optimizer_kwargs,
                lr_scheduler_cls=lr_scheduler_cls,
                lr_scheduler_kwargs=lr_scheduler_kwargs,
                **model_config,
            )

    def save_model(self, modelpath):
        self.model.save(modelpath)

    def load_model(self, modelpath):
        self.model.load(modelpath)

    def train(
        self,
        X_all,
    ):
        ## Only Accept Values form  arrays
        self.TS = TimeSeries.from_values(ARRAY_PACKAGE.array(X_all))
        self.TS_shape = self.TS.values().shape
        # training_data = self.TS.head(self.TS_shape[0] - self.data_embargo)
        ## Use the whole time series as training data
        training_data = self.TS.head(self.TS_shape[0])
        self.model.fit(
            training_data.tail(self.train_size + self.lookback),
            verbose=False,
        )

    def predict(
        self,
    ):
        pred = self.model.predict(
            self.data_embargo,
            series=self.TS,
            verbose=False,
        ).values()[-2:-1, :]
        return pred

    def get_prediction_names(self):
        return [str(i) for i in range(1, 2)]


"""

Alternative implementation of DARTS models using historical forecast, 
the results are much better which I suspect there maybe look-ahead bias? 

DART Models for single CV splits 



"""


class TimeSeriesNNIncrementalAlt:
    def __init__(
        self,
        model_name="DARTS-TransformerModel",
        data_embargo=6,
        train_size=200,
        lookback=50,
        model_config=dict(),
        additional_hyper=dict(),
        **kwargs,
    ):

        self.data_embargo = data_embargo
        self.train_size = train_size
        self.lookback = lookback
        self.model_config = model_config
        self.additional_hyper = additional_hyper
        self.model_name = model_name
        self.model_config = model_config
        self.model_config["input_chunk_length"] = self.lookback
        self.model_config["output_chunk_length"] = self.data_embargo
        self.model_config["random_state"] = self.model_config.get("seed", 0)

        ## Prevent Saturation by stopping the training process when TRAINING_LOSS drops slows down
        train_stopper = EarlyStopping(
            monitor="train_loss",
            patience=20,
            min_delta=0.001,
            mode="min",
        )

        pl_trainer_kwargs = {
            "accelerator": "gpu",
            "devices": -1,
            "auto_select_gpus": True,
            "callbacks": [train_stopper],
        }
        ## ToDo: Set learning rate from additional_hyper
        optimizer_cls = torch.optim.Adam
        optimizer_kwargs = {"lr": 1e-3}
        ## ToDo: How to adjust learning rate for LSTM-like models
        # lr_scheduler_cls = torch.optim.lr_scheduler.StepLR
        # lr_scheduler_kwargs = {"gamma": 0.95, "step_size": 10}
        lr_scheduler_cls = None
        lr_scheduler_kwargs = None

        if self.model_name == "DARTS-TransformerModel":
            self.model = TransformerModel(
                pl_trainer_kwargs=pl_trainer_kwargs,
                optimizer_cls=optimizer_cls,
                optimizer_kwargs=optimizer_kwargs,
                lr_scheduler_cls=lr_scheduler_cls,
                lr_scheduler_kwargs=lr_scheduler_kwargs,
                **model_config,
            )
        if self.model_name == "DARTS-TCNModel":
            self.model = TCNModel(
                pl_trainer_kwargs=pl_trainer_kwargs,
                optimizer_cls=optimizer_cls,
                optimizer_kwargs=optimizer_kwargs,
                lr_scheduler_cls=lr_scheduler_cls,
                lr_scheduler_kwargs=lr_scheduler_kwargs,
                **model_config,
            )
        if self.model_name == "DARTS-BlockRNNModel":
            self.model = BlockRNNModel(
                pl_trainer_kwargs=pl_trainer_kwargs,
                optimizer_cls=optimizer_cls,
                optimizer_kwargs=optimizer_kwargs,
                lr_scheduler_cls=lr_scheduler_cls,
                lr_scheduler_kwargs=lr_scheduler_kwargs,
                **model_config,
            )

    def train(self, X_all):
        ## Only Accept Values form np arrays
        self.TS = TimeSeries.from_values(ARRAY_PACKAGE.array(X_all))
        self.TS_shape = self.TS.values().shape
        print(self.TS_shape)
        ## Prediction
        ## Cannot set start to be the end of time series,
        # so the best would be setting as follows which run the algo twice and get the last row
        assert self.train_size > (self.lookback + self.data_embargo)
        self.preds = self.model.historical_forecasts(
            self.TS,
            train_length=self.train_size - self.lookback,
            start=self.TS_shape[0] - 2,
            retrain=False,
            forecast_horizon=self.data_embargo,
            overlap_end=True,
            last_points_only=True,
            verbose=False,
        ).values()

    def predict(
        self,
    ):
        return ARRAY_PACKAGE.array(self.preds[-1, :])

    def get_prediction_names(self):
        return [f"target-{i}" for i in range(1, 2)]

    def save_model(self, modelpath):
        self.model.save(modelpath)

    def load_model(self, modelpath):
        self.model.load(modelpath)
