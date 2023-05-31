import joblib, os, shutil, datetime, json, glob
import logging, gc, copy
import math

import numpy as np

import dask.array as da
import dask.dataframe as dd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset

from pytorch_lightning import Trainer, LightningModule, seed_everything
from pytorch_lightning.callbacks import (
    LearningRateFinder,
    StochasticWeightAveraging,
    EarlyStopping,
    GradientAccumulationScheduler,
    ModelPruning,
    ModelCheckpoint,
)


from pythor.utils.data import tabular_dataloader

from .nets import SparseMLP, MLP

"""

Torch Lightning Model for Tabular Dataset 

Trainin Process

1. Early Stopping 
2. Find Learning Rate Automatically, Batch Size has to be fixed for all eras to calculate ranking loss
3. Stochastic Weight Averaging
4. Gradient Accumulation 
5. Model Pruning



"""

ARRAY_PACKAGE = np
PYTROCH_TENSOR_CAST = lambda x: torch.tensor(np.array(x).astype(float))


class BatchTensorDataset(torch.utils.data.IterableDataset):
    def __init__(self, X, y, groups):
        self.X = X
        self.y = y
        self.groups = groups

    def __iter__(self):
        for i, df in self.groups.groupby(self.groups.columns[0]):
            batch_index = df.index
            X_batch = self.X.loc[df.index].values.astype(float)
            ## Assume the input values are between 0 to 1
            y_batch = self.y.loc[df.index].values.astype(float) - 0.5
            yield torch.tensor(X_batch), torch.tensor(y_batch)


class PyTorchIncrementalTabularModel:
    def __init__(self, nn_model, config=dict()):

        """
        Args:
            nn_model (LightningModule): Neural Networks implmented as a LightningModule
            config (dict): A dictionary which contains the parameters for training NN
        """

        ## Initialisation
        self.nn_model = nn_model
        self.network = None
        ## Save a copy of the config dictionary as we will modify it during continuous training
        self.config = copy.deepcopy(config)
        self.setup_random()

    def setup_random(self):
        ## For resutls to be consistent in reruns
        seed_everything(self.config.get("seed", 0), workers=True)
        self.device = torch.cuda.current_device()

    def create_callbacks(self):

        callbacks = list()

        ## Save the best model based on validation loss (Needs to be the first callback in order to restore the best model params)
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1, monitor="val_loss", mode="min"
        )
        callbacks.append(checkpoint_callback)

        ## Early Stopping
        patience = self.config.get("patience", 0)
        if patience > 0:
            early_stop_callback = EarlyStopping(
                monitor="val_loss",
                min_delta=0.00,
                patience=patience,
                verbose=False,
                mode="min",
            )
            callbacks.append(early_stop_callback)

        ## Learning Rate Finder
        if self.config.get("use_lr_finder", False):
            lr_callback = LearningRateFinder(
                min_lr=0.001,
                max_lr=0.2,
                num_training_steps=5,  ## Changed to 5 from 20 to reduce search time?
                mode="exponential",
                early_stop_threshold=None,
                update_attr=True,
            )
            callbacks.append(lr_callback)

        ## Gradient Accumulation
        if self.config.get("use_gradient_accumulation", False):
            max_epochs = self.config.get("max_epochs", 10)
            equal_sized_decay = dict()
            steps = self.config.get("grad_acc_steps", 4)
            start = self.config.get("grad_acc_start", 16)
            for i in range(steps):
                epoch = int(i * max_epochs / steps)
                equal_sized_decay[epoch] = int(start * math.pow(0.5, i))
            accumulator = GradientAccumulationScheduler(scheduling=equal_sized_decay)
            callbacks.append(accumulator)

        ## Stochastic Weight Averaging
        swa_lrs = self.config.get("swa_lrs", 0)
        if swa_lrs > 0:
            SWA_callback = StochasticWeightAveraging(swa_lrs=swa_lrs)
            callbacks.append(SWA_callback)

        ## Model Pruning
        if self.config.get("use_model_pruning", False):
            pass

        return callbacks

    def setup_dataloader(
        self,
        tabular_dataloader_func,
        train_tabular_dataloader_state,
        validate_tabular_dataloader_state,
    ):

        logging.info("Creating Data for PyTorch Models")
        ## Load Data without shuffling to maintain roughly division across eras
        ## ToDo: Get target cols from model config instead of dataloader state
        # target_name = train_tabular_dataloader_state.get("target_cols", ["target"])
        target_name = self.config.get("target_cols", ["target"])
        train_data = tabular_dataloader(
            tabular_dataloader_func, train_tabular_dataloader_state
        )
        validate_tabular_dataloader_state[
            "feature_cols"
        ] = train_tabular_dataloader_state["feature_cols"]
        X_train = train_data["features"]
        y_train = train_data["target"][target_name]
        validate_data = tabular_dataloader(
            tabular_dataloader_func, validate_tabular_dataloader_state
        )
        X_validate = validate_data["features"]
        y_validate = validate_data["target"][target_name]

        dataset_train = BatchTensorDataset(
            X_train,
            y_train,
            train_data["groups"],
        )

        dataset_validate = BatchTensorDataset(
            X_validate,
            y_validate,
            validate_data["groups"],
        )

        ## Setting up DataLoader with the datasets
        dataloader_train = DataLoader(
            dataset_train,
            batch_size=1,
            num_workers=0,
            shuffle=False,
            drop_last=False,
            pin_memory=False,  ## Should we use pin memory?
        )

        dataloader_validate = DataLoader(
            dataset_validate,
            batch_size=1,
            num_workers=0,
            shuffle=False,
            drop_last=False,
            pin_memory=False,
        )

        ## Creating Network requires knowing the size of features and targets
        self.config["input_shape"] = X_train.shape[1]
        self.config["output_shape"] = y_train.shape[1]

        logging.info(f"Training NN input {X_train.shape[1]} output {y_train.shape[1]}")
        self.network = self.nn_model(self.config)

        return dataloader_train, dataloader_validate

    """
    The Main Training Loop of PyTorch Lightning Models 
    
    """

    def train(
        self,
        tabular_dataloader_func,
        train_tabular_dataloader_state,
        validate_tabular_dataloader_state,
    ):
        ### If we already have a network, we can reuse the dataloaders
        if self.network is None:
            self.dataloader_train, self.dataloader_validate = self.setup_dataloader(
                tabular_dataloader_func,
                train_tabular_dataloader_state,
                validate_tabular_dataloader_state,
            )

        callbacks = self.create_callbacks()

        ## Clip Gradients as proxy for L1 regularisation?
        self.trainer = Trainer(
            enable_progress_bar=False,
            accelerator="cuda",
            deterministic=False,  ## For Fast Computations
            max_epochs=self.config.get("max_epochs", 10),
            callbacks=callbacks,
            gradient_clip_val=self.config.get("gradient_clip_val", 0.5),
        )

        logging.info("Starts training in PyTorch Model")
        self.trainer.fit(
            self.network,
            self.dataloader_train,
            self.dataloader_validate,
        )

        ## Reload Best Model Weights
        if self.config.get("patience", 0) > 0:
            self.network = self.network.load_from_checkpoint(
                callbacks[0].best_model_path
            )

    """
    Current Implementation it is the same as train 
        For Model Snapshots, no need to change anything for training process 

    """

    def update(
        self,
        tabular_dataloader_func,
        train_tabular_dataloader_state,
        validate_tabular_dataloader_state,
        online_params,
    ):
        self.train(
            tabular_dataloader_func,
            train_tabular_dataloader_state,
            validate_tabular_dataloader_state,
        )

    """
    Predictions from PyTroch model

    Input could be dask/pandas dataframes loaded by tabular_load_func 

    A Recommend usecase would be online inference, which is used by the backtest env to generate prediction at each week. 
    

    """

    def predict_single(self, X, predict_args=dict()):
        dataset_prediction = TensorDataset(PYTROCH_TENSOR_CAST(X))
        dataloader_prediction = DataLoader(
            dataset_prediction,
            batch_size=5000,  ## Just A Number to limit how much data to put into memory at a time.
            num_workers=0,
            shuffle=False,
        )
        trainer = Trainer(
            accelerator="cuda",
            deterministic=False,
            enable_progress_bar=False,
        )
        predictions = trainer.predict(self.network, dataloaders=dataloader_prediction)
        return torch.cat(predictions).numpy()

    def predict(self, X, predict_args=dict()):
        return self.predict_single(X, predict_args)

    """
    Save and Loading Models
    
    """

    def update_config(self):
        self.config["learning_rate"] = self.network.lr

    ## Get Model Representation to be used in other models
    def get_model_parameters(self):
        output = dict()
        self.update_config()
        output["config"] = copy.deepcopy(self.config)
        output["state_dict"] = copy.deepcopy(self.network.state_dict())
        return output

    def save_model(self, path):
        output = self.get_model_parameters()
        torch.save(output, path)

    ## Load Model Representation from python objects, for PyTorch it is dictionary
    def load_model_parameters(self, params):
        self.config = copy.deepcopy(params["config"])
        self.setup_random()
        self.network = self.nn_model(params["config"])
        state_dict = params["state_dict"]
        self.network.load_state_dict(state_dict)
        self.network.eval()

    def load_model(self, path):
        params = torch.load(path)
        self.load_model_parameters(params)
