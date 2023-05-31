"""

Perform a very simple model snapshots for PyTorch Models  



Given a ML model which implments the following methods 
    - __init__(model_params)
    - train(X_train,y_train,X_test,y_test)
    - predict(X_test,predict_args)
    - save_model(model_path)
    - load_model(model_path) 


Parameters (within ml_model_params_batch)

    - base_learner_method='torch-sparsemlp-incremental'
    - base_learner_params_batch = dict()
    - base_learner_params_live = dict()

    - num_iterations 
        How many Snapshots or Ensembles to create
    - reuse_learner 
        For model snapshots, we will reuse learner to continue training, 


Implementation 

    - Model Snapshots 
        - By setting learning rate to be zero and reuse_learner to be true, we are training a single model continuously and then                 

        

Recommened:

Use Early Stopping from a simple PyTorch model so that the snapshots will not overrun
        

"""


MLP_standard_params = {
    "seed": 1000,
    "max_epochs": 20,
    "patience": 5,
    "Weights-L2": 0,  ## Maybe NOT to use this when using sparse layers?
    "swa_lrs": 0,
    "gradient_clip_val": 0.5,  ## Clip to Encourage Slow Learning?
    "num_encoding_layers": 2,  ## Cannot be to more than 6 layers?
    "num_decoding_layers": 2,  ## Cannot be to more than 6 layers?
    "encoding_neuron_scale": 0.8,  ## Set between 0.8 to 1
    "decoding_neuron_scale": 0.8,  ## Set between 0.8 to 1
    "encoding_layer_sparsity": 0.9,
    "decoding_layer_sparsity": 0.9,
    "loss_func": "pearson",
    "proportion": 0.25,  ## Could be set between 0 to 0.5
}


import copy
import os, json, joblib, logging
import itertools

import pandas as pd
import numpy as np
import scipy


from pythor.constants import TABULAR_FORMATS
from pythor.tabular.common import (
    train_model_batch_tabular,
    train_model_live_tabular,
    get_model_pred_tabular,
    load_tabular_model,
    save_tabular_model,
)


ARRAY_PACKAGE = np


class ModelSnapshotEnsemble:
    def __init__(
        self,
        ml_model_params_batch=dict(),
        ml_model_params_live=dict(),
    ):
        ## Machine Learning Models
        self.ml_model_params_batch = ml_model_params_batch
        self.ml_model_params_live = ml_model_params_live
        self.setup_model(ml_model_params_batch)

    def setup_model(self, ml_model_params_batch):

        ## Set Up Model Snapshots
        self.base_learners_collection = list()
        self.num_boosters = ml_model_params_batch.get("num_iterations", 10)
        self.reuse_learners = ml_model_params_batch.get("reuse_learners", True)

        ## Set Up Base Learners
        self.base_learner_method = ml_model_params_batch.get(
            "base_learner_method", "torch-sparsemlp-incremental"
        )
        self.base_learners_params_batch = ml_model_params_batch.get(
            "base_learners_params_batch", MLP_standard_params
        )
        self.base_learners_params_live = ml_model_params_batch.get(
            "base_learners_params_live", dict()
        )

    """
    Train and Prediction Methods 
    
    """

    def train(
        self,
        tabular_dataloader_func,
        train_tabular_dataloader_state,
        validate_tabular_dataloader_state,
    ):
        for round in range(self.num_boosters):
            ## Train Base Learners
            if not self.reuse_learners or round == 0:
                ## Create New Base Learner
                self.current_base_learner = train_model_batch_tabular(
                    self.base_learner_method,
                    self.base_learners_params_batch,
                    tabular_dataloader_func,
                    train_tabular_dataloader_state,
                    validate_tabular_dataloader_state,
                )
            else:
                train_model_live_tabular(
                    self.base_learner_method,
                    self.base_learners_params_live,
                    self.current_base_learner,
                    tabular_dataloader_func,
                    train_tabular_dataloader_state,
                    validate_tabular_dataloader_state,
                )
            ## Copy Model Parameters
            current_model_params = copy.deepcopy(
                self.current_base_learner.get_model_parameters()
            )
            self.base_learners_collection.append(current_model_params)
            ## Check for Early Stopping

            logging.info(f"Snapshot {round} Model {self.base_learner_method}")

    def update(
        self,
        train_data,
        validate_data,
    ):
        ## Not Implemented Currently, no need for model snapshots as we are always updating the weights of NN
        self.train(train_data, validate_data)

    def predict(
        self,
        X_test,
        predict_arg=dict(),
    ):
        ## Get Predictions from the boosters
        predictions = list()
        for round in range(self.num_boosters):
            current_model_params = self.base_learners_collection[round]
            current_base_learner = load_tabular_model(
                self.base_learner_method, current_model_params
            )
            predictions.append(current_base_learner.predict(X_test, predict_arg))
        return np.concatenate(predictions, axis=1)

    ## Get Model Representation to be used in other models
    def get_model_parameters(self):
        output = dict()
        output["ml_model_params_batch"] = self.ml_model_params_batch
        output["ml_model_params_live"] = self.ml_model_params_live
        output["base_learners_collection"] = self.base_learners_collection
        return output

    def save_model(self, path):
        params = self.get_model_parameters()
        joblib.dump(params, path)

    ## Load Model Representation from python objects,
    ## We save all the model parameters in a dictionary for GB-Ensemble
    def load_model_parameters(self, params):
        self.ml_model_params_batch = params["ml_model_params_batch"]
        self.setup_model(params["ml_model_params_batch"])
        self.ml_model_params_live = params["ml_model_params_live"]
        self.base_learners_collection = params["base_learners_collection"]

    def load_model(self, path):
        params = joblib.load(path)
        self.load_model_parameters(params)
