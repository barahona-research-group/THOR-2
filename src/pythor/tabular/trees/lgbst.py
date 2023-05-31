"""
Gradient Boosting Decision Tree Models 




"""
import copy, joblib, logging
from functools import partial
import numpy as np
import pandas as pd
import scipy


import lightgbm as lgb
from lightgbm import LGBMRanker, LGBMRegressor, Booster


import dask.dataframe as dd
import dask.array as da

from pythor.utils.linalg import tail_corr
from pythor.utils.data import tabular_dataloader


EXAMPLE_LIGHTGBM_CONFIG = {
    "early_stopping_round": 200,  ## Better NOT to use early stopping when monitroing validation loss from Corr
    "n_estimators": 5000,
    "learning_rate": 0.01,
    "num_leaves": 128,
    "min_data_in_leaf": 20,
    "min_gain_to_split": 0.0001,
    "feature_fraction": 0.25,
    "feature_fraction_bynode": 1,
    "lambda_l1": 0.0001,
    "lambda_l2": 0.0001,
    "bagging_fraction": 0.5,
    "bagging_freq": 100,
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": None,
    "device_type": "cuda",
    "verbose": -2,
    "num_gpu": 1,
    "max_bin": 15,
    "gpu_use_dp": True,
    "seed": 100,
}


"""
Utility for Ranking Correlation loss for LightGBM models 

"""


def calc_group_sizes_dask(df):
    return [df.get_partition(i).shape[0].compute() for i in range(df.npartitions)]


def calc_group_sizes_pandas(df):
    group_sizes = list()
    for i, g in df.groupby("era"):
        group_sizes.append(df.shape[0])
    return group_sizes


class LightGBMIncrementalRegressor:
    def __init__(
        self,
        config=dict(),
    ):
        self.config = copy.deepcopy(config)
        self.model = None

    def train(
        self,
        tabular_dataloader_func,
        train_tabular_dataloader_state,
        validate_tabular_dataloader_state,
    ):

        # dask_to_numpy = lambda x: np.array(x)
        # dask_to_numpy = lambda x: x.compute()

        ## This works for both pd dataframes and dask dataframes
        dask_to_numpy = lambda x: x.values

        early_stopping_rounds = self.config.get("early_stopping_round", 20)

        logging.info("Creating Data from Dask Loader for LightGBM Models")
        ## LightGBM models have a single target only, so default to use the first one from a given list
        target_name = train_tabular_dataloader_state.get("target_names", ["target"])[0]
        train_data = tabular_dataloader(
            tabular_dataloader_func, train_tabular_dataloader_state
        )
        X_train = dask_to_numpy(train_data["features"])
        y_train = dask_to_numpy(train_data["target"][target_name])
        if early_stopping_rounds > 0:
            validate_data = tabular_dataloader(
                tabular_dataloader_func, validate_tabular_dataloader_state
            )
            X_validate = dask_to_numpy(validate_data["features"])
            y_validate = dask_to_numpy(validate_data["target"][target_name])

        ## Convert into LightGBM group format
        ## As Batch Size is fixed, it can be calculated from train and validate dataloader state
        if train_tabular_dataloader_state.get("use_dask", True):
            group_train = calc_group_sizes_dask(train_data["groups"])
            group_validate = calc_group_sizes_dask(validate_data["groups"])
        else:
            pass

        ## Create Local Copy of Config as modifying objectives
        self.config_train = copy.deepcopy(self.config)
        objective = self.config_train.get("objective", "regression")

        if True:
            print(type(X_train), type(y_train), target_name)

        ## Run LightGBM Boosting Rounds with early stopping, automatically continue to use existing model if there is one

        if early_stopping_rounds > 0:
            validate_result = dict()
            callbacks = [
                lgb.early_stopping(
                    early_stopping_rounds,
                    first_metric_only=False,
                    verbose=True,
                ),
                lgb.record_evaluation(validate_result),
                lgb.log_evaluation(period=100, show_stdv=True),
            ]
            if objective in [
                "lambdarank",
                "rank_xendcg",
                "regression",
            ]:
                model = LGBMRanker(**self.config_train)
                model.fit(
                    X=X_train,
                    y=y_train,
                    group=group_train,
                    eval_set=[(X_validate, y_validate)],
                    eval_names=["validation"],
                    eval_group=[group_validate],
                    eval_metric=[
                        "lambdarank",
                        "regression",
                    ],
                    callbacks=callbacks,
                    init_model=self.model,
                )
            else:
                model = LGBMRegressor(**self.config_train)
                model.fit(
                    X=X_train,
                    y=y_train,
                    eval_set=[(X_validate, y_validate)],
                    eval_names=["validation"],
                    eval_metric=[
                        "regression",
                    ],
                    callbacks=callbacks,
                    init_model=self.model,
                )
            ## Extract Models
            self.model = model.booster_
            self.validate_result = validate_result
            logging.info("LightGBM Model Training with Early Stopping Done")
        else:
            if objective in [
                "lambdarank",
                "rank_xendcg",
                "regression",
            ]:
                model = LGBMRanker(**self.config_train)
                model.fit(
                    X=X_train,
                    y=y_train,
                    group=group_train,
                    init_model=self.model,
                )
            else:
                model = LGBMRegressor(**self.config_train)
                model.fit(
                    X=X_train,
                    y=y_train,
                    init_model=self.model,
                )
            ## Extract Models
            self.model = model.booster_
            logging.info("LightGBM Model Training Done")

    def update(
        self,
        tabular_dataloader_func,
        train_tabular_dataloader_state,
        validate_tabular_dataloader_state,
        online_params,
    ):
        ## ToDo: Find a better ways for continual learning, currently we simply repeat the training process once
        if False:
            self.config["num_iterations"] = online_params.get("num_iterations", 10)
            self.config["early_stopping_round"] = online_params.get(
                "early_stopping_round", 0
            )
        self.train(
            tabular_dataloader_func,
            train_tabular_dataloader_state,
            validate_tabular_dataloader_state,
        )

    def predict(
        self,
        X_test,
        predict_arg=dict(),
    ):
        dask_to_numpy = lambda x: np.array(x)

        if predict_arg.get("use_gpu", False) and False:
            ## Use HummingBird for inference in PyTorch
            ## Disabled due to not able to trauncate trees during predictions
            from hummingbird.ml import convert, load

            inference_model = convert(self.model, "pytorch", device="cuda")
            inference_model = self.model
        else:
            inference_model = self.model
        num_iteration_ratio = predict_arg.get("num_iteration_ratio", 0)
        if num_iteration_ratio != 0:
            num_iteration = int(num_iteration_ratio * self.model.num_trees())
            return inference_model.predict(
                dask_to_numpy(X_test),
                start_iteration=0,
                num_iteration=num_iteration,
            )
        else:
            ## Best iteration used when ratio is 0
            num_iteration = getattr(
                self.model, "best_iteration", self.model.num_trees()
            )
            return inference_model.predict(
                dask_to_numpy(X_test), start_iteration=0, num_iteration=num_iteration
            )

    ## Get Model Representation to be used in other models
    def get_model_parameters(self):
        output = dict()
        output["model_string"] = self.model.model_to_string()
        output["config"] = copy.deepcopy(self.config)
        return output

    def save_model(self, path):
        output = self.get_model_parameters()
        joblib.dump(output, path)

    ## Load Model Representation from python objects, for LightGBM it is string format.
    def load_model_parameters(self, params):
        self.config = params["config"]
        self.model = Booster(model_str=params["model_string"])

    def load_model(self, path):
        params = joblib.load(path)
        self.load_model_parameters(params)
