import os, joblib, copy, logging, json
import numpy as np
import pandas as pd

import xgboost as xgb

from pythor.utils.linalg import tail_corr
from pythor.utils.data import tabular_dataloader


XGBOOST_EXAMPLE_CONFIG = {
    "num_boost_round": 1000,
    "early_stopping_rounds": 100,
    "booster": "gbtree",  ## Could be one of "gbtree/gblinear/dart"
    "min_split_loss": 1e-4,
    "max_depth": 6,
    "min_child_weight": 10,
    "max_delta_step": 0,
    "subsample": 0.5,
    "sampling_method": "uniform",
    "colsample_bytree": 0.5,
    "reg_lambda": 1e-3,
    "reg_alpha": 1e-3,
    "grow_policy": "depthwise",  ## Could be set to "depthwise/lossguide"
    "max_leaves": 128,
    "max_bin": 5,
    "num_parallel_tree": 1,
    "seed": 0,
}


"""
XGBoost Custom Eval Objective 

"""


def clip_normalise(y, clip=1e-4):
    return (y - y.mean()) / max(clip, y.std())


def pearson_rank(predt: np.ndarray, dtrain: xgb.DMatrix):
    no_groups = dtrain.get_group().shape[0]
    raw_group = np.concatenate([[0], np.cumsum(dtrain.get_group())], axis=0).astype(int)
    group_start = raw_group[:-1]
    group_end = raw_group[1:]
    y = dtrain.get_label()
    pearson_rank = 0
    for i in range(no_groups):
        y_era = y[group_start[i] : group_end[i]]
        predt_era = predt[group_start[i] : group_end[i]]
        y_era = clip_normalise(y_era)
        predt_era = clip_normalise(predt_era)
        era_corr = np.corrcoef(y_era, predt_era)[0, 1]
        pearson_rank = pearson_rank + era_corr

    return "pearson_rank", pearson_rank / no_groups


def create_xgboost_dmatrix(output, max_bin, target_name="target", ref=None):
    ## Default to train with the first target
    y_train = output["target"][target_name]
    raw_counts = np.bincount(output["groups"].values.reshape(-1))
    group_sizes = raw_counts[raw_counts > 0]
    ## ToDo: More creative ways to set data weights?
    train = xgb.QuantileDMatrix(
        data=output["features"].values,
        label=y_train.values,
        group=group_sizes,
        max_bin=max_bin,
        ref=ref,
        nthread=10,  ## Set to parallel data loading
    )
    shape = output["features"].shape
    logging.info(f"XGBoost Feature Shape {shape}")
    return train


class XGBoostIncrementalRegressor:
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

        ## Sensible Default parameters for Training XGBoost models
        default_parameters = {
            "verbosity": 0,
            "tree_method": "gpu_hist",
            "gpu_id": 0,  ## Set to Single GPU for now
            "nthread": 10,  ## This is to allow quicker data loading
            "disable_default_eval_metric": True,  ## We use Correlation Rank as custom Eval metric
            "maximize": True,  ## We want to maximise Correlation Rank
        }
        ## Create Local Copy of Config as modifying objectives
        self.config_train = copy.deepcopy(self.config)
        self.config_train.update(default_parameters)
        self.config_train.pop("use_snapshots", False)
        self.config_train.pop("no_snapshots", 10)

        logging.info("Loading Data from Era Folder")
        train_data = tabular_dataloader(
            tabular_dataloader_func, train_tabular_dataloader_state
        )
        validate_tabular_dataloader_state[
            "feature_cols"
        ] = train_tabular_dataloader_state["feature_cols"]
        validate_data = tabular_dataloader(
            tabular_dataloader_func, validate_tabular_dataloader_state
        )
        ## Currently only CrunchDAO has 7 bins
        max_bin = self.config_train.get("max_bin", 7)
        target_name = self.config_train.pop("target_col", "target")
        train_data_xgboost = create_xgboost_dmatrix(
            train_data, max_bin, target_name=target_name
        )
        validate_data_xgboost = create_xgboost_dmatrix(
            validate_data, max_bin, target_name=target_name, ref=train_data_xgboost
        )

        logging.info("Starts Training for XGBoost Models")
        ## Use the Booster object to train directly and allows for continual training
        model = xgb.train(
            params=self.config_train,
            dtrain=train_data_xgboost,
            num_boost_round=self.config_train.get("num_boost_round", 1000),
            evals=[(validate_data_xgboost, "validation")],
            early_stopping_rounds=self.config_train.get("early_stopping_rounds", 100),
            verbose_eval=2000,
            xgb_model=self.model,
            custom_metric=pearson_rank,
            maximize=True,
        )
        ## Get trained CatBoost model
        self.model = model

    def update(
        self,
        tabular_dataloader_func,
        train_tabular_dataloader_state,
        validate_tabular_dataloader_state,
        online_params,
    ):
        self.config_train = copy.deepcopy(online_params)
        # self.config_train['process_type'] = 'update' Use this for updating tree weights
        # self.config_train['updater'] = 'grow_gpu_hist,refresh,prune' ## common separately string
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
        cast_to_numpy = lambda x: np.array(x)
        inference_model = self.model
        inference_model.set_param({"predictor": "cpu_predictor"})
        start_iteration_ratio = predict_arg.get("start_iteration_ratio", 0)
        end_iteration_ratio = predict_arg.get("end_iteration_ratio", 1)
        num_trees = self.model.num_boosted_rounds()
        if start_iteration_ratio != end_iteration_ratio:
            start_iteration = int(start_iteration_ratio * num_trees)
            end_iteration = int(end_iteration_ratio * num_trees)
        else:
            ## Best iteration used when ratio is 0
            start_iteration = 0
            end_iteration = getattr(self.model, "best_iteration")
            if end_iteration is None:
                end_iteration = num_trees
        return inference_model.inplace_predict(
            cast_to_numpy(X_test),
            iteration_range=(start_iteration, end_iteration),
        )

    ## Get Model Representation to be used in other models
    def get_model_parameters(self):
        output = dict()
        output["config"] = copy.deepcopy(self.config)
        output["model_json"] = self.model.save_raw(
            raw_format="ubj",
        )
        return output

    def save_model(self, path):
        output = self.get_model_parameters()
        joblib.dump(output, path)

    ## Load Model Representation from python objects, for LightGBM it is string format.
    def load_model_parameters(self, params):
        self.config = params["config"]
        ## Does not matter much when loading model for use
        self.model = xgb.Booster(params=self.config, model_file=params["model_json"])

    def load_model(self, path):
        params = joblib.load(path)
        self.load_model_parameters(params)
