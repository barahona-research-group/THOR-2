import os, joblib, copy, logging, json
import numpy as np
import pandas as pd

from catboost import CatBoostRanker, CatBoostRegressor, CatBoost, Pool, FeaturesData

from pythor.utils.linalg import tail_corr
from pythor.utils.data import tabular_dataloader

"""
loss_function: 
(pointwise): RMSE, Quantile, "Lq:q=4"
(pairwise): "PairLogit:max_pairs=100000"
(listwise): QueryRMSE, YetiRank 

bootstrap_type: "Bayesian", "Bernoulli",
subsample: float between 0 to 1 (when bootstrap type is NOT Bayesian)

grow_policy: 
"SymmetricTree", "Depthwise", "Lossguide", 

 "score_function": Set to L2 as we cannot use Cosine for Lossguide tree growing policy 

"""

CATBOOST_EXAMPLE_CONFIG = {
    "random_seed": 0,
    "loss_function": "PairLogitPairwise:max_pairs=500000",
    "iterations": 5000,
    "early_stopping_rounds": 5000,
    "learning_rate": 0.001,
    "depth": 8,
    "min_data_in_leaf": 20,
    "max_leaves": 128,
    "grow_policy": "SymmetricTree",
    "score_function": "L2",
    "reg_lambda": 0.001,
    "bagging_temperature": 1,
    "bootstrap_type": "Bayesian",
}


"""
CatBoost Data Format

We do not include feature names within the CatBoost model as we save it within our own format in THOR

"""


def create_catboost_pool(output, loss_function, target_name="target"):
    X_train = FeaturesData(
        num_feature_data=output["features"].values.astype(np.float32),
    )
    ## Default to train with the first target
    y_train = output["target"][target_name].values.reshape(-1, 1)
    queries_train = output["groups"]
    train = Pool(data=X_train, label=y_train, group_id=queries_train)
    return train


class CatBoostIncrementalRegressor:
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

        ## Sensible Default parameters for Training CatBoost models
        default_parameters = {
            "silent": True,
            "allow_writing_files": False,
            "custom_metric": None,
            "eval_metric": None,
            "task_type": "GPU",
        }
        ## Create Local Copy of Config as modifying objectives
        self.config_train = copy.deepcopy(self.config)
        self.config_train.update(default_parameters)

        ## Remove Parameters for controlling tree complexity as CatBoost do not accept extra parameters
        use_snapshots = self.config_train.pop("use_snapshots", False)
        self.config_train.pop("no_snapshots", 10)
        target_name = self.config_train.pop("target_col", "target")

        loss_function = self.config_train.get(
            "loss_function",
            "QueryRMSE",
        )
        ## On CUDA, these loss functions for ranking tasks needs to be fixed under 1023
        if loss_function in [
            "YetiRank",
        ]:
            train_tabular_dataloader_state["batch_size"] = 1000
            validate_tabular_dataloader_state["batch_size"] = 1000

        logging.info("Loading Data from Era Folder")
        train_data = tabular_dataloader(
            tabular_dataloader_func, train_tabular_dataloader_state
        )
        validate_data = tabular_dataloader(
            tabular_dataloader_func, validate_tabular_dataloader_state
        )
        logging.info("Creating Data for CatBoost Models")
        train_data_catboost = create_catboost_pool(
            train_data, loss_function, target_name=target_name
        )
        validate_data_catboost = create_catboost_pool(
            validate_data, loss_function, target_name=target_name
        )

        logging.info("Starts Training for CatBoost Models")
        task = self.config_train.get("task", "ranking")
        if (
            use_snapshots
            or self.config_train["iterations"]
            == self.config_train["early_stopping_rounds"]
        ):
            use_best_model = False
        else:
            use_best_model = True
        if task == "ranking":
            model = CatBoostRanker(**self.config_train)
            model.fit(
                train_data_catboost,
                eval_set=validate_data_catboost,
                init_model=self.model,
                use_best_model=use_best_model,
            )
        else:
            model = CatBoostRegressor(**self.config_train)
            model.fit(
                train_data_catboost,
                eval_set=validate_data_catboost,
                init_model=self.model,
                use_best_model=use_best_model,
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
        start_iteration_ratio = predict_arg.get("start_iteration_ratio", 0)
        end_iteration_ratio = predict_arg.get("end_iteration_ratio", 1)
        num_trees = self.model.tree_count_
        if start_iteration_ratio != end_iteration_ratio:
            start_iteration = int(start_iteration_ratio * num_trees)
            end_iteration = int(end_iteration_ratio * num_trees)
        else:
            start_iteration = 0
            end_iteration = getattr(self.model, "best_iteration_")
            if end_iteration is None:
                end_iteration = num_trees
        return inference_model.predict(
            cast_to_numpy(X_test),
            ntree_start=start_iteration,
            ntree_end=end_iteration,
        )

    ## Get Model Representation to be used in other models
    def get_model_parameters(self):
        output = dict()
        output["config"] = copy.deepcopy(self.config)
        temp_file_name = "catboost_model_temp.json"
        self.model.save_model(temp_file_name, format="json")
        with open(temp_file_name, "r") as f:
            output["model_json"] = json.load(f)
        os.remove(temp_file_name)
        return output

    def save_model(self, path):
        output = self.get_model_parameters()
        joblib.dump(output, path)

    ## Load Model Representation from python objects, for LightGBM it is string format.
    def load_model_parameters(self, params):
        self.config = params["config"]
        ## Does not matter much when loading model for use
        self.model = CatBoostRanker(self.config)
        temp_file_name = "catboost_model_temp.json"
        ## Will Always overwrite this temp file, so please do not run this in parallel
        with open(temp_file_name, "w") as f:
            json.dump(params["model_json"], f)
        self.model.load_model(temp_file_name, format="json")
        os.remove(temp_file_name)

    def load_model(self, path):
        params = joblib.load(path)
        self.load_model_parameters(params)
