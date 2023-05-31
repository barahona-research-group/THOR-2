import copy, os, json, logging, joblib

import pandas as pd
import numpy as np
import scipy

import dask.dataframe as dd
import dask.array as da

from pythor.utils.linalg import tail_corr, numerai_corr
from pythor.utils.stats import update_cumulative_stats
from pythor.utils.ml import save_thor_model, load_thor_model
from pythor.constants import TABULAR_FORMATS, TIMESERIES_FORMATS


from pythor.utils.data import tabular_dataloader, check_file_status

"""
Backtest Envs 

1. Base Class for Temporal Tabular Datasets 
    1. Load Data and the create point-in-time views for training models 
        1. Standard train/validation split for standard tabular data models
        2. Time Series slicing for factor-timing models 

2. Lazy Loading of data 
For Tabular features, we use lazy loading data for training 

3. Walk Forward Prediction

We load data in each era for prediction 



"""
from pythor.constants import TABULAR_MODELS, FACTOR_TIMING_MODELS

TABULAR_PRED_MODELS = [x for v in TABULAR_MODELS.values() for x in v]
TS_PRED_MODELS = [x for v in FACTOR_TIMING_MODELS.values() for x in v]


class TabularBase:
    def __init__(
        self,
        tabular_dataloader_func=None,
        tabular_dataloader_states=dict(),
        timeseries_dataloader_func=None,
        timeseries_dataloader_states=dict(),
        timeseries_dataloader_path_stem=None,
        incremental_learning_params=dict(),
        env_seed=0,
        debug=False,
        live_submission=False,
    ):

        ## ML Models
        self.ml_model = None
        self.model_seq = 0
        self.tabular_performances = list()

        ## Data Loading
        self.tabular_dataloader_func = tabular_dataloader_func
        self.tabular_dataloader_states = tabular_dataloader_states
        self.timeseries_dataloader_func = timeseries_dataloader_func
        self.timeseries_dataloader_states = timeseries_dataloader_states
        self.timeseries_dataloader_path_stem = timeseries_dataloader_path_stem
        self.live_submission = live_submission

        ## Training Size of Tabular Features
        self.train_size = incremental_learning_params.get("train_size", 20)
        self.validate_pct = incremental_learning_params.get("validate_pct", 0.25)
        self.train_pct = incremental_learning_params.get("train_pct", 0.75)

        ## Determine Start and End Eras
        self.model_start = incremental_learning_params.get("model_start", 863)
        self.live_era = incremental_learning_params.get("live_era", 1050)
        self.embargo_size = incremental_learning_params.get("embargo_size", 5)
        self.model_start_era = self.model_start

        ## Continue previous model to train or train a new model from scratch
        self.retrain_freq = incremental_learning_params.get("retrain_freq", 200)
        self.reuse_model = incremental_learning_params.get("reuse_model", False)
        self.retrain_counter = self.retrain_freq - 1

        self.pred_score_func = incremental_learning_params.get(
            "pred_score_func", "correlation"
        )

        self.env_seed = env_seed
        self.debug = debug

        self.features_names = None
        self.target_names = None

        ### Data Loading Logic
        if live_submission:
            self.get_live_data()
        else:
            self.get_backtest_data()

    """
    Data Loading from sources

    1. Backtest Data Loading
    2. Live Data Loading 
    
    """

    def get_backtest_ts_features(self):
        self.timeseries_data_buffer = dict()
        filepath = f"{self.timeseries_dataloader_path_stem}.dataloader"
        data_loader_object = joblib.load(filepath)
        data_loader_buffer = data_loader_object["dataloader_states"]["data_buffer"]
        for key in TIMESERIES_FORMATS:
            self.timeseries_data_buffer[key] = data_loader_buffer.pop(
                key, pd.DataFrame()
            )
            if self.timeseries_data_buffer[key].shape[1] < 1:
                logging.info(f"Time Series Feature {key} is empty")
            else:
                logging.info(f"Time Series Feature {key} loaded for Backtest")

    def get_backtest_data(self):

        ## Load Data By Era
        data_start = max(self.model_start_era - self.embargo_size - self.train_size, 1)
        logging.info(
            f"Backtest starts at Era {data_start}, Train Size {self.train_size}, Embargo {self.embargo_size}"
        )
        ## Load Time Series/Tabular Features
        self.get_backtest_ts_features()

    def get_live_data(self):
        logging.info(f"Loading Data for Live Submission Era {self.live_era}")
        self.get_backtest_ts_features()

    """
    Get Data for training 

    Tabular Features: train/validation split by groups
    Time Series Features: All timeseries data up to data embargo 

    ToDo: Move Data Splits to the individual tabular methods
    
    """

    def get_data_splits_tabular(self):
        ## Calculate the start and end of training/validation eras
        ## Convention is not to include the end point as like python loops
        start_era = max(self.current_era - self.embargo_size - self.train_size, 1)
        end_era = self.current_era - self.embargo_size
        mid_era = int(self.validate_pct * start_era + (1 - self.validate_pct) * end_era)

        if self.validate_pct > 0:
            temp = self.train_pct + self.validate_pct
            train_start_era = int(temp * start_era + (1 - temp) * end_era)
            train_end_era = int(mid_era - self.embargo_size / 2)
            validate_start_era = int(mid_era + self.embargo_size / 2)
            validate_end_era = end_era
        else:
            train_start_era = start_era
            train_end_era = end_era
            validate_start_era = start_era
            validate_end_era = end_era

        logging.info(
            f"Train Start {train_start_era} Train End {train_end_era -1} Validation Start {validate_start_era} Validation End {validate_end_era-1}"
        )

        train_tabular_dataloader_state = copy.deepcopy(self.tabular_dataloader_states)
        train_tabular_dataloader_state["start_era"] = train_start_era
        train_tabular_dataloader_state["end_era"] = train_end_era
        validate_tabular_dataloader_state = copy.deepcopy(
            self.tabular_dataloader_states
        )
        validate_tabular_dataloader_state["start_era"] = validate_start_era
        validate_tabular_dataloader_state["end_era"] = validate_end_era

        return train_tabular_dataloader_state, validate_tabular_dataloader_state

    """
    Get Sliced Time Series Features
    """

    def get_data_slices_timeseries(self):

        ## Extract Time Series Data
        timeseries_data_era = dict()
        data_embargo_index = self.current_era - self.embargo_size
        for key in TIMESERIES_FORMATS:
            if key in self.timeseries_data_buffer.keys():
                logging.info(
                    f"Extract TimeSeries Data {key} from buffer up to {data_embargo_index-1}"
                )
                if self.timeseries_data_buffer[key].shape[0] > 0:
                    df = self.timeseries_data_buffer[key]
                    timeseries_data_era[key] = df[df["era"] < data_embargo_index]

        return timeseries_data_era

    """
    Data Loading Logic During incremental learning 

    We load data > train models > update data buffer > repeat 

    ToDo: Always load data from live dataloader for performances calc but do not add back to backtest buffer 
    (Since we use dask to manage computations)

    """

    def check_feature_consistency(self):
        hyper_filepath = f"{self.model_path_stem}.hyperparameters"
        if os.path.exists(hyper_filepath):
            hyper_parameters = joblib.load(hyper_filepath)
            expected_features = hyper_parameters["features_names"]
            return len(expected_features)
        else:
            return -1

    def get_current_tabular_data(self):

        if self.ml_model_name in TABULAR_PRED_MODELS:
            logging.info(f"Load Data from Tabular data load {self.current_era}")
            backtest_state = copy.deepcopy(self.tabular_dataloader_states)
            backtest_state["start_era"] = self.current_era
            backtest_state["end_era"] = self.current_era + 1
            ## Do not perform resampling during model evaluation
            backtest_state["batch_size"] = -1
            backtest_state["target_projection_strength"] = 0
            tabular_data_era = tabular_dataloader(
                self.tabular_dataloader_func,
                backtest_state,
            )
            if self.features_names is None:
                self.features_names = list(tabular_data_era["features"].columns)
                self.target_names = list(tabular_data_era["target"].columns)
                self.tabular_dataloader_states["feature_cols"] = self.features_names

            ## Check features
            expected_features_no = self.check_feature_consistency()
            if expected_features_no > 0:
                assert expected_features_no == len(self.features_names)

        ## These are the tabular data to combine with feature/factor timing
        elif self.ml_model_name in TS_PRED_MODELS:
            logging.info(f"Loading Data from TS data load {self.current_era}")
            backtest_state = copy.deepcopy(self.timeseries_dataloader_states)
            if self.ml_model_name == "DynamicFN":
                backtest_state["run_DFN"] = True
            else:
                backtest_state["run_DFN"] = False
            backtest_state["calc_feature_performance"] = True
            tabular_data_era = self.timeseries_dataloader_func(
                self.current_era, backtest_state
            )
        else:
            logging.info(f"{self.ml_model_name} has no data loading method implemented")

        return tabular_data_era

    """
    Incremental Learner Training 
    
    """

    def train_incremental_models_tabular(self):
        if self.ml_model == None or self.retrain_counter <= 0:
            self.model_seq += 1
            model_path = f"{self.model_path_stem}-seq{self.model_seq}.model"
            if not self.live_submission and not os.path.exists(model_path):
                ## Can only run update if we already have a model
                if self.reuse_model and self.model_seq > 1:
                    logging.info(f"Update Existing model {model_path}")
                    self.train_model_live()
                else:
                    logging.info(f"Train New model {model_path}")
                    self.train_model_batch()
                save_thor_model(self.ml_model, self.ml_model_name, model_path)
                logging.info(f"New model saved {model_path}")
            else:
                self.ml_model = load_thor_model(self.ml_model_name, model_path)
                # logging.info(f"Loaded model {model_path}")
            self.retrain_counter = self.retrain_freq - 1
        else:
            self.retrain_counter -= 1

    def predict_score_era(
        self,
        tabular_data_era,
        timeseries_data_era,
    ):
        era_str = f"{self.current_era:04d}"
        prediction_file = f"{self.prediction_path_stem}_preds_{era_str}.parquet"
        current_era = self.current_era
        ## Ranked Predictions (-0.5 to 0.5) from Models as a DataFrame
        predictions_era = self.get_model_predictions(
            tabular_data_era, timeseries_data_era
        )
        ## As we will reuse prediction as features,
        # need to make sure the predictions have the same index order as tabular_data_era
        ## And also assume to be zero-mean ranks (-0.5 to 0.5) for target subtraction
        predictions_era = predictions_era.reindex(
            tabular_data_era["groups"].index
        ).fillna(0)
        predictions_era.to_parquet(prediction_file)
        ## ToDo: How about other ways to calculate metrics for Tabular Ranking Tasks
        ## score_func(X,y) where X is (N_observations, No_Models) and y is (N_observations,) and output (K,1)
        if self.pred_score_func == "correlation":
            if self.ml_model_name in TABULAR_PRED_MODELS:
                target_name = self.tabular_dataloader_states.get(
                    "target_scoring", ["target"]
                )[0]
            elif self.ml_model_name in TS_PRED_MODELS:
                target_name = self.timeseries_dataloader_states.get(
                    "target_scoring",
                    ["target"],
                )[0]
            ## Default to use first column from the current data loader
            if not target_name in tabular_data_era["target"].columns:
                target_name = tabular_data_era["target"].columns[0]

            targets = tabular_data_era["target"][target_name].values
            targets_val = targets
            score = numerai_corr(
                predictions_era,
                targets_val,
            ).transpose()  ## Flip back to (1,K)
        ## Assuming Scoring on first target only
        performances_era = pd.DataFrame(
            score, columns=predictions_era.columns, index=[current_era]
        )
        self.tabular_performances.append(performances_era)
        logging.info(
            f"Prediction {current_era} {performances_era.iloc[:,:2].to_dict()} {performances_era.iloc[:,-2:].to_dict()}"
        )

    """
    Reset the Backtest Env to the start 
    """

    def reset(
        self,
    ):
        self.current_era = self.model_start_era

    def check_backtest_end(self):
        if self.current_era >= self.live_era:
            logging.info(f"Backtest Completed at Era {self.current_era}")
            terminated = True
        else:
            terminated = False

        return terminated

    def step(self, action=None):

        era_str = f"{self.current_era:04d}"
        prediction_file = f"{self.prediction_path_stem}_preds_{era_str}.parquet"
        current_era = self.current_era

        ## Only update the live era during live submission for speed up
        if self.live_submission:
            if current_era >= self.live_era:
                needs_update = True
            else:
                needs_update = False
        ## Needs to update the previous era as live data rotates from weekend to weekday rounds
        elif current_era >= self.live_era - 6:
            needs_update = True
        else:
            needs_update = False

        ## Skip Running Backtest for the day if file is recently updated
        days_since_updated = check_file_status(prediction_file, 1)
        if days_since_updated < 0.04:
            needs_update = False

        if self.ml_model_name in TABULAR_PRED_MODELS:
            self.train_incremental_models_tabular()

        if not os.path.exists(prediction_file) or needs_update:
            ## Get Data View of current Era
            tabular_data_era = self.get_current_tabular_data()
            logging.info(f"Currently Running Era {self.current_era}")
            timeseries_data_era = self.get_data_slices_timeseries()
            ## Scoring first and then update model
            self.predict_score_era(tabular_data_era, timeseries_data_era)

        ## Move to Next Era
        terminated = self.check_backtest_end()
        # logging.info(f"Completed Running Era {self.current_era}")
        self.current_era = self.current_era + 1
        return terminated

    """
    Each Incremental Learner Env Needs to implement the following 
        - train_model_batch: Offline Batch Training of Standard Incremental Learning Models
        - train_model_live: Continuous training of Standard Incremental Learning Models
        - get_model_predictions:  Get Tabular Predictions from models

    
    """

    def train_model_batch(self):
        pass

    def train_model_live(self):
        pass

    def get_model_predictions(self):
        pass
