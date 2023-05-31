"""

Factor-timing Models for Temporal Tabular Predictions 

Input: Temporal Tabular Data 

Output: Predictions for temporal tabular predictions 

Scoring: Ranking Correlation 

Factoring Timing: Perform Time Series Prediction which predict the future values of factor return,
Using only the lagged values of factors to predict factors

Tabular >> Feature-Target Cross-Correlation >> Lagged TimeSeries >> Factor Timing >> Linear Weighted Prediction >> Tabular Predictions

Models 
    - StatsRules 
    - TimeSeriesRidge
    - DARTS


"""
import copy, logging, os, joblib
import itertools


import pandas as pd
import numpy as np
import scipy


from pythor.utils.linalg import normalise_preds, matrix_project
from pythor.base.backtest import TabularBase

from .stats import StatsRules
from .ridge import TimeSeriesRidge
from .darts import TimeSeriesDARTS, TimeSeriesNNIncrementalAlt


from pythor.constants import FACTOR_TIMING_MODELS

RIDGE_MODELS = FACTOR_TIMING_MODELS["RIDGE_MODELS"]
DARTS_MODELS = FACTOR_TIMING_MODELS["DARTS_MODELS"]
STATS_MODELS = FACTOR_TIMING_MODELS["STATS_MODELS"]

from .stats import CUMULATIVE_RULES


"""
Factor Timing for Tabular Tasks 

    1. Statistcal Rules 
        - Moving Averages 
    2. Time Series Ridge 
        - Random Fourier/ReLU transformations 
        - Random Signature Transforms 
    3. Deep Learning Models for sequences  
        - Transformer (DARTS)
        - Block RNN (DARTS)
        - Temporal Convolutional Network (DARTS) 
    4. Dynamic Feature Neutralisation 
        - Dynamically Adjust the subset of features for neutralisation 

"""


"""
Dynamic Feature Neutralisation

tabular_preds (N_obs,N_models), factor_timings(N_methods,N_features)

We will use 5 different cumulative rolling stats to select the feature to neutralise. 


The cutoff is based on 4 different methods 
    Top 25\% of featues, Bottom 25\% of feaures,
    Top and Bottom 12.5\% The Middle 25\%  

"""


def dynamic_feature_neutralisation(tabular_preds, factor_timings):
    tabular_preds_output = list()
    ## Extract the feature to neutralise for
    for i in range(factor_timings.shape[0]):
        raw_preds = factor_timings[i, :]
        for c in [(0.25, -0.5), (0.5, -0.25), (0.375, -0.375), (-0.125, 0.125)]:
            cutoff = c[0]
            negcutoff = c[1]
            truncated = normalise_preds(
                raw_preds, axis=1, cutoff=cutoff, negcutoff=negcutoff
            )
            selected_features_index = np.nonzero(truncated)[0]
            selected_features = np.array(tabular_preds["features"])[
                :, selected_features_index
            ].astype(float)
            projected_preds = matrix_project(
                np.array(tabular_preds["model_predictions"]).astype(float),
                selected_features,
            )
            tabular_preds_output.append(projected_preds)
    return np.concatenate(tabular_preds_output, axis=1)


class FactorTiming(TabularBase):
    def __init__(
        self,
        ml_model_name="StatsRules",
        ml_model_params_batch=dict(),
        ml_model_params_live=dict(),
        model_path_stem="StatsRules_0",
        prediction_path_stem="StatsRules_0",
        **kwargs,
    ):

        ## Machine Learning Models
        self.ml_model_name = ml_model_name
        self.ml_model_params_batch = ml_model_params_batch
        self.ml_model_params_live = ml_model_params_live
        self.model_path_stem = model_path_stem
        self.prediction_path_stem = prediction_path_stem
        super().__init__(**kwargs)

    ## TS Model are np array of shape (1,N) N is the number of features
    def get_single_model_pred(
        self,
        tabular_data_era,
        timeseries_data_era,
        ml_model_name,
        target_name="target",
        **args,
    ):

        feature_performances_raw = timeseries_data_era["feature_performances"].fillna(0)
        feature_performances = (
            (
                feature_performances_raw[
                    feature_performances_raw["target_names"] == target_name
                ].sort_values("era")
            )
            .drop(
                [
                    "era",
                    "target_names",
                ],
                axis=1,
            )
            .astype(float)
        )

        if self.debug and False:
            print(
                f"Lookback Shape of Feature Performances {feature_performances.shape}"
            )
        if ml_model_name in STATS_MODELS:
            model = StatsRules(
                ml_model_name=ml_model_name,
                data_embargo=self.embargo_size,
                lookbacks=self.ml_model_params_live.get("lookbacks", None),
                rules=self.ml_model_params_live.get("rules", None),
            )
            model.train(feature_performances)
            raw_preds = model.predict()

        if ml_model_name in RIDGE_MODELS:
            model = TimeSeriesRidge(
                ml_model_name=ml_model_name,
                data_embargo=self.embargo_size,
                train_size=self.train_size,
                lookback=self.ml_model_params_live.get("lookback", 50),
                feature_complexity=self.ml_model_params_live.get("no_feature_sets", 1),
                seed=self.ml_model_params_live.get("seed", 0),
                alphas=self.ml_model_params_live.get(
                    "alphas",
                    [
                        0.0001,
                    ],
                ),
            )
            model.train(feature_performances.values)
            raw_preds = model.predict().numpy(force=True)

        if ml_model_name in DARTS_MODELS:
            ml_model_params_live = copy.deepcopy(self.ml_model_params_live)
            ## Remove parameters to run DARTS
            ml_model_params_live.pop("target_cols", None)
            ml_model_params_live.pop("target_cols", None)
            model = TimeSeriesDARTS(
                model_name=ml_model_name,
                data_embargo=self.embargo_size,
                train_size=self.train_size,
                lookback=ml_model_params_live.pop(
                    "lookback", 50
                ),  ## lookback is not valid argument in model_params so need to remove
                model_config=ml_model_params_live,
                additional_hyper=dict(),
            )
            model.train(feature_performances.values)
            raw_preds = model.predict()

        col_names = model.get_prediction_names()
        col_names = [f"{target_name}-{x}" for x in col_names]

        return raw_preds, col_names

    def process_single_model_predictions(
        self, tabular_data_era, timeseries_data_era, target_name
    ):
        if self.ml_model_name != "EqualWeighted":

            ## Get Factor Timing of TS models of shape (1, no_features) or (no_targets,no_features)
            raw_preds, cols = self.get_single_model_pred(
                tabular_data_era,
                timeseries_data_era,
                self.ml_model_name,
                target_name,
            )

            factor_timing = normalise_preds(raw_preds, axis=1, cutoff=0)

            ## ToDo: We allow to take positive weights only when we combine models, default to use top 25\% of models
            if self.ml_model_params_live.get("positive_weights_only", False):
                pos_weight_cutoff = self.ml_model_params_live.get(
                    "positive_weights_cutoff", 0.25
                )
                factor_timing = np.clip(factor_timing, pos_weight_cutoff, 0.5)

            ## Create Tabular Predictions
            if self.ml_model_name != "DynamicFN":
                factortimed_preds = np.matmul(
                    tabular_data_era["features"].values.astype(float),
                    factor_timing.transpose(),
                )
            else:
                ## Implement Dynamic Feature Neutralisation Logic Here
                factortimed_preds = dynamic_feature_neutralisation(
                    tabular_data_era, factor_timing
                )
                preds_names = self.ml_model_params_live.get("target_cols", None)
                cols = list(
                    itertools.product(
                        preds_names,
                        CUMULATIVE_RULES,
                        [
                            "top",
                            "bottom",
                            "tail",
                            "middle",
                        ],
                    )
                )
                cols = ["-".join(x) for x in cols]
                if len(cols) != factortimed_preds.shape[1]:
                    cols = [
                        f"DFN-{i}" for i in range(1, factortimed_preds.shape[1] + 1)
                    ]

            factortimed_preds_ranked = normalise_preds(
                factortimed_preds.astype(float), axis=0, cutoff=0
            )

            return factortimed_preds_ranked, cols
        else:
            factortimed_preds_ranked = np.mean(
                tabular_data_era["features"], axis=1
            ).values.reshape(-1, 1)
            cols = ["Equal_Weighted"]
            return factortimed_preds_ranked, cols

    """
    Methods to be used in the base Backtest Env 
    
    """

    def train_model_batch(self):
        pass

    def train_model_live(self):
        pass

    def get_model_predictions(self, tabular_data_era, timeseries_data_era):

        target_names = self.ml_model_params_live.get("target_cols", None)
        if target_names is None:
            target_names = timeseries_data_era["feature_performances"][
                "target_names"
            ].unique()
            self.ml_model_params_live["target_cols"] = target_names

        cols = list()
        preds_era_targets = list()
        for target_name in target_names:
            preds_era_target, cols_target = self.process_single_model_predictions(
                tabular_data_era,
                timeseries_data_era,
                target_name,
            )
            preds_era_targets.append(preds_era_target)
            cols.extend(cols_target)

        preds_era_all = np.concatenate(preds_era_targets, axis=1)
        preds_era_rank = normalise_preds(
            preds_era_all,
            axis=0,
            cutoff=0,
        )
        ## We store predictions in pandas as we only perform up to 10k stocks in each prediction
        predictions_era = pd.DataFrame(
            preds_era_rank,
            columns=cols,
            index=tabular_data_era["target"].index,
        )
        ## Add Model Names to prefix
        model_prefix = self.model_path_stem.split("/")[-1] + "-"
        return predictions_era.add_prefix(model_prefix)
