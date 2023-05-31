import json, gc, logging
import numpy as np
import pandas as pd


"""
Statistical Models 

Calculate 



"""

from pythor.utils.stats import update_cumulative_stats

ARRAY_PACKAGE = np

ROLLING_RULES = [
    "Rolling_Mean",
    "Rolling_Vol",
]

CUMULATIVE_RULES = [
    "Cumulative_Mean",
]


class StatsRules:
    def __init__(
        self,
        ml_model_name="StatsRule_1",
        data_embargo=5,
        lookbacks=None,
        rules=None,
        **kwargs,
    ):

        self.data_embargo = data_embargo
        self.lookbacks = lookbacks
        ## Machine Learning Parameters
        self.ml_model_name = ml_model_name
        self.rules = rules
        self.cols = list()

    def cumulative_window_cals(self, alpha, rule):
        if rule == "Cumulative_Mean":
            raw_preds = (
                self.timeseries_df.ewm(
                    alpha=alpha,
                    adjust=False,
                )
                .mean()
                .tail(1)
                .values
            )
        return raw_preds

    """
    Rolling Window Calcs underperform so do not recommend to use, simply put here for completeness 
    
    """

    def rolling_window_cals(self, lookback, rule):
        ## Statistical Rules based on Rolling Size Windows
        if rule == "Rolling_Mean":
            raw_preds = ARRAY_PACKAGE.mean(
                self.timeseries_raw[-1 * lookback :, :], axis=0
            ).reshape(1, -1)
        if rule == "Rolling_Vol":
            raw_preds = (
                ARRAY_PACKAGE.std(
                    self.timeseries_raw[-1 * lookback :, :], axis=0
                ).reshape(1, -1)
                * -1
            )
        return raw_preds

    def train(self, TS_hist):

        self.timeseries_raw = ARRAY_PACKAGE.array(TS_hist)
        self.timeseries_df = TS_hist.copy()

        raw_preds_list = list()

        if self.lookbacks is None:
            lookbacks = [
                0.001,
                0.01,
                0.1,
                1.0,
            ]
        elif type(self.lookbacks) == list:
            lookbacks = self.lookbacks
        else:
            lookbacks = [self.lookbacks]

        if self.rules is None:
            rules = CUMULATIVE_RULES
        elif type(self.rules) == list:
            rules = self.rules
        else:
            rules = [self.rules]

        for lookback in lookbacks:
            for rule in rules:
                if rule in ROLLING_RULES:
                    raw_preds_list.append(self.rolling_window_cals(lookback, rule))
                    self.cols.append(f"{rule}_lookback{lookback}")

        for lookback in lookbacks:
            for rule in rules:
                if rule in CUMULATIVE_RULES:
                    raw_preds_list.append(self.cumulative_window_cals(lookback, rule))
                    self.cols.append(f"{rule}_lookback{lookback}")

        self.raw_preds = ARRAY_PACKAGE.concatenate(raw_preds_list, axis=0)

    def predict(
        self,
    ):
        return self.raw_preds

    def get_prediction_names(self):
        return self.cols
