"""
List of Constants for defining the Incremental Learners 



"""


### Data Formats Currently Support
TABULAR_FORMATS = [
    "features",
    "target",
    "groups",
    "raw_target",  ## Used when real financial data are used where we can calculate actual portfolio returns
]
TIMESERIES_FORMATS = [
    "feature_performances",
]


"""
Different Backtest Programs

Factor Timing: Apply Time Series Prediction techniques to rank the features, then 

"""

INCREMENTAL_LEARNER_ENV = [
    "FactorTiming",
    "TemporalTabular",
    "EnsembleTabular",
]

"""
Different Prediction Models

"""

FACTOR_TIMING_MODELS = {
    "STATS_MODELS": [
        "EqualWeighted",
        "StatsRules",
        "DynamicFN",
    ],
    "RIDGE_MODELS": [
        "TimeSeriesRidge-Ensemble",
        "TimeSeriesRidge-ReLU",
        "TimeSeriesRidge-Fourier",
        "TimeSeriesRidge-Signature",
    ],
    "DARTS_MODELS": [
        "DARTS-TransformerModel",
        "DARTS-TCNModel",
        "DARTS-BlockRNNModel",
    ],
}


TABULAR_MODELS = {
    "XGBoost": [
        "xgboost-regression-tabular",
    ],
    "PyTorch": [
        "torch-MLP-tabular",
        "torch-SparseMLP-tabular",
    ],
    "Basic": [
        "equal_weighted-tabular",
    ],
    "CatBoost": [
        "catboost-regression-tabular",
    ],
    "Ensemble": [
        "ensemble-snapshot-tabular",
    ],
    "LightGBM": [
        "lightgbm-regression-tabular",
    ],
}
