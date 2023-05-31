"""
Loading Machine Learning Objects

"""


from ..constants import FACTOR_TIMING_MODELS, TABULAR_MODELS


from ..tabular.common import save_tabular_model, load_tabular_model
from ..ensemble.common import save_ensemble_model, load_ensemble_model


def save_thor_model(model, model_type, outputpath):
    if (
        model_type
        in TABULAR_MODELS["CatBoost"]
        + TABULAR_MODELS["XGBoost"]
        + TABULAR_MODELS["PyTorch"]
        + TABULAR_MODELS["LightGBM"]
    ):
        save_tabular_model(model, model_type, outputpath)
    if model_type in TABULAR_MODELS["Ensemble"]:
        save_ensemble_model(model, model_type, outputpath)
    return None


def load_thor_model(model_type, outputpath):
    if (
        model_type
        in TABULAR_MODELS["CatBoost"]
        + TABULAR_MODELS["XGBoost"]
        + TABULAR_MODELS["PyTorch"]
        + TABULAR_MODELS["LightGBM"]
    ):
        reg = load_tabular_model(model_type, outputpath)
        return reg
    if model_type in TABULAR_MODELS["Ensemble"]:
        reg = load_ensemble_model(model_type, outputpath)
        return reg
    return None
