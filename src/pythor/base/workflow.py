import copy, os, json, logging, glob

import pandas as pd
import numpy as np
import scipy

import joblib


from pythor.tabular.backtest import TemporalTabular
from pythor.timeseries.backtest import FactorTiming
from pythor.ensemble.backtest import EnsembleTabular

from pythor.utils.data import update_dataloader_timeseries, update_dataloader_tabular


## ToDo: Generate this based on keys from constants.py
BACKTEST_ENV_MAP = {
    "equal_weighted-tabular": TemporalTabular,
    "lightgbm-regression-tabular": TemporalTabular,
    "xgboost-regression-tabular": TemporalTabular,
    "catboost-regression-tabular": TemporalTabular,
    "torch-MLP-tabular": TemporalTabular,
    "torch-SparseMLP-tabular": TemporalTabular,
    "ensemble-snapshot-tabular": EnsembleTabular,
    "EqualWeighted": FactorTiming,
    "StatsRules": FactorTiming,
    "DynamicFN": FactorTiming,
    "TimeSeriesRidge-Ensemble": FactorTiming,
    "TimeSeriesRidge-ReLU": FactorTiming,
    "TimeSeriesRidge-Fourier": FactorTiming,
    "TimeSeriesRidge-Signature": FactorTiming,
    "DARTS-TransformerModel": FactorTiming,
    "DARTS-TCNModel": FactorTiming,
    "DARTS-BlockRNNModel": FactorTiming,
}


"""
Run Benchmark Workflow for incremental learning 


"""


def incremental_tabular_benchmark_base(
    tabular_dataloader_func,
    tabular_dataloader_states,
    timeseries_dataloader_func,
    timeseries_dataloader_states,
    timeseries_dataloader_path_stem=None,
    incremental_learning_params=None,
    backtest_model_params=None,
    debug=False,
    env_seed=0,
    live_submission=False,
):

    ##
    ml_model_name = backtest_model_params.get(
        "ml_model_name", "lightgbm-regression-tabular"
    )
    ml_model_params_batch = backtest_model_params.get("ml_model_params_batch", dict())
    ml_model_params_live = backtest_model_params.get("ml_model_params_live", dict())
    model_path_stem = backtest_model_params.get("model_path_stem", None)
    prediction_path_stem = backtest_model_params.get("prediction_path_stem", None)

    ## Default Names for mdoels/predictions/dataloaders
    default_name = f"{ml_model_name}_{env_seed}"
    if model_path_stem is None:
        model_path_stem = default_name
    if prediction_path_stem is None:
        prediction_path_stem = default_name
    if timeseries_dataloader_path_stem is None:
        timeseries_dataloader_path_stem = default_name

    ## Determine Backtest Env to use from ml_model_name
    tabular_env_method = BACKTEST_ENV_MAP[ml_model_name]

    tabular_env = tabular_env_method(
        ml_model_name=ml_model_name,
        ml_model_params_batch=ml_model_params_batch,
        ml_model_params_live=ml_model_params_live,
        model_path_stem=model_path_stem,
        prediction_path_stem=prediction_path_stem,
        tabular_dataloader_func=tabular_dataloader_func,
        tabular_dataloader_states=tabular_dataloader_states,
        timeseries_dataloader_func=timeseries_dataloader_func,
        timeseries_dataloader_states=timeseries_dataloader_states,
        timeseries_dataloader_path_stem=timeseries_dataloader_path_stem,
        incremental_learning_params=incremental_learning_params,
        debug=debug,
        env_seed=env_seed,
        live_submission=live_submission,
    )

    ## Reset Data Pointer to the start
    tabular_env.reset()
    ## Run Backtest
    terminated = False
    while not terminated:
        action = None
        terminated = tabular_env.step(action)

    ## Hyper-parameters of backtest
    output = dict()
    ### Parameterstabular_env
    output["hyperparameters"] = {
        "incremental_learning_params": copy.deepcopy(incremental_learning_params),
        "tabular_model": ml_model_name,
        "tabular_model_batch_params": copy.deepcopy(ml_model_params_batch),
        "tabular_model_live_params": copy.deepcopy(ml_model_params_live),
        "features_names": getattr(tabular_env, "features_names", None),
        "target_names": getattr(tabular_env, "target_names", None),
        "env_seed": env_seed,
        "model_path_stem": model_path_stem,
        "prediction_path_stem": prediction_path_stem,
        "timeseries_dataloader_path_stem": timeseries_dataloader_path_stem,
    }
    filepath = f"{model_path_stem}.hyperparameters"
    if not os.path.exists(filepath):
        joblib.dump(output["hyperparameters"], filepath)

    return output, tabular_env


"""
Iterative Modelling
"""


def model_files_check(current_learners_hyper):
    valid = True
    for c in current_learners_hyper:
        sample_model_path = c["model_path_stem"] + "-seq1.model"
        if not os.path.exists(sample_model_path):
            valid = False
    return valid


def prediction_file_check(current_learners_hyper, live_era):
    valid = True
    for c in current_learners_hyper:
        current_era = live_era
        era_str = f"{current_era:04d}"
        last_prediction_path = c["prediction_path_stem"] + f"_preds_{era_str}.parquet"
        if not os.path.exists(last_prediction_path):
            valid = False
    return valid


## We pass level as from 1 to N
def update_tabular_dataloader_states(
    base_dataloader_state,
    dataloader_state_model,
    prediction_files,
    layer,
    model_no,  ## Model Number within layer
):
    ## Copy Metadata: project,dataset,folder
    new_dataloader_state = copy.deepcopy(base_dataloader_state)
    new_dataloader_state["previous_layer_files"] = prediction_files
    new_dataloader_state["prediction_layer"] = layer

    ##
    for key in [
        "subsample_freq",
        "batch_size",
        "feature_subsample_ratio",
        "use_given_features",
        "feature_set",
        "data_random_state",
        "feature_random_state",
    ]:
        if key in dataloader_state_model.keys():
            new_dataloader_state[key] = dataloader_state_model[key]
    return new_dataloader_state


def update_incremental_learning_states(
    base_incremental_learning_params,
    model_incremental_learning_params,
):
    ## Copy Metadata: project,dataset,folder
    new_config = copy.deepcopy(base_incremental_learning_params)
    ##
    for key in [
        "retrain_freq",
        "validate_pct",
        "train_pct",
    ]:
        if key in model_incremental_learning_params.keys():
            new_config[key] = model_incremental_learning_params[key]
    return new_config


def update_timeseries_dataloader_states(
    dataloader_state,
    prediction_files,
    layer,
):
    new_dataloader_state = copy.deepcopy(dataloader_state)
    new_dataloader_state["prediction_layer"] = layer
    new_dataloader_state["previous_layer_files"] = prediction_files
    if layer > 1 and False:
        new_dataloader_state["calc_feature_performance"] = True
    return new_dataloader_state


## We pass level as from 1 to N
def update_individual_learner_hyper(
    learner_hyper,
    ensemble_folder,
    level,
    seed,
    timeseries_dataloader_path_stem,
):
    ml_method = learner_hyper["ml_model_name"]
    learner_hyper[
        "model_path_stem"
    ] = f"{ensemble_folder}/models/{ensemble_folder}_layer{level}_{ml_method}_{seed}"
    learner_hyper[
        "prediction_path_stem"
    ] = f"{ensemble_folder}/predictions/{ensemble_folder}_layer{level}_{ml_method}_{seed}"
    learner_hyper[
        "tabular_dataloader_path_stem"
    ] = f"{ensemble_folder}/models/{ensemble_folder}_layer{level}_{ml_method}_{seed}"

    ## TimeSeries Data (which are factor performances from previous layers are shared within a layer)
    learner_hyper["timeseries_dataloader_path_stem"] = timeseries_dataloader_path_stem

    return copy.deepcopy(learner_hyper)


## We pass level as from 1 to N
def create_incremental_learning_params(base_incremental_learning_params, level):
    ## Calculate the approproate starting era of each layer given the first layer parameterrs
    incremental_learning_params = copy.deepcopy(base_incremental_learning_params)
    start = base_incremental_learning_params.get("model_start_shift", 1)
    for i in range(level):
        lag_size = (
            base_incremental_learning_params["embargo_size"]
            + base_incremental_learning_params["train_sizes"][i]
        )
        start = start + lag_size
    incremental_learning_params["embargo_size"] = base_incremental_learning_params[
        "embargo_size"
    ]
    incremental_learning_params["train_size"] = base_incremental_learning_params[
        "train_sizes"
    ][i]
    incremental_learning_params["model_start"] = start
    logging.info(f"Level {level} starts prediction at {start}")
    return incremental_learning_params


"""
Combine Predictions within a layer and return the folder path to get the predictions for the next layer
"""


def combine_prediction_files(
    ensemble_folder,
    layer,
    incremental_learning_params,
    live_submission=False,
    delete_preds=False,
):
    model_start_era = incremental_learning_params.get("model_start")
    live_era = incremental_learning_params.get("live_era")
    embargo_size = incremental_learning_params.get("embargo_size")
    combined_predictions_stem = (
        f"{ensemble_folder}/features/{ensemble_folder}_layer{layer}"
    )

    for era in range(model_start_era, live_era + 1):
        era_str = f"{era:04d}"
        file_names_within_layer = glob.glob(
            f"{ensemble_folder}/predictions/*layer{layer}*{era_str}.parquet"
        )
        combined_predictions = f"{combined_predictions_stem}_{era_str}.parquet"

        ## Only update the live era during live submission for speed up
        if live_submission:
            if era >= live_era:
                needs_update = True
            else:
                needs_update = False
        elif era >= live_era - 1:
            needs_update = True
        else:
            needs_update = False

        if not os.path.exists(combined_predictions) or needs_update:
            output = list()
            for f in file_names_within_layer:
                output.append(pd.read_parquet(f))
            ## Normalise Prediction Ranking as features for next layer
            combined_df = pd.concat(output, axis=1).rank(pct=True) - 0.5
            combined_predictions = f"{combined_predictions_stem}_{era_str}.parquet"
            combined_df.to_parquet(combined_predictions)
            logging.info(f"Combined Layer Predictions for Layer {layer} Era {era}")
            for f in file_names_within_layer:
                if delete_preds and os.path.exists(f):
                    os.remove(f)
        else:
            pass
    return combined_predictions_stem


"""
Iterative Tabular Modelling workflow 

Repeat the following at each level 



1. Set up Parameters
    - incremental learning parameters
        - when to start model
    - dataloader states 
        - tabular/timeseries dataloader needs to read predictions from previous layers
        - timeseries dataloaders will be saved 

2. Run Backtest for each model 
    Run Backtest Pipeline for model 
    Update model performances

3. Check Files 
    - Model files
    - Prediction files

At the end of iterative training, save model hyperparameters


"""


def iterative_tabular_benchmark_base(
    tabular_dataloader_func,
    timeseries_dataloader_func,
    base_tabular_dataloader_states=dict(),
    base_timeseries_dataloader_states=dict(),
    base_incremental_learning_params=dict(),
    learners_hyperparameters=list(),
    learners_tabular_dataloader_states=list(),
    learners_incremental_learning_params=list(),
    ensemble_folder="Ensemble1",
    debug=False,
    live_submission=False,
    **args,
):

    ## Create the folder objects
    if not os.path.exists(ensemble_folder):
        os.mkdir(ensemble_folder)

    performances_folder = f"{ensemble_folder}/performances"

    for subfolder in [
        "performances",
        "dataloaders",
        "models",
        "predictions",
        "features",
    ]:
        folder = f"{ensemble_folder}/{subfolder}"
        if not os.path.exists(folder):
            os.mkdir(folder)

    no_levels = len(learners_hyperparameters)
    prediction_files_layer_all = list()

    for layer in range(no_levels):
        logging.info(f"Start Iterative Backtest at level {layer+1}")

        ## ToDo: Each model within a layer can have different incremental learning params
        incremental_learning_params = create_incremental_learning_params(
            base_incremental_learning_params, layer + 1
        )
        logging.info(f"Updating Incremental Learning Params at level {layer+1}")

        ## Rename Time Series Data Loaders to be unique!
        timeseries_dataloader_path_stem = (
            f"{ensemble_folder}/dataloaders/{ensemble_folder}_TS_Layer{layer+1}"
        )

        timeseries_dataloader_states = update_timeseries_dataloader_states(
            base_timeseries_dataloader_states,
            prediction_files_layer_all,
            layer + 1,
        )
        live_era = incremental_learning_params["live_era"]
        embargo_size = incremental_learning_params["embargo_size"]
        update_dataloader_timeseries(
            timeseries_dataloader_path_stem,
            live_era=live_era,
            embargo_size=embargo_size,
            dataloader_func=timeseries_dataloader_func,
            dataloader_states=timeseries_dataloader_states,
        )
        logging.info(f"Updated Time Series Data Loader at level {layer+1}")

        current_learners_hyper = learners_hyperparameters[layer]
        current_learners_hyper = [
            update_individual_learner_hyper(
                current_learners_hyper[x],
                ensemble_folder,
                layer + 1,
                x,
                timeseries_dataloader_path_stem,
            )
            for x in range(len(current_learners_hyper))
        ]
        logging.info(f"Updated Backtest Model Params at level {layer+1}")

        layer_tabular_dataloader_states = learners_tabular_dataloader_states[layer]
        layer_incremental_learning_states = learners_incremental_learning_params[layer]

        ## Setting Up Backtest Env
        tabular_env = None
        prediction_files_layer = list()
        for x in range(len(current_learners_hyper)):
            ## Update Tabular Data Loader and Incremental Learning states for the Model
            tabular_dataloader_states_model = update_tabular_dataloader_states(
                base_tabular_dataloader_states,
                layer_tabular_dataloader_states[x],
                prediction_files_layer_all,
                layer + 1,
                x,
            )
            incremental_learning_params_model = update_incremental_learning_states(
                incremental_learning_params,
                layer_incremental_learning_states[x],
            )

            output, tabular_env = incremental_tabular_benchmark_base(
                tabular_dataloader_func,
                tabular_dataloader_states_model,
                timeseries_dataloader_func,
                timeseries_dataloader_states,
                timeseries_dataloader_path_stem=timeseries_dataloader_path_stem,
                incremental_learning_params=incremental_learning_params_model,
                backtest_model_params=current_learners_hyper[x],
                debug=debug,
                env_seed=x,
                live_submission=live_submission,
            )
            model_name = current_learners_hyper[x]["ml_model_name"]
            logging.info(f"Updated {model_name} at level {layer+1}")
            tabular_dataloader_path_stem = current_learners_hyper[x][
                "tabular_dataloader_path_stem"
            ]
            ## Save Model Based Tabular Features Data Loader
            update_dataloader_tabular(
                tabular_dataloader_path_stem,
                dataloader_func=tabular_dataloader_func,
                dataloader_states=tabular_dataloader_states_model,
                backtest_env_config=output,
            )
            layer_tabular_dataloader_states[x] = tabular_dataloader_states_model
            logging.info(
                f"Updated Tabular Data Loader for {model_name} at level {layer+1}"
            )
            ## Save Model Performances
            if len(tabular_env.tabular_performances) > 0:
                performances = pd.concat(
                    tabular_env.tabular_performances, axis=0
                ).dropna()
            else:
                performances = pd.DataFrame()
            performance_file = f"{performances_folder}/{ensemble_folder}_layer{layer+1}_{model_name}_{x}.csv"
            if not os.path.exists(performance_file):
                performances.to_csv(performance_file)
            else:
                pass
            prediction_files_layer.append(
                current_learners_hyper[x]["prediction_path_stem"]
            )

        prediction_files_layer_all.extend(prediction_files_layer)

    ## Pipeline Configurations
    pipeline_path = f"{ensemble_folder}/pipeline.joblib"
    pipeline = dict()
    pipeline["tabular_dataloader_func"] = tabular_dataloader_func
    pipeline["timeseries_dataloader_func"] = timeseries_dataloader_func
    pipeline["base_tabular_dataloader_states"] = base_tabular_dataloader_states
    pipeline["base_timeseries_dataloader_states"] = base_timeseries_dataloader_states
    pipeline["base_incremental_learning_params"] = base_incremental_learning_params
    pipeline["learners_hyperparameters"] = learners_hyperparameters
    pipeline["learners_tabular_dataloader_states"] = learners_tabular_dataloader_states
    pipeline[
        "learners_incremental_learning_params"
    ] = learners_incremental_learning_params
    pipeline["ensemble_folder"] = ensemble_folder
    joblib.dump(pipeline, pipeline_path)
    logging.info(f"Saving Pipeline Configurations for {ensemble_folder}")

    logging.info(f"Ensemble Model Training Pipeline {ensemble_folder} done")
    return prediction_files_layer_all


"""
ToDo: Lazy Copy Time Series Loaders from previous ones to new folders


"""
