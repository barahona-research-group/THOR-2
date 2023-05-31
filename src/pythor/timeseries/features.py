import json, gc
import numpy as np
import pandas as pd

import sklearn

from joblib import Parallel, delayed
from sklearn.base import TransformerMixin, BaseEstimator
import torch, signatory

from pythor.utils.linalg import tensor_rolling_calcs

"""
Feature Engineering for Time Series 

Multi-variate Methods 

1. Signature Transform
2. Random Convolution Kernel 
3. Random Fourier Transform (As in Virture of Complexity)

Single-Variate Methods 
4. Statistical Moments 

## Assume Input is a 2D tensor (Time, Features,) or 3D (Time, Features, Entities, )
Most of the test cases are written for 2D tensors 


"""


class TimeSeriesTransformerBase(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        maxlookback=52,
        lookbackscale=0.5,
        no_lookbacks=2,
        **args,
    ):
        self.maxlookback = maxlookback
        self.lookbackscale = lookbackscale
        self.no_lookbacks = no_lookbacks
        if self.maxlookback > 0:
            self.lookbacks = [
                int(self.maxlookback * np.power(self.lookbackscale, i))
                for i in range(self.no_lookbacks)
            ]
        else:
            self.lookbacks = [-1]
        self.data = dict()


"""
Feature Engineering Methods for Univariate TimeSeries

"""


class StatisticalBase(TimeSeriesTransformerBase):
    def __init__(
        self,
        calc_vol=False,
        **args,
    ):
        super(StatisticalBase, self).__init__(**args)
        self.calc_vol = calc_vol

    def transform(self, X):
        output = list()
        X_tensor = torch.as_tensor(X, device=torch.cuda.current_device())
        for l in self.lookbacks:
            if l > 0:
                X_tensor_new = tensor_rolling_calcs(
                    X_tensor,
                    method="mean",
                    window_size=l,
                    step_size=1,
                    time_dimension=0,
                    zero_padding=True,
                )
                output.append(X_tensor_new)
                if self.calc_vol:
                    X_tensor_new = tensor_rolling_calcs(
                        X_tensor,
                        method="vol",
                        window_size=l,
                        step_size=1,
                        time_dimension=0,
                        zero_padding=True,
                    )
                    output.append(X_tensor_new)
            features = torch.concatenate(output, axis=1).float()
            return features


"""
Feature Engineering methods for multi-varaite TimeSeries

"""


class SignatureTransformerBase(TimeSeriesTransformerBase):
    def __init__(
        self,
        signature_level=2,
        max_path_width=4,
        no_features=200,
        seed=0,
        apply_summation=True,
        use_logsig=True,
        **args,
    ):
        super(SignatureTransformerBase, self).__init__(**args)
        self.device = torch.cuda.current_device()
        self.signature_level = signature_level
        self.max_path_width = max_path_width
        self.no_features = no_features
        self.seed = seed
        ## For return series (or any stationary series with mean zero) apply summation to convert to price series
        self.apply_summation = apply_summation
        self.use_logsig = use_logsig

    def set_random_generators(self):
        torch.manual_seed(self.seed)

    def transform(self, X):
        self.set_random_generators()
        if self.apply_summation:
            X = torch.as_tensor(
                X,
                device=self.device,
            )
            X_tensor = torch.cumsum(X, dim=0)
        else:
            X_tensor = torch.as_tensor(
                X,
                device=self.device,
            )
        history_length = X_tensor.shape[0]
        ## Assume X is 2D array, input for signatory is 3D array with batch as first dimension
        sample_tensor = torch.zeros(
            (1, 100, self.max_path_width),
            device=self.device,
        )
        sample_path_class = signatory.Path(sample_tensor, self.signature_level)
        if self.use_logsig:
            no_sigs = sample_path_class.logsignature_channels()
        else:
            no_sigs = sample_path_class.signature_channels()
        self.no_signatures_sets = max(int(self.no_features / no_sigs), 1)
        sigs = torch.zeros(
            (history_length, len(self.lookbacks), self.no_signatures_sets * no_sigs),
            device=self.device,
        )
        ## Simulate batches of random projections
        sampled_cols = torch.randint(
            0,
            X_tensor.shape[1],
            size=(
                self.no_signatures_sets,
                self.max_path_width,
            ),
            device=self.device,
        )
        input_paths = X_tensor[:, sampled_cols].transpose(0, 1)

        ## Build Signatures
        path_class = signatory.Path(input_paths, self.signature_level)
        for l in range(len(self.lookbacks)):
            lookback = self.lookbacks[l]
            for i in range(0, history_length):  ## Minimum Length of path is 2
                end_index = max(2, i)
                ## Set Lookback to be negative values for using all the values up to time X
                if lookback > 0:
                    start_index = max(0, i - lookback)
                    nomralise_size = lookback
                else:
                    start_index = 0
                    nomralise_size = end_index
                ## Normalise Signature by lookback length to scale
                if self.use_logsig:
                    sigs[i, l, :] = (
                        path_class.logsignature(start_index, end_index).reshape(-1)
                    ) / nomralise_size
                else:
                    sigs[i, l, :] = (
                        path_class.signature(start_index, end_index).reshape(-1)
                    ) / nomralise_size
        features = sigs.reshape(history_length, -1)
        return torch.nan_to_num(
            features,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )


### ToDo: How to support expanding window calcs for Random Fourier Transforms?
class RandomFourierBase(TimeSeriesTransformerBase):
    def __init__(
        self,
        seed=0,
        no_rfb_feature_sets=200,
        **args,
    ):
        super(RandomFourierBase, self).__init__(**args)
        self.device = torch.cuda.current_device()
        self.random_seed = seed
        self.no_feature_sets = no_rfb_feature_sets
        self.gammas = torch.tensor(
            [
                0.1,
                0.5,
                1,
                2,
                4,
                8,
                16,
            ],
            device=self.device,
        ).reshape(1, -1)
        self.no_gammas = self.gammas.shape[1]

    def set_random_generators(self):
        torch.manual_seed(self.random_seed)

    def transform(self, X):
        self.set_random_generators()
        ## Assume X is 2D array, create lagged returns
        if self.maxlookback > 0:
            trans = StatisticalBase(
                maxlookback=self.maxlookback,
                lookbackscale=self.lookbackscale,
                no_lookbacks=self.no_lookbacks,
            )
            X_tensor = trans.transform(X)
        else:
            X_mean = np.cumsum(X, axis=0) / (np.arange(X.shape[0]) + 1).reshape(-1, 1)
            X_tensor = torch.as_tensor(
                X_mean, device=torch.cuda.current_device()
            ).float()

        history_length = X_tensor.shape[0]
        features = torch.zeros(
            (
                2,
                self.no_feature_sets,
                history_length,
                self.no_gammas,
            ),
            device=self.device,
        )

        ## Create Weights
        weights = torch.normal(
            0,
            1,
            size=(X.shape[1] * len(self.lookbacks), self.no_feature_sets),
            device=self.device,
        )
        ## (History * no_lagged_features,) (no_lagged_features, no_feature_sets)
        feature_raw = torch.matmul(
            torch.matmul(X_tensor, weights)
            .transpose(
                0,
                1,
            )
            .unsqueeze(-1),
            self.gammas,
        )
        ## (no_feature_sets,History,1) (1, no_gammas)
        features[0, :, :] = torch.sin(feature_raw)
        features[1, :, :] = torch.cos(feature_raw)
        features_row = features.reshape(history_length, -1)
        return torch.nan_to_num(
            features_row,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )


class RandomReLUBase(TimeSeriesTransformerBase):
    def __init__(
        self,
        no_rcb_feature_sets=10,
        seed=0,
        **args,
    ):
        super(RandomReLUBase, self).__init__(**args)
        self.device = torch.cuda.current_device()
        self.no_feature_sets = no_rcb_feature_sets
        self.random_seed = seed

    def set_random_generators(self):
        torch.manual_seed(self.random_seed)

    def transform(self, X):
        self.set_random_generators()
        X_tensor = torch.as_tensor(
            X,
            device=self.device,
        )
        history_length = X_tensor.shape[0]
        features = torch.zeros(
            (
                history_length,
                len(self.lookbacks),
                self.no_feature_sets,
            ),
            device=self.device,
        )
        output = list()
        for l in self.lookbacks:
            X_batch = tensor_rolling_calcs(
                X_tensor,
                method="all",
                window_size=l,
                step_size=1,
                time_dimension=0,
                zero_padding=True,
            ).float()
            feature_weights = torch.normal(
                0,
                1,
                size=(
                    l,
                    self.no_feature_sets,
                ),
                device=self.device,
            )
            ## Size (history length,no_feature_sets)
            weighted_X = torch.matmul(X_batch, feature_weights).mean(axis=1)
            bias = torch.rand(
                size=(
                    history_length,
                    self.no_feature_sets,
                ),
                device=self.device,
            )
            relu_X = torch.clip(weighted_X + bias, 0)
            output.append(relu_X)
        features = torch.concat(output, axis=0)
        features_row = features.reshape(history_length, -1)
        return torch.nan_to_num(
            features_row,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
