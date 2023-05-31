from functools import wraps


import joblib, os, glob, copy
import pandas as pd
import numpy as np

import torch


import scipy.stats


"""
Official Scoring functions from Numerai 

ranked_preds are predictions that are between -0.5 to 0.5 
centered_target is the given Numerai target that is between -2 and 2

"""


def numerai_corr(preds, target):
    ranked_preds = (preds.rank(method="average").values - 0.5) / preds.shape[0]
    gauss_ranked_preds = scipy.stats.norm.ppf(ranked_preds)
    # make targets centered around 0. This assumes the targets have a mean of 0.5
    centered_target = target - 0.5
    # raise both preds and target to the power of 1.5 to accentuate the tails
    preds_p15 = np.sign(gauss_ranked_preds) * np.abs(gauss_ranked_preds) ** 1.5
    target_p15 = np.sign(centered_target) * np.abs(centered_target) ** 1.5
    # finally return the Pearson correlation
    return tail_corr(
        np.array(preds_p15).astype(float), np.array(target_p15).astype(float)
    )


"""
Matrix Algorithms

Implemented in CUDA pipelines 


    - Cross Correlation Matrix 
    - Ridge Regression 
    - Tail Risk
    - Matrix Projection 


"""


"""
Allow for Backward comptability with numpy 

"""


def __cast_to_torch(
    output_numpy,
    collection_of_args,
):
    if type(collection_of_args) == list or type(collection_of_args) == tuple:
        output = list()
        for x in collection_of_args:
            if type(x) == np.ndarray or type(x) == torch.Tensor:
                if torch.cuda.is_available():
                    output.append(
                        torch.as_tensor(x, device=torch.cuda.current_device())
                    )
                else:
                    output.append(torch.as_tensor(x, device=None))
                if type(x) == np.ndarray:
                    output_numpy = True
            else:
                output.append(x)
        return output, output_numpy

    if type(collection_of_args) == dict:
        output = dict()
        for k, x in collection_of_args.items():
            if type(x) == np.ndarray or type(x) == torch.Tensor:
                if torch.cuda.is_available():
                    output[k] = torch.as_tensor(x, device=torch.cuda.current_device())
                else:
                    output[k] = torch.as_tensor(x, device=None)
                if type(x) == np.ndarray:
                    output_numpy = True
            else:
                output[k] = x
        return output, output_numpy


def numpybackward(func):
    @wraps(func)
    def numpy_to_torch(
        *args,
        **kwargs,
    ):
        ## Cast all Numpy Arrays to Torch Tensors
        output_numpy = False
        args_copy, output_numpy, = __cast_to_torch(
            output_numpy,
            args,
        )
        kwargs_copy, output_numpy, = __cast_to_torch(
            output_numpy,
            kwargs,
        )
        ## Return Numpy Array if any of the results is a numpy
        if output_numpy:
            return func(
                *args_copy,
                **kwargs_copy,
            ).numpy(force=True)
        else:
            return func(
                *args_copy,
                **kwargs_copy,
            )

    return numpy_to_torch


@numpybackward
def ridge_solve(
    X,
    Y,
    alphas=[
        0.01,
        0.1,
        1,
        10,
        100,
    ],
):
    U, S, V = torch.linalg.svd(X.T @ X)
    y_tilde = V @ (X.T @ Y)
    b = torch.zeros((len(alphas), X.shape[1], Y.shape[1]))
    for i in range(len(alphas)):
        temp = torch.diag(torch.reciprocal(S + alphas[i])) @ y_tilde
        b[i, :, :] = U @ temp
    return b


@numpybackward
def cross_correlation_mtx(X, Y):
    ## X: N*A Y: N*B
    ## Output A*B
    EXY = torch.matmul(X.transpose(0, 1), Y) / X.shape[0]
    EXEY = torch.matmul(
        torch.mean(X, axis=0).reshape(-1, 1), torch.mean(Y, axis=0).reshape(1, -1)
    )
    VarXVarY = torch.matmul(
        torch.std(X, axis=0).reshape(-1, 1), torch.std(Y, axis=0).reshape(1, -1)
    )
    cov = EXY - EXEY
    correction = X.shape[0] / (X.shape[0] - 1)
    return cov / VarXVarY * correction


@numpybackward
## Input X (N,K) or (N,) Input y (N,1) or (N,) output (K,1)
def tail_corr(X, y, threshold=1):
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    if threshold < 1:
        ranks = (y.argsort().argsort() - 0.5) / y.shape[0]
        tail_ranks = torch.where((ranks >= 1 - threshold) | (ranks <= threshold))[0]
        return cross_correlation_mtx(X[tail_ranks], y[tail_ranks])
    else:
        return cross_correlation_mtx(X, y)


"""
Feature Neutralisation 

Solve Matrix Projection Problem 

"""


@numpybackward
## Input Y(N,K) X(N,M) output (N,K)
def matrix_project(
    Y,
    X,
    proportion=1,
    scale=True,
):
    gram_mtx = torch.matmul(torch.linalg.pinv(X), Y)
    projected_values = Y - proportion * torch.matmul(X, gram_mtx)
    if scale:
        ranks = projected_values.argsort(axis=0,).argsort(
            axis=0,
        )
        return ranks
    else:
        return projected_values


"""
Ranking Predictions 

"""


@numpybackward
##
def normalise_preds(preds, axis=1, cutoff=0, negcutoff=None):
    preds = torch.nan_to_num(preds, posinf=0, neginf=0)
    if len(preds.shape) > 1:
        ranks = ((preds.argsort(axis).argsort(axis) - 0.5) / preds.shape[axis]) - 0.5
    else:
        ranks = ((preds.argsort(0).argsort(0) - 0.5) / preds.shape[0]) - 0.5
    if negcutoff is None:
        negcutoff = -1 * cutoff
    if cutoff >= 0:
        ranks_truncated = ranks - torch.clip(ranks, negcutoff, cutoff)
    else:
        ranks_truncated = torch.clip(ranks, -1 * negcutoff, -1 * cutoff)
    return ranks_truncated


"""
Rolling Statistics 

Calculating Rolling Statistics in PyTorch 

https://stackoverflow.com/questions/63361688/rolling-statistics-in-numpy-or-pytroch


"""


@numpybackward
## For 2D tensors or higher dimensional tensors
def tensor_rolling_calcs(
    data, method="std", window_size=20, step_size=1, time_dimension=0, zero_padding=True
):
    if zero_padding:
        zeros = torch.zeros(
            window_size - 1, *data.shape[1:], device=torch.cuda.current_device()
        )
        processed_data = torch.cat([zeros, data], dim=time_dimension)
    tensor_rolled_data = processed_data.unfold(
        dimension=time_dimension, size=window_size, step=step_size
    )
    batch_dimension = -1
    if method == "all":
        return tensor_rolled_data
    if method == "mean":
        return tensor_rolled_data.mean(dim=batch_dimension)
    if method in [
        "std",
        "vol",
    ]:
        return tensor_rolled_data.std(dim=batch_dimension)
    if method in [
        "skew",
        "kurt",
    ]:
        mean = tensor_rolled_data.mean(dim=batch_dimension).unsqueeze(batch_dimension)
        diffs = tensor_rolled_data - mean
        var = torch.mean(torch.pow(diffs, 2.0), dim=batch_dimension).unsqueeze(
            batch_dimension
        )
        std = torch.pow(var, 0.5)
        zscores = diffs / std
        print(zscores.shape)
        if method == "skew":
            skews = torch.mean(torch.pow(zscores, 3.0), dim=batch_dimension)
            return skews
        elif method == "kurt":
            kurtoses = torch.mean(torch.pow(zscores, 4.0), dim=batch_dimension) - 3.0
            return kurtoses
