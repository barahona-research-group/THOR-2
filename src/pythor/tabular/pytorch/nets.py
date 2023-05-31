import joblib, os, shutil, datetime, json, glob
import logging, gc, copy

import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset

from pytorch_lightning import Trainer, LightningModule, seed_everything
from pytorch_lightning.callbacks import (
    LearningRateFinder,
    StochasticWeightAveraging,
    EarlyStopping,
    GradientAccumulationScheduler,
    ModelPruning,
    ModelCheckpoint,
)

"""
Apply Feature Neutralisation on predictions

Loss Func for ranking tasks 

1. MSE
2. Pearson Corr
3. Spearman Ranking Corr (Unlikley needed as Pearson)

"""
import torchsort


## Gaussian standardised data and apply symmetric cutoffs
def soft_clip(pred, cutoff=0, t=10):
    x = pred - pred.mean()
    x = x / x.norm()
    if cutoff > 0:
        clip_x = torch.sigmoid((x - cutoff) * t) + (1 - torch.sigmoid((x + cutoff) * t))
        return clip_x
    else:
        return x


def feature_neutralisation(Y, X, proportion=1):
    gram_mtx = torch.matmul(torch.linalg.pinv(X), Y)
    projected_values = Y - proportion * torch.matmul(X, gram_mtx)
    return projected_values


## ToDo: Refactor Spearman Correlation for mutiple targets, which gives a 3D tensor
## Needs to work on 2D tensors of size (1,N) as ranking as performed on the last dimension
def spearmanr(pred, target, **kw):
    pred = torchsort.soft_rank(pred, **kw)
    target = torchsort.soft_rank(target, **kw)
    return pearsonr(pred, target, **kw)


def pearsonr(pred, target, **kw):
    pred = soft_clip(pred)
    target = soft_clip(target)
    corr = -1 * (pred * target).sum()
    return corr


"""
PyTorch Lightning models for Tabular Data

1. Multi-Layer Perceptron 

    Architecture: Auto-Encoding FE Layers followed by Funnel-shaped Layers (As in Auto_PyTorch)
   
2, Sparse MLP 

    Sparsity: Replace Standard Linear Layers with Sparse Linear Layers 
    Architecture: Symmetric Layers with the middel layer the largest 


"""


class RankingNN(LightningModule):
    def __init__(self, config):

        super().__init__()
        self.config = config
        self.lr = config.get("learning_rate", 0.001)

        self.create_model()

        ## Need to have this to ensure correct hyper-parameters are loaded
        ## https://github.com/Lightning-AI/lightning/issues/3981
        self.save_hyperparameters()

    ## Create NN Model and save to self.layers
    def create_model(
        self,
    ):
        pass

    def forward(self, x):
        y_hat = self.layers(x.float())
        y_hat_standard = (y_hat - y_hat.mean()) / y_hat.std()
        proportion = self.config.get("proportion", 0.2)
        y_hat_neut = feature_neutralisation(
            y_hat_standard, x.float(), proportion=proportion
        )
        return y_hat_neut

    def calc_loss_func(self, batch, batch_idx):
        ## ToDo: Add more objects to batch if needed?
        ##
        x = batch[0]
        y = batch[1]
        y_hat = self.forward(x.float())
        loss_func_name = self.config.get("loss_func", "pearson")
        if loss_func_name == "spearman":
            loss = spearmanr(
                y_hat.t(),
                y.float().t(),
            )
        elif loss_func_name == "pearson":
            loss = pearsonr(
                y_hat,
                y.float(),
            )
        ## Default to use MSE Loss
        else:
            loss_func = nn.MSELoss()
            loss = loss_func(y_hat, y.float())
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.calc_loss_func(batch, batch_idx)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.calc_loss_func(batch, batch_idx)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        return loss

    ## We use data loaders during prediction so we need to obtain the batch information as follows
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch[0]
        return self.forward(x.float())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,  ## Automatically find by Tuner
            weight_decay=self.config.get(
                "Weights-L2", 0.001
            ),  ## Set L2 regularisation of network weights
        )
        return optimizer


class MLP(RankingNN):
    def __init__(self, config):
        super().__init__(config)

    def create_model(self):

        config = self.config
        input_neuron_sizes = config["input_shape"]
        num_encoding_layers = config.get("num_encoding_layers", 1)
        num_funnel_layers = config.get("num_funnel_layers", 2)
        feature_size = config.get("feature_size", 500)

        self.layers = nn.Sequential()

        ## Auto-Encoding Like layers
        for i in range(
            0,
            num_encoding_layers,
        ):

            self.layers.add_module(
                f"EncodingLayer{i+1}", nn.Linear(input_neuron_sizes, feature_size)
            )
            self.layers.append(nn.ReLU())
            self.layers.add_module(
                f"DecodingLayer{i+1}", nn.Linear(feature_size, input_neuron_sizes)
            )
            self.layers.append(nn.ReLU())
            if config.get("encoding_batchnorm", False):
                self.layers.append(nn.BatchNorm1d(input_neuron_sizes))
            dropout = config.get("encoding_dropout", 0.1)
            if dropout > 0:
                self.layers.append(nn.Dropout(p=dropout))
            feature_size = int(feature_size * config.get("encoding_neuron_scale", 0.8))

        ## Funnel MLP
        neuron_sizes = input_neuron_sizes
        for i in range(
            1,
            num_funnel_layers,
        ):
            new_neuron_sizes = int(
                neuron_sizes * config.get("funnel_neuron_scale", 0.8)
            )
            self.layers.add_module(
                f"FunnelLayer{i}", nn.Linear(neuron_sizes, new_neuron_sizes)
            )
            if config.get("funnel_batchnorm", False):
                self.layers.append(nn.BatchNorm1d(new_neuron_sizes))
            self.layers.append(nn.ReLU())
            dropout = config.get("funnel_dropout", 0.1)
            if dropout > 0:
                self.layers.append(nn.Dropout(p=dropout))
            neuron_sizes = new_neuron_sizes

        ## Final Output Layer
        self.layers.add_module(
            "OutputLayer", nn.Linear(neuron_sizes, config["output_shape"])
        )
        return None


class SparseMLP(RankingNN):
    def __init__(self, config):
        super().__init__(config)

    def create_model(self):

        import sparselinear as sl

        config = self.config
        input_neuron_sizes = config["input_shape"]
        num_encoding_layers = config.get("num_encoding_layers", 2)
        num_decoding_layers = config.get("num_decoding_layers", 2)

        self.layers = nn.Sequential()

        feature_size = input_neuron_sizes

        ## Encoding
        for i in range(
            0,
            num_encoding_layers,
        ):
            new_feature_size = int(
                feature_size * config.get("encoding_neuron_scale", 1.1)
            )
            encoding_layer_sparsity = config.get("encoding_layer_sparsity", 0.9)
            self.layers.add_module(
                f"EncodingLayer{i+1}",
                sl.SparseLinear(
                    feature_size, new_feature_size, sparsity=encoding_layer_sparsity
                ),
            )
            self.layers.append(nn.ReLU())
            feature_size = new_feature_size

        ## Decoding
        for i in range(
            1,
            num_decoding_layers,
        ):
            new_feature_size = int(
                feature_size * config.get("decoding_neuron_scale", 0.9)
            )
            decoding_layer_sparsity = config.get("decoding_layer_sparsity", 0.9)
            self.layers.add_module(
                f"DecodingLayer{i+1}",
                sl.SparseLinear(
                    feature_size, new_feature_size, sparsity=decoding_layer_sparsity
                ),
            )
            self.layers.append(nn.ReLU())
            feature_size = new_feature_size

        ## Final Output Layer
        self.layers.add_module(
            "OutputLayer", nn.Linear(feature_size, config["output_shape"])
        )
        return None
