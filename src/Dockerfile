### Base Image from official PyTorch for CUDA 
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

### Update Essentials
### Build instructions: https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html#build-lightgbm
RUN apt-get update && \
    apt-get install -y build-essential cmake libboost-dev libboost-system-dev libboost-filesystem-dev ocl-icd-libopencl1 clinfo
RUN apt-get install -y git 

### Very Basic Data Science 
RUN pip install numpy scipy
RUN pip install pandas
RUN pip install scipy
RUN pip install scikit-learn statsmodels

### Build LightGBM 
RUN pip install lightgbm --install-option=--cuda

### Neural Network Models (PyTorch)
RUN pip install torch==1.13.1
RUN pip install pytorch-lightning
RUN pip install pytorch-tabnet --no-deps
RUN pip install cvxpylayers

### Neural Network Models (tenforflow)
RUN pip install tensorflow==2.12.*
RUN pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

### Time Series
RUN pip install darts

### Reinfrocement Learning 
RUN pip install gymnasium
RUN pip install "ray[rllib]"
RUN pip install "ray[tune]"
RUN pip install "ray[serve]"

### Sorting Algorithms in PyTorch 
### https://github.com/teddykoker/torchsort, Set up Env Variables for using 
ENV TORCH_CUDA_ARCH_LIST="Pascal;Volta;Turing;Ampere"
RUN pip install torchsort

### Feature Engineering 
### Needs to update when newer supporting torch1.11.0 on PyPI
RUN pip install signatory==1.2.6.1.9.0 --no-deps
### RUN pip install https://github.com/patrick-kidger/signatory/archive/master.zip --no-deps

### Hyper-parameter optimsiation
RUN pip install optuna 

## Sparse Neural Networks
RUN pip install hummingbird-ml
RUN pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
RUN pip install sparselinear


## Benchmark Models (Do not use the dependency specified as conflicts)
RUN pip install autogluon --no-deps
RUN pip install river --no-deps

## Data Parallel
RUN pip install "dask[complete]"

## Jupyter notebooks 
RUN pip install ipykernel jupyterlab notebook 

## Ensembles for Deep Learning (For Future Use)
#RUN pip install tensorflow_decision_forests --upgrade
#RUN pip install torchensemble



### Install Competition API
RUN pip install numerapi==2.12.9 --no-deps
RUN pip install --upgrade crunchdao --no-deps

### Build THOR 
WORKDIR /
COPY pythor/ pythor/
COPY setup.py setup.py
RUN pip install .
WORKDIR /workspace 