import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from aeon.datasets import load_classification

def create_directory(directory_path):

    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)


def load_data(file_name, differentiate=True):
    
    xtrain, ytrain = load_classification(name=file_name, split="train")
    xtest, ytest = load_classification(name=file_name, split="test")

    # aeon to keras shape (N, T, C)
    xtrain = np.swapaxes(xtrain, axis1=1, axis2=2)
    xtest = np.swapaxes(xtest, axis1=1, axis2=2)
    
    ytrain, ytest = encode_labels(ytrain, ytest)
    
    # Channel 1: Global z-normalization of raw data
    xtrain_c1, train_mean_c1, train_std_c1 = global_znormalisation(xtrain)
    xtest_c1, _, _ = global_znormalisation(xtest, mean=train_mean_c1, std=train_std_c1)
    
    if differentiate:
        # Channel 2: Derive -> Global z-normalization
        xtrain_derived = derive(xtrain) # Derived from raw (swapped) data
        xtest_derived = derive(xtest)
        
        xtrain_c2, train_mean_c2, train_std_c2 = global_znormalisation(xtrain_derived)
        xtest_c2, _, _ = global_znormalisation(xtest_derived, mean=train_mean_c2, std=train_std_c2)
        
        # Concatenate channels: (N, T, 2)
        xtrain = np.concatenate([xtrain_c1, xtrain_c2], axis=-1)
        xtest = np.concatenate([xtest_c1, xtest_c2], axis=-1)
    else:
        # Single channel: (N, T, 1)
        xtrain = xtrain_c1
        xtest = xtest_c1

    return xtrain, ytrain, xtest, ytest


def derive(x):
    """
    Computes the first-order derivative (gradient) of the time series along the time axis.
    Uses np.gradient to maintain the same shape as the input.
    """
    return np.gradient(x, axis=1)


def global_min_max_normalisation(x, x_min=None, x_max=None):
    
    # If min and max are not provided, calculate them globally
    if x_min is None or x_max is None:
        x_min = np.min(x)
        x_max = np.max(x)
        
    # Prevent division by zero if the entire dataset is a constant value
    range_val = x_max - x_min
    if range_val == 0.0:
        range_val = 1.0
        
    x_norm = (x - x_min) / range_val
    return x_norm, x_min, x_max


def global_znormalisation(x, mean=None, std=None):
    
    # If mean and std are not provided, calculate them globally
    if mean is None or std is None:
        mean = np.mean(x)
        std = np.std(x)
        
    # Prevent division by zero if the entire dataset is a constant value
    if std == 0.0:
        std = 1.0
        
    x_norm = (x - mean) / std
    return x_norm, mean, std

def znormalisation(x):
    stds = np.std(x, axis=1, keepdims=True)
    if len(stds[stds == 0.0]) > 0:
        stds[stds == 0.0] = 1.0
        return (x - x.mean(axis=1, keepdims=True)) / stds
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True))

def encode_labels(ytrain, ytest):

    labenc = LabelEncoder()
    labenc.fit(ytrain)
    ytain_enc = labenc.transform(ytrain)
    ytest_enc = labenc.transform(ytest)
    
    return ytain_enc, ytest_enc
