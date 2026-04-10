import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from aeon.datasets import load_classification

def create_directory(directory_path):

    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)


def load_data(file_name):
    
    xtrain, ytrain = load_classification(name=file_name, split="train")
    xtest, ytest = load_classification(name=file_name, split="test")

    # aeon to keras shape
    xtrain = np.swapaxes(xtrain, axis1=1, axis2=2)
    xtest = np.swapaxes(xtest, axis1=1, axis2=2)
    
    ytrain, ytest = encode_labels(ytrain, ytest)
    
    # Calculate global constants on train, apply to train
    xtrain, train_min, train_max = global_min_max_normalisation(xtrain)
    
    # Reuse train constants to apply to test
    xtest, _, _ = global_min_max_normalisation(xtest, x_min=train_min, x_max=train_max)

    return xtrain, ytrain, xtest, ytest


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
