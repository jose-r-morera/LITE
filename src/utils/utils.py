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
    
    xtrain = znormalisation(xtrain)
    xtest = znormalisation(xtest)

    return xtrain, ytrain, xtest, ytest


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
