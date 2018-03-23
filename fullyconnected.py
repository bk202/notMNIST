from __future__ import print_function
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

datasets = pickle.load(open("./notMNIST.pickle", "rb"))

train_dataset = datasets["train_dataset"]
train_labels = datasets["train_labels"]
valid_dataset = datasets["valid_dataset"]
valid_labels = datasets["valid_labels"]
test_dataset = datasets["test_dataset"]
test_labels = datasets["test_labels"]
del datasets

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)