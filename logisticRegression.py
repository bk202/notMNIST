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
train_dataset = datasets['train_dataset']
train_labels = datasets['train_labels']
valid_dataset = datasets['valid_dataset']
valid_labels = datasets['valid_labels']
test_dataset = datasets['test_dataset']
test_labels = datasets['test_labels']
del datasets

training_samples = 500

test_start = 100
test_end = 120


logreg = LogisticRegression(C=1e5)

train_dataset = train_dataset.reshape((train_dataset.shape[0], -1))
valid_dataset = valid_dataset.reshape((valid_dataset.shape[0], -1))

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)

logreg.fit(train_dataset[:training_samples], train_labels[:training_samples])

prediction = logreg.predict(train_dataset[test_start:test_end])
print(prediction[:])
print(train_labels[test_start:test_end])