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
import tensorflow as tf
import math as math
#from fullyconnected import reformat, accuracy


def reformat(dataset, labels):
    dataset = dataset.reshape(-1, image_size, image_size, num_channels)
    labels = (np.arange(num_labels)) == labels[:, None].astype(np.float32)
    #labels = labels.reshape(-1, num_labels).astype(np.float32)
    return dataset, labels

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

pickle_file = 'notMNIST.pickle'
image_size = 28
num_labels = 10
batch_size = 128
data_subset = 10000
num_steps = 3001
beta = 0.001

patch_size = 5
num_features = 16
num_channels = 1 #grey scale
num_hidden = 64

graph = tf.Graph()

f = open(pickle_file, 'rb')
datasets = pickle.load(f, encoding='latin1')
train_dataset = datasets['train_dataset']
train_labels = datasets['train_labels']
valid_dataset = datasets['valid_dataset']
valid_labels = datasets['valid_labels']
test_dataset = datasets['test_dataset']
test_labels = datasets['test_labels']
del datasets

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    '''weights'''
    cnn_weights_1 = tf.Variable(
        tf.truncated_normal([patch_size, patch_size, num_channels, num_features], stddev=0.1)
    )
    cnn_biases_1 = tf.Variable(
        tf.zeros(num_features)
    )
    cnn_weights_2 = tf.Variable(
        tf.truncated_normal([patch_size, patch_size, num_features, num_features], stddev=0.1)
    )
    cnn_biases_2 = tf.Variable(
        tf.zeros(num_features)
    )
    fully_weights_3 = tf.Variable(
        tf.truncated_normal([image_size // 4 * image_size // 4 * num_features, num_hidden], stddev=0.1)
        #1 additional CNN layer => duplicate base of division
        #so if CNN layers = 3 => image_size // 8
    )
    fully_biases_3 = tf.Variable(
        tf.zeros(num_hidden)
    )
    output_weights_4 = tf.Variable(
        tf.truncated_normal([num_hidden, num_labels], stddev=0.1)
    )
    output_biases_4 = tf.Variable(
        tf.zeros(num_labels)
    )

    '''Model'''
    def model(data):
        print("data shape: ", data.get_shape().as_list())
        conv = tf.nn.conv2d(data, cnn_weights_1, strides=[1, 1, 1, 1], padding='SAME')
        print("layer1 shape: ", conv.get_shape().as_list())
        hidden = tf.nn.relu(conv + cnn_biases_1)
        pool = tf.nn.max_pool(hidden, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        print("pool1 shape: ", pool.get_shape().as_list())
        conv = tf.nn.conv2d(pool, cnn_weights_2, [1, 1, 1, 1], padding='SAME')
        print("layer2 shape: ", conv.get_shape().as_list())
        hidden = tf.nn.relu(conv + cnn_biases_2)
        pool = tf.nn.max_pool(hidden, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        print("pool2 shape: ", pool.get_shape().as_list())
        shape = pool.get_shape().as_list()
        reshape = tf.reshape(pool, [shape[0], shape[1] * shape[2] * shape[3]])
        print("reshape: ", reshape.get_shape().as_list())
        hidden = tf.nn.relu(tf.matmul(reshape, fully_weights_3) + fully_biases_3)
        print("layer3 shape: ", hidden.get_shape().as_list())
        return tf.matmul(hidden, output_weights_4) + output_biases_4

    logits = model(tf_train_dataset)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

    '''Optimizer'''
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

    '''train prediction'''
    train_prediction = tf.nn.softmax(logits)

    '''validation prediction'''
    logits = model(tf_valid_dataset)
    valid_prediction = tf.nn.softmax(logits)

    '''test prediction'''
    logits = model(tf_test_dataset)
    test_prediction = tf.nn.softmax(logits)

# train_dataset = train_dataset[:500, :]
# train_labels = train_labels[:500]

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print("Initiaizlied")
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset: (offset + batch_size), :]
        batch_labels = train_labels[offset: (offset + batch_size), :]

        feed_dict = {
            tf_train_dataset: batch_data,
            tf_train_labels: batch_labels
        }
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

        if (step % 500 == 0):
            print('Minibatch loss at step %d: %f' % (step, l))
            minibatch_acc = accuracy(predictions, batch_labels)
            print('Minibatch accuracy: %.1f%%' % (minibatch_acc))

            train_acc = accuracy(predictions, train_labels)
            valid_acc = accuracy(valid_prediction.eval(), valid_labels)
            print("Validation accuracy: %.1f%%" % (valid_acc))

    test_acc = accuracy(test_prediction.eval(), test_labels)
    print("Test accuracy: %.1f%%" % (test_acc))