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
    dataset = dataset.reshape(-1, image_size * image_size)
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
num_steps = 14501
beta = 0.001

hidden_nodes_1 = 1024
hidden_nodes_2 = int((hidden_nodes_1 * np.power(0.5, 1)))
hidden_nodes_3 = int((hidden_nodes_1 * np.power(0.5, 2)))
hidden_nodes_4 = int((hidden_nodes_1 * np.power(0.5, 3)))
hidden_nodes_5 = int((hidden_nodes_1 * np.power(0.5, 4)))

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
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

    '''weights'''
    #layer 1
    weights_1 = tf.Variable(
        tf.truncated_normal([image_size * image_size, hidden_nodes_1], stddev=math.sqrt(2.0/(image_size*image_size)))
    )
    biases_1 = tf.Variable(tf.zeros(hidden_nodes_1))

    #layer 2
    weights_2 = tf.Variable(
        tf.truncated_normal([hidden_nodes_1, hidden_nodes_2], stddev=math.sqrt(2.0/hidden_nodes_1))
    )
    biases_2 = tf.Variable(tf.zeros(hidden_nodes_2))

    #layer 3
    weights_3 = tf.Variable(
        tf.truncated_normal([hidden_nodes_2, hidden_nodes_3], stddev=math.sqrt(2.0/hidden_nodes_2))
    )
    biases_3 = tf.Variable(tf.zeros(hidden_nodes_3))

    #layer 4
    weights_4 = tf.Variable(
        tf.truncated_normal([hidden_nodes_3, hidden_nodes_4], stddev=math.sqrt(2.0/hidden_nodes_3))
    )
    biases_4 = tf.Variable(tf.zeros(hidden_nodes_4))

    #layer 5
    weights_5 = tf.Variable(
        tf.truncated_normal([hidden_nodes_4, hidden_nodes_5], stddev=math.sqrt(2.0/hidden_nodes_4))
    )
    biases_5 = tf.Variable(tf.zeros(hidden_nodes_5))

    #output layer
    weights_6 = tf.Variable(
        tf.truncated_normal([hidden_nodes_5, num_labels], stddev=math.sqrt(2.0/hidden_nodes_5))
    )
    biases_6 = tf.Variable(tf.zeros(num_labels))

    '''Model training'''
    logits_1 = tf.matmul(tf_train_dataset, weights_1) + biases_1
    relu_layer = tf.nn.relu(logits_1)
    relu_layer_dropout = tf.nn.dropout(relu_layer, 0.5)
    logits_2 = tf.matmul(relu_layer_dropout, weights_2) + biases_2
    relu_layer = tf.nn.relu(logits_2)
    relu_layer_dropout = tf.nn.dropout(relu_layer, 0.5)
    logits_3 = tf.matmul(relu_layer_dropout, weights_3) + biases_3
    relu_layer = tf.nn.relu(logits_3)
    relu_layer_dropout = tf.nn.dropout(relu_layer, 0.5)
    logits_4 = tf.matmul(relu_layer_dropout, weights_4) + biases_4
    relu_layer = tf.nn.relu(logits_4)
    relu_layer_dropout = tf.nn.dropout(relu_layer, 0.5)
    logits_5 = tf.matmul(relu_layer_dropout, weights_5) + biases_5
    relu_layer = tf.nn.relu(logits_5)
    relu_layer_dropout = tf.nn.dropout(relu_layer, 0.5)
    '''output layer'''
    logits_6 = tf.matmul(relu_layer_dropout, weights_6) + biases_6

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits_6))
    regularizers = tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2) + tf.nn.l2_loss(weights_3) + tf.nn.l2_loss(weights_4) + tf.nn.l2_loss(weights_5) + tf.nn.l2_loss(weights_6)
    loss = tf.reduce_mean(loss + beta * regularizers)

    '''Optimizer'''
    #Decaying learning rate
    # global_step = tf.Variable(0)
    # start_learning_rate = 0.5
    # learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, 100000, 0.96, staircase=True)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    #difference: 93.1% vs 92.8%

    '''train prediction'''
    train_prediction = tf.nn.softmax(logits_6)

    '''validation prediction'''
    logits_1 = tf.matmul(tf_valid_dataset, weights_1) + biases_1
    relu_layer = tf.nn.relu(logits_1)
    logits_2 = tf.matmul(relu_layer, weights_2) + biases_2
    relu_layer = tf.nn.relu(logits_2)
    logits_3 = tf.matmul(relu_layer, weights_3) + biases_3
    relu_layer = tf.nn.relu(logits_3)
    logits_4 = tf.matmul(relu_layer, weights_4) + biases_4
    relu_layer = tf.nn.relu(logits_4)
    logits_5 = tf.matmul(relu_layer, weights_5) + biases_5
    relu_layer = tf.nn.relu(logits_5)
    '''output layer'''
    logits_6 = tf.matmul(relu_layer, weights_6) + biases_6

    valid_prediction = tf.nn.softmax(logits_6)

    '''test prediction'''
    logits_1 = tf.matmul(tf_test_dataset, weights_1) + biases_1
    relu_layer = tf.nn.relu(logits_1)
    logits_2 = tf.matmul(relu_layer, weights_2) + biases_2
    relu_layer = tf.nn.relu(logits_2)
    logits_3 = tf.matmul(relu_layer, weights_3) + biases_3
    relu_layer = tf.nn.relu(logits_3)
    logits_4 = tf.matmul(relu_layer, weights_4) + biases_4
    relu_layer = tf.nn.relu(logits_4)
    logits_5 = tf.matmul(relu_layer, weights_5) + biases_5
    relu_layer = tf.nn.relu(logits_5)
    '''output layer'''
    logits_6 = tf.matmul(relu_layer, weights_6) + biases_6

    test_prediction = tf.nn.softmax(logits_6)

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
