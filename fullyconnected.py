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

graph = tf.Graph()

image_size = 28
num_labels = 10
train_subset = 10000
num_steps = 3001
batch_size = 128
num_nodes = 1024

def reformat(dataset, labels):
    dataset = dataset.reshape(-1, image_size * image_size)
    labels = (np.arange(num_labels)) == labels[:, None].astype(np.float32)
    #labels = labels.reshape(-1, num_labels).astype(np.float32)
    return dataset, labels

datasets = open("./notMNIST.pickle", 'rb')
#add encoding under tensorflow activated
#datasets = pickle.load(datasets)
datasets = pickle.load(datasets, encoding='latin1')

train_dataset = datasets["train_dataset"]
train_labels = datasets["train_labels"]
valid_dataset = datasets["valid_dataset"]
valid_labels = datasets["valid_labels"]
test_dataset = datasets["test_dataset"]
test_labels = datasets["test_labels"]
del datasets

# plt.imshow(train_dataset[0])
# plt.show()

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

with graph.as_default():
    #tf_train_dataset = tf.constant(train_dataset[:train_subset, :])
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
    #tf_train_labels = tf.constant(train_labels[:train_subset])
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    #valid_labels = tf.constant(valid_labels[:train_subset, :])
    tf_test_dataset = tf.constant(test_dataset)
    #test_labels = tf.constant(test_labels[:train_subset, :])

    weights_1 = tf.Variable(tf.truncated_normal([image_size * image_size, num_nodes]))
    biases_1 = tf.Variable(tf.zeros([num_nodes]))
    weights_2 = tf.Variable(tf.truncated_normal([num_nodes, num_labels]))
    biases_2 = tf.Variable(tf.zeros([num_labels]))

    logits_1 = tf.matmul(tf_train_dataset, weights_1) + biases_1
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits_1))
    relu_layer = tf.nn.relu(logits_1)
    logits_2 = tf.matmul(logits_1, weights_2) + biases_2
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits_2))

    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    train_prediction = tf.nn.softmax(logits_2)

    # valid_prediction = tf.nn.softmax(
    #     tf.matmul(tf_valid_dataset, weights_1) + biases_1
    # )
    # test_prediction = tf.nn.softmax(
    #     tf.matmul(tf_test_dataset, weights_1) + biases_1
    # )

    valid_pred_logits_1 = tf.matmul(tf_valid_dataset, weights_1) + biases_1
    valid_pred_relu_layer = tf.nn.relu(valid_pred_logits_1)
    valid_pred_logits_2 = tf.matmul(valid_pred_relu_layer, weights_2) + biases_2
    valid_prediction = tf.nn.softmax(valid_pred_logits_2)

    test_pred_logits_1 = tf.matmul(tf_test_dataset, weights_1) + biases_1
    test_pred_relu_layer = tf.nn.relu(test_pred_logits_1)
    test_pred_logits_2 = tf.matmul(test_pred_relu_layer, weights_2) + biases_2
    test_prediction = tf.nn.softmax(test_pred_logits_2)

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    for step in range(num_steps):
        #_, l, predictions = session.run([optimizer, loss, train_prediction])

        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset+batch_size), :]

        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

        if(step % 100 == 0):
            print('Minibatch loss at step %d: %f' % (step, l))
            minibatch_acc = accuracy(predictions, batch_labels)
            print('Minibatch accuracy: %.1f%%' %(minibatch_acc))

            train_acc = accuracy(predictions, train_labels[:train_subset, :])
            valid_acc = accuracy(valid_prediction.eval(), valid_labels)
            print("Validation accuracy: %.1f%%" % (valid_acc))

    test_acc = accuracy(test_prediction.eval(), test_labels)
    print("Test accuracy: %.1f%%" %(test_acc))