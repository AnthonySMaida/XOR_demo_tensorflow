#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 09:50:54 2017
Revised: June 17, 2017 to include TensorBoard
Tested in TensorFlow 1.1.

@author: maida

2-layer XOR network.
Architecture and known solution wts can be found in Goodfellow et al (2016), Chpt 6
When weights are initialized near known solution, training converges to
zero loss.
When weights are initialized to good practice values, loss does not go
to zero. This is an example of a network, despite its small size, that is
hard to train.
"""

import numpy as np
import tensorflow as tf

LOGDIR = "/tmp/xor_demo/"

num_steps = 20

# initialize wts  and biases near a known solution
wts1 = np.array([[1.2, 1.1],
                 [1, 0.9]], dtype=np.float32)

bias1 = np.array([[ 0.1],
                  [-1]], dtype=np.float32)

wts2 = np.array([[ 1.1, -2]], dtype=np.float32)

bias2 = np.array([[0]], dtype=np.float32)

def layer1(input, name="pretrainedL1"):
    """ Supports custom weight initialization for layer 1."""
    w = tf.Variable(tf.convert_to_tensor(wts1, dtype=tf.float32), name="W")
    b = tf.Variable(tf.convert_to_tensor(bias1, dtype=tf.float32), name="B")
    h = tf.matmul(w, input) + b
    h = tf.nn.relu(h)
    tf.summary.histogram("weights", w)
    tf.summary.histogram("biases", b)
    tf.summary.histogram("activations", h)
    return h

def layer2(input, name="pretrainedL2"):
    """ Supports custom weight initialization for layer 2. """
    w = tf.Variable(tf.convert_to_tensor(wts2, dtype=tf.float32), name="W")
    b = tf.Variable(tf.convert_to_tensor(bias2, dtype=tf.float32), name="B")
    h = tf.matmul(w, input) + b
    tf.summary.histogram("weights", w)
    tf.summary.histogram("biases", b)
    tf.summary.histogram("activations", h)
    return h

def layer(input, size_in, size_out, name="L", act=False):
  """ Creates a fully connected layer, with wts and biases.
      input:    input to the layer.
      size_in:  number of cols in the wt matrix created.
      size_out: number of rows in the wt matrix created.
      name:     used by TensorBoard; 'L' stands for layer.
      act:      If True, include a relu activation function.
      
      Uses broadcasting to pad bias layer from 2x1 to 2x4.
      This allows network to handle 4 training examples at once, 
      but does not duplicate the bias weights."""

  with tf.name_scope(name):
    w = tf.Variable(tf.truncated_normal([size_out,size_in],mean=1.0,stddev=0.2), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[size_out,1]), name="B")
    print('input shape: ', input.get_shape())
    print('w shape: ', w.get_shape())
    print('b shape: ', b.get_shape())
    h = tf.matmul(w, input) + b # uses broadcasting
    if act:
        h = tf.nn.relu(h)
    tf.summary.histogram("weights", w)
    tf.summary.histogram("biases", b)
    tf.summary.histogram("activations", h)
    print('h shape: ', h.get_shape())
    return h
        
def buildModel():
    """ Build a 2-layer network. Two units in the hidden layer with relu
        activation. One unit in the output layer which is linear. 
        All four training examples are presented at once."""
    tf.reset_default_graph()
    # Define unlabeled and labeled data.
    # Transposed design matrix
    # This program doesn't do randomization or shuffling of inputs (problematic?)
    train_dataset = tf.constant([[0, 0, 1, 1],
                                 [0, 1, 0, 1]], dtype=tf.float32)
    # Transposed labels
    train_labels  = tf.constant([[0, 1, 1, 0]], dtype=tf.float32)

    print('train_dataset shape: ', train_dataset.get_shape().as_list())
    
    # Build the 2-layer network
    # Comment out the two lines below to turn off good practice weight 
    # initialization
    h1  = layer(train_dataset, 2, 2, act=True)
    out = layer(h1, 2, 1)
    
# Uncomment the two lines below to support custom weight initialization
#    h1  = layer1(train_dataset)
#    out = layer2(h1)
    
    # Define mean-squared-error loss function
    with tf.name_scope("loss"):
      print('out shape: ', out.get_shape())
      print('labels shape: ', train_labels.get_shape())
      loss = tf.reduce_mean(tf.square(out - train_labels))
      tf.summary.scalar("loss", loss)
      
    # record the output for the first training sample
    with tf.name_scope("output"):
      o1 = out[0,0]
      tf.summary.scalar("o1", o1)
      o2 = out[0,1]
      tf.summary.scalar("o2", o2)
      o3 = out[0,2]
      tf.summary.scalar("o3", o3)
      o4 = out[0,3]
      tf.summary.scalar("o4", o4)
      
    # Choose a gradient descent optimizer
    with tf.name_scope("train"):
      train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
    
    # return items needed for training the model coded as a dictionary
    return {'ts': train_step, 'loss': loss, 'out': out}

def runModel(model):
    """ Unpack the model dictionary, start a session,
        and proceed with training. """
    train_step = model['ts']
    loss = model['loss']
    out  = model['out']
    
    # Create and initialize a (training) session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    # Create graph summary
    writer = tf.summary.FileWriter(LOGDIR + "1")
    writer.add_graph(sess.graph)
    summ = tf.summary.merge_all()

    # Perform 20 steps of training
    for step in range(num_steps):
       _, l, predictions, s = sess.run([train_step, loss, out, summ])
       print('Loss at step %d: %f' % (step, l))
       print(predictions)
       writer.add_summary(s, step) # doesn't seem to work

if __name__ == '__main__':
    runModel(buildModel())

#model = tf.Graph()
#with model.as_default():
#    train_dataset = tf.constant([[0,0],[0,1],[1,0],[1,1]], dtype=tf.float32, name='input')
#    train_labels  = tf.constant([[0,1,1,0]], dtype=tf.float32, name='label')
#    W_layer1      = tf.Variable(tf.truncated_normal([2,2], mean=0.0, stddev=0.2), name='W_layer1')
#    w_layer2      = tf.Variable(tf.truncated_normal([2,1], stddev=0.2), name='w_layer2')
##    bias_layer1   = tf.Variable(tf.zeros([2,1]), name='bias_layer1')
#    bias_layer1   = tf.Variable(tf.zeros([2,4]), name='bias_layer1')
##    bias_layer2   = tf.Variable(tf.zeros([1,1]), name='bias_layer2')
#    bias_layer2   = tf.Variable(tf.zeros([1,4]), name='bias_layer2')
#    
#    net_in_layer1 = tf.matmul(tf.transpose(W_layer1),tf.transpose(train_dataset)) + bias_layer1
#    print("layer1 dims: ", net_in_layer1.get_shape().as_list())
#    activation_h1 = tf.nn.relu(net_in_layer1)
#    
#    net_in_layer2 = tf.matmul(tf.transpose(w_layer2),activation_h1) + bias_layer2
#    loss = tf.reduce_mean(tf.square(net_in_layer2-train_labels))
#    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
#    
#
#
#with tf.Session(graph=model) as sess:
##    tf.initialize_all_variables().run()  # deprecated
#    tf.global_variables_initializer().run() # only works in the most recent version of tf
#    print('Initialized')
#    print("TF version: ", tf.__version__)
#    print('W_layer1: \n', W_layer1.eval())
#    print('bias_layer1: \n', bias_layer1.eval())
#    print('w_layer2: \n', w_layer2.eval())
#    print('bias_layer2: \n', bias_layer2.eval())
#    
#    # Create graph summary
#    writer = tf.summary.FileWriter("/tmp/xor_demo/1")
#    writer.add_graph(sess.graph)
#
#    for step in range(num_steps):
#        _, l, predictions = sess.run([optimizer, loss, net_in_layer2])
#        print('\nLoss at step %d: %f' % (step, l))
#        print('W_layer1: \n', W_layer1.eval())
#        print('bias_layer1: \n', bias_layer1.eval())
#        print('w_layer2: \n', w_layer2.eval())
#        print('bias_layer2: \n', bias_layer2.eval())
#
#        print(predictions)
##        print('Training accuracy: %.1f%%' % accuracy(predictions, train_labels))













