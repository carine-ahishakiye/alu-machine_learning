#!/usr/bin/env python3
"""
15-model.py
Builds, trains, and saves a neural network in TensorFlow using Adam optimization,
mini-batch gradient descent, learning rate decay, and batch normalization.
"""

import numpy as np
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Creates a batch normalization layer in TensorFlow."""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    dense = tf.layers.Dense(units=n, kernel_initializer=init, use_bias=False)
    Z = dense(prev)
    gamma = tf.Variable(tf.ones([1, n]), trainable=True)
    beta = tf.Variable(tf.zeros([1, n]), trainable=True)
    Z_norm = tf.layers.batch_normalization(Z, momentum=0.9, epsilon=1e-8,
                                           center=True, scale=True)
    return activation(Z_norm) if activation else Z_norm


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Creates a learning rate decay operation using inverse time decay."""
    return tf.train.inverse_time_decay(alpha, global_step, decay_step, decay_rate,
                                       staircase=True)


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """Creates the Adam optimization operation in TensorFlow."""
    return tf.train.AdamOptimizer(learning_rate=alpha, beta1=beta1,
                                  beta2=beta2, epsilon=epsilon).minimize(loss)


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """Builds, trains, and saves a neural network model in TensorFlow."""
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid
    m, _ = X_train.shape

    tf.set_random_seed(0)
    x = tf.placeholder(tf.float32, shape=[None, X_train.shape[1]])
    y = tf.placeholder(tf.float32, shape=[None, Y_train.shape[1]])

    prev = x
    for n_nodes, activation in zip(layers, activations):
        prev = create_batch_norm_layer(prev, n_nodes, activation)
    y_pred = prev

    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred)
    global_step = tf.Variable(0, trainable=False)
    alpha_decay = learning_rate_decay(alpha, decay_rate, global_step, 1)
    train_op = create_Adam_op(loss, alpha_decay, beta1, beta2, epsilon)

    correct_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(epochs + 1):
            # Shuffle training data
            perm = np.random.permutation(m)
            X_shuffled = X_train[perm]
            Y_shuffled = Y_train[perm]

            # Evaluate at start of epoch
            train_cost, train_acc = sess.run([loss, accuracy],
                                             feed_dict={x: X_train, y: Y_train})
            valid_cost, valid_acc = sess.run([loss, accuracy],
                                             feed_dict={x: X_valid, y: Y_valid})

            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_acc))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_acc))

            if epoch == epochs:
                break

            # Mini-batch training
            steps = 0
            for start in range(0, m, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                Y_batch = Y_shuffled[start:end]
                steps += 1
                sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})

                if steps % 100 == 0:
                    step_cost, step_acc = sess.run([loss, accuracy],
                                                   feed_dict={x: X_batch, y: Y_batch})
                    print("\tStep {}:".format(steps))
                    print("\t\tCost: {}".format(step_cost))
                    print("\t\tAccuracy: {}".format(step_acc))

        save_path = saver.save(sess, save_path)
        return save_path
