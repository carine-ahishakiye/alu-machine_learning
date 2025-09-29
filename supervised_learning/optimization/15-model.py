#!/usr/bin/env python3
"""15-model.py"""
import numpy as np
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Creates a batch normalization layer in TensorFlow"""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    dense = tf.layers.Dense(units=n, kernel_initializer=init)(prev)
    gamma = tf.Variable(tf.ones([n]), trainable=True)
    beta = tf.Variable(tf.zeros([n]), trainable=True)
    mean, variance = tf.nn.moments(dense, axes=[0])
    bn = tf.nn.batch_normalization(dense, mean, variance, beta, gamma, 1e-8)
    if activation is not None:
        bn = activation(bn)
    return bn


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """Creates an Adam optimization operation"""
    optimizer = tf.train.AdamOptimizer(learning_rate=alpha,
                                       beta1=beta1,
                                       beta2=beta2,
                                       epsilon=epsilon)
    return optimizer.minimize(loss)


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Creates an inverse time decay learning rate in TensorFlow"""
    return tf.train.inverse_time_decay(alpha, global_step,
                                       decay_step,
                                       decay_rate,
                                       staircase=True)


def model(Data_train, Data_valid, layers, activations, alpha=0.001,
          beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=1,
          batch_size=32, epochs=5, save_path='/tmp/model.ckpt'):
    """Builds, trains, and saves a neural network model"""
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid
    n_x = X_train.shape[1]
    n_y = Y_train.shape[1]

    tf.set_random_seed(0)
    np.random.seed(0)

    x = tf.placeholder(tf.float32, shape=[None, n_x])
    y = tf.placeholder(tf.float32, shape=[None, n_y])

    prev = x
    for i, (n, act) in enumerate(zip(layers, activations)):
        prev = create_batch_norm_layer(prev, n, act)

    y_pred = prev
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred)

    global_step = tf.Variable(0, trainable=False)
    alpha_op = learning_rate_decay(alpha, decay_rate, global_step, 1)
    train_op = tf.train.AdamOptimizer(learning_rate=alpha_op,
                                      beta1=beta1,
                                      beta2=beta2,
                                      epsilon=epsilon).minimize(loss,
                                                                global_step=global_step)

    correct_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        m = X_train.shape[0]

        for epoch in range(epochs + 1):
            # Evaluate training and validation cost/accuracy
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

            # Shuffle the training data
            perm = np.random.permutation(m)
            X_shuffled = X_train[perm]
            Y_shuffled = Y_train[perm]

            num_batches = int(np.ceil(m / batch_size))
            step_number = 0

            for batch in range(num_batches):
                start = batch * batch_size
                end = min(start + batch_size, m)
                X_batch = X_shuffled[start:end]
                Y_batch = Y_shuffled[start:end]

                step_number += 1
                sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})

                if step_number % 100 == 0:
                    step_cost, step_acc = sess.run([loss, accuracy],
                                                   feed_dict={x: X_batch,
                                                              y: Y_batch})
                    print("\tStep {}:".format(step_number))
                    print("\t\tCost: {}".format(step_cost))
                    print("\t\tAccuracy: {}".format(step_acc))

        return saver.save(sess, save_path)
