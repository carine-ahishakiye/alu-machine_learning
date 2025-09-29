#!/usr/bin/env python3
"""
Builds, trains, and saves a neural network model in TensorFlow using:
Adam optimization, mini-batch gradient descent,
learning rate decay, and batch normalization.
"""

import numpy as np
import tensorflow as tf
create_batch_norm_layer = __import__('14-batch_norm').create_batch_norm_layer
learning_rate_decay = __import__('12-learning_rate_decay').learning_rate_decay


def model(Data_train, Data_valid, layers, activations, alpha=0.001,
          beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=1,
          batch_size=32, epochs=5, save_path='/tmp/model.ckpt'):
    """
    Builds, trains, and saves a neural network in TensorFlow
    """
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid
    m = X_train.shape[0]

    tf.set_random_seed(0)
    np.random.seed(0)

    x = tf.placeholder(tf.float32, shape=[None, X_train.shape[1]], name='x')
    y = tf.placeholder(tf.float32, shape=[None, Y_train.shape[1]], name='y')

    A_prev = x
    for i in range(len(layers)):
        activation = activations[i]
        A_prev = create_batch_norm_layer(A_prev, layers[i], activation)

    y_pred = A_prev
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred)
    tf.add_to_collection('loss', loss)

    global_step = tf.Variable(0, trainable=False)
    alpha_decay = learning_rate_decay(alpha, decay_rate, global_step, 1)

    optimizer = tf.train.AdamOptimizer(learning_rate=alpha_decay,
                                       beta1=beta1,
                                       beta2=beta2,
                                       epsilon=epsilon)
    train_op = optimizer.minimize(loss, global_step=global_step)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(epochs + 1):
            # Compute metrics for the epoch
            train_cost, train_pred = sess.run([loss, y_pred],
                                              feed_dict={x: X_train, y: Y_train})
            valid_cost, valid_pred = sess.run([loss, y_pred],
                                              feed_dict={x: X_valid, y: Y_valid})

            train_acc = np.mean(np.argmax(train_pred, axis=1) ==
                                np.argmax(Y_train, axis=1))
            valid_acc = np.mean(np.argmax(valid_pred, axis=1) ==
                                np.argmax(Y_valid, axis=1))

            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_acc))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_acc))

            if epoch == epochs:
                break

            # Shuffle training data
            permutation = np.random.permutation(m)
            X_shuffled = X_train[permutation]
            Y_shuffled = Y_train[permutation]

            num_batches = int(np.ceil(m / batch_size))
            step_number = 0

            for i in range(0, m, batch_size):
                step_number += 1
                X_batch = X_shuffled[i:i + batch_size]
                Y_batch = Y_shuffled[i:i + batch_size]

                _, step_pred, step_cost = sess.run([train_op, y_pred, loss],
                                                   feed_dict={x: X_batch, y: Y_batch})

                step_acc = np.mean(np.argmax(step_pred, axis=1) ==
                                   np.argmax(Y_batch, axis=1))

                if step_number % 100 == 0:
                    print("\tStep {}:".format(step_number))
                    print("\t\tCost: {}".format(step_cost))
                    print("\t\tAccuracy: {}".format(step_acc))

        save_path = saver.save(sess, save_path)

    return save_path
