#!/usr/bin/env python3
"""
Trains a loaded neural network model using mini-batch gradient descent
"""

import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5,
                     load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
    Trains a loaded neural network model using mini-batch gradient descent

    Parameters:
    X_train (np.ndarray): training data of shape (m, 784)
    Y_train (np.ndarray): training labels one-hot of shape (m, 10)
    X_valid (np.ndarray): validation data
    Y_valid (np.ndarray): validation labels one-hot
    batch_size (int): mini-batch size
    epochs (int): number of epochs
    load_path (str): path to load model
    save_path (str): path to save trained model

    Returns:
    str: path where the model was saved
    """
    saver = tf.train.import_meta_graph(load_path + '.meta')
    with tf.Session() as sess:
        saver.restore(sess, load_path)

        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        y = graph.get_tensor_by_name("y:0")
        accuracy = graph.get_tensor_by_name("accuracy:0")
        loss = graph.get_tensor_by_name("loss:0")
        train_op = graph.get_operation_by_name("train_op")

        m = X_train.shape[0]

        for epoch in range(epochs + 1):
            train_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            train_accuracy = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
            valid_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            valid_accuracy = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})

            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))

            if epoch == epochs:
                break

            X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)

            for step, start in enumerate(range(0, m, batch_size), 1):
                end = min(start + batch_size, m)
                X_batch = X_shuffled[start:end]
                Y_batch = Y_shuffled[start:end]

                sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})

                if step % 100 == 0 or end == m:
                    step_cost = sess.run(loss, feed_dict={x: X_batch, y: Y_batch})
                    step_accuracy = sess.run(accuracy, feed_dict={x: X_batch, y: Y_batch})
                    print("\tStep {}:".format(step))
                    print("\t\tCost: {}".format(step_cost))
                    print("\t\tAccuracy: {}".format(step_accuracy))

        save_path_final = saver.save(sess, save_path)

    return save_path_final
