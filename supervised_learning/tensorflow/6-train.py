#!/usr/bin/env python3
"""
Trains a neural network classifier and saves the model.
"""
import tensorflow as tf

create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_train_op = __import__('5-create_train_op').create_train_op


def train(X_train, Y_train, X_valid, Y_valid,
          layer_sizes, activations, alpha, iterations,
          save_path="/tmp/model.ckpt"):
    """
    Builds, trains, and saves a neural network classifier.

    Returns:
        The path where the model was saved.
    """
    # Create placeholders
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])

    # Forward propagation
    y_pred = forward_prop(x, layer_sizes, activations)

    # Loss and accuracy
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)

    # Training operation
    train_op = create_train_op(loss, alpha)

    # Add to graph collections
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)

    # Start session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(iterations + 1):
            # Compute training and validation metrics
            train_cost, train_acc = sess.run(
                [loss, accuracy], feed_dict={x: X_train, y: Y_train})
            valid_cost, valid_acc = sess.run(
                [loss, accuracy], feed_dict={x: X_valid, y: Y_valid})

            # Print metrics at intervals
            if i % 100 == 0 or i == iterations:
                print(f"After {i} iterations:")
                print(f"\tTraining Cost: {train_cost}")
                print(f"\tTraining Accuracy: {train_acc}")
                print(f"\tValidation Cost: {valid_cost}")
                print(f"\tValidation Accuracy: {valid_acc}")

            # Perform training step
            if i < iterations:
                sess.run(train_op, feed_dict={x: X_train, y: Y_train})

        # Save the model
        saver = tf.train.Saver()
        save_path = saver.save(sess, save_path)

    return save_path
