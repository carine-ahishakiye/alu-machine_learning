#!/usr/bin/env python3
import tensorflow as tf

def calculate_loss(y, y_pred):
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_pred)
    )
    return loss
