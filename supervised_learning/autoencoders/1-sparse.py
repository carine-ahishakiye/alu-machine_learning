#!/usr/bin/env python3
"""Sparse Autoencoder module"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Creates a sparse autoencoder

    Args:
        input_dims (int): dimensions of the model input
        hidden_layers (list): number of nodes for each hidden layer in the encoder
        latent_dims (int): dimensions of the latent space representation
        lambtha (float): L1 regularization parameter on the encoded output

    Returns:
        encoder (tf.keras.Model): the encoder model
        decoder (tf.keras.Model): the decoder model
        auto (tf.keras.Model): the full autoencoder model
    """
    # Input layer
    input_layer = layers.Input(shape=(input_dims,))

    # Encoder
    x = input_layer
    for nodes in hidden_layers:
        x = layers.Dense(nodes, activation='relu')(x)
    latent = layers.Dense(latent_dims, activation='relu',
                          activity_regularizer=regularizers.l1(lambtha))(x)

    # Decoder
    x = latent
    for nodes in reversed(hidden_layers):
        x = layers.Dense(nodes, activation='relu')(x)
    output_layer = layers.Dense(input_dims, activation='sigmoid')(x)

    # Models
    encoder = models.Model(inputs=input_layer, outputs=latent, name="encoder")
    decoder_input = layers.Input(shape=(latent_dims,))
    x_dec = decoder_input
    for nodes in reversed(hidden_layers):
        x_dec = layers.Dense(nodes, activation='relu')(x_dec)
    decoder_output = layers.Dense(input_dims, activation='sigmoid')(x_dec)
    decoder = models.Model(inputs=decoder_input, outputs=decoder_output, name="decoder")

    auto = models.Model(inputs=input_layer, outputs=output_layer, name="sparse_autoencoder")
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
