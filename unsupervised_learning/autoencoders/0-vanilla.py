#!/usr/bin/env python3
"""
Vanilla Autoencoder module
"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a vanilla autoencoder

    Args:
        input_dims (int): dimensions of the model input
        hidden_layers (list): number of nodes for each hidden layer in the encoder
        latent_dims (int): dimensions of the latent space representation

    Returns:
        encoder (keras.Model): the encoder model
        decoder (keras.Model): the decoder model
        auto (keras.Model): the full autoencoder model
    """
    # Input layer
    input_layer = keras.layers.Input(shape=(input_dims,))

    # Encoder
    x = input_layer
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)
    latent = keras.layers.Dense(latent_dims, activation='relu')(x)

    # Decoder
    x = latent
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    output_layer = keras.layers.Dense(input_dims, activation='sigmoid')(x)

    # Build encoder model
    encoder = keras.Model(inputs=input_layer, outputs=latent, name="encoder")

    # Build decoder model
    decoder_input = keras.layers.Input(shape=(latent_dims,))
    x_dec = decoder_input
    for nodes in reversed(hidden_layers):
        x_dec = keras.layers.Dense(nodes, activation='relu')(x_dec)
    decoder_output = keras.layers.Dense(input_dims, activation='sigmoid')(x_dec)
    decoder = keras.Model(inputs=decoder_input, outputs=decoder_output, name="decoder")

    # Build full autoencoder
    auto = keras.Model(inputs=input_layer, outputs=output_layer,
                       name="vanilla_autoencoder")
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
