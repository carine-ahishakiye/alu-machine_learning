#!/usr/bin/env python3
"""
0-vanilla.py
Builds a vanilla autoencoder using Keras.
"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a vanilla autoencoder.

    Args:
        input_dims (int): dimensions of the input
        hidden_layers (list): number of nodes for each hidden layer in the encoder
        latent_dims (int): dimensions of the latent space representation

    Returns:
        encoder (keras.Model): encoder model
        decoder (keras.Model): decoder model
        auto (keras.Model): full autoencoder model
    """
    # ---------------- Encoder ----------------
    input_layer = keras.layers.Input(shape=(input_dims,))
    x = input_layer
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation='relu')(x)
    latent = keras.layers.Dense(latent_dims, activation='relu')(x)

    encoder = keras.Model(input_layer, latent, name='encoder')

    # ---------------- Decoder ----------------
    latent_input = keras.layers.Input(shape=(latent_dims,))
    x = latent_input
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units, activation='relu')(x)
    output_layer = keras.layers.Dense(input_dims, activation='sigmoid')(x)

    decoder = keras.Model(latent_input, output_layer, name='decoder')

    # ---------------- Autoencoder ----------------
    auto_input = input_layer
    encoded = encoder(auto_input)
    decoded = decoder(encoded)
    auto = keras.Model(auto_input, decoded, name='autoencoder')
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    print("Autoencoder network is built properly")

    return encoder, decoder, auto
