#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.keras import layers, models

def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a vanilla autoencoder

    Args:
        input_dims (int): dimensions of the input
        hidden_layers (list): number of nodes in each hidden layer of encoder
        latent_dims (int): dimensions of latent space

    Returns:
        encoder (Model): encoder model
        decoder (Model): decoder model
        auto (Model): full autoencoder model
    """
    # Encoder
    input_layer = layers.Input(shape=(input_dims,))
    x = input_layer
    for units in hidden_layers:
        x = layers.Dense(units, activation='relu')(x)
    latent = layers.Dense(latent_dims, activation='relu')(x)
    
    encoder = models.Model(input_layer, latent, name='encoder')
    
    # Decoder
    latent_input = layers.Input(shape=(latent_dims,))
    x = latent_input
    for units in reversed(hidden_layers):
        x = layers.Dense(units, activation='relu')(x)
    output_layer = layers.Dense(input_dims, activation='sigmoid')(x)
    
    decoder = models.Model(latent_input, output_layer, name='decoder')
    
    # Autoencoder
    auto_input = input_layer
    encoded = encoder(auto_input)
    decoded = decoder(encoded)
    auto = models.Model(auto_input, decoded, name='autoencoder')
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    
    return encoder, decoder, auto
