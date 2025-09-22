#!/usr/bin/env python3
"""
Variational Autoencoder
Builds a variational autoencoder using Keras.
"""

import tensorflow.keras as keras
from tensorflow.keras import backend as K


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder (VAE).

    Args:
        input_dims (int): Dimensions of the model input
        hidden_layers (list): Nodes for each hidden layer in encoder
        latent_dims (int): Dimensions of latent space representation

    Returns:
        encoder, decoder, auto (keras.Model): Encoder, decoder, and VAE model
    """

    # Sampling function using reparameterization trick
    def sampling(args):
        mu, log_var = args
        epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
        return mu + K.exp(log_var / 2) * epsilon

    # Encoder
    encoder_input = keras.layers.Input(shape=(input_dims,))
    x = encoder_input
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    mu = keras.layers.Dense(latent_dims, activation=None)(x)
    log_var = keras.layers.Dense(latent_dims, activation=None)(x)

    z = keras.layers.Lambda(sampling, output_shape=(latent_dims,))([mu, log_var])
    encoder = keras.Model(encoder_input, [z, mu, log_var], name='encoder')

    # Decoder
    decoder_input = keras.layers.Input(shape=(latent_dims,))
    x = decoder_input
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)

    decoder_output = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(decoder_input, decoder_output, name='decoder')

    # VAE model
    auto_input = encoder_input
    encoded, mu, log_var = encoder(auto_input)
    decoded = decoder(encoded)
    auto = keras.Model(auto_input, decoded, name='vae')

    # Define VAE loss
    reconstruction_loss = keras.losses.binary_crossentropy(auto_input, decoded)
    reconstruction_loss *= input_dims
    kl_loss = 1 + log_var - K.square(mu) - K.exp(log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)

    auto.add_loss(vae_loss)
    auto.compile(optimizer='adam')

    return encoder, decoder, auto
