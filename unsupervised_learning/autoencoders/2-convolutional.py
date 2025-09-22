#!/usr/bin/env python3
"""
Convolutional Autoencoder module
"""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder

    Args:
        input_dims (tuple): dimensions of the model input (H, W, C)
        filters (list): number of filters for each convolutional layer in encoder
        latent_dims (tuple): dimensions of the latent space (H, W, C)

    Returns:
        encoder (keras.Model): encoder model
        decoder (keras.Model): decoder model
        auto (keras.Model): full autoencoder model
    """
    # Encoder
    input_layer = keras.layers.Input(shape=input_dims)
    x = input_layer
    for f in filters:
        x = keras.layers.Conv2D(f, (3, 3), padding='same', activation='relu')(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)

    latent = keras.layers.Conv2D(latent_dims[2], (3, 3), padding='same', activation='relu')(x)

    encoder = keras.Model(inputs=input_layer, outputs=latent, name="encoder")

    # Decoder
    x = latent
    reversed_filters = list(reversed(filters))
    for i, f in enumerate(reversed_filters):
        if i == len(reversed_filters) - 1:
            x = keras.layers.Conv2D(f, (3, 3), padding='valid', activation='relu')(x)
        else:
            x = keras.layers.Conv2D(f, (3, 3), padding='same', activation='relu')(x)
            x = keras.layers.UpSampling2D((2, 2))(x)
    # Add final upsampling to restore original input dimensions
    x = keras.layers.UpSampling2D((2, 2))(x)
    output_layer = keras.layers.Conv2D(input_dims[2], (3, 3),
                                       padding='same', activation='sigmoid')(x)

    decoder_input = keras.layers.Input(shape=latent_dims)
    x_dec = decoder_input
    for i, f in enumerate(reversed_filters):
        if i == len(reversed_filters) - 1:
            x_dec = keras.layers.Conv2D(f, (3, 3), padding='valid', activation='relu')(x_dec)
        else:
            x_dec = keras.layers.Conv2D(f, (3, 3), padding='same', activation='relu')(x_dec)
            x_dec = keras.layers.UpSampling2D((2, 2))(x_dec)
    x_dec = keras.layers.UpSampling2D((2, 2))(x_dec)
    decoder_output = keras.layers.Conv2D(input_dims[2], (3, 3),
                                         padding='same', activation='sigmoid')(x_dec)

    decoder = keras.Model(inputs=decoder_input, outputs=decoder_output, name="decoder")

    # Full autoencoder
    auto = keras.Model(inputs=input_layer, outputs=output_layer, name="conv_autoencoder")
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
