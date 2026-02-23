#!/usr/bin/env python3
"""
Convolutional Autoencoder
Builds a convolutional autoencoder using Keras.
"""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder.

    Args:
        input_dims (tuple): Dimensions of the model input (H, W, C)
        filters (list): List of filters for each encoder conv layer
        latent_dims (tuple): Dimensions of the latent space representation

    Returns:
        encoder, decoder, auto (keras.Model): Encoder, decoder, and autoencoder
    """
    # Encoder
    encoder_input = keras.layers.Input(shape=input_dims)
    x = encoder_input

    for f in filters:
        x = keras.layers.Conv2D(f, (3, 3), padding='same', activation='relu')(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)

    latent = keras.layers.Conv2D(latent_dims[2], (3, 3), padding='same',
                                 activation='relu')(x)
    encoder = keras.Model(inputs=encoder_input, outputs=latent, name='encoder')

    # Decoder
    decoder_input = keras.layers.Input(shape=latent_dims)
    x = decoder_input
    reversed_filters = list(reversed(filters))

    for i, f in enumerate(reversed_filters):
        if i == len(reversed_filters) - 1:
            # second-to-last conv: valid padding
            x = keras.layers.Conv2D(f, (3, 3), padding='valid', activation='relu')(x)
        else:
            x = keras.layers.Conv2D(f, (3, 3), padding='same', activation='relu')(x)
            x = keras.layers.UpSampling2D((2, 2))(x)

    # Final upsampling to match input size
    x = keras.layers.UpSampling2D((2, 2))(x)
    decoder_output = keras.layers.Conv2D(input_dims[2], (3, 3), padding='same',
                                         activation='sigmoid')(x)

    decoder = keras.Model(inputs=decoder_input, outputs=decoder_output, name='decoder')

    # Autoencoder
    auto_input = encoder_input
    encoded = encoder(auto_input)
    decoded = decoder(encoded)
    auto = keras.Model(inputs=auto_input, outputs=decoded, name='autoencoder')
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
