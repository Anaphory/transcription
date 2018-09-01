#!/usr/bin/env python

from pathlib import Path

import numpy

import keras
from keras.models import Model
from keras.optimizers import Adadelta
from keras.losses import mean_squared_error, binary_crossentropy
from keras.layers import Dense, Activation, LSTM, Input
from keras.activations import sigmoid

import tensorflow as tf

import dataset
from hparams import hparams
from phonetic_features import N_FEATURES
import phonetic_features
from spectrogram_to_sound import stft, griffin_lim

# Set hyperparameters

N_SPECTROGRAM = 513
N_LSTM_HIDDEN = 129
N_FEATURES_HIDDEN = 30
FRAME_LENGTH_MS = hparams.frame_length_ms
PREDICTION_TIME_AT_LEAST_MS = 20 # If this is not a multiple of frame length, take the next multiple


prediction_length_windows = -(-PREDICTION_TIME_AT_LEAST_MS // FRAME_LENGTH_MS)


inputs = Input(shape=(None, N_SPECTROGRAM))

lstm_hidden = LSTM(
    input_shape=(None, N_SPECTROGRAM),
    units=N_LSTM_HIDDEN,
    return_sequences=True, # We want the entire sequence, not do something like seq2seq
    )(inputs)

spectrogram_ahead_output = Dense(
    units=N_SPECTROGRAM,
    name="ahead_spectrogram")(lstm_hidden)

spectrogram_behind_output = Dense(
    units=N_SPECTROGRAM,
    name="behind_spectrogram")(lstm_hidden)

model = Model(
    inputs=inputs,
    outputs=[spectrogram_ahead_output, spectrogram_behind_output])

model.compile(
    optimizer=Adadelta(),
    loss=[mean_squared_error, mean_squared_error],
    loss_weights=[1., 1.])

def spectrograms_and_shifted():
    p = prediction_length_windows
    # After
    a = lambda d: d[2 * p:].reshape((1, -1, N_SPECTROGRAM))
    # Before
    b = lambda d: d[:-2 * p].reshape((1, -1, N_SPECTROGRAM))
    # Central
    c = lambda d: d[p:-p].reshape((1, -1, N_SPECTROGRAM))

    for waveform, segments in dataset.wavfile_with_textgrid():
        with tf.Session() as sess:
            spectrogram = sess.run(tf.log(tf.abs(stft(waveform)) + 1e-8))
        yield (
            c(spectrogram),
            [b(spectrogram),
             a(spectrogram)])

data = spectrograms_and_shifted()
while True:
    try:
        model.fit_generator(data, steps_per_epoch=10)
    except StopIteration:
        break

hidden = Dense(N_FEATURES_HIDDEN, activation=sigmoid)(lstm_hidden)
feature_outputs = [Dense(2, activation=sigmoid, name=feature)(hidden)
                    for feature in phonetic_features.features]

feature_model = Model(inputs=inputs, outputs=feature_outputs)
feature_model.compile(
    optimizer=Adadelta(),
    loss=binary_crossentropy)

def spectrograms_and_features():
    for waveform, segments in dataset.wavfile_with_textgrid():
        if segments is None:
            continue
        with tf.Session() as sess:
            try:
                spectrogram = sess.run(tf.log(tf.abs(stft(waveform)) + 1e-8))
            except InvalidArgumentError:
                continue
        if len(segments) > len(spectrogram):
            segments = segments[:len(spectrogram)]
        elif len(segments) < len(spectrogram):
            spectrogram = spectrogram[:len(segments)]

        feature_values = [
            numpy.vstack(
                (segments[:, feature_id],
                1 - segments[:, feature_id])).reshape((1, -1, 2))
            for feature_id in range(N_FEATURES)]
        yield spectrogram.reshape((1, -1, N_SPECTROGRAM)), feature_values

for i in range(40):
    data = spectrograms_and_features()
    while True:
        try:
            history = feature_model.fit_generator(data, steps_per_epoch=10)
        except StopIteration:
            break

# Example prediction
from matplotlib import pyplot as plt

files = [Path(__file__).parent.parent / "data" / "Futbol.ogg"]
for waveform, fts in dataset.wavfile_with_textgrid(files):
    with tf.Session() as sess:
        spectrogram = sess.run(tf.log(tf.abs(stft(waveform)) + 1e-8))
    result = feature_model.predict(spectrogram.reshape((1, -1, N_SPECTROGRAM)))

    plt.subplot(3, 1, 1)
    plt.imshow(fts.T, aspect='auto')
    plt.subplot(3, 1, 2)
    plt.imshow(result[0].T, aspect='auto')
    plt.subplot(3, 1, 3)
    plt.imshow(numpy.log(spectrogram.T), aspect='auto', origin='lower')
    plt.show()
