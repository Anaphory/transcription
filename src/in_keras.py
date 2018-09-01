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

from hparams import hparams as p
from phonetic_features import features, N_FEATURES

inputs = Input(shape=(None, p["n_spectrogram"]))

lstm_hidden = LSTM(
    input_shape=(None, p["n_spectrogram"]),
    units=p["n_lstm_hidden"],
    return_sequences=True, # We want the entire sequence, not do something like seq2seq
    )(inputs)

spectrogram_ahead_output = Dense(
    units=p["n_spectrogram"],
    name="ahead_spectrogram")(lstm_hidden)

spectrogram_behind_output = Dense(
    units=p["n_spectrogram"],
    name="behind_spectrogram")(lstm_hidden)

hidden = Dense(p["n_features_hidden"], activation=sigmoid)(lstm_hidden)

feature_outputs = [Dense(2, activation=sigmoid, name=feature)(hidden)
                    for feature in features]

model = Model(
    inputs=inputs,
    outputs=[spectrogram_ahead_output, spectrogram_behind_output])
model.compile(
    optimizer=Adadelta(),
    loss=[mean_squared_error, mean_squared_error],
    loss_weights=[1., 1.])

feature_model = Model(
    inputs=inputs,
    outputs=feature_outputs)
feature_model.compile(
    optimizer=Adadelta(),
    loss=binary_crossentropy)


data = dataset.ShiftedSpectrogramSequence()
model.fit_generator(data)

for i in range(40):
    data = dataset.SpectrogramFeaturesSequence()
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
    result = feature_model.predict(
        spectrogram.reshape((1, -1, p["n_spectrogram"])))

    plt.subplot(3, 1, 1)
    plt.imshow(fts.T, aspect='auto')
    plt.subplot(3, 1, 2)
    plt.imshow(result[0].T, aspect='auto')
    plt.subplot(3, 1, 3)
    plt.imshow(numpy.log(spectrogram.T), aspect='auto', origin='lower')
    plt.show()
