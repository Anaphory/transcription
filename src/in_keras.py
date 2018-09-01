#!/usr/bin/env python

import numpy

import keras
from keras.models import Model
from keras.optimizers import Adadelta
from keras.losses import mean_squared_logarithmic_error
from keras.layers import Dense, Activation, LSTM, Input

import tensorflow as tf

import dataset
from hparams import hparams
from phonetic_features import N_FEATURES
from spectrogram_to_sound import stft, griffin_lim

# Set hyperparameters

N_SPECTROGRAM = 513
N_LSTM_HIDDEN = 129
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

model = Model(inputs=inputs, outputs=[spectrogram_ahead_output, spectrogram_behind_output])

model.compile(
    optimizer=Adadelta(),
    loss=[mean_squared_logarithmic_error, mean_squared_logarithmic_error],
    loss_weights=[1., 1.])

def spectrograms_and_shifted():
    empty_windows = numpy.zeros((prediction_length_windows, N_SPECTROGRAM))
    b = lambda d: d.reshape((1, -1, N_SPECTROGRAM))

    for waveform, segments in dataset.wavfile_with_textgrid():
        with tf.Session() as sess:
            spectrogram = sess.run(tf.abs(stft(waveform)))
        yield (
            b(numpy.vstack((
                empty_windows,
                spectrogram,
                empty_windows))),
            [b(numpy.vstack((
                empty_windows,
                spectrogram,
                empty_windows))),
            b(numpy.vstack((
                empty_windows,
                spectrogram,
                empty_windows)))])


for i in range(40):
    model.fit_generator(spectrograms_and_shifted(),
                        steps_per_epoch=10)
