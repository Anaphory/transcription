#!/usr/bin/env python

import keras
from keras.models import Sequential
from keras.optimizers import Adadelta
from keras.losses import mean_squared_logarithmic_error
from keras.layers import Dense, Activation, LSTM

import tensorflow as tf

import dataset
from phonetic_features import N_FEATURES
from spectrogram_to_sound import stft, griffin_lim
N_SPECTROGRAM = 513
N_LSTM_HIDDEN = 129

model = Sequential()
model.add(
    LSTM(
        input_shape=(None, N_SPECTROGRAM),
        units=N_LSTM_HIDDEN,
        return_sequences=True, # We want the entire sequence, not do something like seq2seq
    ))

model.add(
    Dense(
        units=N_SPECTROGRAM
    ))

model.compile(
    optimizer=Adadelta(),
    loss=mean_squared_logarithmic_error)

def spectrograms_and_shifted():
    for waveform, segments in dataset.wavfile_with_textgrid():
        with tf.Session() as sess:
            spectrogram = sess.run(stft(waveform))
        yield spectrogram[1:].reshape((1, -1, N_SPECTROGRAM)), spectrogram[:-1].reshape((1, -1, N_SPECTROGRAM))

model.fit_generator(spectrograms_and_shifted(),
                    steps_per_epoch=10)
