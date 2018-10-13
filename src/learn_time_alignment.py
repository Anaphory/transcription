#!/usr/bin/env python

from pathlib import Path

import numpy

import keras
from keras.models import Model
from keras.optimizers import Adadelta
from keras.activations import sigmoid, softmax
from keras.layers import Dense, Activation, LSTM, Input, GRU, Bidirectional, Concatenate
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy

from keras import backend as K

from hparams import hparams
import dataset

# Model parameters

inputs = Input(shape=(None, hparams["n_spectrogram"]))

lstmf, lstmb = Bidirectional(
    LSTM(
        input_shape=(None, hparams["n_spectrogram"]),
        units=hparams["n_lstm_hidden"],
        dropout=0.2,
        return_sequences=True,
    ), merge_mode=None)(inputs)

merger = keras.layers.Concatenate(axis=-1)([lstmf, lstmb])

lstmf, lstmb = Bidirectional(
    LSTM(
        input_shape=(None, hparams["n_spectrogram"]),
        units=hparams["n_lstm_hidden"],
        dropout=0.2,
        return_sequences=True,
    ), merge_mode=None)(merger)

merger = keras.layers.Concatenate(axis=-1)([lstmf, lstmb])


lstm = Bidirectional(
    LSTM(
        input_shape=(None, hparams["n_spectrogram"]),
        units=hparams["n_lstm_hidden"],
        dropout=0.2,
        return_sequences=True,
    ), merge_mode='sum')(merger)

output = Dense(
    units=len(dataset.SEGMENTS),
    activation=softmax)(lstm)

model = Model(
    inputs=inputs,
    outputs=[output])
model.compile(
    optimizer=Adadelta(),
    loss=[categorical_crossentropy],
    metrics=[categorical_accuracy])

time_aligned_data = dataset.TimeAlignmentSequence(batch_size=3)

print(model.evaluate_generator(time_aligned_data))
for _ in range(100):
    model.fit_generator(time_aligned_data)
    print(model.evaluate_generator(time_aligned_data))
