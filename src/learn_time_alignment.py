#!/usr/bin/env python

import sys
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

connector = inputs

for l in hparams["n_lstm_hidden"]:
    lstmf, lstmb = Bidirectional(
        LSTM(
            units=l,
            dropout=0.2,
            return_sequences=True,
        ), merge_mode=None)(connector)

    connector = keras.layers.Concatenate(axis=-1)([lstmf, lstmb])

output = Dense(
    units=len(dataset.SEGMENTS),
    activation=softmax)(connector)

model = Model(
    inputs=inputs,
    outputs=[output])
model.compile(
    optimizer=Adadelta(),
    loss=[categorical_crossentropy],
    metrics=[categorical_accuracy])

time_aligned_data = dataset.TimeAlignmentSequence(batch_size=3)

old_e = 0
for e in range(0, 251, 5):
    model.fit_generator(
        time_aligned_data, epochs=e, initial_epoch=old_e)
    old_e = e
    print(model.evaluate_generator(time_aligned_data))

# Example prediction
for file in time_aligned_data.files:
    x, y = dataset.TimeAlignmentSequence(files=[file])[0]
    pred = model.predict(x)

    before = None, None
    lines = [[], []]
    for segment, expected in zip(pred.argmax(2)[0],
                                y.argmax(2)[0]):
        segment = dataset.SEGMENTS[segment]
        expected = dataset.SEGMENTS[expected]
        if (segment, expected) == before:
            pass
        elif segment != before[0]:
            if expected == before[1]:
                lines[0].append(segment)
                lines[1].append(" "*len(segment))
            else:
                if len(segment) < len(expected):
                    lines[0].append(segment + " " * (len(expected) - len(segment)))
                    lines[1].append(expected)
                else:
                    lines[0].append(segment)
                    lines[1].append(expected + " " * (len(segment) - len(expected)))
        else:
            lines[0].append(" "*len(expected))
            lines[1].append(expected)
        before = (segment, expected)

    print("".join(lines[0]))
    print("".join(lines[1]))
    print()
