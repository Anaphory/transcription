#!/usr/bin/env python

import sys
from pathlib import Path

import numpy

import keras
from keras.models import Model
from keras.optimizers import Adadelta, SGD
from keras.activations import sigmoid, softmax
from keras.layers import Dense, Activation, LSTM, Input, GRU, Bidirectional, Concatenate, Lambda
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy

from keras import backend as K

from hparams import hparams

import dataset

time_aligned_data = dataset.TimeAlignmentSequence(batch_size=3)
string_data = dataset.ToStringSequence(batch_size=1)

# Model parameters

inputs = Input(shape=(None, hparams["n_spectrogram"]))
labels = Input(shape=[string_data.max_len])
input_length = Input(shape=[1], dtype='int64')
label_length = Input(shape=[1], dtype='int64')

connector = inputs

for l in hparams["n_lstm_hidden"]:
    print(l)
    lstmf, lstmb = Bidirectional(
        LSTM(
            units=l,
            dropout=0.3,
            return_sequences=True,
        ), merge_mode=None)(connector)

    connector = keras.layers.Concatenate(axis=-1)([lstmf, lstmb])

output = Dense(
    units=len(dataset.SEGMENTS)+1,
    activation=softmax)(connector)


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

loss_out = Lambda(
    ctc_lambda_func, output_shape=(1,),
    name='ctc')([output, labels, input_length, label_length])

model = Model(
    inputs=[inputs],
    outputs=[output])
model.compile(
    optimizer=Adadelta(),
    loss=[categorical_crossentropy],
    metrics=[categorical_accuracy])

ctc_model = Model(
    inputs=[inputs, labels, input_length, label_length],
    outputs=[loss_out])
ctc_model.compile(loss={'ctc': lambda y_true, y_pred: y_pred},
                  optimizer=SGD(
                      lr=0.02,
                      decay=1e-6,
                      momentum=0.9,
                      nesterov=True,
                      clipnorm=5))

old_e = 0
for e in range(0, 200, 5):
    if e < 20:
        model.fit_generator(
            time_aligned_data, epochs=e, initial_epoch=old_e)
        old_e = e
        print(model.evaluate_generator(time_aligned_data))
    else:
        ctc_model.fit_generator(
            string_data, epochs=e, initial_epoch=old_e)
        old_e = e
        print(ctc_model.evaluate_generator(string_data))


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
