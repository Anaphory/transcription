#!/usr/bin/env python

import sys
from pathlib import Path
import itertools

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


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def labels_to_text(labels):
    ret = []
    for c in labels:
        if c == len(dataset.SEGMENTS):  # CTC Blank
            ret.append("")
        else:
            ret.append(dataset.SEGMENTS[c])
    return ret


# Define all inputs
inputs = Input(shape=(None, hparams["n_spectrogram"]))
labels = Input(shape=[hparams["max_string_length"]])
input_length = Input(shape=[1], dtype='int64')
label_length = Input(shape=[1], dtype='int64')

# Construct the core model: Stack some LSTM layers
connector = inputs
for l in hparams["n_lstm_hidden"]:
    print(l)
    lstmf, lstmb = Bidirectional(
        LSTM(
            units=l,
            dropout=0.1,
            return_sequences=True,
        ), merge_mode=None)(connector)

    connector = keras.layers.Concatenate(axis=-1)([lstmf, lstmb])

output = Dense(
    units=len(dataset.SEGMENTS)+1,
    activation=softmax)(connector)

# Compile the core model
model = Model(
    inputs=[inputs],
    outputs=[output])
model.compile(
    optimizer=Adadelta(),
    loss=[categorical_crossentropy],
    metrics=[categorical_accuracy])


def decode_batch(word_batch, test_func=model.predict):
    ret = []
    for i in range(len(word_batch)):
        item, target = word_batch[i]
        out = test_func(item)
        for t, output in zip(target, out):
            actual = [k for k, g in itertools.groupby(numpy.argmax(t, 1))]
            out_best = [k for k, g in itertools.groupby(numpy.argmax(output, 1))]
            ret.append((labels_to_text(actual),
                        labels_to_text(out_best)))
    return ret


# Stick connectionist temporal classification on the end of the core model
loss_out = Lambda(
    ctc_lambda_func, output_shape=(1,),
    name='ctc')([output, labels, input_length, label_length])

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

# Prepare the data

data_files = [f for f in dataset.DATA_PATH.glob("*.TextGrid")]

# Shuffle the list of files
data_files.sort(key=lambda x: numpy.random.random())
# Inverse floor, in order to get the ceiling operation, to make sure that at least one entry is in the validation set.
n_test = -int(-0.1 * len(data_files))
assert len(data_files) > 2 * n_test

training = data_files[2 * n_test:]
test = data_files[:n_test]
validation = data_files[n_test:2 * n_test]
time_aligned_data = dataset.TimeAlignmentSequence(
    batch_size=3, files=training)
validation_data = dataset.TimeAlignmentSequence(
    batch_size=3, files=validation)
test_data = dataset.TimeAlignmentSequence(
    batch_size=3, files=test)

string_data = dataset.ToStringSequence(batch_size=2, files=training)

# Start training, first with time aligned data, then with pure output sequences
old_e = 0
for e in range(0, 5000, 8):
    string_data = dataset.ChoppedStringSequence(
        chunk=5+e//4, batch_size=2, files=training)
    if e < 20:
        model.fit_generator(
            time_aligned_data, epochs=e, initial_epoch=old_e,
            validation_data=validation_data)
        old_e = e
    else:
        ctc_model.fit_generator(
            string_data, epochs=e, initial_epoch=old_e)
        old_e = e
    for x, y in decode_batch(validation_data):
        print(''.join(x), "\t", ''.join(y))


# Example prediction
for file in data_files:
    x, y = dataset.TimeAlignmentSequence(files=[file])[0]
    pred = model.predict(x)

    before = None, None
    lines = [[], []]
    for segment, expected in zip(pred.argmax(2)[0],
                                y.argmax(2)[0]):
        try:
            segment = dataset.SEGMENTS[segment]
        except IndexError: # NULL segment
            segment = ''
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
