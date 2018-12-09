#!/usr/bin/env python

import sys
from pathlib import Path
import itertools
from datetime import datetime

import numpy
from matplotlib import pyplot as plt

import keras
from keras.models import Model
from keras.optimizers import Adadelta, SGD
from keras.activations import sigmoid, softmax
from keras.layers import Dense, Activation, LSTM, Input, GRU, Bidirectional, Concatenate, Lambda
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.callbacks import TensorBoard

from keras import backend as K

from hparams import hparams

import dataset


timestamp = datetime.now


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def labels_to_text(labels):
    ret = []
    for c in labels:
        if c == -1 or c == len(dataset.SEGMENTS):  # CTC Blank
            ret.append('')
        else:
            ret.append(dataset.SEGMENTS[c])
    return ret


def decode_batch(word_batch, test_func):
    ret = []
    for i in range(len(word_batch)):
        item, target = word_batch[i]
        try:
            out = test_func(item)
            for t, output in zip(target, out):
                actual = [k for k, g in itertools.groupby(numpy.argmax(t, 1))]
                out_best = [k for k, g in itertools.groupby(numpy.argmax(output, 1))]
                ret.append((labels_to_text(actual),
                            labels_to_text(out_best)))
        except ValueError:
            out = test_func(item[0])
            for t, l, output in zip(item[1], item[3], out):
                actual = [int(i) for i in t[:l]]
                out_best = [k for k, g in itertools.groupby(numpy.argmax(output, 1))]
                ret.append((labels_to_text(actual),
                            labels_to_text(out_best)))
    return ret

# Define all inputs
inputs = Input(name='spectrograms',
               shape=(None, hparams["n_spectrogram"]))
labels = Input(name='target_labels',
               shape=[hparams["max_string_length"]], dtype='int64')
input_length = Input(name='len_spectrograms',
                     shape=[1], dtype='int64')
label_length = Input(name='len_target_labels',
                     shape=[1], dtype='int64')

# Construct the core model: Stack some LSTM layers
connector = inputs

for l in hparams["n_lstm_hidden"]:
    lstmf, lstmb = Bidirectional(
        GRU(
            units=l,
            dropout=0.1,
            return_sequences=True,
        ), merge_mode=None)(connector)

    connector = keras.layers.Concatenate(axis=-1)([lstmf, lstmb])

output = Dense(
    name='framewise_labels',
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

model.summary()

# Stick connectionist temporal classification on the end of the core model
paths = K.function(
    [inputs, input_length],
    K.ctc_decode(output, input_length[0], greedy=True, top_paths=4)[0])

loss_out = Lambda(
    ctc_lambda_func, output_shape=(1,),
    name='ctc')([output, labels, input_length, label_length])

ctc_model = Model(
    inputs=[inputs, labels, input_length, label_length],
    outputs=[loss_out])
ctc_model.compile(
    loss={'ctc': lambda y_true, y_pred: y_pred},
    optimizer=SGD(
        lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5))

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

string_data = dataset.ToStringSequence(batch_size=3, files=training)

# Make a TensorBoard
log_dir = "log{:}".format(timestamp())
tensorboard = TensorBoard(log_dir=log_dir)

# Start training, first with time aligned data, then with pure output sequences
old_e = 0
for e in range(0, 5000, 2):
    # For the purpose of this training sequence, use growing chopped-up parts
    # of the actual string sequences. There is likely a better way to do it,
    # but with callbacks, this looked really strange.
    string_data = dataset.ChoppedStringSequence(
        chunk_size=30+e//4, batch_size=3, files=training)
    ctc_model.fit_generator(
        string_data, epochs=e, initial_epoch=old_e,
        callbacks=[tensorboard])
    old_e = e

    # Do some visual validation
    plt.figure()
    j = 1
    for i in range(4):
        (xs, labels, l_x, l_labels), y = string_data[i]
        for x, ys, target, l, lx in zip(
                xs, paths([xs, [l_x]])[0], labels, l_labels, l_x):
            plt.subplot(2, 7, j); j += 1
            target = ''.join(labels_to_text(target[:l]))
            pred = ''.join(i or '_' for i in labels_to_text(ys[:l]))
            d = model.predict([[x]])[0]
            plt.imshow(d.T, aspect='auto')
            plt.yticks(ticks=range(len(dataset.SEGMENTS)+1),
                       labels=dataset.SEGMENTS + ["ε"])
            # plt.imshow(x[:lx].T[::-1], vmin=-20, vmax=0,
            #           aspect='auto')
            # plt.axis('off')
            # plt.xlabel(target)
            plt.text(0, 0, target, horizontalalignment='left', verticalalignment='top')
            plt.text(0, 4, pred, horizontalalignment='left', verticalalignment='top')
    plt.savefig("{:}/prediction-{:d}.pdf".format(log_dir, e))
    plt.close()
