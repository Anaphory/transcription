#!/usr/bin/env python

import sys
from pathlib import Path
import itertools
from datetime import datetime

import numpy
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import keras
from keras.models import Model
from keras.optimizers import Adadelta, SGD
from keras.activations import sigmoid, softmax
from keras.layers import Dense, Activation, LSTM, Input, GRU, Bidirectional, Concatenate, Lambda, SimpleRNN
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.callbacks import TensorBoard

from keras import backend as K

from hparams import hparams

import dataset


def timestamp():
    return datetime.now().isoformat()


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred[:, hparams["chop"]:, :], input_length - hparams["chop"], label_length)

def labels_to_text(labels, index=0):
    ret = []
    for c in labels:
        if c == -1 or c == len(dataset.features[index]):  # CTC Blank
            ret.append('.')
        else:
            ret.append(list(dataset.features[index])[c])
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
labels = [Input(#name='target_labels',
                shape=[237], dtype='int64')
          for _ in dataset.features]
input_length = Input(name='len_spectrograms',
                     shape=[1], dtype='int64')
label_length = Input(name='len_target_labels',
                     shape=[1], dtype='int64')

# Construct the core model: Stack some LSTM layers
connector = inputs

for l in hparams["n_lstm_hidden"]:
    lstmf, lstmb = Bidirectional(
        LSTM(
            units=l,
            dropout=0.05,
            return_sequences=True,
        ), merge_mode=None)(connector)

    connector = keras.layers.Concatenate(axis=-1)([lstmf, lstmb])

outputs = [Dense(
    # name='framewise_labels',
    units=len(f)+1,
    activation=softmax)(connector)
          for f in dataset.features]

# Compile the core model
model = Model(
    inputs=[inputs],
    outputs=outputs)
model.compile(
    optimizer=Adadelta(),
    loss=categorical_crossentropy,
    metrics=[categorical_accuracy])

model.summary()

# Stick connectionist temporal classification on the end of the core model
paths = [K.function(
    [inputs, input_length],
    K.ctc_decode(output, input_length[..., 0], greedy=True)[0])
         for output in outputs]

loss_out = [Lambda(
    ctc_lambda_func, output_shape=(1,),
    #name='ctc'
)([output, label, input_length, label_length])
            for output, label in zip(outputs, labels)]

ctc_model = Model(
    inputs=[inputs] + labels + [input_length, label_length],
    outputs=loss_out)
ctc_model.compile(
    loss=[lambda y_true, y_pred: y_pred for _ in dataset.features],
    optimizer=SGD(
        lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5))


# Make a TensorBoard
log_dir = "log{:}".format(timestamp())
tensorboard = TensorBoard(log_dir=log_dir)

# Start learning on the whole dataset

# Prepare the data
data_files = [f.with_suffix(".txt") for f in dataset.DATA_PATH.glob("**/*.txt")
              if f.with_suffix(".txt").exists()
              if any([f.with_suffix(audio).exists() for audio in [".ogg", ".oga", ".mp3", ".wav"]])]

# Shuffle the list of files
data_files.sort(key=lambda x: numpy.random.random())
# Inverse floor, in order to get the ceiling operation, to make sure that at least one entry is in the validation set.
n_test = -int(-0.1 * len(data_files))
# assert len(data_files) > 2 * n_test

N = 50000
data_files = data_files[:N]
training = data_files[2 * n_test:]
test = data_files[:n_test]
validation = data_files[n_test:2 * n_test]
transcribed_data = dataset.TranscribedSequence(
    batch_size=10, files=training)
validation_data = dataset.TranscribedSequence(
    batch_size=10, files=validation)
test_data = dataset.TranscribedSequence(
    batch_size=10, files=test)

k = len(paths) + 1

# Start training, first with time aligned data, then with pure output sequences
old_e = 0
for e in range(0, 50000, 20):
    # For the purpose of this training sequence, use growing chopped-up parts
    # of the actual string sequences. There is likely a better way to do it,
    # but with callbacks, this looked really strange.
    ctc_model.fit_generator(
        transcribed_data, epochs=e, initial_epoch=old_e,
        shuffle=True,
        callbacks=[tensorboard])
    old_e = e

    # Do some visual validation
    plt.figure(figsize=(21, 30))
    j = 1

    index = min(23, len(transcribed_data))
    x, y = transcribed_data[index]
    _, transcription = zip(*transcribed_data.human_readable(index))
    xs, labels, l_x, l_labels = x[0], x[1:-2], x[-2], x[-1]
    for t, (x, lx) in enumerate(zip(
            xs, l_x)):
        if j >= 7 * k:
            break

        plt.subplot(7, k, j); j += 1
        plt.imshow(x.T, aspect='auto')

        plt.text(1, 4, transcription[t], horizontalalignment='left', verticalalignment='top')

        for p, predictor in enumerate(paths):
            plt.subplot(7, k, j); j += 1
            pred = ''.join(i or '.' for i in labels_to_text(predictor([[x], [[lx]]])[0][0], p))
            d = model.predict([[x]])[p][0]


            plt.imshow(d.T, aspect='auto')
            plt.yticks(ticks=list(dataset.features[p].values()),
                       labels=list(dataset.features[p])+["ε"])

            plt.text(1, 1, ''.join(labels_to_text(labels[p][t], p)), horizontalalignment='left', verticalalignment='top')
            plt.text(1, 4, pred, horizontalalignment='left', verticalalignment='top')
    plt.savefig("{:}/prediction-{:09d}.pdf".format(log_dir, e))
    plt.close()
