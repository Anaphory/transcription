#!/usr/bin/env python

from dataset import audio_dataset
from model import lstm_network
import tensorflow as tf

dataset = audio_dataset.padded_batch(5, padded_shapes=[None])

iterator = tf.data.Iterator.from_structure(dataset.output_types,
                                           dataset.output_shapes)

signals = iterator.get_next()

magnitude_spectrograms = tf.abs(tf.contrib.signal.stft(
            signals,
            frame_length=1024, # The normal sample rate is 20000Hz, so this is 20 ms
            # Periodic Hann is the default window
            frame_step=128)) # 6.4 ms

output, loss, ahead, behind = lstm_network(magnitude_spectrograms)

train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

init_op = tf.global_variables_initializer()

training_iterator = iterator.make_initializer(dataset)
validation_iterator = iterator.make_initializer(dataset)

with tf.Session() as sess:
    sess.run(init_op)

    sess.run(training_iterator)
    while True:
        try:
            sess.run(train_op)
        except tf.errors.OutOfRangeError:
            break

    sess.run(validation_iterator)
    while True:
        try:
            ahead_sound, behind_sound = sess.run([ahead, behind])
            # invert stft
        except tf.errors.OutOfRangeError:
            break





